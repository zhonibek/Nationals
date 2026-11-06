'use strict';

// A task on top of the tactical environment: one task spec (from the text parser, Nemotron or the task sampler)
// decides which candidate macro-actions are allowed, when scoring stops and parking starts, and when the task is
// over. The app's task runner and the multi-task trainer both play through this class, so a trained policy meets
// exactly the task rules it learned under.
//
// Feature set v4 = the v3 candidate features plus a task block (what the task asks for and how much of it is
// left), so one network can serve very different tasks. Each step is also scored by the task's goal; every number
// comes from the Override rules engine's score. Nothing here trains.

const OverrideGame = require('../simulator/override');
const {TacticalEnvironment, FEATURE_COUNT, FEATURE_COUNT_V3} = require('./tactical_environment');

const MATCH_SECONDS = OverrideGame.constants.MATCH_SECONDS;
const DRIVER_SECONDS = OverrideGame.constants.DRIVER_SECONDS;
// With an open-ended scoring task followed by parking, scoring stops this many seconds before the end of the match.
const PARK_RESERVE_SECONDS = 12;
const MAX_LOOP_STEPS = 200;
const SCORING_KINDS = new Set(['score-pin', 'score-cup', 'load-score-pin', 'load-score-cup']);
const TASK_FEATURE_COUNT = 22;
const FEATURE_COUNT_V4 = FEATURE_COUNT_V3 + TASK_FEATURE_COUNT;
const TASK_FEATURE_SETS = Object.freeze(['v2', 'v3', 'v4']);
// Task reward (feature set v4 training). Open-ended play keeps the score-margin reward; a target task pays for
// progress and completion and charges game time, so finishing sooner is better; parking pays per robot parked.
const TASK_REWARD = Object.freeze({timeCostPerSecond: 0.2, perPlacement: 4, perPoint: 0.5, completion: 10, failure: 10,
  perParkedRobot: 5, rejected: -2});

function actionParts(id) {
  const [kind, robotId, target, goal] = id.split(':');
  return {kind, robotId, target: target ?? null, goal: goal ?? null};
}

// Whether a candidate macro-action fits the task in the current phase ('score' or 'park').
function allows(spec, phase, id, alliance) {
  const {kind, goal} = actionParts(id);
  if (phase === 'park') return kind === 'park';
  if (spec.mode === 'play') return true;
  if (!SCORING_KINDS.has(kind)) return false;
  if (spec.objects !== 'any' && !kind.endsWith(spec.objects)) return false;
  if (spec.source === 'loader' && !kind.startsWith('load-score-')) return false;
  if (spec.source === 'field' && kind.startsWith('load-score-')) return false;
  if (spec.goals === 'neutral' && !goal?.startsWith('g-neutral-')) return false;
  if (spec.goals === 'alliance' && !goal?.startsWith(`g-${alliance}-`)) return false;
  if (spec.goals === 'tall' && goal !== 'g-neutral-tall') return false;
  return true;
}

const SPEC_VALUES = Object.freeze({mode: ['play', 'score', 'park'], objects: ['any', 'pin', 'cup'],
  source: ['any', 'loader', 'field'], goals: ['any', 'neutral', 'alliance', 'tall']});

// The allowlisted shape every task spec must have, whoever produced it (parser, Nemotron or sampler).
function validateSpec(spec) {
  if (!spec || typeof spec !== 'object') throw Error('A task spec is required');
  for (const [key, values] of Object.entries(SPEC_VALUES)) {
    if (!values.includes(spec[key])) throw Error(`Task ${key} must be one of ${values.join(', ')}`);
  }
  const integer = (value, minimum, maximum, label) => {
    if (value !== null && (!Number.isInteger(value) || value < minimum || value > maximum)) {
      throw Error(`${label} must be from ${minimum} to ${maximum}`);
    }
  };
  integer(spec.count, 1, 30, 'The number of scoring actions');
  integer(spec.points, 1, 500, 'The points target');
  integer(spec.seconds, 1, DRIVER_SECONDS, 'The time limit in seconds');
  if (typeof spec.park !== 'boolean') throw Error('Task park must be true or false');
  if (spec.mode === 'park' && !spec.park) throw Error('A park task must park');
  if (spec.mode !== 'score' && (spec.count !== null || spec.points !== null)) throw Error('Only a scoring task has count or points targets');
  if (spec.mode === 'play' && spec.park) throw Error('A play task does not park');
  return spec;
}

const oneHot = (value, values) => values.map(candidate => (candidate === value ? 1 : 0));

class TaskEnvironment {
  // `environment` is a TacticalEnvironment whose feature set the policy reads: v2 or v3 candidates are passed through
  // unchanged (the original single-goal policies); for v4 pass a v3 environment and featureSet 'v4'.
  constructor({spec, environment, featureSet} = {}) {
    this.spec = validateSpec(spec);
    this.environment = environment ?? new TacticalEnvironment({featureSet: featureSet === 'v4' ? 'v3' : featureSet ?? 'v2'});
    this.featureSet = featureSet ?? this.environment.featureSet;
    if (!TASK_FEATURE_SETS.includes(this.featureSet)) throw Error(`Unknown tactical feature set ${this.featureSet}`);
    const base = this.featureSet === 'v4' ? 'v3' : this.featureSet;
    if (this.environment.featureSet !== base) {
      throw Error(`Feature set ${this.featureSet} needs a ${base} tactical environment, not ${this.environment.featureSet}`);
    }
  }

  reset(seed, alliance = 'red') {
    this.observation = this.environment.reset(seed, alliance);
    this.alliance = alliance;
    this.seed = seed;
    this.clockStart = this.observation.clock;
    this.ownStart = this.observation.score[alliance];
    this.marginStart = this.observation.score.margin;
    this.parked = new Set();
    this.banned = new Set();
    this.phase = this.spec.mode === 'park' ? 'park' : 'score';
    this.scoringActions = 0;
    this.rejectedActions = 0;
    this.stopReason = null;
    this.loopSteps = 0;
    this.taskReturn = 0;
    this.log = [];
    this.done = false;
    this.candidates = [];
    this._advance();
    return this.observe();
  }

  get targeted() { return this.spec.count !== null || this.spec.points !== null; }

  gained() { return this.observation.score[this.alliance] - this.ownStart; }

  targetsMet() {
    return (this.spec.count === null || this.scoringActions >= this.spec.count) &&
      (this.spec.points === null || this.gained() >= this.spec.points);
  }

  // Switches to parking when the task parks afterwards; otherwise ends the task with `reason`. True when it ended.
  _finishScoring(reason) {
    if (this.spec.park) { this.phase = 'park'; this.banned = new Set(); return false; }
    this.stopReason = reason;
    return true;
  }

  _end(reason) {
    if (reason !== undefined) this.stopReason ??= reason;
    this.done = true;
    this.candidates = [];
  }

  // The checks before each decision, as the original runTask loop made them, then the allowed candidates.
  _advance() {
    const spec = this.spec;
    for (;;) {
      if (this.loopSteps >= MAX_LOOP_STEPS) { this._end(); return; }
      this.loopSteps += 1;
      const observation = this.observation;
      if (observation.done) { this._end(this.phase === 'park' && this.parked.size ? 'task-complete' : 'match-over'); return; }
      if (this.phase === 'score') {
        const timeUp = spec.seconds !== null && observation.clock - this.clockStart >= spec.seconds - 1e-9;
        if (this.targeted && this.targetsMet() && this._finishScoring('task-complete')) { this._end(); return; }
        if (this.phase === 'score' && timeUp && this._finishScoring('time-limit')) { this._end(); return; }
        if (this.phase === 'score' && !this.targeted && spec.park && observation.clock >= MATCH_SECONDS - PARK_RESERVE_SECONDS) {
          this._finishScoring();
        }
      }
      const candidates = observation.actions.filter(action => !this.banned.has(action.id) &&
        allows(spec, this.phase, action.id, this.alliance) && !(this.phase === 'park' && this.parked.has(actionParts(action.id).robotId)));
      if (!candidates.length) {
        if (this.phase === 'score' && !this._finishScoring('no-allowed-action')) continue;
        if (this.phase === 'park') this.stopReason = this.parked.size ? 'task-complete' : 'no-allowed-action';
        this._end();
        return;
      }
      this.candidates = candidates;
      return;
    }
  }

  // What the task asks for and how much of it is left; the same block is appended to every candidate (v4).
  taskFeatures() {
    const spec = this.spec;
    const clock = this.observation.clock;
    const countLeft = spec.count === null ? 0 : Math.max(0, spec.count - this.scoringActions) / 10;
    const pointsLeft = spec.points === null ? 0 : Math.max(0, spec.points - this.gained()) / 100;
    const secondsLeft = spec.seconds === null ? 0 : Math.max(0, spec.seconds - (clock - this.clockStart)) / DRIVER_SECONDS;
    return [...oneHot(spec.mode, SPEC_VALUES.mode), ...oneHot(spec.objects, SPEC_VALUES.objects),
      ...oneHot(spec.source, SPEC_VALUES.source), ...oneHot(spec.goals, SPEC_VALUES.goals),
      spec.count === null ? 0 : 1, Math.min(1, countLeft), spec.points === null ? 0 : 1, Math.min(1, pointsLeft),
      spec.seconds === null ? 0 : 1, Math.min(1, secondsLeft), spec.park ? 1 : 0, this.phase === 'park' ? 1 : 0,
      this.parked.size / 2];
  }

  observe() {
    const task = this.taskFeatures();
    const actions = this.done ? [] : this.candidates.map(action => ({id: action.id,
      features: this.featureSet === 'v4' ? [...action.features, ...task] : action.features}));
    return {...this.observation, actions, done: this.done, featureSet: this.featureSet,
      task: {spec: this.spec, phase: this.phase, scoringActions: this.scoringActions, parkedRobots: [...this.parked],
        ownPointsGained: this.gained(), stopReason: this.done ? this.stopReason ?? 'decision-cap' : null,
        achieved: this.done ? this.achieved() : null, taskReturn: this.taskReturn,
        secondsUsed: Number((this.observation.clock - this.clockStart).toFixed(3))}};
  }

  // The task's score for one step: see TASK_REWARD. `result` is the tactical environment's step result.
  _reward(result, {counted, parkedNow, ownBefore}) {
    const spec = this.spec;
    // Open play keeps the original single-goal reward (score-margin change, -2 for a rejected action).
    if (spec.mode === 'play') return result.reward;
    if (result.rejectedAction) return TASK_REWARD.rejected - TASK_REWARD.timeCostPerSecond * result.durationSeconds;
    if (this.phase === 'park') {
      return (parkedNow ? TASK_REWARD.perParkedRobot : 0) - TASK_REWARD.timeCostPerSecond * result.durationSeconds;
    }
    const ownGain = result.observation.score[this.alliance] - ownBefore;
    if (!this.targeted) return ownGain;
    let progress = 0;
    if (spec.count !== null && counted && this.scoringActions <= spec.count) progress += TASK_REWARD.perPlacement;
    if (spec.points !== null) {
      const before = ownBefore - this.ownStart;
      progress += TASK_REWARD.perPoint * Math.max(0, Math.min(ownGain, spec.points - before));
    }
    return progress - TASK_REWARD.timeCostPerSecond * result.durationSeconds;
  }

  step(actionId) {
    if (this.done) throw Error('The task is already complete');
    if (!this.candidates.some(action => action.id === actionId)) throw Error('Action is not allowed by the task right now');
    const spec = this.spec;
    const clock = this.observation.clock;
    const ownBefore = this.observation.score[this.alliance];
    const wasTargetsMet = this.targeted && this.targetsMet();
    const candidateCount = this.candidates.length;
    const phase = this.phase;
    const result = this.environment.step(actionId);
    const parts = actionParts(actionId);
    let counted = false;
    let parkedNow = false;
    if (result.rejectedAction) { this.rejectedActions += 1; this.banned.add(actionId); }
    else {
      this.banned = new Set();
      if (SCORING_KINDS.has(parts.kind)) { this.scoringActions += 1; counted = true; }
      if (parts.kind === 'park') { this.parked.add(parts.robotId); parkedNow = true; }
    }
    this.observation = result.observation;
    let reward = this._reward(result, {counted, parkedNow, ownBefore});
    if (this.targeted && !wasTargetsMet && this.targetsMet() && phase === 'score') reward += TASK_REWARD.completion;
    this._advance();
    if (this.done) {
      if (this.targeted && !this.targetsMet()) reward -= TASK_REWARD.failure;
      if (spec.park && !this.parked.size) reward -= TASK_REWARD.failure;
    }
    this.taskReturn += reward;
    const row = {decision: this.log.length + 1, phase, clock: Number(clock.toFixed(2)), id: actionId, kind: parts.kind,
      robotId: parts.robotId, points: result.reward, ownPoints: this.observation.score[this.alliance] - ownBefore,
      seconds: Number(result.durationSeconds.toFixed(3)), rejected: Boolean(result.rejectedAction), score: result.score,
      taskReward: Number(reward.toFixed(4)), candidates: candidateCount};
    this.log.push(row);
    return {observation: this.observe(), reward, engineReward: result.reward, durationSeconds: result.durationSeconds,
      score: result.score, done: this.done, rejectedAction: Boolean(result.rejectedAction), row};
  }

  // Read-only engine previews of the currently allowed candidates (for the non-learning tactics).
  previewAllowed() {
    const allowed = new Set(this.candidates.map(action => action.id));
    return this.environment.previewCandidates(action => allowed.has(action.id));
  }

  achieved() {
    const spec = this.spec;
    if (spec.mode === 'park') return this.parked.size > 0;
    if (spec.mode === 'play') return ['match-over', 'time-limit', 'decision-cap'].includes(this.stopReason ?? 'decision-cap');
    return (this.targeted ? this.targetsMet() : this.scoringActions > 0) && (!spec.park || this.parked.size > 0);
  }

  summary() {
    const observation = this.observation;
    return {seed: this.seed, alliance: this.alliance, task: this.spec, achieved: this.achieved(),
      stopReason: this.stopReason ?? 'decision-cap', scoringActions: this.scoringActions, rejectedActions: this.rejectedActions,
      parkedRobots: [...this.parked], ownPointsGained: this.gained(), marginChange: observation.score.margin - this.marginStart,
      decisions: this.log.length, taskReturn: this.taskReturn, clockStart: Number(this.clockStart.toFixed(2)),
      clockEnd: Number(observation.clock.toFixed(2)), finalScore: observation.score, timing: observation.timing,
      movementModel: observation.movementModel};
  }
}

module.exports = {TaskEnvironment, TASK_FEATURE_COUNT, FEATURE_COUNT, FEATURE_COUNT_V3, FEATURE_COUNT_V4, TASK_FEATURE_SETS,
  TASK_REWARD, SCORING_KINDS, PARK_RESERVE_SECONDS, MAX_LOOP_STEPS, SPEC_VALUES, actionParts, allows, validateSpec};
