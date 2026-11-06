'use strict';

// Multi-task training support (feature set v4): every episode seed fixes one random task, so the trainer, its
// held-out evaluation and the non-learning comparison all play the same task on the same start pose.
// The non-learning tactics choose only among the moves the task allows, using rules-engine previews.

const {TacticalEnvironment} = require('./tactical_environment');
const {TaskEnvironment} = require('./tactical_task_environment');
const {sampleTask} = require('./tactical_task');
const {choose, POLICIES} = require('./tactical_baselines');

const TASK_SEED_MASK = 0x7A5C3E1D;
const MAX_TASK_DRAWS = 20;
// 'task-greedy' is the strongest simple task solver here: the team's loader-first tactic for open scoring,
// the quickest legal placement for a placement count, the best points per second for a points target,
// and the quickest park. It is the warm-start teacher and the bar a multi-task policy must clear.
const TASK_POLICIES = Object.freeze({...POLICIES,
  'task-greedy': 'Per task: loader-first for open scoring, quickest placement for a count, best points per second for a points target, quickest park'});
const TASK_DEMONSTRATION_POLICIES = Object.freeze(['task-greedy', 'loader-cycle']);

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

// Training seeds are consecutive, and an LCG's first draws for consecutive seeds are nearly equal, so the seed is
// hashed first (murmur3 finalizer); otherwise neighbouring episodes would get almost the same task.
function taskRandom(seed) {
  let hash = (seed ^ TASK_SEED_MASK) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 16), 0x85ebca6b) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35) >>> 0;
  return seededRandom((hash ^ (hash >>> 16)) >>> 0);
}

const quickest = rows => [...rows].sort((first, second) => first.seconds - second.seconds ||
  second.points - first.points || first.id.localeCompare(second.id))[0] || null;

function chooseForTask(policy, task, rows, random) {
  if (policy !== 'task-greedy') return choose(policy, rows, random);
  const legal = rows.filter(row => row.legal);
  if (!legal.length) return null;
  const spec = task.spec;
  if (task.phase === 'park' || spec.count !== null) return quickest(legal);
  if (spec.points !== null) return choose('greedy-all', legal, random);
  return choose('loader-cycle', legal, random);
}

// One reusable v3 tactical environment; each reset draws the episode's task from its seed. A drawn task with no
// allowed move at the start is redrawn (up to MAX_TASK_DRAWS times) from the same seeded stream.
class SampledTaskEnvironment {
  constructor(options = {}) {
    this.base = new TacticalEnvironment({...options, featureSet: 'v3'});
    this.featureSet = 'v4';
  }

  reset(seed, alliance = 'red', spec = null) {
    const random = taskRandom(seed);
    let observation;
    for (let draw = 0; draw < MAX_TASK_DRAWS; draw += 1) {
      const taskSpec = spec ?? sampleTask(random);
      this.task = new TaskEnvironment({spec: taskSpec, environment: this.base, featureSet: 'v4'});
      observation = this.task.reset(seed, alliance);
      if (spec || observation.actions.length) break;
    }
    return observation;
  }

  step(actionId) {
    const {observation, reward, durationSeconds, score, done, rejectedAction} = this.task.step(actionId);
    return {observation, reward, durationSeconds, score, done, rejectedAction};
  }

  // The teacher's choice in the current state (null when the task is over or nothing allowed is legal).
  demonstrate(policy = 'task-greedy') {
    if (!TASK_DEMONSTRATION_POLICIES.includes(policy)) throw Error(`Unsupported task demonstration policy ${policy}`);
    if (this.task.done) return null;
    return chooseForTask(policy, this.task, this.task.previewAllowed(), null);
  }
}

// Plays the seed's task with one non-learning tactic. When nothing allowed is legal it takes the first allowed
// move anyway (the engine rejects it and the task bans it), exactly as a learned policy would be treated.
function playTaskBaseline({policy, seed, alliance = 'red', environmentOptions = {}}) {
  if (!Object.hasOwn(TASK_POLICIES, policy)) throw Error(`Unknown tactical policy ${policy}`);
  const environment = new SampledTaskEnvironment(environmentOptions);
  let observation = environment.reset(seed, alliance);
  const random = seededRandom((seed ^ 0x9E3779B9) >>> 0);
  while (!observation.done) {
    const pick = chooseForTask(policy, environment.task, environment.task.previewAllowed(), random);
    observation = environment.step(pick ? pick.id : observation.actions[0].id).observation;
  }
  return environment.task.summary();
}

function evaluateTaskBaselines({seeds, alliance = 'red', policies = Object.keys(TASK_POLICIES), environmentOptions = {}} = {}) {
  if (!Array.isArray(seeds) || !seeds.length) throw Error('Provide evaluation seeds');
  const results = {};
  for (const policy of policies) {
    const rows = seeds.map(seed => playTaskBaseline({policy, seed, alliance, environmentOptions}));
    results[policy] = {description: TASK_POLICIES[policy], seeds, alliance,
      taskReturns: rows.map(row => row.taskReturn), achieved: rows.map(row => row.achieved),
      margins: rows.map(row => row.finalScore.margin), secondsUsed: rows.map(row => Number((row.clockEnd - row.clockStart).toFixed(3))),
      tasks: rows.map(row => row.task.summary),
      meanTaskReturn: rows.reduce((sum, row) => sum + row.taskReturn, 0) / rows.length,
      successRate: rows.filter(row => row.achieved).length / rows.length};
  }
  return results;
}

module.exports = {TASK_SEED_MASK, TASK_POLICIES, TASK_DEMONSTRATION_POLICIES, SampledTaskEnvironment, chooseForTask,
  playTaskBaseline, evaluateTaskBaselines, taskRandom};
