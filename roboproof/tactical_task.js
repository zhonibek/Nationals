'use strict';

// "Give Tactical AI a task": a typed task (English or Russian) becomes an allowlisted task spec. The trained tactical
// policy then plays the abstract driver period choosing only among candidate macro-actions that fit the task, and
// the Override rules engine scores every step. A task the parser does not understand is refused, never guessed.

const OverrideGame = require('../simulator/override');
const {TacticalEnvironment} = require('./tactical_environment');

const MAX_TASK_LENGTH = 300;
const MATCH_SECONDS = OverrideGame.constants.MATCH_SECONDS;
const DRIVER_SECONDS = OverrideGame.constants.DRIVER_SECONDS;
// With an open-ended scoring task followed by parking, scoring stops this many seconds before the end of the match.
const PARK_RESERVE_SECONDS = 12;
const MAX_LOOP_STEPS = 200;
const SCORING_KINDS = new Set(['score-pin', 'score-cup', 'load-score-pin', 'load-score-cup']);
const EXAMPLES = ['score 3 match loads into the neutral goals, then park', 'забей 2 кубка с поля за 30 секунд',
  'score Pins until we gain 40 points', 'play the driver period for the best score', 'припаркуй роботов'];

const NUMBER_WORDS = {one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  'один': 1, 'одну': 1, 'одна': 1, 'одного': 1, 'два': 2, 'две': 2, 'три': 3, 'четыре': 4, 'пять': 5, 'шесть': 6,
  'семь': 7, 'восемь': 8, 'девять': 9, 'десять': 10};

const PATTERNS = {
  pin: /\bpins?\b|пин/u,
  cup: /\bcups?\b|кубк|кубок|стакан/u,
  loader: /match[ -]?loads?|\bloads?\b|loader|loading[ -]zone|загрузк|загрузо|лоадер/u,
  field: /\bfield\b|с поля|полев/u,
  neutral: /neutral|нейтрал/u,
  alliance: /alliance goals?|own goals?|our goals?|сво(?:и|ю|их|ей|ими) (?:цел|ворот)|союзн/u,
  tall: /\btall\b|\bcent(?:er|re) goal|\bmiddle goal|высок|центральн/u,
  park: /\bpark|парк/u,
  play: /\bplay\b|maximi[sz]e|\bmax(?:imum)?\b|\bwin\b|whole match|full match|играй|сыграй|максим|побед|весь матч/u,
  score: /scor|deliver|\bplace\b|\bput\b|bring|забе|заби|положи|постав|набер|набра|принес/u,
  points: /(\d+)\s*(?:points?\b|pts\b|очк)/u,
  seconds: /(\d+)\s*(?:seconds?\b|secs?\b|s\b|сек)/u,
  count: /(\d+)\s*(?:(?:match[ -]?)?loads?\b|pins?\b|cups?\b|objects?\b|times\b|загрузк\p{L}*|загрузо\p{L}*|пин\p{L}*|кубк\p{L}*|кубок|объект\p{L}*|раз\b)/u,
  verbCount: /(?:\bscore|забей|забить|положи|поставь)\s+(\d+)\b/u
};

function normalize(text) {
  return text.toLowerCase().replace(/ё/gu, 'е').split(/([^\p{L}\p{N}]+)/u)
    .map(token => Object.hasOwn(NUMBER_WORDS, token) ? String(NUMBER_WORDS[token]) : token).join('');
}

function bounded(match, minimum, maximum, label) {
  if (!match) return null;
  const value = Number(match[1]);
  if (!Number.isInteger(value) || value < minimum || value > maximum) throw Error(`${label} must be from ${minimum} to ${maximum}`);
  return value;
}

function only(first, second) {
  return first && !second ? 'first' : second && !first ? 'second' : null;
}

function describe(spec) {
  if (spec.mode === 'play') {
    return `Play the driver period for the best score with every action the policy knows${spec.seconds ? `, for ${spec.seconds} s` : ''}`;
  }
  if (spec.mode === 'park') return 'Park the robots';
  const object = {any: 'objects', pin: 'Pins', cup: 'Cups'}[spec.objects];
  const what = spec.source === 'loader' ? `${spec.objects === 'any' ? '' : spec.objects === 'pin' ? 'Pin ' : 'Cup '}match loads`
    : spec.source === 'field' ? `field ${object}` : object;
  const goals = {any: ' in goals', neutral: ' in neutral goals', alliance: ' in alliance goals', tall: ' in the tall goal'}[spec.goals];
  // "Place": a legal placement counts toward the task even when the rules award no points for it yet (a Cup without a Pin).
  return `Place ${spec.count ?? 'as many'} ${what}${goals}` + (spec.points ? ` until the alliance gains ${spec.points} points` : '') +
    (spec.seconds ? ` within ${spec.seconds} s` : '') + (spec.park ? ', then park' : '');
}

function parseTask(text) {
  if (typeof text !== 'string' || !text.trim() || text.length > MAX_TASK_LENGTH) {
    throw Error(`Describe the task in 1 to ${MAX_TASK_LENGTH} characters`);
  }
  const t = normalize(text);
  const has = name => PATTERNS[name].test(t);
  const objects = {first: 'pin', second: 'cup'}[only(has('pin'), has('cup'))] ?? 'any';
  const source = {first: 'loader', second: 'field'}[only(has('loader'), has('field'))] ?? 'any';
  const goals = has('tall') ? 'tall' : {first: 'neutral', second: 'alliance'}[only(has('neutral'), has('alliance'))] ?? 'any';
  const count = bounded(t.match(PATTERNS.count) || t.match(PATTERNS.verbCount), 1, 30, 'The number of scoring actions');
  const points = bounded(t.match(PATTERNS.points), 1, 500, 'The points target');
  const seconds = bounded(t.match(PATTERNS.seconds), 1, DRIVER_SECONDS, 'The time limit in seconds');
  const park = has('park');
  const filtered = objects !== 'any' || source !== 'any' || goals !== 'any';
  const scoring = has('score') || filtered || count !== null || points !== null;
  let mode;
  if (has('play') && !filtered && count === null && points === null) mode = 'play';
  else if (scoring) mode = 'score';
  else if (park) mode = 'park';
  else throw Error(`Task not understood. Try for example: ${EXAMPLES.map(example => `"${example}"`).join(', ')}`);
  const spec = {mode, objects, source, goals, count, points, seconds, park: mode === 'park' || (mode === 'score' && park)};
  return {...spec, summary: describe(spec)};
}

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

function runTask({spec, policy, seed, alliance = 'red', environment = new TacticalEnvironment()}) {
  if (!policy || typeof policy.choose !== 'function') throw Error('A trained tactical policy is required');
  let observation = environment.reset(seed, alliance);
  const clockStart = observation.clock;
  const ownStart = observation.score[alliance];
  const marginStart = observation.score.margin;
  const parked = new Set();
  const log = [];
  let banned = new Set();
  let phase = spec.mode === 'park' ? 'park' : 'score';
  let scoringActions = 0;
  let rejectedActions = 0;
  let stopReason = null;
  const gained = () => observation.score[alliance] - ownStart;
  const targetsMet = () => (spec.count === null || scoringActions >= spec.count) && (spec.points === null || gained() >= spec.points);
  const finishScoring = reason => {
    if (spec.park) { phase = 'park'; banned = new Set(); return false; }
    stopReason = reason;
    return true;
  };

  for (let step = 0; step < MAX_LOOP_STEPS; step += 1) {
    if (observation.done) { stopReason = phase === 'park' && parked.size ? 'task-complete' : 'match-over'; break; }
    if (phase === 'score') {
      const timeUp = spec.seconds !== null && observation.clock - clockStart >= spec.seconds - 1e-9;
      const targeted = spec.count !== null || spec.points !== null;
      if (targeted && targetsMet() && finishScoring('task-complete')) break;
      if (phase === 'score' && timeUp && finishScoring('time-limit')) break;
      if (phase === 'score' && !targeted && spec.park && observation.clock >= MATCH_SECONDS - PARK_RESERVE_SECONDS) finishScoring();
    }
    const candidates = observation.actions.filter(action => !banned.has(action.id) && allows(spec, phase, action.id, alliance) &&
      !(phase === 'park' && parked.has(actionParts(action.id).robotId)));
    if (!candidates.length) {
      if (phase === 'score' && !finishScoring('no-allowed-action')) continue;
      if (phase === 'park') stopReason = parked.size ? 'task-complete' : 'no-allowed-action';
      break;
    }
    const pick = policy.choose(candidates);
    const clock = observation.clock;
    const ownBefore = observation.score[alliance];
    const result = environment.step(pick.action.id);
    const parts = actionParts(pick.action.id);
    if (result.rejectedAction) { rejectedActions += 1; banned.add(pick.action.id); }
    else {
      banned = new Set();
      if (SCORING_KINDS.has(parts.kind)) scoringActions += 1;
      if (parts.kind === 'park') parked.add(parts.robotId);
    }
    observation = result.observation;
    log.push({decision: log.length + 1, phase, clock: Number(clock.toFixed(2)), id: pick.action.id, kind: parts.kind,
      robotId: parts.robotId, points: result.reward, ownPoints: observation.score[alliance] - ownBefore,
      seconds: Number(result.durationSeconds.toFixed(3)), rejected: Boolean(result.rejectedAction), score: result.score,
      policyScore: Number(pick.score.toFixed(4)), candidates: candidates.length});
  }
  stopReason ??= 'decision-cap';
  const targeted = spec.count !== null || spec.points !== null;
  const achieved = spec.mode === 'park' ? parked.size > 0
    : spec.mode === 'play' ? ['match-over', 'time-limit', 'decision-cap'].includes(stopReason)
      : (targeted ? targetsMet() : scoringActions > 0) && (!spec.park || parked.size > 0);
  return {seed, alliance, task: spec, achieved, stopReason, scoringActions, rejectedActions, parkedRobots: [...parked],
    ownPointsGained: gained(), marginChange: observation.score.margin - marginStart, decisions: log.length,
    clockStart: Number(clockStart.toFixed(2)), clockEnd: Number(observation.clock.toFixed(2)), finalScore: observation.score,
    timing: observation.timing, movementModel: observation.movementModel, log};
}

module.exports = {MAX_TASK_LENGTH, EXAMPLES, parseTask, allows, runTask};
