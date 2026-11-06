'use strict';

// "Give Tactical AI a task": a typed task (English or Russian) becomes an allowlisted task spec. The trained tactical
// policy then plays the abstract driver period choosing only among candidate macro-actions that fit the task, and
// the Override rules engine scores every step. A task the parser does not understand is refused, never guessed.

const OverrideGame = require('../simulator/override');
const {TacticalEnvironment} = require('./tactical_environment');
const {TaskEnvironment, allows, validateSpec} = require('./tactical_task_environment');

const MAX_TASK_LENGTH = 300;
const DRIVER_SECONDS = OverrideGame.constants.DRIVER_SECONDS;
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

// Plays one task with a trained policy through the same TaskEnvironment the multi-task trainer uses.
// `environment` is the underlying TacticalEnvironment (its feature set must match the policy's base features).
function runTask({spec, policy, seed, alliance = 'red', environment}) {
  if (!policy || typeof policy.choose !== 'function') throw Error('A trained tactical policy is required');
  const featureSet = policy.featureSet ?? 'v2';
  environment ??= new TacticalEnvironment({featureSet: featureSet === 'v4' ? 'v3' : featureSet});
  // A v3 policy reads the rules-engine preview features and a v4 policy also the task; other features would not fit.
  const base = featureSet === 'v4' ? 'v3' : featureSet;
  if (base !== (environment.featureSet ?? 'v2')) {
    throw Error(`The tactical policy needs ${featureSet} features, but the environment produces ${environment.featureSet}`);
  }
  const task = new TaskEnvironment({spec: validateSpec({...spec}), environment, featureSet});
  let observation = task.reset(seed, alliance);
  const log = [];
  while (!observation.done) {
    const pick = policy.choose(observation.actions);
    const {row, observation: next} = task.step(pick.action.id);
    const {taskReward, ...rest} = row;
    log.push({...rest, policyScore: Number(pick.score.toFixed(4)), ...(featureSet === 'v4' ? {taskReward} : {})});
    observation = next;
  }
  const {taskReturn, ...summary} = task.summary();
  return {...summary, task: spec, ...(featureSet === 'v4' ? {taskReturn} : {}), log};
}

// Random task specs for multi-task training, drawn from the same allowlisted shape the parser produces.
// `random` returns floats in [0, 1).
function sampleTask(random) {
  const weighted = entries => {
    let draw = random() * entries.reduce((sum, [, weight]) => sum + weight, 0);
    for (const [value, weight] of entries) { draw -= weight; if (draw < 0) return value; }
    return entries.at(-1)[0];
  };
  const integer = (minimum, maximum, step = 1) => minimum + step * Math.floor(random() * (Math.floor((maximum - minimum) / step) + 1));
  const mode = weighted([['play', 0.2], ['score', 0.7], ['park', 0.1]]);
  let spec;
  if (mode === 'play') {
    spec = {mode, objects: 'any', source: 'any', goals: 'any', count: null, points: null,
      seconds: random() < 0.5 ? null : integer(15, 100, 5), park: false};
  } else if (mode === 'park') {
    spec = {mode, objects: 'any', source: 'any', goals: 'any', count: null, points: null, seconds: null, park: true};
  } else {
    const target = weighted([['count', 0.4], ['points', 0.3], ['none', 0.3]]);
    spec = {mode, objects: weighted([['any', 0.4], ['pin', 0.3], ['cup', 0.3]]),
      source: weighted([['any', 0.4], ['loader', 0.3], ['field', 0.3]]),
      goals: weighted([['any', 0.4], ['neutral', 0.25], ['alliance', 0.2], ['tall', 0.15]]),
      count: target === 'count' ? integer(1, 8) : null, points: target === 'points' ? integer(10, 80, 5) : null,
      seconds: random() < 0.4 ? integer(10, 90, 5) : null, park: random() < 0.3};
  }
  validateSpec(spec);
  return {...spec, summary: describe(spec)};
}

module.exports = {MAX_TASK_LENGTH, EXAMPLES, parseTask, allows, describe, runTask, sampleTask};
