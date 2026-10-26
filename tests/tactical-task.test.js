'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createPolicy} = require('../roboproof/tactical_policy');
const {parseTask, allows, runTask} = require('../roboproof/tactical_task');
const {fixturePolicyRecord} = require('./helpers/tactical-policy-fixture');

const pick = (spec, ...keys) => Object.fromEntries(keys.map(key => [key, spec[key]]));

test('typed tasks in English and Russian become allowlisted task specs', () => {
  assert.deepEqual(pick(parseTask('score 3 match loads into the neutral goals, then park'), 'mode', 'objects', 'source', 'goals', 'count', 'park'),
    {mode: 'score', objects: 'any', source: 'loader', goals: 'neutral', count: 3, park: true});
  assert.deepEqual(pick(parseTask('забей 2 кубка с поля за 30 секунд'), 'mode', 'objects', 'source', 'count', 'seconds', 'park'),
    {mode: 'score', objects: 'cup', source: 'field', count: 2, seconds: 30, park: false});
  assert.deepEqual(pick(parseTask('положи три пина в центральную цель'), 'objects', 'goals', 'count'), {objects: 'pin', goals: 'tall', count: 3});
  assert.deepEqual(pick(parseTask('набери 50 очков'), 'mode', 'points', 'count'), {mode: 'score', points: 50, count: null});
  assert.deepEqual(pick(parseTask('Забей пять загрузок в свои цели и припаркуйся'), 'source', 'goals', 'count', 'park'),
    {source: 'loader', goals: 'alliance', count: 5, park: true});
  assert.equal(parseTask('припаркуй роботов').mode, 'park');
  assert.equal(parseTask('play the driver period for the best score').mode, 'play');
  assert.equal(parseTask('score pins and cups').objects, 'any');
  assert.match(parseTask('score 3 match loads into the neutral goals, then park').summary, /^Place 3 match loads in neutral goals, then park$/);
});

test('tasks that are unclear, empty, too long or out of range are refused instead of guessed', () => {
  assert.throws(() => parseTask('hello there'), /Task not understood/);
  assert.throws(() => parseTask('   '), /1 to 300 characters/);
  assert.throws(() => parseTask('x'.repeat(301)), /1 to 300 characters/);
  assert.throws(() => parseTask(42), /1 to 300 characters/);
  assert.throws(() => parseTask('score 99 cups'), /from 1 to 30/);
  assert.throws(() => parseTask('score cups within 500 seconds'), /from 1 to 105/);
});

test('only candidate macro-actions that fit the task are allowed', () => {
  const loads = parseTask('score match loads into the neutral goals');
  assert.equal(allows(loads, 'score', 'load-score-cup:red-1:loader-red-n:g-neutral-e', 'red'), true);
  assert.equal(allows(loads, 'score', 'load-score-cup:red-1:loader-red-n:g-red-s', 'red'), false);
  assert.equal(allows(loads, 'score', 'score-pin:red-1:pin-01:g-neutral-e', 'red'), false);
  assert.equal(allows(loads, 'score', 'toggle:red-1:toggle-west', 'red'), false);
  assert.equal(allows(loads, 'score', 'park:red-1', 'red'), false);
  assert.equal(allows(loads, 'park', 'park:red-1', 'red'), true);
  const own = parseTask('score pins into our goals');
  assert.equal(allows(own, 'score', 'score-pin:blue-2:pin-04:g-blue-n', 'blue'), true);
  assert.equal(allows(own, 'score', 'score-pin:blue-2:pin-04:g-red-s', 'blue'), false);
  assert.equal(allows(own, 'score', 'score-cup:blue-2:cup-04:g-blue-n', 'blue'), false);
  assert.equal(allows(parseTask('play for the maximum'), 'score', 'toggle:red-1:toggle-west', 'red'), true);
});

test('the policy carries out a counted task, then parks, using only allowed actions', () => {
  const policy = createPolicy(fixturePolicyRecord());
  const spec = parseTask('score 2 match loads into the neutral goals, then park');
  const result = runTask({spec, policy, seed: 101, alliance: 'red'});
  assert.equal(result.achieved, true);
  assert.equal(result.stopReason, 'task-complete');
  assert.equal(result.scoringActions, 2);
  assert(result.parkedRobots.length > 0);
  const scoring = result.log.filter(row => row.phase === 'score' && !row.rejected);
  assert.equal(scoring.length, 2);
  assert(scoring.every(row => /^load-score-(pin|cup):red-\d:loader-red-[ns]:g-neutral-/.test(row.id)), JSON.stringify(scoring));
  const firstPark = result.log.findIndex(row => row.phase === 'park');
  assert(firstPark > 0 && result.log.slice(firstPark).every(row => row.kind === 'park'));
  assert(result.log.every(row => row.candidates > 0 && Number.isFinite(row.policyScore)));
  assert.equal(result.task, spec);
});

test('park-only, time-limited and points tasks stop for the right reason', () => {
  const policy = createPolicy(fixturePolicyRecord());
  const park = runTask({spec: parseTask('park the robots'), policy, seed: 101, alliance: 'blue'});
  assert.equal(park.achieved, true);
  assert(park.log.every(row => row.kind === 'park'));
  assert.deepEqual([...park.parkedRobots].sort(), ['blue-1', 'blue-2']);

  const timed = runTask({spec: parseTask('score pins within 10 seconds'), policy, seed: 101, alliance: 'red'});
  assert(['time-limit', 'no-allowed-action'].includes(timed.stopReason), timed.stopReason);
  assert(timed.log.every(row => row.kind === 'score-pin' || row.kind === 'load-score-pin'));
  if (timed.stopReason === 'time-limit') assert(timed.clockEnd - timed.clockStart >= 10);

  const points = runTask({spec: parseTask('gain 10 points'), policy, seed: 101, alliance: 'red'});
  if (points.achieved) assert(points.ownPointsGained >= 10);
  else assert.notEqual(points.stopReason, 'task-complete');

  assert.throws(() => runTask({spec: parseTask('park'), policy: null, seed: 1}), /trained tactical policy/);
});
