'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {TacticalEnvironment} = require('../roboproof/tactical_environment');
const {TaskEnvironment, FEATURE_COUNT_V3, FEATURE_COUNT_V4, TASK_FEATURE_COUNT, TASK_REWARD, allows, validateSpec} =
  require('../roboproof/tactical_task_environment');
const {parseTask, sampleTask} = require('../roboproof/tactical_task');
const {SampledTaskEnvironment, chooseForTask, evaluateTaskBaselines, playTaskBaseline, taskRandom} =
  require('../roboproof/tactical_task_training');

function playGreedy(task, observation) {
  while (!observation.done) {
    const pick = chooseForTask('task-greedy', task, task.previewAllowed(), null);
    observation = task.step(pick ? pick.id : observation.actions[0].id).observation;
  }
  return task.summary();
}

test('v4 candidates carry the v3 features plus the task block, and only allowed moves are offered', () => {
  // Pin match loads are not available at the start of the driver period, Cup loads are.
  const spec = parseTask('score 2 cups from the loader');
  const task = new TaskEnvironment({spec, environment: new TacticalEnvironment({featureSet: 'v3'}), featureSet: 'v4'});
  const observation = task.reset(5, 'blue');
  assert.equal(FEATURE_COUNT_V4, FEATURE_COUNT_V3 + TASK_FEATURE_COUNT);
  assert.ok(observation.actions.length > 0);
  const block = task.taskFeatures();
  assert.equal(block.length, TASK_FEATURE_COUNT);
  for (const action of observation.actions) {
    assert.equal(action.features.length, FEATURE_COUNT_V4);
    assert.deepEqual(action.features.slice(FEATURE_COUNT_V3), block);
    assert.ok(allows(spec, 'score', action.id, 'blue'), action.id);
    assert.ok(action.id.startsWith('load-score-cup:'), action.id);
  }
  assert.equal(new TaskEnvironment({spec: parseTask('score 2 pins from the loader'), featureSet: 'v4',
    environment: new TacticalEnvironment({featureSet: 'v3'})}).reset(5, 'blue').done, true);
  assert.throws(() => new TaskEnvironment({spec, environment: new TacticalEnvironment({featureSet: 'v2'}), featureSet: 'v4'}),
    /needs a v3 tactical environment/);
});

test('a placement-count task pays per placement and on completion, charges game time, and records success', () => {
  const task = new TaskEnvironment({spec: parseTask('score 2 cups from the loader'), featureSet: 'v4',
    environment: new TacticalEnvironment({featureSet: 'v3'})});
  const summary = playGreedy(task, task.reset(3, 'red'));
  assert.equal(task.rejectedActions, 0);
  assert.equal(summary.achieved, true);
  assert.equal(summary.stopReason, 'task-complete');
  assert.equal(summary.scoringActions, 2);
  const seconds = task.log.reduce((sum, row) => sum + row.seconds, 0);
  const expected = 2 * TASK_REWARD.perPlacement + TASK_REWARD.completion - TASK_REWARD.timeCostPerSecond * seconds;
  assert.ok(Math.abs(summary.taskReturn - expected) < 0.01, `${summary.taskReturn} vs ${expected}`);
  assert.ok(Math.abs(summary.taskReturn - task.log.reduce((sum, row) => sum + row.taskReward, 0)) < 1e-3);
});

test('a task that runs out of time is penalized, and open play keeps the score-margin reward', () => {
  const failing = new TaskEnvironment({spec: parseTask('score 30 cups within 10 seconds'), featureSet: 'v4',
    environment: new TacticalEnvironment({featureSet: 'v3'})});
  const failed = playGreedy(failing, failing.reset(9, 'red'));
  assert.equal(failed.achieved, false);
  assert.equal(failed.stopReason, 'time-limit');
  assert.ok(failed.decisions > 0 && failing.rejectedActions === 0);
  const usedSeconds = failing.log.reduce((sum, row) => sum + row.seconds, 0);
  const expected = TASK_REWARD.perPlacement * failed.scoringActions - TASK_REWARD.timeCostPerSecond * usedSeconds - TASK_REWARD.failure;
  assert.ok(Math.abs(failed.taskReturn - expected) < 0.01, `${failed.taskReturn} vs ${expected}`);
  const play = new TaskEnvironment({spec: parseTask('play the driver period for the best score'), featureSet: 'v4',
    environment: new TacticalEnvironment({featureSet: 'v3'})});
  playGreedy(play, play.reset(9, 'red'));
  assert.ok(play.log.length > 10);
  assert.ok(play.log.every(row => row.taskReward === row.points));
});

test('robots already standing in midfield count as parked, with the same parking reward', () => {
  const task = new TaskEnvironment({spec: parseTask('score 4 pins and cups as fast as possible, then park'), featureSet: 'v4',
    environment: new TacticalEnvironment({featureSet: 'v3'})});
  task.reset(2500001, 'red');
  const moves = ['load-score-cup:red-1:loader-red-s:g-neutral-tall', 'score-pin:red-1:pin-01:g-neutral-tall',
    'load-score-cup:red-2:loader-red-n:g-neutral-tall', 'score-pin:red-2:pin-02:g-neutral-tall'];
  let last;
  for (const id of moves) last = task.step(id);
  const summary = task.summary();
  assert.equal(summary.achieved, true);
  assert.equal(summary.stopReason, 'task-complete');
  assert.deepEqual(summary.parkedRobots.sort(), ['red-1', 'red-2']);
  // The last placement completes the count and leaves both robots in the central goal's midfield: completion bonus
  // plus the parking reward for each robot, minus game time.
  const expected = TASK_REWARD.perPlacement + TASK_REWARD.completion + 2 * TASK_REWARD.perParkedRobot -
    TASK_REWARD.timeCostPerSecond * last.durationSeconds;
  assert.ok(Math.abs(last.reward - expected) < 1e-9, `${last.reward} vs ${expected}`);
});

test('sampled tasks are valid, deterministic per seed and cover every task kind', () => {
  const kinds = new Set();
  for (let seed = 1; seed <= 300; seed += 1) {
    const spec = sampleTask(taskRandom(seed));
    assert.doesNotThrow(() => validateSpec(spec));
    assert.deepEqual(sampleTask(taskRandom(seed)), spec);
    kinds.add(spec.mode);
    if (spec.count !== null) kinds.add('count');
    if (spec.points !== null) kinds.add('points');
    if (spec.seconds !== null) kinds.add('seconds');
    if (spec.mode === 'score' && spec.park) kinds.add('score-then-park');
  }
  assert.deepEqual([...kinds].sort(), ['count', 'park', 'play', 'points', 'score', 'score-then-park', 'seconds']);
  const environment = new SampledTaskEnvironment();
  const first = environment.reset(1234, 'red');
  const again = new SampledTaskEnvironment().reset(1234, 'red');
  assert.deepEqual(first.task.spec, again.task.spec);
  assert.deepEqual(first.actions.map(action => action.id), again.actions.map(action => action.id));
  assert.throws(() => validateSpec({...first.task.spec, mode: 'dance'}), /mode must be one of/);
  assert.throws(() => validateSpec({...parseTask('score 3 pins'), count: 0}), /from 1 to 30/);
});

test('non-learning tactics play the same seeded tasks, and the teacher picks only allowed legal moves', () => {
  const seeds = [11, 12, 13, 14];
  const results = evaluateTaskBaselines({seeds, alliance: 'blue', policies: ['task-greedy', 'loader-cycle', 'random']});
  for (const result of Object.values(results)) {
    assert.equal(result.taskReturns.length, 4);
    assert.deepEqual(result.tasks, results['task-greedy'].tasks);
  }
  assert.deepEqual(playTaskBaseline({policy: 'task-greedy', seed: 12, alliance: 'blue'}).taskReturn, results['task-greedy'].taskReturns[1]);
  const environment = new SampledTaskEnvironment();
  const observation = environment.reset(21, 'red');
  const pick = environment.demonstrate('task-greedy');
  if (pick) assert.ok(observation.actions.some(action => action.id === pick.id));
  assert.throws(() => environment.demonstrate('random'), /Unsupported task demonstration policy/);
});

test('the v4 bridge plays sampled tasks, demonstrates, compares baselines and closes', async () => {
  const bridge = spawn(process.execPath, [path.join(__dirname, '../roboproof/tactical_learning/bridge.js'), '--features', 'v4'],
    {stdio: ['pipe', 'pipe', 'pipe']});
  const lines = [];
  let buffer = '';
  const waiters = [];
  bridge.stdout.on('data', chunk => {
    buffer += chunk;
    let index;
    while ((index = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      (waiters.shift() || (value => lines.push(value)))(JSON.parse(line));
    }
  });
  const request = payload => new Promise(resolve => { waiters.push(resolve); bridge.stdin.write(JSON.stringify(payload) + '\n'); });
  const reset = await request({operation: 'reset', seed: 77, alliance: 'red'});
  assert.equal(reset.ok, true);
  assert.ok(reset.observation.task.spec.summary);
  if (reset.observation.actions.length) {
    assert.equal(reset.observation.actions[0].features.length, FEATURE_COUNT_V4);
    const demonstrated = await request({operation: 'demonstrate'});
    assert.equal(demonstrated.policy, 'task-greedy');
    const step = await request({operation: 'step', actionId: demonstrated.actionId ?? reset.observation.actions[0].id});
    assert.equal(step.ok, true);
    assert.equal(typeof step.reward, 'number');
  }
  const baselines = await request({operation: 'baselines', seeds: [77, 78], alliance: 'red'});
  assert.equal(baselines.ok, true);
  assert.equal(baselines.baselines['task-greedy'].taskReturns.length, 2);
  const closed = await request({operation: 'close'});
  assert.equal(closed.ok, true);
  await new Promise(resolve => bridge.on('exit', resolve));
});
