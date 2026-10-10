'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {DEFAULT_TASK} = require('../simulator/motion');
const {Playback} = require('../simulator/motion-replay');
const Policy = require('../simulator/policy');
const {policyFromSnapshot} = require('../roboproof/motion-learner');
const {runPolicy, replay} = require('../roboproof/motion');
const runner = loadHeadless();

function model(probes = [144], featureTransform = 'deadline-context-v1') {
  const featureInputSize = {'identity-v1': 34, 'reference-frame-v1': 35,
    'history-context-v1': 144, 'deadline-context-v1': 145}[featureTransform];
  const layers = [[featureInputSize, 32], [32, 32], [32, 8]].map(([inputs, outputs]) => ({
    weight: Array.from({length: outputs}, () => Array(inputs).fill(0)), bias: Array(outputs).fill(0)}));
  probes.forEach((feature, axis) => {
    layers[0].weight[axis][feature] = 1;
    layers[1].weight[axis][axis] = 1;
    layers[2].weight[axis][axis] = 1;
  });
  const data = {schemaVersion: 1, algorithm: 'ppo-beta-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width: 32, featureTransform, featureInputSize,
    scales: Array(featureInputSize).fill(1), layers, identity: runner.identity,
    trainingSeed: 503, steps: 10, updates: 1, trained: true};
  if (featureInputSize >= 144) Object.assign(data, {historyFrames: 4, initialContextSize: 4});
  if (featureTransform === 'deadline-context-v1') data.deadlineContext = {
    schemaVersion: 1, settlingReserveSeconds: 2, input: 'public-reference-duration-at-reset'};
  return data;
}

const decode = data => policyFromSnapshot(data, createHash('sha256').update(JSON.stringify(data)).digest('hex'));

function assertActions(actual, features) {
  assert.equal(actual.length, 4);
  for (let axis = 0; axis < 4; axis++) {
    const logit = Math.tanh(Math.tanh(Math.max(-10, Math.min(10, features[axis] ?? 0))));
    const alpha = Math.max(logit, 0) + Math.log1p(Math.exp(-Math.abs(logit))) + 1;
    const beta = Math.log(2) + 1;
    const expected = 2 * alpha / (alpha + beta) - 1;
    assert(Math.abs(actual[axis] - expected) < 1e-12, `action axis ${axis}: ${actual[axis]} != ${expected}`);
  }
}

test('deadline scalar occupies only the last feature and uses current public progress with clipping', () => {
  const policy = decode(model());
  policy.act.reset({referenceDurationSeconds: 8});
  for (const [referenceSeconds, remainingSeconds, scalar] of [
    [2, 12, 0.6], [5, 12, 0.3], [8, 12, 0.25], [10, 12, 0.25],
    [0, 3, 1], [7.995, 2, 0.5], [7.995, -1, 0.5], [-1, 12, 0.9], [0, 10, 1]
  ]) {
    const raw = Array(34).fill(0);
    raw[18] = remainingSeconds;
    raw[25] = referenceSeconds;
    const original = [...raw];
    assertActions(policy.act(Object.freeze(raw)), [scalar]);
    assert.deepEqual(raw, original);
  }
  assert.throws(() => policy.act(Array(35).fill(0)), /sensor/);
  const raw = Array(34).fill(0);
  raw[25] = NaN;
  assert.throws(() => policy.act(raw), /sensor/);
});

test('deadline history keeps four chronological 35-feature frames and the initial raw target error', () => {
  const velocity = decode(model([3, 38, 73, 108]));
  const heading = decode(model([34, 69, 104, 139]));
  const initial = decode(model([140, 141, 142, 143]));
  for (const policy of [velocity, heading, initial]) policy.act.reset({referenceDurationSeconds: 10});
  const target = [0.2, -0.3, 0.4, -0.5];
  for (let tick = 0; tick < 6; tick++) {
    const raw = Array(34).fill(0);
    raw[2] = 0.1 * (tick + 1);
    raw[3] = 0.05 * (tick + 1);
    raw[4] = 0.1 * (tick + 1);
    raw[21] = Math.PI / 2;
    raw[18] = 10;
    raw[25] = tick * 0.1;
    raw.splice(10, 4, ...target.map(value => value + tick * 0.1));
    Object.freeze(raw);
    const frameTicks = [Math.max(0, tick - 3), Math.max(0, tick - 2), Math.max(0, tick - 1), tick];
    assertActions(velocity.act(raw), frameTicks.map(frame => -0.1 * (frame + 1)));
    assertActions(heading.act(raw), frameTicks.map(frame => Math.cos(Math.PI / 2 - 0.1 * (frame + 1))));
    assertActions(initial.act(raw), target);
  }
});

test('deadline inference requires a public reset, permits zero duration and clears on context-free reset', () => {
  const policy = decode(model());
  const raw = Array(34).fill(0);
  raw[18] = 10;
  assert.throws(() => policy.act(raw), /public reference duration.*reset/i);
  const context = {referenceDurationSeconds: 5};
  policy.act.reset(context);
  context.referenceDurationSeconds = 100;
  assertActions(policy.act(raw), [0.625]);
  policy.act.reset({referenceDurationSeconds: 0});
  assertActions(policy.act(raw), [0.25]);
  policy.act.reset();
  assert.throws(() => policy.act(raw), /public reference duration.*reset/i);
});

test('invalid resets discard the prior public duration and all episode history before failing closed', () => {
  const policy = decode(model([3, 140, 144]));
  const first = Array(34).fill(0);
  first[3] = 0.1;
  first[10] = 0.2;
  first[18] = 10;
  const next = [...first];
  next[3] = 0.8;
  next[10] = 0.9;
  for (const context of [null, {}, 5, [], {referenceDurationSeconds: undefined},
    {referenceDurationSeconds: null}, {referenceDurationSeconds: '5'}, {referenceDurationSeconds: true},
    {referenceDurationSeconds: -0.01}, {referenceDurationSeconds: NaN},
    {referenceDurationSeconds: Infinity}, {referenceDurationSeconds: -Infinity}]) {
    policy.act.reset({referenceDurationSeconds: 8});
    policy.act(first);
    for (let tick = 0; tick < 5; tick++) policy.act(next);
    assert.throws(() => policy.act.reset(context), /invalid public reference duration/i);
    assert.throws(() => policy.act(next), /public reference duration.*reset/i);
    policy.act.reset({referenceDurationSeconds: 4});
    assertActions(policy.act(first), [0.1, 0.2, 0.5]);
  }
});

test('deadline snapshots reject incompatible schemas, context definitions, widths and scalar scales', () => {
  const valid = model();
  assert.equal(valid.featureInputSize, 145);
  assert.equal(valid.scales.length, 145);
  assert.equal(valid.scales[144], 1);
  for (const change of [data => { data.featureTransform = 'deadline-context-v2'; },
    data => { delete data.deadlineContext; }, data => { data.deadlineContext = null; },
    data => { data.deadlineContext.schemaVersion = 2; }, data => { data.deadlineContext.schemaVersion = '1'; },
    data => { data.deadlineContext.settlingReserveSeconds = 1; },
    data => { data.deadlineContext.input = 'physical-truth'; }, data => { data.deadlineContext.extra = true; },
    data => { data.historyFrames = 3; }, data => { delete data.historyFrames; },
    data => { data.initialContextSize = 5; }, data => { delete data.initialContextSize; },
    data => { delete data.featureInputSize; }, data => { data.featureInputSize = 144; },
    data => { data.featureInputSize = 146; }, data => { data.scales.pop(); }, data => { data.scales.push(1); },
    data => { data.scales[144] = 0.5; }, data => { data.scales[144] = 2; },
    data => { data.layers[0].weight[0].pop(); }, data => { data.layers[0].weight[0].push(0); }]) {
    const data = structuredClone(valid);
    change(data);
    assert.throws(() => decode(data), /(?:feature|model|weights)/i);
  }
});

test('older 34, 35 and 144-feature actors retain inference and optional reset compatibility', () => {
  const raw = Array(34).fill(0);
  raw[3] = 0.3;
  for (const featureTransform of ['identity-v1', 'reference-frame-v1', 'history-context-v1']) {
    const data = model([3], featureTransform);
    const policy = decode(data);
    assertActions(policy.act(raw), [0.3]);
    policy.act.reset();
    assertActions(policy.act(raw), [0.3]);
    policy.act.reset({referenceDurationSeconds: 3});
    assertActions(policy.act(raw), [0.3]);
    if (featureTransform === 'identity-v1') {
      delete data.featureTransform;
      delete data.featureInputSize;
      assertActions(decode(data).act(raw), [0.3]);
    }
  }
});

test('original runPolicy supplies only reset-time public planner duration and isolates repeated episodes', () => {
  const policy = decode(model([3, 140, 144]));
  const contexts = [];
  const act = raw => {
    assert(contexts.length > 0);
    assert.equal(raw.length, 34);
    return policy.act(raw);
  };
  act.reset = context => { contexts.push(structuredClone(context)); policy.act.reset(context); };
  const firstTask = {...DEFAULT_TASK, goal: {xIn: 3, yIn: 8, headingDeg: 15}, deadlineSeconds: 0.15};
  const secondTask = {...DEFAULT_TASK, goal: {xIn: 40, yIn: -23, headingDeg: -25}, deadlineSeconds: 0.2};
  const expected = [firstTask, secondTask, firstTask].map(task => {
    const episode = runner.createEpisode();
    episode.reset(93, {}, task, {controlMode: 'policy'});
    return {referenceDurationSeconds: episode.referenceAdapter.duration};
  });
  const run = task => runPolicy({seed: 93, task, act, options: {policyIdentity: policy.identity}});
  const first = run(firstTask);
  const second = run(secondTask);
  assert.notEqual(expected[0].referenceDurationSeconds, expected[1].referenceDurationSeconds);
  assert.deepEqual(run(firstTask), first);
  assert.deepEqual(contexts, expected);
  for (const report of [first, second]) {
    assert.equal(report.ticks, Math.round(report.task.deadlineSeconds / 0.01));
    assert(report.actions.every(entry => entry.action.values.length === 4));
    assert(report.transitions.every(transition => Policy.vector(transition.observation).length === 34));
    assert.equal(Object.hasOwn(report, 'deadlineContext'), false);
  }
});

test('deadline actions replay unchanged in the original Simulator without any model inference or reset', () => {
  for (const recordTransitions of [true, false]) {
    const policy = decode(model([144]));
    let inferences = 0, resets = 0, replaying = false;
    const act = raw => {
      assert.equal(replaying, false);
      inferences++;
      return policy.act(raw);
    };
    act.reset = context => {
      assert.equal(replaying, false);
      resets++;
      policy.act.reset(context);
    };
    const report = runPolicy({seed: 94, task: {...DEFAULT_TASK, deadlineSeconds: 0.2}, act,
      options: {recordTransitions, policyIdentity: policy.identity}});
    assert.equal(resets, 1);
    assert.equal(inferences, 4);
    replaying = true;
    policy.act.reset();
    assert.throws(() => policy.act(Array(34).fill(0)), /public reference duration.*reset/i);
    assert.deepEqual(replay(JSON.parse(JSON.stringify(report))), report);
    if (recordTransitions) {
      const playback = new Playback(report, () => runner.createEpisode().controlFactory());
      assert.equal(playback.sim.constructor.name, 'VexRobotSimulator');
      playback.sim.isPaused = false;
      while (!playback.finished) playback.step();
      assert.equal(playback.verified, true);
      assert.deepEqual(playback.episode.actions, report.actions);
    }
    assert.equal(resets, 1);
    assert.equal(inferences, 4);
  }
});
