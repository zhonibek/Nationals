'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {policyFromSnapshot} = require('../roboproof/motion-learner');
const {runPolicy, replay} = require('../roboproof/motion');

function model() {
  const frame = [1.524, 1.524, 3.142, 0.0254, 0.0254, 0.0873, 3, 3, 3, 3,
    0.02032, 0.02032, 0.035, 1, 1, 1, 1, 1, 10, 1.524, 1.524, 1, 2, 2, 4, 10, 1,
    0.114, 0.114, 0.1, 0.08, 0.08, 0.25, 40, 1];
  const scales = Array.from({length: 4}, () => frame).flat().concat([1.524, 1.524, 1, 1, 1]).map(Math.fround);
  const layers = [[145, 32], [32, 32], [32, 8]].map(([inputs, outputs]) => ({
    weight: Array.from({length: outputs}, () => Array(inputs).fill(0)), bias: Array(outputs).fill(0)}));
  [10, 108, 110, 117].forEach((feature, axis) => {
    layers[0].weight[axis][feature] = 1;
    layers[1].weight[axis][axis] = 1;
    layers[2].weight[axis][axis] = 1;
  });
  return {schemaVersion: 1, algorithm: 'ppo-beta-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width: 32, featureTransform: 'settling-normalized-deadline-v1',
    featureInputSize: 145, historyFrames: 4, initialContextSize: 4, scales, layers, identity: loadHeadless().identity,
    trainingSeed: 981, steps: 10, updates: 1, trained: true,
    deadlineContext: {schemaVersion: 1, settlingReserveSeconds: 2, input: 'public-reference-duration-at-reset'},
    normalizationDefinition: {schemaVersion: 1, method: 'sensor-settling-units-v1', historyFrames: 4,
      frameScales: {'3': 0.0254, '4': 0.0254, '5': 0.0873, '10': 0.02032, '11': 0.02032, '12': 0.035},
      heading: 'Goal-heading sine divided by0.035; cosine retained at scale1, not a success classifier',
      initialContext: 'Unchanged field-scale initial sensor goal context', clip: [-10, 10],
      scope: 'Fixed public sensor/reference units only; no evaluator truth, fitted statistics or new input capacity'}};
}

const decode = data => policyFromSnapshot(data, createHash('sha256').update(JSON.stringify(data)).digest('hex'));
const expectedAxis = feature => {
  const logit = Math.tanh(Math.tanh(Math.max(-10, Math.min(10, feature))));
  const alpha = Math.max(logit, 0) + Math.log1p(Math.exp(-Math.abs(logit))) + 1;
  return 2 * alpha / (alpha + Math.log(2) + 1) - 1;
};

test('six sensor units use fixed float32 scales with unchanged four-frame padding, eviction and clipping', () => {
  const data = model(), policy = decode(data);
  const raw = Array(34).fill(0);
  raw[18] = 10; raw[13] = 1;
  assert.throws(() => policy.act(raw), /reference duration/);
  policy.act.reset({referenceDurationSeconds: 4});
  for (let index = 0; index < 24; index++) {
    raw[10] = index * 0.02032; raw[3] = index * 0.00635;
    raw[5] = -index * 0.01; raw[12] = index * 0.0042;
    raw[13] = Math.sqrt(1 - raw[12] ** 2);
    const result = policy.act(raw);
    const probes = [Math.max(0, index - 3) * 0.02032 / data.scales[10], raw[3] / data.scales[108],
      raw[5] / data.scales[110], raw[12] / data.scales[117]];
    probes.forEach((probe, axis) => assert.ok(Math.abs(result[axis] - expectedAxis(probe)) < 1e-12));
  }
});

test('initial goal context remains at field units and reset clears every temporal state', () => {
  const data = model();
  data.layers[0].weight[0][10] = 0;
  data.layers[0].weight[0][140] = 1;
  const policy = decode(data);
  const raw = Array(34).fill(0);
  raw[18] = 10; raw[13] = 1; raw[10] = 0.02032;
  policy.act.reset({referenceDurationSeconds: 3});
  const first = policy.act(raw);
  raw[10] = 1;
  assert.equal(policy.act(raw)[0], first[0]);
  assert.ok(Math.abs(first[0] - expectedAxis(0.02032 / data.scales[140])) < 1e-12);
  policy.act.reset({referenceDurationSeconds: 3});
  assert.ok(Math.abs(policy.act(raw)[0] - expectedAxis(1 / data.scales[140])) < 1e-12);
  policy.act.reset();
  assert.throws(() => policy.act(raw), /reference duration/);
});

test('normalization metadata and every frame/context scale fail closed rather than accepting arbitrary units', () => {
  const mutations = [data => { data.normalizationDefinition = null; },
    data => { data.normalizationDefinition.frameScales['10'] = 1; },
    data => { data.normalizationDefinition.clip = [-100, 100]; },
    data => { data.scales[3] *= 2; }, data => { data.scales[115] *= 2; },
    data => { data.scales[140] = 0.02032; }, data => { data.scales[13] = 0.035; },
    data => { data.historyFrames = 16; }, data => { data.featureInputSize = 565; },
    data => { data.scales[3] = NaN; }];
  for (const mutate of mutations) {
    const data = model();
    mutate(data);
    assert.throws(() => decode(data), /mismatch|Unsupported/);
  }
  const policy = decode(model());
  policy.act.reset({referenceDurationSeconds: 1});
  assert.throws(() => policy.act(Array(34)), /sensor/);
  assert.throws(() => policy.act(Array(34).fill(NaN)), /sensor/);
});

test('fixed-unit toy actor executes and replays only through the original Simulator and controller', () => {
  const policy = decode(model());
  const report = runPolicy({seed: 98005, act: policy.act,
    options: {policyIdentity: {...policy.identity, id: 'test-fixture-not-trained-movement-evidence'}}});
  assert.deepEqual(report.identity, loadHeadless().identity);
  assert.ok(['success', 'time_limit'].includes(report.reason));
  assert.ok(report.transitions.length > 0);
  const replayed = replay(report);
  assert.deepEqual(replayed.metrics, report.metrics);
});
