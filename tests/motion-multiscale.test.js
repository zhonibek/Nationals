'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {policyFromSnapshot} = require('../roboproof/motion-learner');
const {runPolicy, replay} = require('../roboproof/motion');
const profiles = ['coarse-augmentation-deadline-v1', 'multiscale-deadline-v1'];
const unitsFor = profile => profile === profiles[0] ? [2, 2, 4, 1.524, 1.524, 1] : [0.0254, 0.0254, 0.0873, 0.02032, 0.02032, 0.035];

function model(profile) {
  const frame = [1.524, 1.524, 3.142, 2, 2, 4, 3, 3, 3, 3,
    1.524, 1.524, 1, 1, 1, 1, 1, 1, 10, 1.524, 1.524, 1, 2, 2, 4, 10, 1,
    0.114, 0.114, 0.1, 0.08, 0.08, 0.25, 40, 1, 1, 1, 1, 1, 1, 1];
  const scales = Array.from({length: 4}, () => frame).flat().concat([1.524, 1.524, 1, 1, 1]).map(Math.fround);
  const layers = [[169, 32], [32, 32], [32, 8]].map(([inputs, outputs]) => ({
    weight: Array.from({length: outputs}, () => Array(inputs).fill(0)), bias: Array(outputs).fill(0)}));
  [10, 158, 161, 163].forEach((feature, axis) => {
    layers[0].weight[axis][feature] = 1;
    layers[1].weight[axis][axis] = 1;
    layers[2].weight[axis][axis] = 1;
  });
  return {schemaVersion: 1, algorithm: 'ppo-beta-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width: 32, featureTransform: profile,
    featureInputSize: 169, frameInputSize: 41, historyFrames: 4, initialContextSize: 4, scales, layers, identity: loadHeadless().identity,
    trainingSeed: 991, steps: 10, updates: 1, trained: true,
    deadlineContext: {schemaVersion: 1, settlingReserveSeconds: 2, input: 'public-reference-duration-at-reset'},
    normalizationDefinition: {schemaVersion: 1, method: 'bounded-sensor-augmentation-v1', historyFrames: 4,
      baseFrameSize: 35, augmentedFrameSize: 41, sourceIndices: [3, 4, 5, 10, 11, 12], units: unitsFor(profile),
      transform: 'tanh(reference-frame-feature/unit)', baseChannels: 'Unchanged field-scale reference-frame35',
      addedChannelScales: [1, 1, 1, 1, 1, 1], initialContext: 'Unchanged field-scale initial sensor goal context', clip: [-10, 10],
      scope: 'Derived public sensor/reference channels only; additional169-input capacity, no evaluator truth or fitted statistics'}};
}

const decode = data => policyFromSnapshot(data, createHash('sha256').update(JSON.stringify(data)).digest('hex'));
const expectedAxis = feature => {
  const logit = Math.tanh(Math.tanh(Math.max(-10, Math.min(10, feature))));
  const alpha = Math.max(logit, 0) + Math.log1p(Math.exp(-Math.abs(logit))) + 1;
  return 2 * alpha / (alpha + Math.log(2) + 1) - 1;
};

test('capacity-matched bounded channels retain coarse values, temporal order and independent local-unit formulas', () => {
  assert.deepEqual(model(profiles[0]).layers, model(profiles[1]).layers);
  assert.deepEqual(model(profiles[0]).scales, model(profiles[1]).scales);
  for (const profile of profiles) {
    const data = model(profile), policy = decode(data), units = unitsFor(profile);
    const raw = Array(34).fill(0);
    raw[18] = 10; raw[13] = 1;
    assert.throws(() => policy.act(raw), /reference duration/);
    policy.act.reset({referenceDurationSeconds: 4});
    for (let index = 0; index < 24; index++) {
      raw[10] = index * 0.15; raw[3] = index * 0.00635; raw[12] = index * 0.0042;
      raw[13] = Math.sqrt(1 - raw[12] ** 2);
      const result = policy.act(raw);
      const probes = [Math.max(0, index - 3) * 0.15 / data.scales[10], Math.tanh(raw[3] / units[0]),
        Math.tanh(raw[10] / units[3]), Math.tanh(raw[12] / units[5])];
      probes.forEach((probe, axis) => assert.ok(Math.abs(result[axis] - expectedAxis(probe)) < 1e-12));
    }
  }
});

test('initial field context stays frozen and reset clears every derived history frame', () => {
  for (const profile of profiles) {
    const data = model(profile);
    data.layers[0].weight[0][10] = 0;
    data.layers[0].weight[0][164] = 1;
    const policy = decode(data), raw = Array(34).fill(0);
    raw[18] = 10; raw[13] = 1; raw[10] = 0.02032;
    policy.act.reset({referenceDurationSeconds: 3});
    const first = policy.act(raw);
    raw[10] = 1;
    assert.equal(policy.act(raw)[0], first[0]);
    assert.ok(Math.abs(first[0] - expectedAxis(0.02032 / data.scales[164])) < 1e-12);
    policy.act.reset({referenceDurationSeconds: 3});
    assert.ok(Math.abs(policy.act(raw)[0] - expectedAxis(1 / data.scales[164])) < 1e-12);
    policy.act.reset();
    assert.throws(() => policy.act(raw), /reference duration/);
  }
});

test('augmentation formulas, units, channel order and both coarse/local scales are bound and fail closed', () => {
  const mutations = [data => { data.normalizationDefinition.units[3] *= 2; },
    data => { data.normalizationDefinition.sourceIndices.reverse(); },
    data => { data.normalizationDefinition.transform = 'identity'; },
    data => { data.scales[10] = 0.02032; }, data => { data.scales[35] = 2; },
    data => { data.scales[164] = 1; }, data => { data.frameInputSize = 35; },
    data => { data.featureInputSize = 145; }, data => { data.historyFrames = 16; },
    data => { data.normalizationDefinition = null; }];
  for (const profile of profiles) {
    for (const mutate of mutations) {
      const data = model(profile);
      mutate(data);
      assert.throws(() => decode(data), /mismatch/);
    }
    const policy = decode(model(profile));
    policy.act.reset({referenceDurationSeconds: 1});
    assert.throws(() => policy.act(Array(34)), /sensor/);
    assert.throws(() => policy.act(Array(34).fill(NaN)), /sensor/);
  }
});

test('both toy representations execute and replay through the original Simulator, not another physics model', () => {
  for (const profile of profiles) {
    const policy = decode(model(profile));
    const report = runPolicy({seed: 99007, act: policy.act,
      options: {policyIdentity: {...policy.identity, id: 'test-fixture-not-trained-movement-evidence'}}});
    assert.deepEqual(report.identity, loadHeadless().identity);
    assert.ok(['success', 'time_limit'].includes(report.reason));
    assert.ok(report.transitions.length > 0);
    assert.deepEqual(replay(report).metrics, report.metrics);
  }
});
