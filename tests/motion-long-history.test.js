'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {policyFromSnapshot} = require('../roboproof/motion-learner');
const {boundedAuditJsonSha256} = require('../roboproof/motion-holdout');

function model() {
  const layers = [[565, 32], [32, 32], [32, 8]].map(([inputs, outputs]) => ({
    weight: Array.from({length: outputs}, () => Array(inputs).fill(0)), bias: Array(outputs).fill(0)}));
  [3, 528, 560, 564].forEach((feature, axis) => {
    layers[0].weight[axis][feature] = 1;
    layers[1].weight[axis][axis] = 1;
    layers[2].weight[axis][axis] = 1;
  });
  return {schemaVersion: 1, algorithm: 'ppo-beta-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width: 32, featureTransform: 'long-history-deadline-v1', featureInputSize: 565,
    historyFrames: 16, initialContextSize: 4, scales: Array(565).fill(1), layers, identity: loadHeadless().identity,
    trainingSeed: 931, steps: 10, updates: 1, trained: true,
    deadlineContext: {schemaVersion: 1, settlingReserveSeconds: 2, input: 'public-reference-duration-at-reset'}};
}

const decode = data => policyFromSnapshot(data, createHash('sha256').update(JSON.stringify(data)).digest('hex'));
const expectedAxis = feature => {
  const logit = Math.tanh(Math.tanh(Math.max(-10, Math.min(10, feature))));
  const alpha = Math.max(logit, 0) + Math.log1p(Math.exp(-Math.abs(logit))) + 1;
  return 2 * alpha / (alpha + Math.log(2) + 1) - 1;
};

test('fixed sixteen-frame inference pads, evicts, freezes initial context and resets', () => {
  const policy = decode(model());
  const raw = Array(34).fill(0);
  raw[18] = 10; raw[10] = 0.2;
  assert.throws(() => policy.act(raw), /reference duration/);
  policy.act.reset({referenceDurationSeconds: 4});
  for (let index = 0; index < 24; index++) {
    raw[3] = index * 0.1; raw[25] = index * 0.05;
    const result = policy.act(raw);
    const probes = [Math.max(0, index - 15) * 0.1, raw[3], 0.2, Math.max(0.25, (4 - raw[25]) / 8)];
    result.forEach((value, axis) => assert.ok(Math.abs(value - expectedAxis(probes[axis])) < 1e-12));
    raw[10] += 0.1;
  }
  policy.act.reset({referenceDurationSeconds: 4});
  const first = policy.act(raw);
  assert.equal(first[0], first[1]);
  assert.ok(Math.abs(first[2] - expectedAxis(raw[10])) < 1e-12);
});

test('long-window schema rejects mismatches and sparse raw observations', () => {
  for (const change of [value => { value.historyFrames = 4; }, value => { value.historyFrames = 32; },
    value => { value.featureInputSize = 145; }, value => { value.initialContextSize = 5; },
    value => { value.scales[564] = 2; }, value => { delete value.deadlineContext; },
    value => { value.observationSize = 35; }, value => { value.layers[0].weight[0].pop(); }]) {
    const data = model(); change(data);
    assert.throws(() => decode(data));
  }
  const policy = decode(model());
  policy.act.reset({referenceDurationSeconds: 4});
  assert.throws(() => policy.act(Array(34)), /sensor/);
  for (const bad of [NaN, Infinity, null, true]) {
    const raw = Array(34).fill(0); raw[3] = bad;
    assert.throws(() => policy.act(raw), /sensor/);
  }
});

test('a bounded long-window actor fits the existing per-candidate holdout audit', () => {
  const data = model();
  assert.equal(boundedAuditJsonSha256(data).length, 64);
  assert.ok(Buffer.byteLength(JSON.stringify(data)) < 1024 * 1024);
  const policy = decode(data);
  policy.act.reset({referenceDurationSeconds: 3});
  const raw = Array(34).fill(0); raw[18] = 6;
  assert.ok(policy.act(raw).every(value => Number.isFinite(value) && value >= -1 && value <= 1));
});
