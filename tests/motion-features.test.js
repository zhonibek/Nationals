'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {policyFromSnapshot} = require('../roboproof/motion-learner');

function model() {
  const layers = [[35, 32], [32, 32], [32, 8]].map(([inputs, outputs]) => ({
    weight: Array.from({length: outputs}, () => Array(inputs).fill(0)), bias: Array(outputs).fill(0)}));
  layers[0].weight[0][19] = 1;
  layers[1].weight[0][0] = 1;
  layers[2].weight[0][0] = 1;
  return {schemaVersion: 1, algorithm: 'ppo-beta-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width: 32, featureTransform: 'reference-frame-v1', featureInputSize: 35,
    scales: Array(35).fill(1), layers, identity: loadHeadless().identity, trainingSeed: 503, steps: 10, updates: 1, trained: true};
}
const decode = value => policyFromSnapshot(value, createHash('sha256').update(JSON.stringify(value)).digest('hex'));

test('reference-frame inference consumes exactly 34 raw values without mutation or privileged inputs', () => {
  const policy = decode(model());
  const observation = Array(34).fill(0);
  observation[20] = 1;
  observation[21] = Math.PI / 2;
  const original = [...observation];
  const action = policy.act(observation);
  assert.deepEqual(observation, original);
  assert.equal(action.length, 4);
  assert(action[0] < 0);
  assert(action.every(value => Number.isFinite(value) && value >= -1 && value <= 1));
  assert.throws(() => policy.act([...observation, 1]), /sensor/);
});

test('unsupported transforms or mismatched feature widths fail closed', () => {
  for (const change of [data => { data.featureTransform = 'privileged-truth'; },
    data => { data.featureTransform = null; },
    data => { delete data.featureInputSize; }, data => { data.featureInputSize = 34; },
    data => { data.scales.pop(); }, data => { data.layers[0].weight[0].pop(); }]) {
    const data = model();
    change(data);
    assert.throws(() => decode(data), /(?:feature|model|weights)/i);
  }
});

test('temporal decoder keeps bounded history, resets explicitly and rejects altered definitions', () => {
  const data = model();
  data.featureTransform = 'history-context-v1';
  data.featureInputSize = 144;
  data.historyFrames = 4;
  data.initialContextSize = 4;
  data.scales = Array(144).fill(1);
  data.layers[0].weight = Array.from({length: 32}, () => Array(144).fill(0));
  data.layers[0].weight[0][3] = 1;
  data.layers[0].weight[0][140] = 1;
  const policy = decode(data);
  const first = Array(34).fill(0);
  first[3] = 0.1;
  first[10] = 0.2;
  const initial = policy.act(first);
  const next = [...first];
  next[3] = 0.8;
  next[10] = 0.9;
  assert.deepEqual(policy.act(next), initial);
  for (let tick = 0; tick < 4; tick++) policy.act(next);
  assert.notDeepEqual(policy.act(next), initial);
  policy.act.reset();
  assert.deepEqual(policy.act(first), initial);
  for (const change of [value => { value.historyFrames = 3; }, value => { value.initialContextSize = 5; },
    value => { value.scales.pop(); }]) {
    const altered = structuredClone(data);
    change(altered);
    assert.throws(() => decode(altered), /(?:temporal|model)/i);
  }
});
