'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {policyFromSnapshot} = require('../roboproof/motion-learner');

function model(algorithm) {
  return {schemaVersion: 1, algorithm, policyContractVersion: 1, observationSize: 34, actionSize: 4,
    width: 32, featureTransform: 'identity-v1', featureInputSize: 34, scales: Array(34).fill(1),
    layers: [[34, 32], [32, 32], [32, 8]].map(([inputs, outputs]) => ({
      weight: Array.from({length: outputs}, () => Array(inputs).fill(0)), bias: Array(outputs).fill(0)})),
    identity: loadHeadless().identity, trainingSeed: 18, steps: 0, updates: 0, trained: false};
}

const decode = data => policyFromSnapshot(data, createHash('sha256').update(JSON.stringify(data)).digest('hex'));

test('same bounded Beta mean decodes with explicit TD3 identity instead of pretending to be PPO', () => {
  const original = decode(model('ppo-beta-reference-v1'));
  const deterministic = decode(model('td3-beta-mean-reference-v1'));
  const observation = Array(34).fill(0);
  assert.deepEqual(deterministic.act(observation), original.act(observation));
  assert.equal(original.identity.id, 'ppo-seed-18-update-0');
  assert.equal(deterministic.identity.id, 'td3-seed-18-update-0');
  assert.equal(deterministic.identity.kind, 'untrained');
  assert.equal(deterministic.data.algorithm, 'td3-beta-mean-reference-v1');
  assert.equal(deterministic.data.observationSize, 34);
  assert.equal(deterministic.data.actionSize, 4);
});

test('new learner kind does not accept an unknown algorithm, different controller or malformed weights', () => {
  assert.throws(() => decode(model('td3-pretend-ppo')), /Unsupported/);
  const differentController = model('td3-beta-mean-reference-v1');
  differentController.identity = {};
  assert.throws(() => decode(differentController), /identity mismatch/);
  const malformed = model('td3-beta-mean-reference-v1');
  malformed.layers[2].weight[0][0] = NaN;
  assert.throws(() => decode(malformed), /Malformed/);
});
