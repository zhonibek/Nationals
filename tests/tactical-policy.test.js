'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {FEATURE_COUNT} = require('../roboproof/tactical_environment');
const {createPolicy, loadPolicy} = require('../roboproof/tactical_policy');
const {fixturePolicyRecord} = require('./helpers/tactical-policy-fixture');

const features = value => Array.from({length: FEATURE_COUNT}, (_, index) => index === 0 ? value : 0.25);

test('the exported tactical actor scores candidates with two tanh layers and a linear head', () => {
  const policy = createPolicy(fixturePolicyRecord());
  assert.ok(Math.abs(policy.score(features(0.8)) - (Math.tanh(Math.tanh(0.8)) + 0.5)) < 1e-12);
  const choice = policy.choose([{id: 'low', features: features(-1)}, {id: 'high', features: features(2)}, {id: 'mid', features: features(0)}]);
  assert.equal(choice.action.id, 'high');
  assert.equal(policy.width, 16);
  assert.equal(policy.provenance.runId, 'fixture-run');
  assert.deepEqual([policy.algorithmId, policy.ppo], ['actor-critic', false]);
  const ppo = createPolicy(fixturePolicyRecord({overrides: {algorithmId: 'ppo-clip-v1', algorithm: 'PPO fixture'}}));
  assert.deepEqual([ppo.algorithmId, ppo.ppo, ppo.algorithm], ['ppo-clip-v1', true, 'PPO fixture']);
  assert.throws(() => policy.score(features(0).slice(1)), /finite numbers/);
  assert.throws(() => policy.score(features(Number.NaN)), /finite numbers/);
});

test('policy files with the wrong model, shapes, values or claims are refused', () => {
  const broken = {
    'wrong schema': {schema: 'other'},
    'wrong model': {model: 'some-other-model'},
    'wrong feature count': {featureCount: FEATURE_COUNT + 1},
    'too narrow': {width: 8},
    'wrong activation': {activation: 'relu'},
    'claims improvement': {learnedImprovementVerified: true},
    'claims qualification': {independentQualificationPerformed: true}
  };
  for (const [label, overrides] of Object.entries(broken)) {
    assert.throws(() => createPolicy(fixturePolicyRecord({overrides})), Error, label);
  }
  assert.throws(() => createPolicy(fixturePolicyRecord({overrides: {algorithmId: 'trpo'}})), /Unknown tactical policy algorithm trpo/);
  const shape = fixturePolicyRecord();
  shape.layers.hidden2.weight.pop();
  assert.throws(() => createPolicy(shape), /hidden2.weight/);
  const nan = fixturePolicyRecord();
  nan.layers.hidden1.bias[3] = Number.NaN;
  assert.throws(() => createPolicy(nan), /hidden1.bias/);
  const head = fixturePolicyRecord();
  head.layers.policy.bias = 'x';
  assert.throws(() => createPolicy(head), /policy.bias/);
});

test('loadPolicy reads an exported file and refuses a missing one', t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tactical-policy-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const file = path.join(directory, 'policy.json');
  fs.writeFileSync(file, JSON.stringify(fixturePolicyRecord({width: 20})));
  assert.equal(loadPolicy(file).width, 20);
  assert.throws(() => loadPolicy(path.join(directory, 'missing.json')), /ENOENT/);
});
