'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {policy, policyFromSnapshot, readJson} = require('../roboproof/motion-learner');
const {createLocalFilesystemHoldoutAudit} = require('../roboproof/motion-holdout');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

function fixture(width = 32, longHistory = false) {
  const features = longHistory ? 565 : 145;
  return {schemaVersion: 1, algorithm: 'td3-beta-mean-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width, hiddenLayers: 10, trainingSeed: 21, steps: 0, updates: 0, trained: false,
    featureTransform: longHistory ? 'long-history-deadline-v1' : 'deadline-context-v1', featureInputSize: features,
    historyFrames: longHistory ? 16 : 4, initialContextSize: 4, scales: Array(features).fill(1),
    deadlineContext: {schemaVersion: 1, settlingReserveSeconds: 2, input: 'public-reference-duration-at-reset'},
    identity: structuredClone(loadHeadless().identity),
    layers: [[features, width], ...Array.from({length: 9}, () => [width, width]), [width, 8]].map(([inputs, outputs], layerIndex) => ({
      weight: Array.from({length: outputs}, (_, row) => Array.from({length: inputs}, (_, column) =>
        Math.fround(Math.sin(column * 0.013 + row * 0.019 + layerIndex) * 0.125))), bias: Array(outputs).fill(0)}))};
}

function directory(context) {
  const parent = path.resolve(__dirname, '../.cache/depth-tests');
  fs.mkdirSync(parent, {recursive: true});
  const target = fs.mkdtempSync(path.join(parent, 'depth-'));
  context.after(() => {
    assert.equal(path.dirname(path.resolve(target)), parent);
    fs.rmSync(target, {recursive: true, force: true});
  });
  return target;
}

test('only declared two/ten layer topology is admitted', () => {
  const data = fixture();
  for (const depth of [null, false, true, 0, 1, 3, 9, 11, 2.5, '10']) {
    assert.throws(() => policyFromSnapshot({...data, hiddenLayers: depth}, 'a'.repeat(64)), /hidden layer count/);
  }
  assert.throws(() => policyFromSnapshot({...data, layers: data.layers.slice(0, 3)}, 'a'.repeat(64)), /bounded motion model/);
  const absent = structuredClone(data);
  delete absent.hiddenLayers;
  assert.throws(() => policyFromSnapshot(absent, 'a'.repeat(64)), /bounded motion model/);
});

test('all ten hidden layers affect inference and late layers are validated', () => {
  const data = fixture(), raw = Array(34).fill(0.1);
  raw[18] = 8; raw[25] = 0.4;
  const original = policyFromSnapshot(data, 'a'.repeat(64));
  original.act.reset({referenceDurationSeconds: 4});
  const action = original.act(raw);
  const changed = structuredClone(data);
  changed.layers[9].bias.fill(1);
  const alternate = policyFromSnapshot(changed, 'b'.repeat(64));
  alternate.act.reset({referenceDurationSeconds: 4});
  assert.notDeepEqual(alternate.act(raw), action);
  changed.layers[9].weight[0][0] = Infinity;
  assert.throws(() => policyFromSnapshot(changed, 'b'.repeat(64)), /Malformed motion model weights/);
});

test('largest supported deep actor freezes and reveals with candidate-only limits', context => {
  const target = directory(context), data = fixture(256, true), bytes = JSON.stringify(data);
  const filename = path.join(target, 'actor.json');
  assert.ok(Buffer.byteLength(bytes) > 8 * 1024 * 1024 && Buffer.byteLength(bytes) < 24 * 1024 * 1024);
  fs.writeFileSync(filename, bytes);
  assert.throws(() => readJson(filename, 4 * 1024 * 1024), /size budget/);
  assert.equal(policy(filename, hash(bytes)).data.hiddenLayers, 10);
  const audit = createLocalFilesystemHoldoutAudit(path.join(target, 'audit'), {
    readCurrentSourceRuntimeIdentity: () => ({sources: {synthetic: 'a'.repeat(64)}, runtime: {node: process.version}})});
  audit.commitFreshCorpus({cases: [{id: 'synthetic-only', seed: 21}]});
  audit.freezeCandidateFilesBeforeReveal([{id: 'future-depth-fixture', path: filename}]);
  const revealed = audit.consumeAndRevealOnce();
  assert.equal(revealed.candidates[0].sha256, hash(bytes));
  assert.equal(revealed.candidates[0].json.layers.length, 11);
  assert.throws(() => audit.consumeAndRevealOnce(), /already consumed/);
  fs.appendFileSync(filename, ' '.repeat(24 * 1024 * 1024 + 1 - Buffer.byteLength(bytes)));
  assert.throws(() => policy(filename), /size budget/);
});
