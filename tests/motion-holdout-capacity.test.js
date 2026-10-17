'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {createLocalFilesystemHoldoutAudit, boundedAuditJsonSha256, AUDIT_LIMITS} = require('../roboproof/motion-holdout');

function fixture(context) {
  const parent = path.resolve(__dirname, '../.cache/holdout-capacity-tests');
  fs.mkdirSync(parent, {recursive: true});
  const directory = fs.mkdtempSync(path.join(parent, 'capacity-'));
  context.after(() => {
    assert.equal(path.dirname(path.resolve(directory)), parent);
    fs.rmSync(directory, {recursive: true, force: true});
  });
  const auditDirectory = path.join(directory, 'audit');
  const audit = createLocalFilesystemHoldoutAudit(auditDirectory, {
    readCurrentSourceRuntimeIdentity: () => ({sources: {synthetic: 'a'.repeat(64)},
      runtime: {node: process.version, v8: process.versions.v8, platform: process.platform, arch: process.arch}})});
  audit.commitFreshCorpus({cases: [{id: 'synthetic-only', seed: 71}]});
  return {directory, auditDirectory, audit};
}

function policy(longHistory) {
  const features = longHistory ? 565 : 145;
  return {schemaVersion: 1, algorithm: 'td3-beta-mean-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width: 256, trainingSeed: 71, steps: 0, updates: 0, trained: false,
    featureTransform: longHistory ? 'long-history-deadline-v1' : 'deadline-context-v1', featureInputSize: features,
    historyFrames: longHistory ? 16 : 4, initialContextSize: 4, scales: Array(features).fill(1),
    deadlineContext: {schemaVersion: 1, settlingReserveSeconds: 2, input: 'public-reference-duration-at-reset'},
    identity: structuredClone(loadHeadless().identity),
    layers: [[features, 256], [256, 256], [256, 8]].map(([inputs, outputs], layerIndex) => ({
      weight: Array.from({length: outputs}, (_, row) => Array.from({length: inputs}, (_, column) =>
        Math.fround(Math.sin((column + 1) * 0.013 + row * 0.019 + layerIndex) * 0.125))),
      bias: Array(outputs).fill(0)}))};
}

for (const longHistory of [false, true]) test(`validated width256 actor can freeze and reveal with long history=${longHistory}`, context => {
  const setup = fixture(context), data = policy(longHistory);
  assert.throws(() => boundedAuditJsonSha256(data), /structural limit|byte limit/);
  const filename = path.join(setup.directory, 'policy.json');
  const bytes = JSON.stringify(data);
  if (longHistory) assert.ok(Buffer.byteLength(bytes) > AUDIT_LIMITS.jsonBytes);
  fs.writeFileSync(filename, bytes);
  const frozen = setup.audit.freezeCandidateFilesBeforeReveal([{id: 'synthetic-model', path: filename}]);
  assert.equal(frozen.candidates[0].sha256, createHash('sha256').update(bytes).digest('hex'));
  const revealed = setup.audit.consumeAndRevealOnce();
  assert.deepEqual(revealed.candidates[0].json, data);
  assert.throws(() => setup.audit.consumeAndRevealOnce(), /already consumed/);
});

test('generic candidates keep the unchanged structural limit', context => {
  const setup = fixture(context), filename = path.join(setup.directory, 'generic.json');
  fs.writeFileSync(filename, JSON.stringify({weights: Array(AUDIT_LIMITS.nodes + 1).fill(0)}));
  assert.throws(() => setup.audit.freezeCandidateFilesBeforeReveal([{id: 'generic', path: filename}]), /structural limit|bounded dense JSON/);
  assert.equal(fs.existsSync(path.join(setup.auditDirectory, 'candidates.json')), false);
  assert.equal(setup.audit.readAuditStatus().consumed, null);
});

test('malformed motion candidates cannot claim the larger model budget', context => {
  const setup = fixture(context), filename = path.join(setup.directory, 'malformed.json');
  const data = policy(false);
  data.width = 999;
  fs.writeFileSync(filename, JSON.stringify(data));
  assert.throws(() => setup.audit.freezeCandidateFilesBeforeReveal([{id: 'malformed', path: filename}]), /Unsupported bounded motion model/);
  assert.equal(setup.audit.readAuditStatus().consumed, null);
});
