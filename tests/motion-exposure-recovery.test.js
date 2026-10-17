'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const fixture = require('./fixtures/motion-exposure-recovery-catalog.json');
const {RECOVERY_DESCRIPTOR, validateEnvelopeCatalog} = require('../roboproof/motion-exposure-recovery');
const {collectExposureLedger} = require('../roboproof/motion-exposures');

const copy = () => structuredClone(fixture);

test('recovery catalog binds all 45 ordered potential shards and the full offline corpus', () => {
  const {manifest, audit} = copy();
  const inventory = validateEnvelopeCatalog(manifest, audit);
  assert.equal(inventory.length, 47);
  assert.equal(new Set(inventory).size, 47);
  assert.equal(manifest.shards.reduce((sum, row) => sum + row.count, 0), 43008);
  assert(inventory.includes('offline-corpus/exposure.json'));
});

test('catalog rejects missing, reordered, duplicated or rebound exclusion shards', () => {
  const mutations = [
    manifest => manifest.shards.pop(),
    manifest => manifest.shards.reverse(),
    manifest => { manifest.shards[1] = structuredClone(manifest.shards[0]); },
    manifest => { manifest.shards[0].path = '../outside.json'; },
    manifest => { manifest.shards[0].firstOrdinal = 1; },
    manifest => { manifest.shards[14].count = 335; },
    manifest => { manifest.shards[0].trainingSeed = 136002; },
    manifest => { manifest.shards[0].sha256 = '0'.repeat(64); },
    manifest => { manifest.offlineCorpus.count = 127; },
    manifest => { manifest.budget.maximumGeneratedWorlds = 14335; },
    manifest => { manifest.trainingSeeds.reverse(); }
  ];
  for (const mutate of mutations) {
    const {manifest, audit} = copy();
    mutate(manifest);
    assert.throws(() => validateEnvelopeCatalog(manifest, audit));
  }
});

test('catalog cannot turn metadata reconstruction into measured queries or a qualification verdict', () => {
  for (const field of ['actualQueriesInferred', 'modelOrResultRecoveryClaimed']) {
    const {manifest, audit} = copy();
    manifest[field] = true;
    assert.throws(() => validateEnvelopeCatalog(manifest, audit));
  }
  for (const field of ['actualQueriesInferred', 'modelOrResultRecoveryClaimed', 'qualificationClaimed', 'lossRegistryModified']) {
    const {manifest, audit} = copy();
    audit[field] = true;
    assert.throws(() => validateEnvelopeCatalog(manifest, audit));
  }
  const {manifest, audit} = copy();
  audit.lossStateRemains = 'resolved';
  assert.throws(() => validateEnvelopeCatalog(manifest, audit));
});

test('a verified flag cannot replace the exact original auditor, input package, manifest or runtime', () => {
  const mutations = [
    audit => { audit.auditor.sha256 = '0'.repeat(64); },
    audit => { audit.sourcePackage.sha256 = '0'.repeat(64); },
    audit => { audit.envelopeManifest.sha256 = '0'.repeat(64); },
    audit => { audit.runtime.python = '3.13.15'; },
    audit => { audit.counters.worldRowsCompared--; },
    audit => { delete audit.envelopeFiles['offline-corpus/exposure.json']; },
    audit => { audit.envelopeFiles['extra.json'] = {bytes: 1, sha256: '0'.repeat(64)}; }
  ];
  for (const mutate of mutations) {
    const {manifest, audit} = copy();
    mutate(audit);
    assert.throws(() => validateEnvelopeCatalog(manifest, audit));
  }
});

function workspace(context, patch = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-recovery-'));
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const write = (relative, value) => {
    const filename = path.join(root, relative);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, JSON.stringify(value));
  };
  const world = {seed: 1, task: {start: {xIn: 0, yIn: 0, headingDeg: 0}, goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 10}};
  for (const name of ['motion-evaluation-worlds.json', 'motion-evaluation-worlds-v2.json']) write(`tests/fixtures/${name}`, {cases: [world]});
  const marker = {schemaVersion: 1, specification: 'unavailable-motion-exposure-v1', state: 'unresolved',
    trainingSeeds: [136001, 136002, 136003], observedRunIds: ['95663b0d-135b-4bc0-b36d-57d1699296f7', 'cb350f34-c592-4703-a2b9-73ec08a0b8c1'],
    lastVerifiedObservation: '2026-10-16T10:17:44.382861+00:00', lossObservedAt: '2026-10-16T16:35:58.364892+00:00',
    study: 'episode-balanced-remote-study-20261016', recovery: {...RECOVERY_DESCRIPTOR}, ...patch};
  write('roboproof/motion-exposure-losses.json', {schemaVersion: 1, specification: 'unavailable-motion-exposure-registry-v1', losses: [marker]});
  return {root, write, read: () => collectExposureLedger({root})};
}

test('a fresh checkout with only an admission descriptor still fails without the real preserved evidence', context => {
  const saved = workspace(context);
  assert.throws(saved.read, /ENOENT/);
});

test('a tiny self-certified replacement cannot impersonate the pinned historical inspection', context => {
  const saved = workspace(context);
  saved.write(RECOVERY_DESCRIPTOR.historicalInspectionFile, {verified: true, completeForIndependentFinal: true, sourceFiles: [], worldSeeds: []});
  assert.throws(saved.read, /changed evidence/);
});

test('admission cannot be rebound to another event, seed, source directory or audit', context => {
  for (const patch of [{trainingSeeds: [136001]}, {observedRunIds: []}, {lossObservedAt: 'later'}, {study: 'another-study'},
    {recovery: {...RECOVERY_DESCRIPTOR, auditSha256: '0'.repeat(64)}},
    {recovery: {...RECOVERY_DESCRIPTOR, envelopeDirectory: '../outside'}}, {recovery: {...RECOVERY_DESCRIPTOR, ignoreFailures: true}}]) {
    const saved = workspace(context, patch);
    assert.throws(saved.read, /training exposure is unavailable/);
  }
});

test('the descriptor is immutable and explicitly preserves unresolved-event semantics', () => {
  assert.equal(Object.isFrozen(RECOVERY_DESCRIPTOR), true);
  assert.equal(Object.hasOwn(RECOVERY_DESCRIPTOR, 'state'), false);
  assert.throws(() => { RECOVERY_DESCRIPTOR.auditSha256 = '0'.repeat(64); }, TypeError);
});
