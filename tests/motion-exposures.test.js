'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {createLocalFilesystemHoldoutAudit} = require('../roboproof/motion-holdout');
const {collectExposureLedger, inspectRetainedExposureHistory, assertFreshCases, geometryFingerprint, snapshotExposureLedger,
  readExposureSnapshot, assertExposureSnapshotCurrent} = require('../roboproof/motion-exposures');

const task = (distance = 24, deadlineSeconds = 10) => ({start: {xIn: 0, yIn: 0, headingDeg: 0},
  goal: {xIn: 0, yIn: distance, headingDeg: 0}, deadlineSeconds});
const world = (seed, distance = 24) => ({id: `world-${seed}`, seed, task: task(distance)});

function fixture(context) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-exposures-'));
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const write = (relative, value) => {
    const filename = path.join(root, relative);
    fs.mkdirSync(path.dirname(filename), {recursive: true});
    fs.writeFileSync(filename, JSON.stringify(value));
    return filename;
  };
  write('tests/fixtures/motion-evaluation-worlds.json', {cases: [world(1, 12)]});
  write('tests/fixtures/motion-evaluation-worlds-v2.json', {cases: [world(2, 13)]});
  write('roboproof/runs/motion/learning/example/run.json', {models: [{history: [{worlds: [world(3, 14), {seed: 4}]}]}]});
  write('roboproof/runs/motion/research/example/trial-protocol.json', {development: {cases: [world(5, 15)]}});
  return {root, write, read: () => collectExposureLedger({root})};
}

test('ledger binds saved training, development and historical tests without inventing legacy geometry', context => {
  const saved = fixture(context);
  const ledger = saved.read();
  assert.deepEqual(ledger.worldSeeds, [1, 2, 3, 4, 5]);
  assert.equal(ledger.bodyGeometries.length, 4);
  assert.equal(ledger.seedOnlyRecords, 1);
  assert.equal(ledger.legacyGeometryFullyKnown, false);
  assert.equal(ledger.sourceFiles.length, 4);
  assert.equal(assertFreshCases([world(9, 30)], ledger).verified, true);
  assert.throws(() => assertFreshCases([world(3, 30)], ledger), /overlaps/);
  assert.throws(() => assertFreshCases([world(9, 14)], ledger), /overlaps/);
  assert.throws(() => assertFreshCases([{...world(9, 14), task: task(14, 6)}], ledger), /overlaps/);
});

test('body-relative geometry detects translated/rotated variants and ignores deadlines', () => {
  const translated = {start: {xIn: 5, yIn: 6, headingDeg: 0}, goal: {xIn: 5, yIn: 30, headingDeg: 0}, deadlineSeconds: 6};
  const rotated = {start: {xIn: 0, yIn: 0, headingDeg: 90}, goal: {xIn: 24, yIn: 0, headingDeg: 90}, deadlineSeconds: 6};
  assert.equal(geometryFingerprint(translated), geometryFingerprint(task()));
  assert.equal(geometryFingerprint(rotated), geometryFingerprint(task()));
  assert.notEqual(geometryFingerprint(task(25)), geometryFingerprint(task()));
});

test('default TD3 training and nested research probes join the independent exclusion ledger', context => {
  const saved = fixture(context);
  saved.write('roboproof/runs/motion/td3/default-run/run.json', {models: [{history: [{worlds: [world(31, 35)]}]}]});
  saved.write('roboproof/runs/motion/research/probe/resumed/run/exposure.json', {cases: [world(32, 36)]});
  const ledger = saved.read();
  assert.equal(ledger.worldSeeds.includes(31), true);
  assert.equal(ledger.worldSeeds.includes(32), true);
  assert.equal(ledger.bodyGeometries.includes(geometryFingerprint(task(35))), true);
  assert.equal(ledger.sourceFiles.some(file => file.path.includes('/motion/td3/')), true);
  assert.throws(() => assertFreshCases([world(31, 40)], ledger), /overlaps/);
  assert.throws(() => assertFreshCases([{...world(33, 35), task: task(35, 6)}], ledger), /overlaps/);
  assert.throws(() => assertFreshCases([world(32, 40)], ledger), /overlaps/);
  assert.equal(assertFreshCases([world(34, 37)], ledger).verified, true);
});

test('GVF TD3 started and completed worlds join exposure exclusions, including interrupted episodes', context => {
  const saved = fixture(context), relative = 'roboproof/runs/motion/gvf-td3/fit-1/exposures.jsonl';
  const filename = path.join(saved.root, relative), sourceManifestSha256 = 'a'.repeat(64);
  fs.mkdirSync(path.dirname(filename), {recursive: true});
  const first = world(60, 25), interrupted = world(61, 26);
  const events = [
    {event: 'started', episode: 1, world: first, sourceManifestSha256},
    {event: 'completed', episode: 1, world: first, sourceManifestSha256, decisionCount: 12},
    {event: 'started', episode: 2, world: interrupted, sourceManifestSha256},
    {event: 'partial-unresolved', episode: 2, world: interrupted, sourceManifestSha256, decisions: 3}
  ];
  fs.writeFileSync(filename, events.map(value => JSON.stringify(value)).join('\n') + '\n');
  const ledger = saved.read();
  assert.equal(ledger.worldSeeds.includes(60), true);
  assert.equal(ledger.worldSeeds.includes(61), true);
  assert.equal(ledger.roleRecordCounts['gvf-training'], 2);
  assert.equal(ledger.sourceFiles.some(file => file.path === relative), true);
  assert.throws(() => assertFreshCases([world(60, 40)], ledger), /overlaps/);
  assert.throws(() => assertFreshCases([world(99, 25)], ledger), /overlaps/);
  assert.equal(assertFreshCases([world(62, 27)], ledger).verified, true);
});

test('incomplete or mismatched GVF JSONL exposure events fail closed', context => {
  const saved = fixture(context), relative = 'roboproof/runs/motion/gvf-td3/fit-1/exposures.jsonl';
  const filename = path.join(saved.root, relative), sourceManifestSha256 = 'b'.repeat(64);
  fs.mkdirSync(path.dirname(filename), {recursive: true});
  const first = world(70, 25), other = world(71, 26);
  const start = {event: 'started', episode: 1, world: first, sourceManifestSha256};
  fs.writeFileSync(filename, JSON.stringify(start));
  assert.throws(saved.read, /Incomplete GVF exposure JSONL record/);
  fs.writeFileSync(filename, [start, {event: 'completed', episode: 1, world: other,
    sourceManifestSha256, decisionCount: 12}].map(value => JSON.stringify(value)).join('\n') + '\n');
  assert.throws(saved.read, /does not match its recorded start/);
});

test('broad development cases independently exclude both their seeds and body-relative geometries', context => {
  const saved = fixture(context);
  saved.write('roboproof/runs/motion/research/broad-only/trial-protocol.json', {
    development: {cases: [world(40, 41)]}, broadDevelopment: {cases: [world(41, 42), world(42, 43)]}
  });
  const ledger = saved.read();
  assert.deepEqual(ledger.worldSeeds, [1, 2, 3, 4, 5, 40, 41, 42]);
  assert.equal(ledger.observedWorldRecords, 8);
  assert.equal(ledger.roleRecordCounts.development, 4);
  assert.equal(ledger.bodyGeometries.includes(geometryFingerprint(task(42))), true);
  assert.equal(ledger.bodyGeometries.includes(geometryFingerprint(task(43))), true);
  assert.throws(() => assertFreshCases([world(41, 50)], ledger), /overlaps/);
  const translated = {start: {xIn: 5, yIn: 6, headingDeg: 0},
    goal: {xIn: 5, yIn: 48, headingDeg: 0}, deadlineSeconds: 6};
  assert.throws(() => assertFreshCases([{...world(50, 50), task: translated}], ledger), /overlaps/);
  assert.equal(assertFreshCases([world(51, 51)], ledger).verified, true);
});

test('broad-only additions invalidate a frozen exposure snapshot before reveal', context => {
  const saved = fixture(context), directory = path.join(saved.root, 'reserved-final');
  snapshotExposureLedger(directory, {root: saved.root});
  saved.write('roboproof/runs/motion/research/broad-only/trial-protocol.json', {broadDevelopment: {cases: [world(43, 45)]}});
  assert.throws(() => assertExposureSnapshotCurrent(directory, {root: saved.root}), /before reveal/);
  assert.throws(() => assertFreshCases([world(43, 50)], saved.read()), /overlaps/);
});

test('malformed worlds in broad development fail closed rather than silently disappearing', context => {
  const saved = fixture(context);
  const filename = 'roboproof/runs/motion/research/broad-only/trial-protocol.json';
  saved.write(filename, {broadDevelopment: {cases: [{seed: -1}]}});
  assert.throws(saved.read, /uint32/);
  saved.write(filename, {broadDevelopment: {cases: [{seed: 44, task: {}}]}});
  assert.throws(saved.read);
});

test('every declared exposure case collection must be an array', context => {
  const saved = fixture(context);
  const filename = 'roboproof/runs/motion/research/malformed/trial-protocol.json';
  for (const value of [{cases: {}}, {development: {cases: null}}, {broadDevelopment: {cases: {}}}]) {
    saved.write(filename, value);
    assert.throws(saved.read, /cases.*array/i);
  }
});

test('lost remote training exposure blocks independent exclusion instead of disappearing with its runtime', context => {
  const saved = fixture(context);
  saved.write('roboproof/runs/motion/research/lost/exposure-loss.json', {
    schemaVersion: 1, specification: 'unavailable-motion-exposure-v1', state: 'unresolved', trainingSeeds: [136001, 136002, 136003]
  });
  assert.throws(saved.read, /training exposure is unavailable/);
  const directory = path.join(saved.root, 'new-final');
  assert.throws(() => snapshotExposureLedger(directory, {root: saved.root}), /training exposure is unavailable/);
  assert.equal(fs.existsSync(directory), false);
});

test('fresh actor files and replacement runtime metadata cannot resolve missing old exposures', context => {
  const saved = fixture(context);
  saved.write('roboproof/runs/motion/research/lost/exposure-loss.json', {
    schemaVersion: 1, specification: 'unavailable-motion-exposure-v1', state: 'unresolved', trainingSeeds: [136001]
  });
  saved.write('roboproof/runs/motion/research/replacement/run.json', {models: [{history: [{worlds: [world(200, 60)]}]}]});
  saved.write('roboproof/runs/motion/research/replacement/runtime.json', {trainingCompleted: true, allSeedsPassed: true});
  assert.throws(saved.read, /training exposure is unavailable/);
});

test('versioned loss registry also protects a checkout without ignored remote run files', context => {
  const saved = fixture(context);
  saved.write('roboproof/motion-exposure-losses.json', {
    schemaVersion: 1, specification: 'unavailable-motion-exposure-registry-v1',
    losses: [{schemaVersion: 1, specification: 'unavailable-motion-exposure-v1', state: 'unresolved', trainingSeeds: [136001]}]
  });
  assert.throws(saved.read, /training exposure is unavailable/);
});

test('retained-history inspection preserves missing events but cannot qualify an independent final', context => {
  const saved = fixture(context);
  const marker = {schemaVersion: 1, specification: 'unavailable-motion-exposure-v1', state: 'unresolved', trainingSeeds: [136001]};
  saved.write('roboproof/motion-exposure-losses.json', {
    schemaVersion: 1, specification: 'unavailable-motion-exposure-registry-v1', losses: [marker]
  });
  saved.write('roboproof/runs/motion/research/lost/exposure-loss.json', marker);
  const inspection = inspectRetainedExposureHistory({root: saved.root});
  assert.equal(inspection.specification, 'incomplete-retained-motion-history-inspection-v1');
  assert.equal(inspection.completeForIndependentFinal, false);
  assert.deepEqual(inspection.unavailableLossMarkers, [marker, marker]);
  assert.deepEqual(inspection.worldSeeds, [1, 2, 3, 4, 5]);
  assert.equal(inspection.sourceFiles.some(file => file.role === 'unavailable-training-exposure-registry'), true);
  assert.throws(() => assertFreshCases([world(9, 30)], inspection), /valid exposure ledger/);
  assert.throws(saved.read, /training exposure is unavailable/);
  assert.throws(() => collectExposureLedger({root: saved.root, onUnavailable: () => {}}), /training exposure is unavailable/);
  assert.throws(() => snapshotExposureLedger(path.join(saved.root, 'must-not-create'), {root: saved.root}), /training exposure is unavailable/);
  assert.equal(fs.existsSync(path.join(saved.root, 'must-not-create')), false);
});

test('diagnostic inspection cannot silently accept a self-dismissed loss event', context => {
  const saved = fixture(context);
  saved.write('roboproof/runs/motion/research/lost/exposure-loss.json', {
    schemaVersion: 1, specification: 'unavailable-motion-exposure-v1', state: 'resolved', trainingSeeds: [136001]
  });
  assert.throws(() => inspectRetainedExposureHistory({root: saved.root}), /cannot be silently dismissed/);
});

test('malformed versioned loss registries cannot silently omit lost training', context => {
  const saved = fixture(context);
  for (const registry of [{}, {schemaVersion: 1, specification: 'unavailable-motion-exposure-registry-v1', losses: null},
    {schemaVersion: 1, specification: 'unavailable-motion-exposure-registry-v1', losses: Array(33).fill({})}]) {
    saved.write('roboproof/motion-exposure-losses.json', registry);
    assert.throws(saved.read, /Invalid unavailable training-exposure registry/);
  }
});

test('malformed or self-dismissed loss markers fail closed', context => {
  const saved = fixture(context), filename = 'roboproof/runs/motion/research/lost/exposure-loss.json';
  const valid = {schemaVersion: 1, specification: 'unavailable-motion-exposure-v1', state: 'unresolved', trainingSeeds: [136001]};
  for (const patch of [{state: 'resolved'}, {trainingSeeds: []}, {trainingSeeds: [136001, 136001]},
    {trainingSeeds: [-1]}, {trainingSeeds: [true]}, {schemaVersion: 2}, {specification: 'ignore-me'}]) {
    saved.write(filename, {...valid, ...patch});
    assert.throws(saved.read, /Invalid unavailable training-exposure marker/);
  }
});

test('unconsumed cases are never read while consumed cases join the exclusion ledger', context => {
  const saved = fixture(context);
  const directory = path.join(saved.root, 'roboproof/runs/motion/holdout/used');
  const identity = {sources: {'fixture.js': 'a'.repeat(64)}, runtime: {node: 'fixture'}};
  const audit = createLocalFilesystemHoldoutAudit(directory, {readCurrentSourceRuntimeIdentity: () => identity});
  audit.commitFreshCorpus({cases: [world(6, 16)]});
  assert.equal(saved.read().worldSeeds.includes(6), false);
  const candidate = saved.write('candidate.json', {fixture: true});
  audit.freezeCandidateFilesBeforeReveal([{id: 'fixture', path: candidate}]);
  audit.consumeAndRevealOnce();
  assert.equal(saved.read().worldSeeds.includes(6), true);
  const untouched = saved.write('roboproof/runs/motion/holdout/untouched/commitment.json', {});
  fs.writeFileSync(untouched, 'malformed secret corpus must not be inspected');
  assert.equal(saved.read().worldSeeds.includes(6), true);
});

test('new saved exposures change the fingerprint rather than silently retaining an old ledger', context => {
  const saved = fixture(context);
  const first = assertFreshCases([world(9, 30)], saved.read()).ledgerSha256;
  saved.write('roboproof/runs/motion/research/new/exposure.json', {cases: [world(8, 19)]});
  const second = assertFreshCases([world(9, 30)], saved.read()).ledgerSha256;
  assert.notEqual(first, second);
});

test('malformed retained worlds and missing historical fixtures fail closed', context => {
  const saved = fixture(context);
  saved.write('roboproof/runs/motion/learning/example/run.json', {models: [{history: [{worlds: [{seed: -1}]}]}]});
  assert.throws(saved.read, /uint32/);
  saved.write('roboproof/runs/motion/learning/example/run.json', {models: [{history: [{worlds: []}]}]});
  fs.unlinkSync(path.join(saved.root, 'tests/fixtures/motion-evaluation-worlds-v2.json'));
  assert.throws(saved.read);
});

function retainedHistoryEntries(context, saved, count, symbolic = false) {
  const directory = path.join(saved.root, 'roboproof/runs/motion/research/example');
  const originalRead = fs.readdirSync;
  const entries = Array.from({length: count}, (_, index) => ({name: `ignored-${String(index).padStart(5, '0')}.json`,
    isDirectory: () => false, isFile: () => true, isSymbolicLink: () => symbolic && index === count - 1}));
  context.mock.method(fs, 'readdirSync', (filename, options) => {
    const actual = originalRead(filename, options);
    return path.resolve(filename) === directory ? [...actual, ...entries] : actual;
  });
}

test('large retained histories keep every later exposure instead of truncating the ledger', context => {
  const saved = fixture(context);
  saved.write('roboproof/runs/motion/research/example/worlds.json', {cases: [world(99, 44)]});
  const expected = saved.read();
  retainedHistoryEntries(context, saved, 16001);
  assert.deepEqual(saved.read(), expected);
  assert.throws(() => assertFreshCases([world(99, 46)], saved.read()), /overlaps/);
  assert.throws(() => assertFreshCases([world(100, 44)], saved.read()), /overlaps/);
});

test('expanded history traversal still fails closed at its finite entry budget', context => {
  const saved = fixture(context);
  retainedHistoryEntries(context, saved, 32769);
  assert.throws(saved.read, /entry budget/);
});

test('large histories do not hide symbolic links behind irrelevant files', context => {
  const saved = fixture(context);
  retainedHistoryEntries(context, saved, 16001, true);
  assert.throws(saved.read, /symlink guard/);
});

test('exclusive exposure snapshots detect later evidence changes before final reveal', context => {
  const saved = fixture(context), directory = path.join(saved.root, 'reserved-final');
  const snapshot = snapshotExposureLedger(directory, {root: saved.root});
  assert.equal(assertExposureSnapshotCurrent(directory, {root: saved.root}).ledgerSha256, snapshot.ledgerSha256);
  assert.equal(readExposureSnapshot(directory).ledgerSha256, snapshot.ledgerSha256);
  assert.throws(() => snapshotExposureLedger(directory, {root: saved.root}), /exist/i);
  saved.write('roboproof/runs/motion/research/new/exposure.json', {cases: [world(11, 20)]});
  assert.throws(() => assertExposureSnapshotCurrent(directory, {root: saved.root}), /before reveal/);
});
