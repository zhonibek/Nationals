'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {createLocalFilesystemHoldoutAudit} = require('../roboproof/motion-holdout');
const {collectExposureLedger, assertFreshCases, geometryFingerprint, snapshotExposureLedger,
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

test('exclusive exposure snapshots detect later evidence changes before final reveal', context => {
  const saved = fixture(context), directory = path.join(saved.root, 'reserved-final');
  const snapshot = snapshotExposureLedger(directory, {root: saved.root});
  assert.equal(assertExposureSnapshotCurrent(directory, {root: saved.root}).ledgerSha256, snapshot.ledgerSha256);
  assert.equal(readExposureSnapshot(directory).ledgerSha256, snapshot.ledgerSha256);
  assert.throws(() => snapshotExposureLedger(directory, {root: saved.root}), /exist/i);
  saved.write('roboproof/runs/motion/research/new/exposure.json', {cases: [world(11, 20)]});
  assert.throws(() => assertExposureSnapshotCurrent(directory, {root: saved.root}), /before reveal/);
});
