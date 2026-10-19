'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const Experiment = require('../roboproof/motion-experiment');

test('movement commitments bind the optional episode replay implementation', () => {
  const filename = 'roboproof/motion_learning/episode_replay.py';
  const source = fs.readFileSync(path.join(__dirname, '..', filename), 'utf8').replace(/\r\n/g, '\n');
  assert.equal(Experiment.identity().sources[filename], createHash('sha256').update(source).digest('hex'));
});

test('fresh mixed-domain commitment reports only metadata before candidates are frozen', context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-final-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const result = Experiment.prepare(directory, {shortFamilies: 1, fieldFamilies: 1, fullFamilies: 1});
  assert.equal(result.caseCount, 12);
  assert.equal(result.corpus, undefined);
  assert.equal(result.identity, undefined);
  assert.match(result.casesSha256, /^[a-f0-9]{64}$/);
  assert.throws(() => Experiment.audit(directory).consumeAndRevealOnce(), /frozen before/);
  const protocol = JSON.parse(fs.readFileSync(path.join(directory, 'protocol.json')));
  assert.equal(protocol.acceptance.independentTrainingSeedsRequired, 3);
  assert.equal(protocol.acceptance.minimumMeanTimeOrEffortImprovementFraction, 0.05);
  assert.match(protocol.knownExposure.ledgerSha256, /^[a-f0-9]{64}$/);
  assert.equal(protocol.knownExposure.verified, true);
  assert.equal(fs.existsSync(path.join(directory, 'exposure-ledger.json')), true);
});

test('current source candidates from relative directories expose absolute artifact paths', () => {
  const fsSource = fs.readFileSync(path.join(__dirname, '../roboproof/motion-experiment.js'), 'utf8');
  assert.match(fsSource, /function modelsForRun\(directory, update\)\s*\{\s*assertEvaluationHost\(\);\s*directory = path\.resolve\(directory\)/);
});

test('movement campaigns require the approved remote and training source hashes are path-bound', () => {
  assert.throws(() => Experiment.assertEvaluationHost('win32', {ROBOPROOF_APPROVED_REMOTE: '1'}), /approved remote Linux host/);
  assert.throws(() => Experiment.assertEvaluationHost('linux', {}), /approved remote Linux host/);
  assert.doesNotThrow(() => Experiment.assertEvaluationHost('linux', {ROBOPROOF_APPROVED_REMOTE: '1'}));
  const filename = 'roboproof/motion_learning/td3_train.py';
  const source = fs.readFileSync(path.join(__dirname, '..', filename), 'utf8').replace(/^\uFEFF/, '').replace(/\r\n/g, '\n');
  const sha256 = createHash('sha256').update(source).digest('hex');
  assert.equal(Experiment.validateTrainingSources({[filename]: sha256}), true);
  assert.throws(() => Experiment.validateTrainingSources({'../outside.py': sha256}), /Training source differs/);
  assert.throws(() => Experiment.validateTrainingSources({[filename]: 'f'.repeat(64)}), /Training source differs/);
});
