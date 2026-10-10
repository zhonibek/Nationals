'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const Experiment = require('../roboproof/motion-experiment');

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
  assert.match(fsSource, /function modelsForRun\(directory, update\)\s*\{\s*directory = path\.resolve\(directory\)/);
});
