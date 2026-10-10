'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash, randomUUID} = require('node:crypto');
const {createLocalFilesystemHoldoutAudit} = require('../roboproof/motion-holdout');
const {readIndependentEvaluation} = require('../roboproof/independent-learning');
const {aggregate, suite} = require('../roboproof/motion-evaluation');
const Learner = require('../roboproof/motion-learner');
const {assertFreshCases} = require('../roboproof/motion-exposures');

const hash = bytes => createHash('sha256').update(bytes).digest('hex');

function fixture(context, passes = false, withExposure = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-independent-view-'));
  context.after(() => fs.rmSync(root, {recursive: true, force: true}));
  const learningDirectory = path.join(root, 'learning'), holdoutDirectory = path.join(root, 'holdout');
  const runId = randomUUID(), holdoutId = randomUUID(), directory = path.join(holdoutDirectory, holdoutId);
  const runDirectory = path.join(learningDirectory, runId);
  fs.mkdirSync(runDirectory, {recursive: true});
  const identity = {sources: {'fixture.js': hash('fixture source')}, runtime: {node: 'fixture'}};
  const options = {holdoutDirectory, readCurrentIdentity: () => identity};
  const audit = createLocalFilesystemHoldoutAudit(directory, {readCurrentSourceRuntimeIdentity: options.readCurrentIdentity});
  const cases = [{id: 'fixture-one', seed: 10}, {id: 'fixture-two', seed: 11}].map((world, position) => ({...world,
    task: {start: {xIn: 0, yIn: 0, headingDeg: 0}, goal: {xIn: 0, yIn: 20 + position * 10, headingDeg: 0}, deadlineSeconds: 10}}));
  const ledger = {schemaVersion: 1, specification: 'known-local-motion-exposure-v1', worldSeeds: [], bodyGeometries: [],
    seedOnlyRecords: 0, legacyGeometryFullyKnown: true, scope: 'Fixture-only local exclusion'};
  const knownExposure = withExposure ? assertFreshCases(cases, ledger) : null;
  audit.commitFreshCorpus({cases, acceptance: suite(2).acceptance, ...(knownExposure ? {knownExposure} : {})});
  if (knownExposure) fs.writeFileSync(path.join(directory, 'exposure-ledger.json'), JSON.stringify(ledger));
  const candidates = [201, 202, 203].flatMap(seed => ['initial', 'learned'].map(kind => {
    const filename = path.join(runDirectory, `${kind}-${seed}.json`);
    fs.writeFileSync(filename, JSON.stringify({fixture: true, seed, kind}));
    return {id: `${kind}-${seed}`, path: filename};
  }));
  audit.freezeCandidateFilesBeforeReveal(candidates);
  const revealed = audit.consumeAndRevealOnce();
  const measured = (reason, effort) => ({reason, metrics: {elapsedSeconds: 5, effortProxyVAs: effort,
    contactSeconds: 0, positionErrorMeters: reason === 'success' ? 0.01 : 0.05}});
  const models = [201, 202, 203].map(seed => {
    const rows = cases.map((world, position) => ({...world,
      baseline: measured(position ? 'time_limit' : 'success', 10),
      neutral: measured(position ? 'time_limit' : 'success', 10),
      untrained: measured('time_limit', 10), learned: measured(passes ? 'success' : 'time_limit', 8)}));
    const summary = kind => aggregate(rows.map(row => row[kind]));
    const baseline = summary('baseline'), neutral = summary('neutral'), untrained = summary('untrained'), learned = summary('learned');
    return {seed, policySha256: revealed.candidates.find(candidate => candidate.id === `learned-${seed}`).sha256,
      initialSha256: revealed.candidates.find(candidate => candidate.id === `initial-${seed}`).sha256,
      baseline, neutral, untrained, learned, results: rows,
      acceptance: Learner.acceptance(rows, baseline, learned, untrained, suite(2).acceptance)};
  });
  const result = {schemaVersion: 1, caseCount: 2, identity, models, learnedImprovementVerified: passes,
    ...(knownExposure ? {knownExposureExclusion: knownExposure} : {}),
    holdoutAudit: {commitment: revealed.commitment, consumed: revealed.consumed}, privateSource: 'Never expose this'};
  const evaluationPath = path.join(directory, 'evaluation.json'), indexPath = path.join(runDirectory, 'independent-final.json');
  const saveResult = () => {
    const bytes = JSON.stringify(result);
    fs.writeFileSync(evaluationPath, bytes);
    fs.writeFileSync(indexPath, JSON.stringify({schemaVersion: 1, runId, holdoutId, evaluationSha256: hash(bytes)}));
  };
  saveResult();
  const status = {available: true, runId, identity, models: models.map(model => ({seed: model.seed,
    policySha256: model.policySha256, initialSha256: model.initialSha256}))};
  const read = () => readIndependentEvaluation(status, learningDirectory, options);
  return {status, options, read, result, saveResult, indexPath, directory, learningDirectory};
}

test('consumed negative evidence is hash/checkpoint bound, compact, read-only and cannot promote', context => {
  const saved = fixture(context);
  const before = fs.readFileSync(saved.indexPath);
  const summary = saved.read();
  assert.equal(summary.available, true);
  assert.equal(summary.sourceRuntimeMatches, true);
  assert.equal(summary.gateReportedPassed, false);
  assert.equal(summary.policyPromotionAllowed, false);
  assert.deepEqual(summary.models.map(model => model.learned.success), [0, 0, 0]);
  assert.deepEqual(summary.models.map(model => model.successRegressions), [1, 1, 1]);
  assert.equal(summary.models[0].baseline.success, 1);
  assert(!JSON.stringify(summary).includes('Never expose this'));
  assert.equal(summary.models[0].results, undefined);
  assert.deepEqual(fs.readFileSync(saved.indexPath), before);
});

test('a reported positive research gate never grants policy promotion', context => {
  const saved = fixture(context, true);
  assert.equal(saved.read().gateReportedPassed, true);
  assert.equal(saved.read().policyPromotionAllowed, false);
});

test('new final evidence verifies the bound exclusion snapshot and fails closed when it changes', context => {
  const saved = fixture(context, true, true);
  assert.equal(saved.read().available, true);
  assert.equal(saved.read().policyPromotionAllowed, false);
  const filename = path.join(saved.directory, 'exposure-ledger.json');
  const ledger = JSON.parse(fs.readFileSync(filename));
  ledger.worldSeeds.push(10);
  fs.writeFileSync(filename, JSON.stringify(ledger));
  assert.equal(saved.read().available, false);
});

test('changed source retains historical evidence with an explicit mismatch label', context => {
  const saved = fixture(context);
  saved.options.readCurrentIdentity = () => ({...saved.status.identity, runtime: {node: 'changed'}});
  const summary = saved.read();
  assert.equal(summary.available, true);
  assert.equal(summary.sourceRuntimeMatches, false);
  assert.equal(summary.policyPromotionAllowed, false);
});

for (const [label, change] of [
  ['checkpoint mismatch', saved => { saved.status.models[0].policySha256 = hash('other checkpoint'); }],
  ['missing consumption marker', saved => { fs.unlinkSync(path.join(saved.directory, 'consumed.json')); }],
  ['corrupted result bytes', saved => { fs.appendFileSync(path.join(saved.directory, 'evaluation.json'), ' '); }],
  ['changed aggregate', saved => { saved.result.models[0].learned.success = 1; saved.saveResult(); }],
  ['missing case', saved => { saved.result.models[0].results.pop(); saved.saveResult(); }],
  ['changed gate', saved => { saved.result.models[0].acceptance.passed = true; saved.saveResult(); }],
  ['changed case identity', saved => { saved.result.models[0].results[0].seed++; saved.saveResult(); }],
  ['duplicate seed', saved => { saved.status.models[1].seed = saved.status.models[0].seed; }],
  ['path escape index', saved => { const index = JSON.parse(fs.readFileSync(saved.indexPath)); index.holdoutId = '../escape'; fs.writeFileSync(saved.indexPath, JSON.stringify(index)); }],
]) {
  test(`${label} fails closed without leaking file contents`, context => {
    const saved = fixture(context);
    change(saved);
    const summary = saved.read();
    assert.equal(summary.available, false);
    assert.equal(summary.policyPromotionAllowed, false);
    assert.equal(summary.models, undefined);
  });
}

test('an unavailable learner or missing index never loads unrelated holdouts', context => {
  const saved = fixture(context);
  assert.equal(readIndependentEvaluation({available: false}, saved.learningDirectory).available, false);
  assert.equal(readIndependentEvaluation(saved.status).message, 'No independent final linked to these checkpoints');
  fs.unlinkSync(saved.indexPath);
  assert.equal(saved.read().available, false);
});
