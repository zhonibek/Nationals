'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {compareCorpus, progressJournal, requireApprovedRemote, selectCohort} = require('../roboproof/gvf-motion-experiment');
const {suite} = require('../roboproof/motion-evaluation');

function report(reason, elapsedSeconds, effortProxyVAs, contactSeconds = 0) {
  return {reason, metrics: {elapsedSeconds, effortProxyVAs, contactSeconds, positionErrorMeters: 0}};
}

function fakeModels() {
  return [50101, 50102, 50103].map(seed => ({seed, runId: `run-${seed}`,
    initialSha256: `${seed}`.padStart(64, '0'), learnedSha256: `${seed + 1}`.padStart(64, '0'),
    actorWeightsChanged: true,
    initial: {identity: {kind: 'untrained', id: `initial-${seed}`}, act: () => [0, 0, 0]},
    learned: {identity: {kind: 'learned', id: `learned-${seed}`, sha256: 'a'.repeat(64)}, act: () => [0, 0, 0]}}));
}

test('GVF corpus comparison requires all three trained actors to beat baseline and initial policies', () => {
  const corpus = {specification: 'mock-development', scope: 'test only', corpusSha256: 'b'.repeat(64),
    cases: [1, 2, 3].map((seed, index) => ({id: `case-${seed}`, seed, task: {}, configuration: {}, index})),
    acceptance: suite(2).acceptance};
  const models = fakeModels();
  let replayCount = 0, caseCount = 0;
  const episodes = [];
  const result = compareCorpus(corpus, models, {
    baseline(seed) { return seed < 3 ? report('success', 10, 10) : report('timeout', 20, 20); },
    zeroResidual(seed) { return seed < 3 ? report('success', 10, 10) : report('timeout', 20, 20); },
    policy({seed, options}) {
      if (options.policyIdentity.kind === 'learned') return report('success', 8, 8);
      return seed < 3 ? report('success', 10, 10) : report('timeout', 20, 20);
    },
    replay() { replayCount++; },
    onEpisode(record) { episodes.push(record); },
    onCase() { caseCount++; }
  });
  assert.equal(caseCount, 3);
  assert.equal(replayCount, 24);
  assert.equal(episodes.length, 24);
  assert.deepEqual(episodes.slice(0, 8).map(record => record.policyId), [
    'existing-controller', 'gvf-zero-residual', 'initial-50101', 'learned-50101',
    'initial-50102', 'learned-50102', 'initial-50103', 'learned-50103'
  ]);
  assert.ok(episodes.every(record => record.replayVerified && record.caseId && record.report.reason));
  assert.equal(result.exactReplayEpisodes, 24);
  assert.equal(result.learnedImprovementVerified, true);
  assert.equal(result.measuredEfficiencyEvidence, true);
  assert.ok(result.models.every(model => model.acceptance.passed && model.actorWeightsChanged));
});

test('evaluation progress journal fsyncs append-only records and refuses reuse', context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gvf-progress-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const filename = path.join(directory, 'evaluation-progress.jsonl');
  const journal = progressJournal(filename, {kind: 'test-progress-v1', caseCount: 1});
  journal.append({type: 'episode', caseIndex: 0, policyId: 'candidate', replayVerified: true});
  assert.equal(journal.recordCount, 2);
  journal.close();
  const records = fs.readFileSync(filename, 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(records[0].type, 'started');
  assert.deepEqual(records[1], {type: 'episode', caseIndex: 0, policyId: 'candidate', replayVerified: true});
  assert.throws(() => progressJournal(filename, {kind: 'test-progress-v1'}), /EEXIST/);
});

test('GVF corpus comparison does not qualify actors without strict improvement', () => {
  const corpus = {specification: 'mock-development', scope: 'test only', corpusSha256: 'c'.repeat(64),
    cases: [{id: 'case-1', seed: 1, task: {}, configuration: {}}], acceptance: suite(2).acceptance};
  const result = compareCorpus(corpus, fakeModels(), {
    baseline: () => report('success', 10, 10),
    zeroResidual: () => report('success', 10, 10),
    policy: ({options}) => report('success', options.policyIdentity.kind === 'learned' ? 9 : 10,
      options.policyIdentity.kind === 'learned' ? 9 : 10),
    replay() {}
  });
  assert.equal(result.learnedImprovementVerified, false);
  assert.equal(result.measuredEfficiencyEvidence, false);
});

test('remote evaluator entry guard rejects the local workstation', () => {
  assert.throws(requireApprovedRemote, /Approved Linux remote required/);
  assert.throws(() => compareCorpus({cases: [{id: 'local', seed: 1, task: {}, configuration: {}}]}, fakeModels()),
    /Approved Linux remote required/);
});

test('512x10 is the primary three-seed cohort and historical 64x2 remains explicit', () => {
  const primary = [50512, 50513, 50514].map(seed => ({config: {
    algorithm: 'td3-tanh-gvf-residual-v1', seed, networkWidth: 512, hiddenLayers: 10}}));
  assert.equal(selectCohort(primary).id, 'primary-gvf512x10-v1');
  const historical = [50101, 50102, 50103].map(seed => ({config: {
    algorithm: 'td3-tanh-gvf-residual-v1', seed, networkWidth: 64, hiddenLayers: 2}}));
  assert.equal(selectCohort(historical).id, 'historical-gvf64x2-v1');
  assert.throws(() => selectCohort(primary.map((row, index) => index ? row : ({config: {...row.config, hiddenLayers: 2}}))),
    /predeclared/);
  assert.throws(() => selectCohort(primary.map((row, index) => index ? row : ({config: {...row.config, seed: 50101}}))),
    /predeclared/);
});
