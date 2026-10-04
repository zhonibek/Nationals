'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {baseline, runPolicy} = require('./motion');
const {loadHeadless} = require('../simulator/headless');
const {normalizeTask} = require('../simulator/motion');
const Policy = require('../simulator/policy');
const frozenFiles = ['motion-evaluation-worlds.json', 'motion-evaluation-worlds-v2.json'];
const hash = value => crypto.createHash('sha256').update(value).digest('hex');

function suite(version = 2) {
  if (![1, 2].includes(version)) throw Error('Unsupported frozen evaluation suite');
  const filename = path.join(__dirname, '../tests/fixtures', frozenFiles[version - 1]);
  const bytes = fs.readFileSync(filename, 'utf8').replace(/\r\n/g, '\n');
  const data = JSON.parse(bytes);
  const ids = new Set();
  for (const entry of data.cases) {
    if (ids.has(entry.id) || typeof entry.id !== 'string' || !Number.isInteger(entry.seed) || entry.seed < 0 || entry.seed > 0xffffffff) throw Error('Invalid frozen world identity');
    ids.add(entry.id);
    normalizeTask(entry.task);
  }
  return {...data, corpusSha256: hash(bytes)};
}

function assertTrainingWorld(world) {
  for (const version of [1, 2]) for (const entry of suite(version).cases) {
    if (world.id === entry.id || world.seed === entry.seed ||
      JSON.stringify(normalizeTask(world.task)) === JSON.stringify(normalizeTask(entry.task))) {
      throw Error('Frozen evaluation world/task is excluded from training and tuning');
    }
  }
  return world;
}

function aggregate(reports) {
  const mean = key => reports.reduce((sum, report) => sum + report.metrics[key], 0) / reports.length;
  const success = reports.filter(report => report.reason === 'success').length;
  return {count: reports.length, success, failed: reports.length - success, successRate: success / reports.length,
    meanElapsedSecondsAllWorlds: mean('elapsedSeconds'), meanEffortProxyVAsAllWorlds: mean('effortProxyVAs'),
    meanContactSecondsAllWorlds: mean('contactSeconds'), meanPositionErrorMetersAllWorlds: mean('positionErrorMeters'),
    interpretation: 'All worlds included, including timeouts; this is not universal safety or physical performance'};
}

function evaluate(version = 2) {
  const corpus = suite(version);
  const baselineReports = [];
  const adapterReports = [];
  const results = [];
  for (const world of corpus.cases) {
    const original = baseline(world.seed, world.task, world.configuration);
    const adapter = runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
      options: {recordTransitions: false, policyIdentity: {kind: 'scripted', id: 'nominal-reference-fixture'}}});
    baselineReports.push(original);
    adapterReports.push(adapter);
    results.push({id: world.id, group: world.group ?? world.id, seed: world.seed,
      resolvedConfiguration: original.configuration, task: world.task,
      baseline: {reason: original.reason, metrics: original.metrics}, adapter: {reason: adapter.reason, metrics: adapter.metrics}});
  }
  return {schemaVersion: 1, evaluatedAt: new Date().toISOString(), corpusSha256: corpus.corpusSha256,
    suiteVersion: version, identity: loadHeadless().identity, policyContractVersion: Policy.CONTRACT_VERSION,
    baseline: aggregate(baselineReports), adapter: aggregate(adapterReports), results,
    regressions: results.filter(row => row.baseline.reason === 'success' && row.adapter.reason !== 'success').map(row => row.id),
    acceptance: corpus.acceptance ?? {primary: 'success rate'}, motionPolicyTrained: false,
    learnedImprovementVerified: false, interpretation: 'Pre-training controller/reference-adapter comparison only. No learned policy was evaluated.'};
}

module.exports = {suite, assertTrainingWorld, aggregate, evaluate};
