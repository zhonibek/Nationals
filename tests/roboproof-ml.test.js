'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {makeScenario, clone, hash} = require('../roboproof/core');
const {runScenario} = require('../roboproof/sim');
const {exportDataset, grouping} = require('../roboproof/ml/export-dataset');
const {confirm} = require('../roboproof/ml/confirm');
const result = runScenario(makeScenario(9921));
const options = {allowLegacy: true, current: result};

test('ML export requires explicit legacy opt-in and excludes report holdout and candidate', () => {
  assert.throws(() => exportDataset([result], {current: result}), /Canonical/);
  const dataset = exportDataset({baseline: {results: [result]}, search: {results: [{}]}, holdout: [{}], candidate: [{}]}, options);
  assert.equal(dataset.rows.length, 1);
  assert.equal(dataset.canonical_ready, false);
  assert.equal(dataset.engine_family, 'legacy-reduced-v1');
  assert.equal(dataset.rows[0].telemetry_sha256, null);
  assert.deepEqual(JSON.parse(dataset.rows[0].scenario_json), result.scenario);
  assert.match(exportDataset({kind: 'ml-confirmed-experiments', results: [result], summary: {}}, options).sampling, /biased/);
});

test('ML exporter rejects stale, malformed, infrastructure and inconsistent observations', () => {
  for (const update of [
    {simulation_version: 'stale'}, {controller_wasm_sha256: 'stale'}, {backend: 'rocm'},
    {scenario_sha256: 'corrupt'}, {categories: ['WORKER_TIMEOUT']}, {passed: !result.passed},
    {categories: ['SUCCESS', 'TIMEOUT']}, {metrics: {...result.metrics, steps: 0}},
    {metrics: {...result.metrics, endpoint_error: null}}, {metrics: {...result.metrics, saturation_fraction: NaN}}
  ]) assert.throws(() => exportDataset([{...result, ...update}], options));
});

test('ML deduplication ignores renamed scenario IDs and rejects conflicting labels', () => {
  const renamed = clone(result);
  renamed.scenario.scenario_id = 'same-world';
  renamed.scenario_sha256 = hash(JSON.stringify(renamed.scenario));
  const dataset = exportDataset([result, renamed], options);
  assert.equal(dataset.rows.length, 1);
  assert.equal(dataset.deduplicated_rows, 1);
  renamed.metrics.endpoint_error += 0.01;
  assert.throws(() => exportDataset([result, renamed], options), /conflicting/);
});

test('paired controllers, nearby worlds and recorded mutation lineages share grouping keys', () => {
  const paired = clone(result.scenario);
  paired.controller.reference_time_scale = 1.5;
  assert.deepEqual(grouping(result.scenario), grouping(paired));
  paired.environment.friction += 0.000001;
  paired.random_seed++;
  assert.equal(grouping(result.scenario)[1], grouping(paired)[1]);
  assert.equal(grouping(result.scenario, 'root')[2], 'lineage:root');
  const mutated = clone(result);
  mutated.scenario.scenario_id = 'evo-1-1-0';
  mutated.scenario_sha256 = hash(JSON.stringify(mutated.scenario));
  assert.throws(() => exportDataset([mutated], options), /lineage/);
  assert.equal(exportDataset([{...mutated, lineage_root: 'root'}], options).rows[0].lineage_root, 'root');
});

test('observed numerical failure preserves missing regression targets', () => {
  const failed = {...result, categories: ['NUMERICAL_FAILURE', 'TIMEOUT'], passed: false,
    metrics: {...result.metrics, endpoint_error: null, final_heading_error: null, completion_time: null}};
  const dataset = exportDataset([failed], options);
  assert.equal(dataset.rows[0].metrics.endpoint_error, null);
  assert.deepEqual(dataset.rows[0].categories, ['NUMERICAL_FAILURE', 'TIMEOUT']);
});

test('learned selections require fresh source hashes and real simulation confirmation', () => {
  const selection = {kind: 'ml-experimental-acquisition', engine_family: 'legacy-reduced-v1', canonical_ready: false,
    expected_provenance: result, scenarios: [result.scenario], predictions: [{scenario_id: result.scenario.scenario_id, category_probabilities: {TIMEOUT: 1}}]};
  assert.throws(() => confirm(selection), /opted-in/);
  assert.throws(() => confirm({...selection, expected_provenance: {...result, simulation_version: 'stale'}}, options), /stale/);
  assert.throws(() => confirm({...selection, predictions: []}, options), /alignment/);
  const report = confirm(selection, options);
  assert.deepEqual(report.results[0].metrics, result.metrics);
  assert.equal(report.results[0].passed, true);
  assert.equal(report.predictions[0].category_probabilities.TIMEOUT, 1);
  assert.equal(report.canonical_ready, false);
});
