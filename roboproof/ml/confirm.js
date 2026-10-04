'use strict';

const fs = require('node:fs');
const {performance} = require('node:perf_hooks');
const {provenance, validate, hash, writeJson} = require('../core');
const {runScenario} = require('../sim');
const {summary} = require('../analysis');

function confirm(selection, {allowLegacy = false} = {}) {
  if (!allowLegacy || selection?.canonical_ready !== false || selection?.engine_family !== 'legacy-reduced-v1' || selection?.kind !== 'ml-experimental-acquisition') throw Error('Only explicitly opted-in legacy acquisition can currently be confirmed');
  if (!Array.isArray(selection.scenarios) || selection.scenarios.length < 1 || selection.scenarios.length > 1000) throw Error('Confirmation budget must be 1-1000');
  if (new Set(selection.scenarios.map(scenario => scenario.scenario_id)).size !== selection.scenarios.length) throw Error('Duplicate selected scenario IDs');
  if (!Array.isArray(selection.predictions) || selection.predictions.length !== selection.scenarios.length || selection.predictions.some((prediction, index) => prediction.scenario_id !== selection.scenarios[index].scenario_id)) throw Error('Prediction/scenario alignment mismatch');
  const current = provenance();
  for (const key of ['simulation_version', 'controller_wasm_sha256']) {
    if (selection.expected_provenance?.[key] !== current[key]) throw Error(`Model evidence is stale: ${key}`);
  }
  const started = performance.now();
  const results = selection.scenarios.map(scenario => runScenario(validate(scenario)));
  const seconds = (performance.now() - started) / 1000;
  return {
    ...current, schema_version: 1, kind: 'ml-confirmed-experiments', backend: 'cpu-wasm', canonical_ready: false,
    model_sha256: selection.model_sha256, selection_sha256: hash(JSON.stringify(selection)),
    package_sha256: selection.package_sha256, predictions: selection.predictions, model_training: selection.model_training || null,
    results, summary: summary(results), seconds, benchmark: {seconds, simulations_per_second: results.length / seconds, gpu: 'NOT MEASURED'},
    distribution: 'ML-selected plus random exploration; biased experimental cohort, not a population reliability estimate',
    interpretation: 'Observed legacy CPU-WASM results; predictions did not determine pass/fail. No search-efficiency or physical-safety claim.'
  };
}

function main(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const key = argv[index];
    if (key === '--allow-legacy') options.allowLegacy = true;
    else if (['--input', '--out'].includes(key) && argv[index + 1]) options[key.slice(2)] = argv[++index];
    else throw Error(`Unknown or incomplete option: ${key}`);
  }
  if (!options.input || !options.out) throw Error('--input and --out are required');
  if (fs.existsSync(options.out)) throw Error('Output already exists; preserve evidence');
  const report = confirm(JSON.parse(fs.readFileSync(options.input, 'utf8').replace(/^\uFEFF/, '')), options);
  writeJson(options.out, report);
  console.log(JSON.stringify({summary: report.summary, seconds: report.seconds, canonical_ready: false}));
}

if (require.main === module) {
  try { main(process.argv.slice(2)); } catch (error) { console.error(error.message); process.exitCode = 1; }
}

module.exports = {confirm};
