'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {spec, validate, hash, provenance, writeJson} = require('../core');
const contract = require('./contract.json');

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]));
  return value;
}

function grouping(scenario, lineageRoot) {
  const environment = Object.fromEntries(Object.entries(spec.ranges).map(([name, [minimum, maximum]]) => [
    name, Math.round((scenario.environment[name] - minimum) / (maximum - minimum) / contract.near_duplicate_resolution)
  ]));
  const world = hash(JSON.stringify(stable({environment, task: scenario.task})));
  return [`seed:${scenario.random_seed}`, `world-bin:${world}`, ...(lineageRoot ? [`lineage:${lineageRoot}`] : [])];
}

function exportDataset(input, {allowLegacy = false, current = provenance()} = {}) {
  if (!allowLegacy) throw Error('Canonical engine is not ready; experimental export requires --allow-legacy');
  const results = Array.isArray(input) ? input : input?.baseline?.results || input?.results || (input?.scenario ? [input] : null);
  if (!Array.isArray(results) || !results.length) throw Error('Expected completed CPU-WASM results or a report baseline');
  if (results.length > 100000) throw Error('Dataset row limit is 100000');
  const seen = new Map();
  const rows = results.map(result => {
    const scenario = validate(result.scenario);
    const serialized = JSON.stringify(scenario);
    if (result.backend !== 'cpu-wasm') throw Error('Only verified CPU-WASM legacy results can currently be exported');
    for (const key of ['simulation_version', 'controller_wasm_sha256']) {
      if (result[key] !== current[key]) throw Error(`Stale or mixed provenance: ${key}`);
    }
    if (hash(serialized) !== result.scenario_sha256) throw Error('Scenario hash mismatch');
    if (!Array.isArray(result.categories) || !result.categories.length || new Set(result.categories).size !== result.categories.length || result.categories.some(name => !['SUCCESS', ...contract.classification_targets].includes(name))) throw Error('Invalid observed categories');
    if (typeof result.passed !== 'boolean' || result.passed !== (result.categories.length === 1 && result.categories[0] === 'SUCCESS') || (!result.passed && result.categories.includes('SUCCESS'))) throw Error('Inconsistent success label');
    if (!result.metrics || !Number.isInteger(result.metrics.steps) || result.metrics.steps < 1) throw Error('Incomplete run is not a robot-failure label');
    const metrics = Object.fromEntries(contract.regression_targets.map(name => {
      const value = result.metrics[name];
      const missingAllowed = name === 'completion_time' || (result.categories.includes('NUMERICAL_FAILURE') && ['endpoint_error', 'final_heading_error'].includes(name));
      if (!(value === null && missingAllowed) && (!Number.isFinite(value) || value < 0)) throw Error(`Invalid observed metric: ${name}`);
      return [name, value];
    }));
    const thresholds = scenario.task.thresholds;
    const observed = {
      PATH_DIVERGENCE: metrics.max_path_deviation > thresholds.path,
      HEADING_INSTABILITY: metrics.final_heading_error > thresholds.heading,
      OSCILLATION: metrics.oscillations >= thresholds.oscillations,
      TIMEOUT: metrics.completion_time === null,
      ENDPOINT_FAILURE: metrics.endpoint_error > thresholds.endpoint,
      LOCALIZATION_DIVERGENCE: metrics.max_localization_error > thresholds.localization,
      MOTOR_SATURATION: metrics.saturation_fraction > thresholds.saturation_fraction
    };
    if (Object.entries(observed).some(([name, failed]) => result.categories.includes(name) !== failed)) throw Error('Category/metric label audit failed');
    const lineageRoot = result.lineage_root || null;
    if (lineageRoot !== null && (typeof lineageRoot !== 'string' || !lineageRoot)) throw Error('Invalid lineage root');
    if (scenario.scenario_id.startsWith('evo-') && !lineageRoot) throw Error('Evolutionary results require recorded lineage; legacy reports lack it');
    const identity = hash(JSON.stringify(stable({...scenario, scenario_id: undefined})));
    const labels = JSON.stringify(stable({metrics, categories: [...result.categories].sort()}));
    if (seen.has(identity) && (seen.get(identity).labels !== labels || seen.get(identity).lineageRoot !== lineageRoot)) throw Error('Duplicate scenario has conflicting observations or lineage');
    seen.set(identity, {labels, lineageRoot});
    return {
      row_id: identity, scenario_json: serialized, scenario_sha256: result.scenario_sha256,
      group_keys: grouping(scenario, lineageRoot), lineage_root: lineageRoot,
      categories: result.categories, metrics,
      telemetry_sha256: result.telemetry ? hash(JSON.stringify(result.telemetry)) : null
    };
  });
  const unique = [...new Map(rows.map(row => [row.row_id, row])).values()];
  return {
    schema_version: 1, engine_family: contract.engine_family, canonical_ready: false,
    scenario_spec_sha256: hash(fs.readFileSync(path.join(__dirname, '../scenario-spec.json'), 'utf8').replace(/\r\n/g, '\n')),
    contract_sha256: hash(fs.readFileSync(path.join(__dirname, 'contract.json'), 'utf8').replace(/\r\n/g, '\n')),
    provenance: Object.fromEntries(['software_version', 'simulation_version', 'source_hashes', 'controller_wasm_sha256', 'controller_source_hashes', 'runtime', 'platform', 'reproducibility'].map(key => [key, current[key]])),
    sampling: input?.kind === 'ml-confirmed-experiments' ? 'ML-selected confirmed cohort; biased, not a population distribution' : input?.baseline?.results || input?.summary ? 'report-baseline-only; search/candidate/holdout excluded' : 'supplied-results; distribution not inferred',
    deduplicated_rows: results.length - unique.length, rows: unique,
    limitations: contract.restrictions
  };
}

function main(argv) {
  if (argv.includes('--help')) {
    console.log('node roboproof/ml/export-dataset.js --input report.json --out dataset.json --allow-legacy');
    return;
  }
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    if (argv[index] === '--allow-legacy') options.allowLegacy = true;
    else if (['--input', '--out'].includes(argv[index]) && argv[index + 1]) {
      const key = argv[index].slice(2);
      options[key] = argv[++index];
    }
    else throw Error(`Unknown or incomplete option: ${argv[index]}`);
  }
  if (!options.input || !options.out) throw Error('--input and --out are required');
  if (fs.existsSync(options.out)) throw Error('Dataset already exists; preserve evidence and use a new output');
  const dataset = exportDataset(JSON.parse(fs.readFileSync(options.input, 'utf8').replace(/^\uFEFF/, '')), options);
  writeJson(options.out, dataset);
  console.log(JSON.stringify({rows: dataset.rows.length, canonical_ready: false, out: path.resolve(options.out)}));
}

if (require.main === module) {
  try { main(process.argv.slice(2)); } catch (error) { console.error(error.message); process.exitCode = 1; }
}

module.exports = {exportDataset, grouping};
