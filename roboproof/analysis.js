'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {ROOT, spec, random, normal, clamp, clone, validate, hash, provenance} = require('./core');
const {runScenario} = require('./sim');

function summary(results) {
  const categories = {};
  for (const result of results) for (const category of result.categories) categories[category] = (categories[category] || 0) + 1;
  const passed = results.filter(result => result.passed).length;
  return {count: results.length, passed, failed: results.length - passed, robustness_percent: results.length ? 100 * passed / results.length : null, categories, interpretation: 'Observed pass fraction within this finite scenario set, not a universal safety score'};
}

function compare(before, after) {
  if (before.length !== after.length) throw Error('Regression sets must have identical size');
  let fixed = 0, regressions = 0, unchangedFailures = 0, unchangedPasses = 0;
  const changes = [];
  for (let index = 0; index < before.length; index++) {
    const original = before[index], candidate = after[index];
    const world = value => JSON.stringify({id: value.scenario.scenario_id, seed: value.scenario.random_seed, environment: value.scenario.environment, task: value.scenario.task});
    if (world(original) !== world(candidate)) throw Error('Regression comparison requires identical worlds and seeds in the same order');
    if (!original.passed && candidate.passed) { fixed++; changes.push({id: original.scenario.scenario_id, change: 'FIXED'}); }
    else if (original.passed && !candidate.passed) { regressions++; changes.push({id: original.scenario.scenario_id, change: 'REGRESSION'}); }
    else if (original.passed) unchangedPasses++;
    else unchangedFailures++;
  }
  return {before: summary(before), after: summary(after), failures_fixed: fixed, new_regressions: regressions, unchanged_failures: unchangedFailures, unchanged_passes: unchangedPasses, net_improvement: fixed - regressions, changes};
}

function batch(scenarios, onProgress = () => {}) {
  return scenarios.map((scenario, index) => {
    const result = runScenario(scenario);
    onProgress(index + 1, scenarios.length);
    return result;
  });
}

function adversarial(initial, {generations = 2, population = 8, seed = 2026, onProgress = () => {}} = {}) {
  if (!initial.length) throw Error('Adversarial search needs seed results');
  if (!Number.isInteger(generations) || generations < 0 || generations > 100) throw Error('Invalid generations');
  if (!Number.isInteger(population) || population < 1 || population > 10000) throw Error('Invalid population');
  const next = random(seed);
  const all = [];
  let elite = [...initial].sort((left, right) => right.failure_score - left.failure_score).slice(0, Math.min(4, initial.length));
  const history = [{generation: 0, worst_score: elite[0].failure_score}];
  for (let generation = 1; generation <= generations; generation++) {
    const children = Array.from({length: population}, (_, index) => {
      const parent = elite[Math.floor(next() * elite.length)].scenario;
      const scenario = clone(parent);
      scenario.scenario_id = `evo-${seed}-${generation}-${index}`;
      scenario.random_seed = Math.floor(next() * 4294967296);
      for (const [name, [minimum, maximum]] of Object.entries(spec.ranges)) {
        scenario.environment[name] = clamp(scenario.environment[name] + normal(next) * (maximum - minimum) * 0.15, minimum, maximum);
      }
      return validate(scenario);
    });
    const results = batch(children, onProgress);
    all.push(...results);
    elite = [...elite, ...results].sort((left, right) => right.failure_score - left.failure_score).slice(0, 4);
    history.push({generation, worst_score: elite[0].failure_score, failures: results.filter(result => !result.passed).length});
  }
  return {results: all, history, worst: elite[0], algorithm: 'Elitist bounded Gaussian mutation; population selected on documented failure score'};
}

function evidenceTools(result) {
  const current = provenance();
  for (const key of ['simulation_version', 'controller_wasm_sha256']) {
    if (result[key] !== current[key]) throw Error(`Cannot investigate stale results: ${key} mismatch`);
  }
  const full = result.telemetry ? result : runScenario(result.scenario, {telemetry: true});
  const log = [];
  const tools = {
    get_scenario() { return {scenario: full.scenario, parameter_bounds: spec.ranges}; },
    find_first_divergence() {
      const point = full.telemetry.find(sample => sample.path_error > full.scenario.task.thresholds.path || sample.localization_error > full.scenario.task.thresholds.localization);
      return point || {observation: 'No path/localization threshold crossing', final_metrics: full.metrics};
    },
    get_telemetry_range({start = 0, end = full.scenario.task.duration, limit = 30} = {}) {
      if (![start, end, limit].every(Number.isFinite) || end < start || limit < 1 || limit > 100) throw Error('Invalid telemetry range');
      const rows = full.telemetry.filter(sample => sample.time >= start && sample.time <= end);
      const stride = Math.max(1, Math.ceil(rows.length / limit));
      return rows.filter((_, index) => index % stride === 0).slice(0, limit);
    },
    inspect_controller() {
      return {source: 'include/subsystems/control/Cascade.hpp', text: fs.readFileSync(path.join(ROOT, 'include/subsystems/control/Cascade.hpp'), 'utf8'), pid_terms: 'NOT EXPORTED by current WASM ABI; do not infer P/I/D separately from aggregate output'};
    },
    inspect_source({filename, start = 1, count = 100} = {}) {
      const allowed = ['include/subsystems/control/Cascade.hpp', 'src/subsystems/control/HolonomicMotion.cpp', 'src/lemlib/chassis/odom.cpp'];
      if (!allowed.includes(filename) || !Number.isInteger(start) || start < 1 || !Number.isInteger(count) || count < 1 || count > 150) throw Error('Unsupported source or line range');
      const source = fs.readFileSync(path.join(ROOT, filename), 'utf8');
      return {filename, start, sha256: hash(source.replace(/\r\n/g, '\n')), lines: source.split(/\r?\n/).slice(start - 1, start - 1 + count)};
    },
    inspect_pid_terms() { return {status: 'UNAVAILABLE', reason: 'The current production WASM ABI does not export individual P/I/D/feedforward terms.', available: ['command', 'target_wheel', 'measured_wheel']}; },
    compare_runs({scenario} = {}) {
      const alternative = runScenario(validate(clone(scenario)));
      return {baseline: {scenario: full.scenario, metrics: full.metrics, categories: full.categories}, alternative: {scenario: alternative.scenario, metrics: alternative.metrics, categories: alternative.categories}, score_delta: alternative.failure_score - full.failure_score};
    },
    run_parameter_sweep({parameter, values} = {}) {
      if (!(parameter in spec.ranges) || !Array.isArray(values) || values.length < 1 || values.length > 12) throw Error('Invalid bounded parameter sweep');
      return values.map(value => {
        const scenario = clone(full.scenario);
        scenario.environment[parameter] = value;
        const alternative = runScenario(validate(scenario));
        return {value, passed: alternative.passed, failure_score: alternative.failure_score, metrics: alternative.metrics, categories: alternative.categories};
      });
    },
    run_scenario() { const rerun = runScenario(full.scenario); return {metrics: rerun.metrics, categories: rerun.categories, reproducible: hash(JSON.stringify(rerun.metrics)) === hash(JSON.stringify(full.metrics))}; }
  };
  function call(name, args = {}) {
    if (!Object.hasOwn(tools, name)) throw Error(`Unknown investigation tool: ${name}`);
    const output = tools[name](args);
    log.push({tool: name, arguments: args, result: output});
    return output;
  }
  return {call, log, names: Object.keys(tools)};
}

function diagnose(result) {
  const investigator = evidenceTools(result);
  investigator.call('get_scenario');
  const divergence = investigator.call('find_first_divergence');
  investigator.call('inspect_controller');
  const reproduction = investigator.call('run_scenario');
  const sweeps = {};
  for (const parameter of ['friction', 'encoder_latency', 'battery_voltage']) {
    sweeps[parameter] = investigator.call('run_parameter_sweep', {parameter, values: [...new Set([spec.ranges[parameter][0], spec.nominal[parameter], spec.ranges[parameter][1]])]});
  }
  const sensitivities = Object.entries(sweeps).map(([parameter, rows]) => ({parameter, score_span: Math.max(...rows.map(row => row.failure_score)) - Math.min(...rows.map(row => row.failure_score)), results: rows})).sort((left, right) => right.score_span - left.score_span);
  
  const dominant = sensitivities[0]?.parameter || null;
  const divTime = divergence.time;
  const categories = result.categories || [];

  const sweepSummary = Object.entries(sweeps).map(([parameter, rows]) => ({parameter,
    tested: rows.length, failed: rows.filter(row => !row.passed).length}));
  const failures = sweepSummary.reduce((total, entry) => total + entry.failed, 0);
  const tested = sweepSummary.reduce((total, entry) => total + entry.tested, 0);
  const sweepObservation = `${failures} of ${tested} one-at-a-time sweep runs failed. These finite samples do not measure stability margins or joint-parameter robustness.`;
  const inference = result.passed
    ? `The recorded scenario passed. ${sweepObservation}`
    : `The recorded scenario failed (${categories.join(', ')}). ${sweepObservation} ${dominant} has the largest observed failure-score span; this is a sensitivity hypothesis, not an established root cause.`;

  return {
    mode: 'deterministic-evidence-investigator',
    llm_status: 'NOT CONNECTED; no neural or LLM inference in this diagnosis',
    failure: result.categories,
    first_divergence: divTime ?? null,
    observations: {
      metrics: result.metrics,
      reproducible: reproduction.reproducible,
      localization: result.scenario.task.localization,
      divergence_instant: divTime !== undefined ? `First path/localization threshold crossing at t = ${divTime.toFixed(2)}s` : 'No observed path/localization threshold crossing',
      dominant_vulnerability: `${dominant} (score span: ${sensitivities[0].score_span.toFixed(2)})`
    },
    inference,
    confidence: 'UNCALIBRATED; deterministic sensitivity analysis is not causal proof',
    sweep_summary: sweepSummary,
    sensitivities,
    tool_log: investigator.log
  };
}

module.exports = {summary, compare, batch, adversarial, evidenceTools, diagnose};
