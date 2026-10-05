'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const {makeScenario, sampleScenarios, clone, writeJson, hash, provenance, spec} = require('./core');
const {runScenario} = require('./sim');
const {summary, compare, batch, adversarial, diagnose} = require('./analysis');

function args(argv) {
  const options = {command: argv[0] || 'help'};
  for (let index = 1; index < argv.length; index += 2) {
    if (!argv[index].startsWith('--') || argv[index + 1] === undefined) throw Error(`Expected --option value: ${argv[index]}`);
    options[argv[index].slice(2)] = argv[index + 1];
  }
  const allowed = {
    help: [], nominal: ['out'], sample: ['count', 'seed', 'out'], run: ['scenarios', 'count', 'seed', 'out'],
    replay: ['file', 'out'], demo: ['count', 'holdout', 'seed', 'generations', 'population', 'out'],
    benchmark: ['counts', 'seed', 'out'], diagnose: ['file', 'out'],
    verify: ['scenarios', 'candidate', 'holdout', 'seed', 'out'],
    'motion-baseline': ['seed', 'task', 'configuration', 'out'], 'motion-replay': ['file', 'out'],
    'motion-check': ['out'], 'motion-evaluate': ['suite', 'out'], 'motion-compare-learned': ['directory', 'out']
  };
  if (!Object.hasOwn(allowed, options.command)) throw Error(`Unknown command: ${options.command}`);
  for (const key of Object.keys(options)) if (key !== 'command' && !allowed[options.command].includes(key)) throw Error(`Unknown option: --${key}`);
  return options;
}

function number(options, key, fallback, minimum = 1, maximum = 100000) {
  const value = options[key] === undefined ? fallback : Number(options[key]);
  if (!Number.isInteger(value) || value < minimum || value > maximum) throw Error(`--${key} must be an integer in [${minimum}, ${maximum}]`);
  return value;
}

const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8').replace(/^\uFEFF/, ''));
function loadScenarios(filename) {
  const data = read(filename);
  const scenarios = Array.isArray(data) ? data : data.scenarios;
  if (!Array.isArray(scenarios) || !scenarios.length) throw Error('Expected nonempty scenario array or {scenarios: [...]}');
  const keys = scenarios.map(scenario => scenario.scenario_id);
  if (new Set(keys).size !== keys.length) throw Error('Scenario ids must be unique within a batch');
  return scenarios;
}

function progress(label) {
  let last = 0;
  return (done, total) => {
    if (done === total || performance.now() - last > 5000) { process.stderr.write(`${label}: ${done}/${total}\n`); last = performance.now(); }
  };
}

function timedBatch(scenarios, label) {
  const start = performance.now();
  const results = batch(scenarios, progress(label));
  const seconds = (performance.now() - start) / 1000;
  return {results, summary: summary(results), seconds, simulations_per_second: scenarios.length / seconds};
}

function experiment(scenarios, candidate, before, label) {
  const after = timedBatch(scenarios.map(scenario => ({...clone(scenario), controller: {...scenario.controller, ...candidate}})), label);
  return {comparison: compare(before, after.results), results: after.results, seconds: after.seconds};
}

function verifyReport(scenarios, candidate, holdout, original) {
  const training = original || timedBatch(scenarios, 'baseline');
  const trainingAfter = experiment(scenarios, candidate, training.results, 'candidate');
  const heldBefore = timedBatch(holdout, 'holdout baseline');
  const heldAfter = experiment(holdout, candidate, heldBefore.results, 'holdout candidate');
  return {candidate, training: trainingAfter.comparison, holdout: heldAfter.comparison, before: training.results, after: trainingAfter.results, holdout_before: heldBefore.results, holdout_after: heldAfter.results, accepted: trainingAfter.comparison.net_improvement > 0 && heldAfter.comparison.net_improvement > 0 && trainingAfter.comparison.new_regressions === 0 && heldAfter.comparison.new_regressions === 0, acceptance_rule: 'Positive net improvement on training and unseen holdout, zero regressions on both; no production merge'};
}

function main(argv = process.argv.slice(2)) {
  const options = args(argv);
  const out = path.resolve(options.out || 'roboproof/runs/latest');
  const seed = number(options, 'seed', 42, 0, 0xffffffff);
  if (options.command === 'motion-compare-learned') {
    const result = require('./motion-learner').evaluateLearned(options.directory);
    if (options.out) writeJson(options.out, result);
    console.log(JSON.stringify({runId: result.runId, learnedImprovementVerified: result.learnedImprovementVerified,
      seeds: result.models.map(row => ({seed: row.seed, baseline: row.baseline.success,
        untrained: row.untrained.success, learned: row.learned.success, gatePassed: row.acceptance.passed}))}));
    return;
  }
  if (options.command === 'motion-check' || options.command === 'motion-evaluate') {
    const result = options.command === 'motion-check' ? require('./motion-readiness').verify()
      : require('./motion-evaluation').evaluate(number(options, 'suite', 2, 1, 2));
    const destination = options.out || `roboproof/runs/motion/${options.command === 'motion-check' ? 'readiness' : 'evaluation'}.json`;
    writeJson(destination, result);
    console.log(JSON.stringify({report: path.resolve(destination), readyForBoundedCpuExperiment: result.readyForBoundedCpuExperiment,
      baseline: result.baseline, adapter: result.adapter, motionPolicyTrained: false}));
    return;
  }
  if (options.command === 'motion-baseline' || options.command === 'motion-replay') {
    const motion = require('./motion');
    const report = options.command === 'motion-baseline'
      ? motion.baseline(seed, options.task ? read(options.task) : undefined, options.configuration ? read(options.configuration) : undefined)
      : motion.replay(read(options.file));
    const destination = options.out || 'roboproof/runs/motion/baseline.json';
    writeJson(destination, report);
    console.log(JSON.stringify({report: path.resolve(destination), canonicalSimulator: true, reason: report.reason, metrics: report.metrics}));
    return;
  }
  if (options.command === 'help') {
    console.log('Training preparation: motion-check --out readiness.json\nmotion-evaluate --suite 2 --out evaluation.json');
    console.log('Motion learner: python -m roboproof.motion_learning.train --seed 42 --seed-count 3 --max-seconds 180\nmotion-compare-learned --directory roboproof/runs/motion/learning');
    console.log('Original Simulator: motion-baseline --seed 42 --task task.json --configuration configuration.json --out report.json\nmotion-replay --file report.json --out replay.json');
    console.log('RoboProof: node roboproof/cli.js <command> [--option value]\nnominal --out result.json\nsample --count 64 --seed 42 --out scenarios.json\nrun --scenarios scenarios.json --out directory\nreplay --file counterexample.json --out replay.json\ndemo --count 64 --holdout 32 --generations 2 --population 8 --out directory\nverify --scenarios scenarios.json --candidate candidate.json --holdout 32 --out directory\ndiagnose --file counterexample.json --out diagnosis.json\nbenchmark --counts 1,10,100 --out benchmark.json');
    return;
  }
  if (options.command === 'nominal') {
    const result = runScenario(makeScenario(seed), {telemetry: true});
    writeJson(options.out || 'roboproof/runs/nominal.json', result);
    console.log(JSON.stringify({passed: result.passed, metrics: result.metrics}));
    if (!result.passed) process.exitCode = 2;
    return;
  }
  if (options.command === 'sample') {
    writeJson(options.out || 'roboproof/runs/scenarios.json', {distribution: spec.distribution, scenarios: sampleScenarios(number(options, 'count', 64), seed)});
    return;
  }
  if (options.command === 'replay') {
    if (!options.file) throw Error('--file is required');
    const original = read(options.file);
    const current = provenance();
    for (const key of ['simulation_version', 'controller_wasm_sha256', 'runtime', 'platform']) if (original[key] !== current[key]) throw Error(`Replay provenance mismatch: ${key}`);
    const result = runScenario(original.scenario, {telemetry: true});
    const keys = ['scenario_sha256', 'metrics', 'categories', 'final_truth', 'final_estimate', 'failure_score'];
    const digest = value => hash(JSON.stringify(keys.map(key => value[key])));
    result.replay_matches = digest(result) === digest(original);
    writeJson(options.out || 'roboproof/runs/replay.json', result);
    console.log(JSON.stringify({replay_matches: result.replay_matches, scenario_id: result.scenario.scenario_id}));
    if (!result.replay_matches) process.exitCode = 2;
    return;
  }
  if (options.command === 'diagnose') {
    if (!options.file) throw Error('--file is required');
    writeJson(options.out || 'roboproof/runs/diagnosis.json', diagnose(read(options.file)));
    return;
  }
  if (options.command === 'benchmark') {
    const counts = (options.counts || '1,10,100').split(',').map(Number);
    if (counts.some(count => !Number.isInteger(count) || count < 1 || count > 100000)) throw Error('Invalid benchmark counts');
    runScenario(makeScenario(seed));
    const rows = counts.map(count => {
      const timed = timedBatch(sampleScenarios(count, seed), `benchmark ${count}`);
      return {count, seconds: timed.seconds, simulations_per_second: timed.simulations_per_second, steps_per_second: timed.results.reduce((total, result) => total + result.metrics.steps, 0) / timed.seconds};
    });
    writeJson(options.out || 'roboproof/runs/benchmark.json', {...provenance(), backend: 'cpu-wasm', gpu: 'NOT MEASURED', duration_per_scenario: spec.task.duration, rows});
    console.log(JSON.stringify(rows));
    return;
  }
  const scenarios = options.scenarios ? loadScenarios(options.scenarios) : sampleScenarios(number(options, 'count', 64), seed);
  if (options.command === 'verify') {
    if (!options.scenarios || !options.candidate) throw Error('verify requires --scenarios and --candidate');
    const candidate = read(options.candidate);
    const report = verifyReport(scenarios, candidate, sampleScenarios(number(options, 'holdout', 32), (seed ^ 0x5bd1e995) >>> 0));
    writeJson(path.join(out, 'verification.json'), report);
    console.log(JSON.stringify({training: report.training, holdout: report.holdout, accepted: report.accepted}));
    return;
  }
  const started = performance.now();
  const baseline = timedBatch(scenarios, 'Monte Carlo');
  const worst = [...baseline.results].sort((left, right) => right.failure_score - left.failure_score)[0];
  const report = {schema_version: 1, ...provenance(), backend: 'cpu-wasm', nominal: runScenario(makeScenario(seed)), distribution: spec.distribution, scenarios, results: baseline.results, summary: baseline.summary, benchmark: {seconds: baseline.seconds, simulations_per_second: baseline.simulations_per_second, gpu: 'NOT MEASURED'}, worst_scenario_id: worst.scenario.scenario_id};
  let counterexample = worst;
  if (options.command === 'demo') {
    const hunt = adversarial(baseline.results, {seed: (seed ^ 0xa1b2c3d4) >>> 0, generations: number(options, 'generations', 2, 0, 100), population: number(options, 'population', 8), onProgress: progress('adversarial')});
    report.search = {history: hunt.history, results: hunt.results, algorithm: hunt.algorithm};
    counterexample = hunt.worst;
    report.diagnosis = diagnose(counterexample);
    const tuning = [];
    for (const reference_time_scale of [1.25, 1.5, 2]) {
      const candidate = {reference_time_scale};
      const trial = experiment(scenarios, candidate, baseline.results, `tuning ${reference_time_scale}`);
      tuning.push({candidate, ...trial});
    }
    tuning.sort((left, right) => right.comparison.after.passed - left.comparison.after.passed || left.candidate.reference_time_scale - right.candidate.reference_time_scale);
    const selected = tuning[0];
    const holdout = sampleScenarios(number(options, 'holdout', 32), (seed ^ 0x5bd1e995) >>> 0);
    const heldBefore = timedBatch(holdout, 'holdout baseline');
    const heldAfter = experiment(holdout, selected.candidate, heldBefore.results, 'holdout candidate');
    const searchAfter = experiment(hunt.results.map(result => result.scenario), selected.candidate, hunt.results, 'adversarial regression');
    report.verification = {candidate: selected.candidate, training: selected.comparison, holdout: heldAfter.comparison, adversarial: searchAfter.comparison, tuning: tuning.map(trial => ({candidate: trial.candidate, comparison: trial.comparison})), after: selected.results, holdout_before: heldBefore.results, holdout_after: heldAfter.results, adversarial_after: searchAfter.results, accepted: selected.comparison.net_improvement > 0 && heldAfter.comparison.net_improvement > 0 && selected.comparison.new_regressions === 0 && heldAfter.comparison.new_regressions === 0 && searchAfter.comparison.new_regressions === 0};
    report.proposal = {type: 'isolated-controller-configuration', hypothesis: 'Slower reference reduces demanded acceleration and wheel slip, potentially improving encoder odometry under traction limits.', expected_effect: 'Lower endpoint and localization error at the cost of longer motion time; fixed scenario deadline remains unchanged.', risk: 'May introduce timeouts or worsen drift; no production configuration is changed.', required_tests: ['nominal', 'same training worlds', 'unseen holdout worlds', 'adversarial counterexamples', 'physical robot'], status: report.verification.accepted ? 'PASSES LOCAL MODEL GATES; REQUIRES PHYSICAL TEST' : 'REJECTED BY LOCAL MODEL GATES', diff: `- reference_time_scale: 1\n+ reference_time_scale: ${selected.candidate.reference_time_scale}\n`};
    writeJson(path.join(out, 'candidate.json'), selected.candidate);
  }
  const replay = runScenario(counterexample.scenario, {telemetry: true});
  report.worst_scenario_id = replay.scenario.scenario_id;
  report.counterexample = replay;
  report.elapsed_seconds = (performance.now() - started) / 1000;
  writeJson(path.join(out, 'scenarios.json'), {distribution: spec.distribution, scenarios});
  writeJson(path.join(out, 'counterexample.json'), replay);
  writeJson(path.join(out, 'report.json'), report);
  console.log(JSON.stringify({report: path.join(out, 'report.json'), summary: report.summary, proposal: report.proposal?.status}));
}

if (require.main === module) {
  try { main(); } catch (error) { console.error(`RoboProof: ${error.message}`); process.exitCode = 1; }
}
module.exports = {main, args, loadScenarios, verifyReport};
