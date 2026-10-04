'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {makeScenario, sampleScenarios, clone, spec, validate, random, provenance} = require('../roboproof/core');
const {runScenario, physics, forward, pathDistance} = require('../roboproof/sim');
const {compare, summary, adversarial, evidenceTools, diagnose} = require('../roboproof/analysis');
const {args} = require('../roboproof/cli');

test('nominal real WASM controller completes and replays exactly', () => {
  const scenario = makeScenario(701);
  const original = runScenario(scenario, {telemetry: true});
  const replay = runScenario(scenario, {telemetry: true});
  assert.deepEqual(replay, original);
  assert.equal(original.passed, true);
  assert.equal(original.backend, 'cpu-wasm');
  assert.equal(original.telemetry.length, 1000);
  assert(original.metrics.endpoint_error < scenario.task.thresholds.endpoint);
  assert(original.metrics.max_localization_error > 0.01);
  assert(original.telemetry.some(row => row.truth[0] !== row.estimate[0]));
  assert.equal(original.controller_wasm_sha256, provenance().controller_wasm_sha256);
});

test('seed and full scenario define independent reproducible noisy worlds', () => {
  const scenario = makeScenario(22, {environment: {encoder_noise: 0.006, encoder_dropout: 0.04, imu_noise: 0.01}, task: {duration: 1, localization: 'encoders-imu'}});
  const first = runScenario(scenario, {telemetry: true});
  const second = runScenario(scenario, {telemetry: true});
  assert.deepEqual(first, second);
  const different = clone(scenario);
  different.random_seed++;
  assert.notDeepEqual(runScenario(different).final_estimate, first.final_estimate);
  assert.deepEqual(sampleScenarios(5, 18), sampleScenarios(5, 18));
  assert.notDeepEqual(sampleScenarios(5, 18), sampleScenarios(5, 19));
});

test('initial physical error is never secretly injected into controller feedback', () => {
  const scenario = makeScenario(1, {environment: {initial_x_error: 0.02, initial_heading_error: 0.035}, task: {duration: 0.1}});
  const result = runScenario(scenario, {telemetry: true});
  assert.deepEqual(result.telemetry[0].estimate, [0, 0, 0]);
  assert.deepEqual(result.telemetry[0].truth, [0.02, 0, 0.035]);
  const baseline = runScenario({...scenario, task: {...scenario.task, localization: 'ground-truth-baseline'}}, {telemetry: true});
  assert.deepEqual(baseline.telemetry[0].truth, baseline.telemetry[0].estimate);
});

test('latency means delayed command and delayed samples, not random jitter', () => {
  const scenario = makeScenario(4, {environment: {control_latency: 0.04, encoder_latency: 0.03}, task: {duration: 0.2}});
  const result = runScenario(scenario, {telemetry: true});
  for (let index = 0; index < 4; index++) assert.deepEqual(result.telemetry[index].applied, [0, 0, 0, 0]);
  for (let index = 4; index < result.telemetry.length; index++) assert.deepEqual(result.telemetry[index].applied, result.telemetry[index - 4].command);
  for (let index = 3; index < result.telemetry.length; index++) assert.deepEqual(result.telemetry[index].measured_wheel, result.telemetry[index - 3].wheel_speed);
});

test('physics has finite response, traction limit, battery effect and rotational inertia', () => {
  const zero = () => ({pose: [0, 0, 0], velocity: [0, 0, 0], wheel: [0, 0, 0, 0]});
  const state = zero();
  physics(state, [12, 12, 12, 12], spec.nominal, 0.001);
  assert(state.wheel[0] > 0 && state.wheel[0] < spec.nominal.max_wheel_speed / 10);
  const low = zero(), high = zero(), slippery = zero(), heavy = zero(), light = zero();
  for (let index = 0; index < 100; index++) {
    physics(low, [12, 12, 12, 12], {...spec.nominal, battery_voltage: 9}, 0.01);
    physics(high, [12, 12, 12, 12], spec.nominal, 0.01);
    physics(slippery, [12, 12, 12, 12], {...spec.nominal, friction: 0}, 0.01);
    physics(heavy, [12, 12, -12, -12], {...spec.nominal, inertia: 1}, 0.01);
    physics(light, [12, 12, -12, -12], {...spec.nominal, inertia: 0.1}, 0.01);
  }
  assert(high.pose[1] > low.pose[1]);
  assert.deepEqual(slippery.pose, [0, 0, 0]);
  assert(slippery.wheel[0] > 0.5);
  assert(Math.abs(light.velocity[2]) > Math.abs(heavy.velocity[2]));
});

test('low-traction counterexample is independently reproducible', () => {
  const scenario = sampleScenarios(3, 42)[2];
  const result = runScenario(scenario);
  assert.equal(result.passed, false);
  assert(result.categories.includes('LOCALIZATION_DIVERGENCE'));
  assert.deepEqual(result.metrics, runScenario(scenario).metrics);
  assert(result.metrics.max_localization_error > scenario.task.thresholds.localization);
});

test('scenario validation rejects impossible, nonfinite and incomplete worlds', () => {
  for (const battery_voltage of [-800, NaN, Infinity, 30]) assert.throws(() => makeScenario(1, {environment: {battery_voltage}}));
  for (const random_seed of [-1, 0.5, 2 ** 32, NaN]) assert.throws(() => makeScenario(random_seed));
  for (const dt of [0, -1, 0.1, NaN]) assert.throws(() => makeScenario(1, {task: {dt}}));
  const missing = makeScenario();
  delete missing.environment.friction;
  assert.throws(() => validate(missing));
  assert.throws(() => makeScenario(1, {environment: {typo: 4}}));
  assert.throws(() => makeScenario(1, {task: {localization: 'secret-truth'}}));
  assert.throws(() => sampleScenarios(0));
  assert.throws(() => sampleScenarios(1.5));
  assert.throws(() => args(['demo', '--sead', '42']));
});

test('regression accounting exposes fixes and new failures', () => {
  const worlds = sampleScenarios(4);
  const make = flags => flags.map((passed, index) => ({scenario: worlds[index], passed, categories: passed ? ['SUCCESS'] : ['TIMEOUT']}));
  const result = compare(make([false, true, false, true]), make([true, false, false, true]));
  assert.equal(result.failures_fixed, 1);
  assert.equal(result.new_regressions, 1);
  assert.equal(result.unchanged_failures, 1);
  assert.equal(result.unchanged_passes, 1);
  assert.equal(result.net_improvement, 0);
  const changed = make([true, true, true, true]);
  changed[0] = {...changed[0], scenario: {...changed[0].scenario, random_seed: 5}};
  assert.throws(() => compare(make([true, true, true, true]), changed));
  assert.equal(summary([]).robustness_percent, null);
});

test('search stays within physical bounds and retains worst counterexample', () => {
  const original = runScenario(makeScenario(1, {task: {duration: 0.2}}));
  const hunt = adversarial([original], {generations: 2, population: 2, seed: 81});
  assert.equal(hunt.results.length, 4);
  for (const result of hunt.results) validate(result.scenario);
  for (let index = 1; index < hunt.history.length; index++) assert(hunt.history[index].worst_score >= hunt.history[index - 1].worst_score);
  assert.deepEqual(hunt, adversarial([original], {generations: 2, population: 2, seed: 81}));
});

test('investigation tools restrict source and sweep access', () => {
  const result = runScenario(makeScenario(7, {task: {duration: 0.1}}), {telemetry: true});
  const investigator = evidenceTools(result);
  assert.equal(investigator.call('run_scenario').reproducible, true);
  assert.throws(() => investigator.call('execute_shell'));
  assert.throws(() => investigator.call('run_parameter_sweep', {parameter: 'battery_voltage', values: [-800]}));
  assert.throws(() => investigator.call('get_telemetry_range', {limit: 100000}));
  assert.equal(investigator.log.length, 1);
});

test('passing nominal diagnosis discloses failed sweeps and never claims neural or causal proof', () => {
  const result = runScenario(makeScenario(42));
  assert.equal(result.passed, true);
  assert.equal(result.canonicalSimulator, false);
  assert.equal(result.engine_family, 'legacy-reduced-v1');
  const diagnosis = diagnose(result);
  assert.equal(diagnosis.mode, 'deterministic-evidence-investigator');
  assert.match(diagnosis.llm_status, /^NOT CONNECTED/);
  assert.match(diagnosis.confidence, /^UNCALIBRATED/);
  const failures = diagnosis.sweep_summary.reduce((total, entry) => total + entry.failed, 0);
  assert(failures > 0);
  assert.match(diagnosis.inference, new RegExp(`${failures} of .* sweep runs failed`));
  assert.doesNotMatch(diagnosis.inference, /remain intact|Primary failure root cause/);
});

test('geometry reports spatial deviation, independently of timing', () => {
  assert.equal(pathDistance([0.5, 0, 0], [0, 0, 0], [1, 0, 0]), 0);
  assert.equal(pathDistance([0.5, 0.2, 0], [0, 0, 0], [1, 0, 0]), 0.2);
  assert.deepEqual(forward([1, 1, -1, -1], 0.2), [0, 0, 5]);
  const next = random(0);
  for (let index = 0; index < 1000; index++) assert(next() > 0 && next() < 1);
});
