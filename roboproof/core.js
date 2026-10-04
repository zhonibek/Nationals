'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const spec = require('./scenario-spec.json');
const ROOT = path.resolve(__dirname, '..');
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const clone = value => JSON.parse(JSON.stringify(value));
const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));

function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

function normal(next) {
  return Math.sqrt(-2 * Math.log(next())) * Math.cos(2 * Math.PI * next());
}

function finite(value, name, minimum, maximum) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw Error(`${name} must be finite in [${minimum}, ${maximum}]`);
  }
}

function validate(scenario) {
  if (!scenario || scenario.schema_version !== 1) throw Error('Unsupported scenario schema');
  if (typeof scenario.scenario_id !== 'string' || !scenario.scenario_id) throw Error('Missing scenario_id');
  finite(scenario.random_seed, 'random_seed', 0, 0xffffffff);
  if (!Number.isInteger(scenario.random_seed)) throw Error('random_seed must be an integer');
  if (!scenario.environment || !scenario.task || !scenario.controller) throw Error('Incomplete scenario');
  for (const [name, bounds] of Object.entries(spec.ranges)) finite(scenario.environment[name], name, ...bounds);
  for (const name of Object.keys(scenario.environment)) if (!(name in spec.nominal)) throw Error(`Unknown parameter: ${name}`);
  for (const name of ['start', 'goal']) {
    if (!Array.isArray(scenario.task[name]) || scenario.task[name].length !== 3) throw Error(`Invalid ${name}`);
    scenario.task[name].forEach(value => finite(value, name, -100, 100));
  }
  finite(scenario.task.dt, 'dt', 0.005, 0.05);
  finite(scenario.task.duration, 'duration', scenario.task.dt, 60);
  if (!['encoders', 'encoders-imu', 'ground-truth-baseline'].includes(scenario.task.localization)) throw Error('Unknown localization mode');
  for (const name of Object.keys(spec.task.thresholds)) finite(scenario.task.thresholds[name], name, 0.000001, 1000);
  finite(scenario.controller.radius, 'controller.radius', 0.1, 0.3);
  finite(scenario.controller.max_wheel_speed, 'controller.max_wheel_speed', 0.1, 2);
  finite(scenario.controller.reference_time_scale, 'reference_time_scale', 1, 4);
  if (scenario.controller.adapter !== 'nationals-wasm') throw Error('Unsupported controller adapter');
  return scenario;
}

function makeScenario(seed = 1, overrides = {}) {
  return validate({
    schema_version: 1,
    scenario_id: overrides.scenario_id || `scenario-${seed}`,
    random_seed: seed,
    environment: {...spec.nominal, ...overrides.environment},
    task: {...clone(spec.task), ...overrides.task, thresholds: {...spec.task.thresholds, ...overrides.task?.thresholds}},
    controller: {...spec.controller, ...overrides.controller}
  });
}

function sampleScenarios(count, seed = 42) {
  finite(count, 'count', 1, 100000);
  if (!Number.isInteger(count)) throw Error('count must be an integer');
  finite(seed, 'seed', 0, 0xffffffff);
  if (!Number.isInteger(seed)) throw Error('seed must be an integer');
  const next = random(seed);
  return Array.from({length: count}, (_, index) => {
    const environment = {};
    for (const [name, [minimum, maximum]] of Object.entries(spec.ranges)) environment[name] = minimum + next() * (maximum - minimum);
    return makeScenario(Math.floor(next() * 4294967296), {scenario_id: `mc-${seed}-${index}`, environment});
  });
}

function provenance() {
  const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'simulator/control-build.json'), 'utf8').replace(/^\uFEFF/, ''));
  for (const [filename, expected] of Object.entries(manifest.sources)) {
    if (hash(fs.readFileSync(path.join(ROOT, filename), 'utf8').replace(/\r\n/g, '\n')) !== expected) throw Error(`Stale production WASM: rebuild ${filename}`);
  }
  const wasm = hash(fs.readFileSync(path.join(ROOT, 'simulator/control.wasm')));
  if (wasm !== manifest.wasm) throw Error('WASM does not match build manifest');
  const files = ['core.js', 'sim.js', 'scenario-spec.json'];
  const sources = Object.fromEntries(files.map(filename => [filename, hash(fs.readFileSync(path.join(__dirname, filename), 'utf8').replace(/\r\n/g, '\n'))]));
  let revision = 'UNVERIFIED';
  try { revision = execFileSync('git', ['rev-parse', 'HEAD'], {cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore']}).trim(); } catch {}
  return {engine_family: 'legacy-reduced-v1', canonicalSimulator: false, task_type: 'legacy-point-to-point-stress-test', software_version: revision, simulation_version: hash(JSON.stringify(sources)), source_hashes: sources, controller_wasm_sha256: wasm, controller_source_hashes: manifest.sources, runtime: process.version, platform: `${process.platform}/${process.arch}`, reproducibility: 'Same source hashes, scenario, runtime and backend; cross-backend numeric tolerance required'};
}

function writeJson(filename, value) {
  const resolved = path.resolve(filename);
  fs.mkdirSync(path.dirname(resolved), {recursive: true});
  fs.writeFileSync(resolved, JSON.stringify(value, null, 2) + '\n');
}

module.exports = {ROOT, spec, hash, clone, clamp, wrap, random, normal, finite, validate, makeScenario, sampleScenarios, provenance, writeJson};
