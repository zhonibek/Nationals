'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const {VexRobotSimulator} = require('../simulator/engine');
const {loadHeadless} = require('../simulator/headless');
const {DEFAULT_TASK, FIXED_DT} = require('../simulator/motion');
const {baseline, replay} = require('../roboproof/motion');
const Control = require('../simulator/control-runtime');
const root = path.resolve(__dirname, '..');
const compiled = new WebAssembly.Module(fs.readFileSync(path.join(root, 'simulator/control.wasm')));
const fixture = require('./fixtures/original-simulator-traces.json');
const state = sim => [sim.x, sim.y, sim.theta, sim.Vx, sim.Vy, sim.w, ...Object.values(sim.odom),
  ...sim.odomVelocity, ...sim.wheelOmega, ...sim.motorVolts, ...sim.motorCurrents, sim.lastMotionResult];
const runner = loadHeadless();

function browserRealm() {
  const context = vm.createContext({console});
  for (const filename of ['robot-config.js', 'control-runtime.js', 'override-geometry.js', 'override-dynamics.js',
    'override-autonomy.js', 'override.js', 'engine.js', 'policy.js', 'motion.js']) {
    vm.runInContext(fs.readFileSync(path.join(root, 'simulator', filename), 'utf8'), context, {filename});
  }
  return context;
}

test('original engine imports without DOM, renderer, timers or reduced RoboProof physics', () => {
  const realm = browserRealm();
  assert.equal(vm.runInContext('typeof document', realm), 'undefined');
  assert.equal(vm.runInContext('typeof setTimeout', realm), 'undefined');
  assert.equal(vm.runInContext('typeof THREE', realm), 'undefined');
  const sim = new VexRobotSimulator(Control.fromModule(compiled));
  assert.equal(sim.override.constructor.name, 'OverrideGame');
  assert.equal(sim.massKg, 6.8);
  assert.equal(sim.wheelRadiusM, 3.25 * 0.0254 / 2);
  assert.equal(runner.identity.canonicalSimulator, true);
});

for (const entry of fixture.cases) {
  test(`unchanged original trace: ${entry.id}`, () => {
    const sim = new VexRobotSimulator(Control.fromModule(compiled));
    if (entry.id === 'match') sim.startGame('match');
    else { sim.setPose(...entry.start); sim.queueAction({...entry.action}); sim.isRunning = true; }
    const hash = crypto.createHash('sha256');
    for (let tick = 0; tick < entry.ticks; tick++) {
      sim.update(fixture.dt);
      const row = sim.fleet ? [state(sim), ...Array.from(sim.fleet.entries.values(), value => state(value.sim)),
        sim.override.clock, sim.override.phase, sim.override.score()] : state(sim);
      hash.update(JSON.stringify(row) + '\n');
    }
    assert.equal(hash.digest('hex'), entry.sha256);
    assert.equal(JSON.stringify(state(sim)), JSON.stringify(entry.final));
    if (entry.score) assert.deepEqual(sim.override.score(), entry.score);
  });
}

test('same episode contract produces exact browser-realm/headless transitions', () => {
  const realm = browserRealm();
  const browser = new realm.NationalsMotion.MotionEpisode(() => realm.ProductionControl.fromModule(compiled));
  const headless = runner.createEpisode();
  const configuration = {massKg: [6, 7], muLong: [0.7, 0.9], batteryInternalR: [0.02, 0.06]};
  const task = {start: {xIn: -8, yIn: -8, headingDeg: 170}, goal: {xIn: 8, yIn: 8, headingDeg: -170}, deadlineSeconds: 10};
  browser.reset(789, configuration, task);
  headless.reset(789, configuration, task);
  browser.applyAction({type: 'pose', ...task.goal});
  headless.applyAction({type: 'pose', ...task.goal});
  while (!headless.terminated && !headless.truncated) {
    headless.observe();
    headless.observe();
    const left = headless.step();
    const right = browser.step();
    assert.deepEqual(structuredClone(right), left);
    assert.deepEqual(structuredClone(state(browser.sim)), state(headless.sim));
  }
  assert.deepEqual(structuredClone(browser.report()), headless.report());
});

test('reset and interleaved environments do not share WASM, physics, RNG, clocks or queues', () => {
  const first = runner.createEpisode(), second = runner.createEpisode(), reference = runner.createEpisode();
  const configuration = {massKg: [5, 8], muLong: [0.65, 0.95]};
  for (const episode of [first, second, reference]) episode.reset(123, configuration);
  assert.notEqual(first.sim.productionControl.e.memory.buffer, second.sim.productionControl.e.memory.buffer);
  first.applyAction({type: 'pose', ...DEFAULT_TASK.goal});
  reference.applyAction({type: 'pose', ...DEFAULT_TASK.goal});
  second.applyAction({type: 'pose', xIn: 24, yIn: 0, headingDeg: 90});
  for (let tick = 0; tick < 200; tick++) {
    const actual = first.step();
    second.step();
    assert.deepEqual(actual, reference.step());
  }
  const oldControl = first.sim.productionControl;
  first.sim.intakeVoltage = 12000;
  first.sim.isPaused = true;
  first.sim.triggerRumble('.');
  first.reset(123, configuration);
  const fresh = runner.createEpisode();
  fresh.reset(123, configuration);
  assert.notEqual(oldControl.e.memory.buffer, first.sim.productionControl.e.memory.buffer);
  assert.deepEqual(first.report(), fresh.report());
  assert.deepEqual(first.observe(), fresh.observe());
  assert.equal(first.sim.intakeVoltage, 0);
  assert.equal(first.sim.isPaused, false);
  assert.equal(first.sim.rumbleActive, false);
  first.applyAction({type: 'pose', ...DEFAULT_TASK.goal});
  fresh.applyAction({type: 'pose', ...DEFAULT_TASK.goal});
  for (let tick = 0; tick < 100; tick++) assert.deepEqual(first.step(), fresh.step());
});

test('observations contain encoder estimates, never ground-truth pose or velocity', () => {
  const episode = runner.createEpisode();
  episode.reset();
  episode.sim.x = 12;
  episode.sim.Vx = 2;
  const observation = episode.observe();
  assert.equal(observation.pose[0], 0);
  assert.equal(observation.velocity[0], 0);
  assert.equal(observation.targetError[0], 0);
  assert.notEqual(episode.metrics().positionErrorMeters, 24 * 0.0254);
});

test('actual settling is success; deadlines are truncation; reward uses measured truth', () => {
  const episode = runner.createEpisode();
  const task = {start: {xIn: 0, yIn: 0, headingDeg: 179}, goal: {xIn: 0, yIn: 0, headingDeg: -179}, deadlineSeconds: 1};
  episode.reset(42, {}, task);
  let transition;
  for (let tick = 0; tick < 15; tick++) transition = episode.step();
  assert.equal(transition.terminated, true);
  assert.equal(transition.truncated, false);
  assert.equal(transition.info.reason, 'success');
  assert.equal(transition.rewardComponents.success, 1);
  assert.throws(() => episode.step(), /ended/);
  episode.reset(42, {}, {...DEFAULT_TASK, deadlineSeconds: 0.02});
  episode.sim.odom.y = 24;
  transition = episode.step();
  assert.equal(transition.rewardComponents.progress, 0);
  transition = episode.step();
  assert.equal(transition.truncated, true);
  assert.equal(transition.terminated, false);
  assert.equal(transition.info.reason, 'time_limit');
});

test('unsupported randomness, NaNs, invalid dt and active-controller replacement are rejected', () => {
  const episode = runner.createEpisode();
  assert.throws(() => episode.reset(-1), /uint32/);
  assert.throws(() => episode.reset(42, {batteryVoltage: 10}), /Unsupported/);
  assert.throws(() => episode.reset(42, {gyroDriftRate: 10}), /Unsupported/);
  assert.throws(() => episode.reset(42, {massKg: [8, 5]}), /must be in/);
  assert.throws(() => episode.reset(42, {}, {...DEFAULT_TASK, deadlineSeconds: 0.015}), /multiple/);
  episode.reset();
  episode.applyAction({type: 'pose', ...DEFAULT_TASK.goal});
  episode.step();
  const controller = episode.sim.productionControl;
  assert.throws(() => episode.applyAction({type: 'pose', ...DEFAULT_TASK.goal}), /already active/);
  assert.equal(episode.sim.productionControl, controller);
  assert.throws(() => episode.applyAction({type: 'pose', xIn: NaN, yIn: 24, headingDeg: 0}), /must be in/);
  assert.equal(episode.sim.isRunning, false);
  assert.equal(episode.sim.commandedWheelVoltages, null);
  assert.throws(() => episode.step(0.02), /fixed dt/);
  assert.equal(episode.ticks, 1);
  episode.sim.odom.x = NaN;
  const failed = episode.step();
  assert.equal(failed.terminated, true);
  assert.equal(failed.info.reason, 'controller_or_sensor_fault');
  assert(Number.isFinite(failed.reward));
});

test('rumble uses simulated time and the shared engine clears derived reset state', () => {
  const sim = new VexRobotSimulator(Control.fromModule(compiled));
  sim.triggerRumble('.');
  sim.update(FIXED_DT);
  assert.equal(sim.rumbleActive, true);
  for (let tick = 0; tick < 30; tick++) sim.update(FIXED_DT);
  assert.equal(sim.rumbleActive, false);
  sim.motorTorques.fill(1);
  sim.wheelTractionForces.fill(2);
  sim.plannedSplineVisual.push({x: 12, y: 12});
  sim.contactSeconds = 3;
  sim.resetSimulation();
  assert(sim.motorTorques.every(value => value === 0));
  assert(sim.wheelTractionForces.every(value => value === 0));
  assert.deepEqual(sim.plannedSplineVisual, []);
  assert.equal(sim.contactSeconds, 0);
});

test('RoboProof runs and replays the original Simulator, rejects altered motion or identities', () => {
  const report = baseline();
  assert.equal(report.reason, 'success');
  assert.equal(report.identity.canonicalSimulator, true);
  assert.match(report.learningStatus, /does not train/);
  assert.deepEqual(replay(report), report);
  assert.throws(() => replay({...report, totalReward: 999}), /does not reproduce/);
  assert.throws(() => replay({...report, identity: {...report.identity, controllerWasmSha256: 'forged'}}), /identity mismatch/);
  assert.throws(() => replay({...report, actions: [{tick: -1, action: {type: 'stop'}}]}), /action tick/);
});
