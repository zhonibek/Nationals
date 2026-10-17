'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Gvf = require('../simulator/gvf-policy');
const Legacy = require('../simulator/policy');
const plain = value => JSON.parse(JSON.stringify(value));
const requested = overrides => ({controlMode: Gvf.CONTRACT_NAME, policyContractName: Gvf.CONTRACT_NAME,
  policyContractVersion: Gvf.CONTRACT_VERSION, ...overrides});

class LifecycleDouble {
  constructor(control) {
    this.productionControl = control;
    this.override = {};
    Object.assign(this, {massKg: 6.8, moiKgM2: 0.145, muLong: 0.8, muLat: 0.08, kSlip: 650, batteryInternalR: 0.04});
  }
  resetSimulation() {
    this.setPose(0, 0, 0);
    this.odomVelocity = [0, 0, 0];
    this.wheelOmega = [0, 0, 0, 0];
    this.wheelRadiusM = 0.03;
    this.motorVolts = [0, 0, 0, 0];
    this.motorCurrents = [0, 0, 0, 0];
    this.contactSeconds = this.Vx = this.Vy = this.w = 0;
    this.isRunning = false;
    this.lastMotionResult = 'Running';
    this.queued = [];
  }
  setPose(xIn, yIn, headingDeg) {
    this.x = xIn; this.y = yIn; this.theta = headingDeg;
    this.odom = {x: xIn, y: yIn, theta: headingDeg};
  }
  queueAction(action) { this.queued.push(action); this.currentAction = action; }
  update() { this.onTick?.(); }
}

function fixture() {
  const state = {initializations: 0, samples: 0, unavailable: false, rejected: false};
  const control = {gvf() { state.initializations++; return !state.rejected; },
    gvfReference(feedback) {
      state.samples++;
      return state.unavailable ? null : {reference: [feedback[0], feedback[1], feedback[2], 0, 0.2, 0],
        parameter: 0.25, remainingMeters: 0.4, crossTrackMeters: 0, complete: true};
    }, duration() { return 1; }, reference(start) { return [...start, 0, 0, 0]; }};
  const context = {module: {exports: {}}, require(name) {
    if (name === './engine') return {VexRobotSimulator: LifecycleDouble};
    if (name === './policy') return Legacy;
    if (name === './gvf-policy') return Gvf;
    throw Error('Unexpected lifecycle-test dependency');
  }};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../simulator/motion.js'), 'utf8'), context);
  const {MotionEpisode, DEFAULT_TASK} = context.module.exports;
  return {episode: new MotionEpisode(() => control), state, task: plain(DEFAULT_TASK), motion: context.module.exports};
}

function bridgeFixture() {
  const current = fixture();
  const context = {module: {exports: {}}, require(name) {
    if (name === '../simulator/headless') return {loadHeadless: () => ({identity: {canonicalSimulator: true}, createEpisode: () => current.episode})};
    if (name === '../simulator/policy') return Legacy;
    if (name === '../simulator/gvf-policy') return Gvf;
    if (name === '../simulator/motion') return current.motion;
    throw Error('Unexpected mock bridge dependency');
  }};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../roboproof/motion-bridge.js'), 'utf8'), context);
  const bridge = context.module.exports.createBridge({controlMode: 'gvf-residual'});
  let sequence = 0;
  return {...current, request(op, fields = {}) {
    return bridge.handle({protocolVersion: 1, id: sequence++, op, ...fields}).result;
  }};
}

test('GVF episode requires explicit contract identity and frozen cadence, not merely a 34-vector', () => {
  const valid = Gvf.options(requested());
  assert.equal(valid.policyIntervalTicks, 5);
  assert.equal(valid.actionTimeoutTicks, 10);
  assert.equal(valid.recordTransitions, true);
  for (const changes of [{policyContractVersion: 1}, {policyContractName: 'policy'}, {controlMode: 'policy'},
    {policyIntervalTicks: 4}, {actionTimeoutTicks: 9}, {recordTransitions: 1}, {pace: 1},
    {policyIdentity: {kind: 'learned', id: 'unbound'}}, {policyIdentity: {kind: 'untrained', id: 'bad/id'}}]) {
    assert.throws(() => Gvf.options(requested(changes)));
  }
  assert.throws(() => Gvf.options({controlMode: Gvf.CONTRACT_NAME}), /Explicit/);
  assert.throws(() => Legacy.options(requested()), /Unsupported/);
  const identity = {kind: 'learned', id: 'bound-actor', sha256: 'a'.repeat(64)};
  const parsed = Gvf.options(requested({policyIdentity: identity}));
  identity.id = 'mutated';
  assert.equal(parsed.policyIdentity.id, 'bound-actor');
});

test('mock lifecycle queues one persistent native-GVF reference and refreshes current sensors without observation side effects', () => {
  const {episode, state} = fixture();
  const initial = episode.reset(7, {}, undefined, requested());
  assert.equal(Gvf.vector(initial).length, 34);
  assert.equal(state.initializations, 1);
  assert.equal(episode.referenceAdapter, null);
  const sampleCount = state.samples;
  assert.deepEqual(episode.observe(), initial);
  assert.equal(state.samples, sampleCount);
  episode.sim.onTick = () => { episode.sim.odom.x += 1; };
  const first = episode.gvfResidualStep([0, 0, 0]);
  assert.equal(first.info.physicsTicks, 5);
  assert.equal(first.observation.pose[0], 5 * 0.0254);
  assert.equal(first.observation.baseReference[0], first.observation.pose[0]);
  assert.equal(first.observation.actionAgeTicks, 5);
  assert.equal(episode.gvfResidualAdapter.tick, 5);
  assert.equal(episode.sim.queued.length, 1);
  assert.equal(episode.sim.queued[0].type, 'gvfResidualReference');
  assert.equal(first.terminated, false);
  assert.equal(first.truncated, false);
  episode.gvfResidualStep([0.25, -0.25, 0]);
  assert.equal(episode.sim.queued.length, 1);
  assert.equal(state.initializations, 1);
  const report = episode.report();
  assert.equal(report.policyContractName, Gvf.CONTRACT_NAME);
  assert.equal(report.policyContractVersion, 2);
  assert.deepEqual(plain(report.actions.map(row => row.action.type)), ['gvf-residual', 'gvf-residual']);
});

test('mock lifecycle keeps truth and configuration outside actor inputs at fixed sensors', () => {
  const {episode} = fixture();
  episode.reset(8, {}, undefined, requested());
  const before = Gvf.vector(episode.observe());
  episode.sim.x = 40;
  episode.sim.y = -40;
  episode.sim.theta = 170;
  episode.sim.Vx = 5;
  episode.sim.contactSeconds = 99;
  episode.sim.massKg = 12;
  episode.configuration.massKg = 12;
  assert.deepEqual(Gvf.vector(episode.observe()), before);
  assert.notEqual(episode.metrics().headingErrorRadians, 0);
});

test('legacy actions, malformed values, repeated and off-cadence actions neutralize the new mode', () => {
  for (const action of [{type: 'policy', values: [1, 0, 0, 0]}, {type: 'gvf-residual', values: [0, NaN, 0]},
    {type: 'gvf-residual', values: [1, 0, 0, 0]}, {type: 'pose', xIn: 0, yIn: 24, headingDeg: 0}]) {
    const {episode} = fixture();
    episode.reset(9, {}, undefined, requested());
    assert.throws(() => episode.applyAction(action));
    assert.equal(episode.sim.isRunning, false);
    assert.equal(episode.step().info.reason, 'stopped');
  }
  for (const ticks of [0, 1]) {
    const {episode} = fixture();
    episode.reset(10, {}, undefined, requested());
    episode.applyAction({type: 'gvf-residual', values: [0, 0, 0]});
    if (ticks) episode.step();
    assert.throws(() => episode.applyAction({type: 'gvf-residual', values: [0, 0, 0]}), /cadence/);
    assert.equal(episode.sim.isRunning, false);
    assert.equal(episode.step().info.reason, 'stopped');
  }
});

test('missing and stale commands retain fault penalties and valid terminal action ages', () => {
  const missing = fixture().episode;
  missing.reset(11, {}, undefined, requested());
  const absent = missing.step();
  assert.equal(absent.info.reason, 'stale_policy_action');
  assert.equal(absent.rewardComponents.fault, -1);
  const {episode} = fixture();
  episode.reset(12, {}, undefined, requested());
  episode.applyAction({type: 'gvf-residual', values: [0, 0, 0]});
  for (let tick = 0; tick < 10; tick++) assert.equal(episode.step().terminated, false);
  const stale = episode.step();
  assert.equal(stale.info.reason, 'stale_policy_action');
  assert.equal(stale.rewardComponents.fault, -1);
  assert.equal(stale.observation.actionAgeTicks, 11);
  assert.equal(Gvf.vector(stale.observation).length, 34);
  assert.equal(episode.sim.isRunning, false);
});

test('native sample and sensor faults fall back explicitly and cannot produce successful terminal observations', () => {
  for (const fault of ['native', 'sensor']) {
    const {episode, state} = fixture();
    episode.reset(13, {}, undefined, requested());
    if (fault === 'native') state.unavailable = true;
    else episode.sim.onTick = () => { episode.sim.wheelOmega[0] = NaN; };
    const result = episode.gvfResidualStep([0, 0, 0]);
    assert.equal(result.terminated, true);
    assert.equal(result.info.reason, 'controller_or_sensor_fault');
    assert.equal(result.info.observationValid, false);
    assert.match(result.info.observationFallback, /last valid/);
    assert.equal(result.rewardComponents.fault, -1);
    assert.equal(result.vector.every(Number.isFinite), true);
    assert.equal(episode.sim.isRunning, false);
  }
  const {episode, state} = fixture();
  state.rejected = true;
  assert.throws(() => episode.reset(14, {}, undefined, requested()), /initialization failed/);
  assert.equal(episode.terminated, true);
  assert.equal(episode.sim.isRunning, false);
  assert.throws(() => episode.applyAction({type: 'gvf-residual', values: [0, 0, 0]}), /ended/);
});

test('mock lifecycle preserves reward aggregation, 0.01-second deadline and fifteen-tick settlement', () => {
  const partial = fixture();
  partial.task.deadlineSeconds = 0.03;
  partial.episode.reset(15, {}, partial.task, requested());
  const result = partial.episode.gvfResidualStep([0, 0, 0]);
  assert.equal(result.truncated, true);
  assert.equal(result.info.reason, 'time_limit');
  assert.equal(result.info.physicsTicks, 3);
  assert.equal(result.vector[17], 0);
  const transitions = partial.episode.report().transitions;
  assert.equal(result.reward, transitions.reduce((sum, row) => sum + row.reward, 0));
  assert(transitions.every(row => row.rewardComponents.time === -0.0001));
  assert.equal(result.rewardComponents.time, transitions.reduce((sum, row) => sum + row.rewardComponents.time, 0));
  const settled = fixture();
  settled.task.goal = {...settled.task.start};
  settled.episode.reset(16, {}, settled.task, requested());
  settled.episode.gvfResidualStep([0, 0, 0]);
  settled.episode.gvfResidualStep([0, 0, 0]);
  const completed = settled.episode.gvfResidualStep([0, 0, 0]);
  assert.equal(completed.info.reason, 'success');
  assert.equal(settled.episode.ticks, 15);
  assert.equal(completed.rewardComponents.success, 1);
});

test('reset clears residual state and legacy episode shape and mode remain explicit', () => {
  const {episode} = fixture();
  const first = episode.reset(17, {}, undefined, requested());
  episode.gvfResidualStep([1, -1, 1]);
  const again = episode.reset(17, {}, undefined, requested());
  assert.deepEqual(again, first);
  assert.equal(episode.ticks, 0);
  assert.throws(() => episode.policyStep([1, 0, 0, 0]), /policy mode/);
  const legacy = episode.reset(17, {}, undefined, {controlMode: 'policy'});
  assert.equal(Legacy.vector(legacy).length, 34);
  assert.equal(episode.gvfResidualAdapter, null);
  assert.equal(episode.report().policyContractVersion, 1);
  assert.equal(Object.hasOwn(episode.report(), 'policyContractName'), false);
  assert.throws(() => episode.gvfResidualStep([0, 0, 0]), /explicit GVF/);
  const scripted = episode.reset();
  assert.equal(Object.hasOwn(scripted, 'policyState'), false);
  assert.equal(Object.hasOwn(scripted, 'contractName'), false);
  assert.equal(episode.options.controlMode, 'scripted');
});

test('actual bridge and mocked episode lifecycle preserve three-action steps and idempotent terminal stop', () => {
  const current = bridgeFixture();
  assert.deepEqual(plain(current.request('stop')), {stopped: true, requiresReset: true});
  const reset = current.request('reset', {seed: 18, task: {...current.task, deadlineSeconds: 0.03}});
  assert.equal(reset.contractName, Gvf.CONTRACT_NAME);
  assert.equal(reset.vector.length, 34);
  assert.equal(Object.hasOwn(reset, 'referenceDurationSeconds'), false);
  const terminal = current.request('step', {action: [0, 0, 0]});
  assert.equal(terminal.truncated, true);
  assert.equal(terminal.info.physicsTicks, 3);
  const before = plain(current.episode.report());
  assert.deepEqual(plain(current.request('stop')), {stopped: true, requiresReset: true});
  assert.deepEqual(plain(current.episode.report()), before);
  assert.throws(() => current.request('step', {action: [0, 0, 0]}), /Reset/);
});

test('bridge contract/reset failure neutralizes without hiding the original error or consuming extra ticks', () => {
  const current = bridgeFixture();
  current.request('reset', {seed: 19});
  current.request('step', {action: [0, 0, 0]});
  const tick = current.episode.ticks;
  assert.throws(() => current.request('reset', {seed: 20, options: {policyContractVersion: 1}}), /Process-fixed/);
  assert.equal(current.episode.sim.isRunning, false);
  assert.equal(current.episode.ticks, tick);
  assert.throws(() => current.request('step', {action: [0, 0, 0]}), /Reset/);
  current.request('reset', {seed: 21});
  assert.throws(() => current.request('step', {action: [1, 0, 0, 0]}), /3 finite/);
  assert.equal(current.episode.reason, 'stopped');
  assert.equal(current.episode.ticks, 1);
  current.request('stop');
  assert.equal(current.episode.ticks, 1);
});
