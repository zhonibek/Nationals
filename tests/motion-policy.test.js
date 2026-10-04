'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {loadHeadless} = require('../simulator/headless');
const {DEFAULT_TASK} = require('../simulator/motion');
const Policy = require('../simulator/policy');
const {baseline, runPolicy, replay} = require('../roboproof/motion');
const Control = require('../simulator/control-runtime');
const runner = loadHeadless();
const root = path.resolve(__dirname, '..');

test('policy actions preserve controller state and never teleport or reset per decision', () => {
  const episode = runner.createEpisode();
  episode.reset(42, {}, DEFAULT_TASK, {controlMode: 'policy'});
  const controller = episode.sim.productionControl;
  let resets = 0;
  const originalReset = controller.reset.bind(controller);
  controller.reset = () => { resets++; originalReset(); };
  for (let decision = 0; decision < 20; decision++) {
    const before = [episode.sim.x, episode.sim.y, episode.sim.theta];
    episode.applyAction({type: 'policy', values: [1, decision % 2 ? 0.5 : -0.5, 0, 0]});
    assert.deepEqual([episode.sim.x, episode.sim.y, episode.sim.theta], before);
    for (let tick = 0; tick < 5; tick++) episode.step();
  }
  assert.equal(resets, 1);
  assert.equal(controller, episode.sim.productionControl);
  assert(episode.sim.y > 0);
  assert.equal(episode.ticks, 100);
});

test('policy reference pace, residual speeds and accelerations obey frozen limits', () => {
  const episode = runner.createEpisode();
  episode.reset(91, {}, DEFAULT_TASK, {controlMode: 'policy', recordTransitions: false});
  let previous = episode.referenceAdapter.observe(0);
  for (let decision = 0; decision < 35; decision++) {
    episode.applyAction({type: 'policy', values: decision % 2 ? [-1, -1, 1, -1] : [1, 1, -1, 1]});
    for (let tick = 0; tick < 5; tick++) {
      episode.step();
      const state = episode.referenceAdapter.observe(0);
      assert(state.pace >= Policy.LIMITS.minimumPace && state.pace <= 1);
      assert(Math.abs(state.pace - previous.pace) <= Policy.LIMITS.paceRate * 0.01 + 1e-12);
      for (let axis = 0; axis < 3; axis++) {
        const speed = axis === 2 ? Policy.LIMITS.offsetYawRate : Policy.LIMITS.offsetSpeedMetersPerSecond;
        const acceleration = axis === 2 ? Policy.LIMITS.offsetYawAcceleration : Policy.LIMITS.offsetAcceleration;
        assert(Math.abs(state.offsetVelocity[axis]) <= speed);
        assert(Math.abs(state.offsetVelocity[axis] - previous.offsetVelocity[axis]) <= acceleration * 0.01 + 1e-12);
      }
      assert(state.reference.slice(0, 2).every(value => Math.abs(value) <= Policy.LIMITS.fieldMeters));
      previous = state;
    }
  }
});

test('full policy transition log and JSON round trip replay exactly without executing the policy', () => {
  const report = runPolicy({options: {policyIdentity: {kind: 'scripted', id: 'nominal-reference-fixture'}}});
  assert.equal(report.reason, 'success');
  assert.equal(report.transitions.length, report.ticks);
  assert.equal(report.options.controlMode, 'policy');
  assert.match(report.learningStatus, /not implemented/);
  assert.deepEqual(replay(JSON.parse(JSON.stringify(report))), report);
  const total = report.transitions.reduce((sum, row) => sum + row.reward, 0);
  assert.equal(total, report.totalReward);
  for (let index = 1; index < report.transitions.length; index++) {
    const before = report.transitions[index].observation;
    const after = report.transitions[index - 1].nextObservation;
    assert.deepEqual(before, after);
  }
  const altered = structuredClone(report);
  altered.transitions[3].reward += 1;
  assert.throws(() => replay(altered), /does not reproduce/);
});

test('fixed policy vectors expose sensors and adapter state, not evaluator truth', () => {
  const episode = runner.createEpisode();
  episode.reset(5, {}, DEFAULT_TASK, {controlMode: 'policy'});
  const vector = Policy.vector(episode.observe());
  assert.equal(vector.length, 34);
  episode.sim.x = 35;
  episode.sim.Vx = 3;
  assert.deepEqual(Policy.vector(episode.observe()), vector);
  assert.throws(() => Policy.vector({}), /policy-mode/);
});

test('missing/stale actions, malformed vectors and off-cadence commands fail safely', () => {
  const episode = runner.createEpisode();
  episode.reset(2, {}, DEFAULT_TASK, {controlMode: 'policy'});
  assert.equal(episode.step().info.reason, 'stale_policy_action');
  episode.reset(2, {}, DEFAULT_TASK, {controlMode: 'policy'});
  episode.applyAction({type: 'policy', values: [1, 0, 0, 0]});
  for (let tick = 0; tick < 10; tick++) episode.step();
  assert.equal(episode.step().info.reason, 'stale_policy_action');
  assert.equal(episode.sim.isRunning, false);
  for (const values of [[NaN, 0, 0, 0], [2, 0, 0, 0], [1, 0], [1, 0, 0, Infinity]]) {
    episode.reset(2, {}, DEFAULT_TASK, {controlMode: 'policy'});
    assert.throws(() => episode.applyAction({type: 'policy', values}), /four finite/);
    assert.equal(episode.sim.isRunning, false);
    assert.equal(episode.step().info.reason, 'stopped');
  }
  episode.reset(2, {}, DEFAULT_TASK, {controlMode: 'policy'});
  episode.policyStep([1, 0, 0, 0]);
  episode.step();
  assert.throws(() => episode.applyAction({type: 'policy', values: [1, 0, 0, 0]}), /cadence/);
  assert.equal(episode.step().info.reason, 'stopped');
  assert.throws(() => episode.reset(2, {}, DEFAULT_TASK, {controlMode: 'policy', policyIntervalTicks: 0}));
  assert.throws(() => episode.reset(2, {}, DEFAULT_TASK, {policyIdentity: {kind: 'learned', id: 'fake'}}), /checkpoint hash/);
});

test('policyStep sums true rewards and distinguishes short time limits from terminal success', () => {
  const task = {...DEFAULT_TASK, deadlineSeconds: 0.03};
  const episode = runner.createEpisode();
  episode.reset(8, {}, task, {controlMode: 'policy'});
  const result = episode.policyStep([1, 0, 0, 0]);
  assert.equal(result.truncated, true);
  assert.equal(result.terminated, false);
  assert.equal(result.info.physicsTicks, 3);
  assert.equal(result.info.elapsedSeconds, 0.03);
  assert.equal(result.reward, episode.totalReward);
  assert.equal(result.reward, Object.values(result.rewardComponents).reduce((sum, value) => sum + value, 0));
  assert.throws(() => episode.policyStep([1, 0, 0, 0]), /ended/);
});

test('sensor policy actions produce exact browser-realm/headless original-engine parity', () => {
  const realm = vm.createContext({console});
  for (const filename of ['robot-config.js', 'control-runtime.js', 'override-geometry.js', 'override-dynamics.js',
    'override-autonomy.js', 'override.js', 'engine.js', 'policy.js', 'motion.js']) {
    vm.runInContext(fs.readFileSync(path.join(root, 'simulator', filename), 'utf8'), realm);
  }
  const compiled = new WebAssembly.Module(fs.readFileSync(path.join(root, 'simulator/control.wasm')));
  const browser = new realm.NationalsMotion.MotionEpisode(() => realm.ProductionControl.fromModule(compiled));
  const headless = runner.createEpisode();
  const configuration = {massKg: [5, 8], muLong: [0.5, 0.95]};
  const task = {...DEFAULT_TASK, deadlineSeconds: 0.5};
  for (const episode of [browser, headless]) episode.reset(999, configuration, task, {controlMode: 'policy'});
  for (let decision = 0; decision < 10; decision++) {
    const action = [decision % 2 ? -1 : 1, 0.5, -0.2, 0.3];
    assert.deepEqual(structuredClone(browser.policyStep(action)), headless.policyStep(action));
  }
  assert.deepEqual(structuredClone(browser.report()), headless.report());
  assert.notEqual(browser.sim.productionControl.e.memory.buffer, headless.sim.productionControl.e.memory.buffer);
});

test('bounded random policy can finish an episode and reset without leaking state', () => {
  let seed = 71;
  const report = runPolicy({task: {...DEFAULT_TASK, deadlineSeconds: 0.3}, act: () => Array.from({length: 4}, () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296 * 2 - 1;
  })});
  assert.equal(report.truncated, true);
  assert.equal(report.ticks, 30);
  assert.deepEqual(replay(report), report);
  const episode = runner.createEpisode();
  episode.reset(1, {}, DEFAULT_TASK, {controlMode: 'policy'});
  episode.policyStep([1, 1, 1, 1]);
  episode.reset(1, {}, DEFAULT_TASK, {controlMode: 'policy'});
  assert.deepEqual(episode.referenceAdapter.offset, [0, 0, 0]);
  assert.equal(episode.actions.length, 0);
  assert.equal(episode.transitions.length, 0);
  assert.equal(episode.lastPolicyTick, null);
});

test('policy interface does not change existing scripted baseline behavior', () => {
  const report = baseline();
  assert.equal(report.reason, 'success');
  assert.equal(report.options.policyIdentity.kind, 'scripted');
  assert.equal(report.options.controlMode, 'scripted');
  assert.equal(report.transitions.length, 0);
  assert.deepEqual(replay(report), report);
  assert(Control);
});
