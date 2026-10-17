'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Control = require('../simulator/control-runtime');
const Legacy = require('../simulator/policy');
const Policy = require('../simulator/gvf-policy');
const {GvfResidualAdapter, LIMITS, ZERO_ACTION} = Policy;
const sensors = (heading = 0) => [0, 0, heading, 0, 0, 0, 0, 0, 0, 0];
const near = (actual, expected, tolerance = 1e-12) => assert(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

function nativeMock(reference = [0, 0.2, 0, 0, 0.2, 0]) {
  const buffer = new Float64Array(64), calls = {initializations: [], feedback: [], resets: 0};
  const fixture = {reference: [...reference], parameter: 0.4, remainingMeters: 0.3, crossTrackMeters: 0.02};
  let valid = true;
  const control = new Control({exports: {
    memory: {buffer: buffer.buffer}, control_buffer: () => 0, control_reset: () => { calls.resets++; },
    control_gvf: (...values) => { calls.initializations.push(values); return valid ? 1 : 0; },
    control_gvf_reference: () => {
      calls.feedback.push(Array.from(buffer.slice(6, 16)));
      if (!valid) return 0;
      buffer.set(fixture.reference);
      buffer.set([fixture.parameter, fixture.remainingMeters, fixture.crossTrackMeters, 0], 18);
      return 1;
    }
  }}, null);
  for (const name of ['duration', 'reference', 'bezierReference', 'configure', 'step']) {
    Object.defineProperty(control, name, {get() { throw Error(`Unexpected native ${name}`); }});
  }
  return {control, calls, fixture, setValid(value) { valid = value; }};
}

test('separate UMD contract has explicit identity, frozen zero action and the existing engineering ceilings', () => {
  assert.equal(Policy.CONTRACT_NAME, 'gvf-residual');
  assert.equal(Policy.CONTRACT_VERSION, 2);
  assert.equal(Policy.ACTION_SIZE, 3);
  assert.equal(Policy.OBSERVATION_SIZE, 34);
  assert.equal(Policy.OBSERVATION_FIELDS.reduce((sum, field) => sum + field[1], 0), 34);
  assert.deepEqual(ZERO_ACTION, [0, 0, 0]);
  assert(Object.isFrozen(ZERO_ACTION));
  assert.equal(Policy.DEFAULT_ACTION, ZERO_ACTION);
  for (const name of Object.keys(Legacy.LIMITS).filter(name => !/pace/i.test(name))) assert.equal(LIMITS[name], Legacy.LIMITS[name]);
  assert.equal(Legacy.CONTRACT_VERSION, 1);
  assert.equal(Legacy.ACTION_SIZE, 4);
  assert.match(Policy.CONTRACT.notes.join(' '), /request ceilings are not physical/);
  const realm = vm.createContext({Date: {now() { throw Error('No clock'); }}});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../simulator/gvf-policy.js'), 'utf8'), realm);
  assert.equal(realm.NationalsGvfPolicy.CONTRACT_NAME, Policy.CONTRACT_NAME);
  const mock = nativeMock();
  const adapter = new realm.NationalsGvfPolicy.GvfResidualAdapter(mock.control, sensors(), [0, 0.6, 0]);
  assert.deepEqual(Array.from(adapter.advance([0, 0, 0], sensors(), 0.01, 1)), mock.fixture.reference);
  assert.deepEqual(Array.from(realm.NationalsGvfPolicy.vector(adapter.observe(1))), Policy.vector(structuredClone(adapter.observe(1))));
});

test('actions reject malformed, sparse, nonfinite, legacy and out-of-range values without coercion', () => {
  const invalid = [null, undefined, {}, '0,0,0', new Float64Array(3), [], [0, 0], [1, 0, 0, 0],
    Array(3), [0, , 0], [NaN, 0, 0], [0, Infinity, 0], [0, 0, -Infinity], ['0', 0, 0], [false, 0, 0],
    [1.000001, 0, 0], [0, -1.000001, 0]];
  for (const value of invalid) assert.throws(() => Policy.normalizeAction(value), /action/);
  const action = [-1, 1, 0.5], normalized = Policy.normalizeAction(action);
  assert.deepEqual(normalized, action);
  normalized[0] = 0;
  assert.equal(action[0], -1);
});

test('native initialization uses measured SI start, inches for the cubic and raw radian headings only once', () => {
  const mock = nativeMock(), feedback = [0.254, -0.254, 2, 0.1, 0.2, 0.3, 1, 2, 3, 4];
  const adapter = new GvfResidualAdapter(mock.control, feedback, [0.508, 0.508, -2]);
  assert.equal(mock.calls.initializations.length, 1);
  const initial = mock.calls.initializations[0];
  for (let index = 0; index < 4; index++) {
    near(initial[index * 2], 10 + 10 * index / 3);
    near(initial[index * 2 + 1], -10 + 30 * index / 3);
  }
  assert.deepEqual(initial.slice(8), [2, -2]);
  assert.deepEqual(mock.calls.feedback[0], feedback);
  adapter.advance(ZERO_ACTION, feedback, 0.01, 1);
  adapter.advance(ZERO_ACTION, feedback, 0.01, 2);
  assert.equal(mock.calls.initializations.length, 1);
  assert.equal(mock.calls.resets, 1);
});

test('zero residual preserves the exact unmodified base, including headings crossing pi, with no clock pace', () => {
  const mock = nativeMock([0.1, 0.2, 5 * Math.PI / 4, 0.1, 0.2, 0.3]);
  const feedback = sensors(3 * Math.PI / 4);
  const adapter = new GvfResidualAdapter(mock.control, feedback, [0, 0.6, -3 * Math.PI / 4]);
  assert.deepEqual(adapter.reference, mock.fixture.reference);
  for (let tick = 1; tick <= 6; tick++) {
    assert.deepEqual(adapter.advance(ZERO_ACTION, feedback, tick % 2 ? 0.1 : 0.01, tick), mock.fixture.reference);
    const observation = adapter.observe(tick % 2 ? 100 : 0, tick);
    assert.deepEqual(observation.baseReference, mock.fixture.reference);
    assert.deepEqual(observation.appliedFieldOffset, [0, 0, 0]);
    assert.deepEqual(observation.offsetVelocity, [0, 0, 0]);
    assert(!('pace' in observation));
    assert(!('referenceSeconds' in observation));
  }
  mock.fixture.reference[1] = 0.4;
  mock.fixture.parameter = 0.7;
  const changed = [...feedback]; changed[1] = 0.4;
  assert.deepEqual(adapter.advance(ZERO_ACTION, changed, 0.01, 7), mock.fixture.reference);
  assert.deepEqual(mock.calls.feedback.at(-1), changed);
  near(adapter.observe(0).projection, 0.7);
});

test('observation reads and returned arrays are pure detached snapshots, never native samples or filter steps', () => {
  const mock = nativeMock(), feedback = sensors(), goal = [0, 0.6, 0];
  const adapter = new GvfResidualAdapter(mock.control, feedback, goal);
  const action = [1, -1, 1];
  const reference = adapter.advance(action, feedback, 0.01, 1);
  const expected = adapter.observe(2, 1), calls = mock.calls.feedback.length;
  feedback[0] = goal[0] = action[0] = reference[0] = 999;
  for (let read = 0; read < 8; read++) {
    const observed = adapter.observe(2, 1);
    assert.deepEqual(observed, expected);
    const packed = Policy.vector(observed);
    observed.pose[0] = observed.baseReference[0] = observed.appliedFieldOffset[0] = 999;
    observed.offsetVelocity[0] = observed.previousAction[0] = packed[0] = 999;
  }
  assert.equal(mock.calls.feedback.length, calls);
  assert.equal(adapter.tick, 1);
  assert.deepEqual(adapter.observe(2, 1), expected);
});

test('post-physics sensor refresh keeps observations current without advancing the filter or control', () => {
  const mock = nativeMock(), adapter = new GvfResidualAdapter(mock.control, sensors(), [0, 0.6, 0]);
  const commanded = adapter.advance([1, 0, 0], sensors(), 0.01, 1);
  const before = adapter.observe(2, 1), resets = mock.calls.resets;
  const fresh = [0.02, 0.03, 0.1, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1];
  mock.fixture.parameter = 0.5;
  mock.fixture.reference[0] = 0.02;
  adapter.refreshSensors(fresh);
  const after = adapter.observe(1.99, 1);
  assert.deepEqual(after.pose, fresh.slice(0, 3));
  assert.deepEqual(after.velocity, fresh.slice(3, 6));
  assert.deepEqual(after.wheelRimSpeeds, fresh.slice(6, 10));
  assert.deepEqual(after.appliedFieldOffset, before.appliedFieldOffset);
  assert.deepEqual(after.offsetVelocity, before.offsetVelocity);
  assert.deepEqual(after.previousAction, before.previousAction);
  assert.deepEqual(after.baseReference, mock.fixture.reference);
  assert.equal(after.projection, 0.5);
  assert.deepEqual(adapter.reference, commanded);
  assert.equal(adapter.tick, 1);
  assert.equal(mock.calls.resets, resets);
  assert.equal(mock.calls.initializations.length, 1);
  const calls = mock.calls.feedback.length;
  assert.deepEqual(adapter.observe(1.99, 1), after);
  assert.equal(mock.calls.feedback.length, calls);
  mock.setValid(false);
  assert.throws(() => adapter.refreshSensors(sensors()), /unavailable/);
  assert.deepEqual(adapter.observe(1.99, 1), after);
});

test('sensor schema packs the distinct 34 features in declared order using measured body goal error', () => {
  const mock = nativeMock(), feedback = [0.1, 0.2, Math.PI / 2, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
  const adapter = new GvfResidualAdapter(mock.control, feedback, [0.4, 0.6, Math.PI]);
  adapter.advance([0.25, -0.5, 0.75], feedback, 0.01, 1);
  const observed = adapter.observe(4, 1), packed = Policy.vector(observed);
  const expected = [...feedback, -0.4, 0.3, 1, 0, 0.25, -0.5, 0.75, 4,
    ...mock.fixture.reference, 0.4, 0.3, 0.02, ...observed.appliedFieldOffset, ...observed.offsetVelocity, 1];
  assert.equal(packed.length, 34);
  packed.forEach((value, index) => near(value, expected[index]));
  assert.notDeepEqual(packed.slice(18, 24), adapter.reference);
});

test('length cannot identify the contract and truth/configuration or malformed schema fields are rejected', () => {
  const mock = nativeMock(), adapter = new GvfResidualAdapter(mock.control, sensors(), [0, 0.6, 0]);
  const observation = adapter.observe(1);
  for (const value of [Array(34).fill(0), {}, {...observation, contractVersion: 1},
    {...observation, contractName: 'policy'}, {...observation, observationSchema: 'legacy34'}]) {
    assert.throws(() => Policy.vector(value), /Explicit GVF residual/);
  }
  for (const name of ['truth', 'configuration', 'seed', 'metrics', 'policyState', 'pace']) {
    assert.throws(() => Policy.vector({...observation, [name]: {massKg: 9}}), /declared sensor/);
  }
  for (const changes of [{pose: Array(3)}, {velocity: [0, 0, NaN]}, {wheelRimSpeeds: [0]},
    {targetError: [0, 0, 0]}, {previousAction: [1, 0, 0, 0]}, {remainingSeconds: -1},
    {baseReference: [0]}, {projection: 2}, {remainingArcMeters: -1}, {crossTrackMeters: -1},
    {appliedFieldOffset: [NaN, 0, 0]}, {offsetVelocity: [0, 0]}, {actionAgeTicks: 0.5}]) {
    assert.throws(() => Policy.vector({...observation, ...changes}));
  }
  Object.defineProperty(mock.control, 'configuration', {get() { throw Error('No configuration'); }});
  const sensorsOnly = sensors();
  Object.defineProperty(sensorsOnly, 'truth', {get() { throw Error('No truth'); }});
  adapter.advance(ZERO_ACTION, sensorsOnly, 0.01, 1);
  assert.deepEqual(adapter.observe(1).pose, [0, 0, 0]);
  assert.throws(() => adapter.advance(ZERO_ACTION, {odom: {}, truth: {}}, 0.01, 2), /Sensor feedback/);
});

test('nominal field filters obey existing component rate/acceleration and body offset request ceilings', () => {
  const mock = nativeMock([0, 0, 0, 0, 0, 0]), feedback = sensors(Math.PI / 4);
  const adapter = new GvfResidualAdapter(mock.control, feedback, [0, 0, 0]);
  let previous = adapter.observe(1);
  for (let tick = 1; tick <= 180; tick++) {
    const action = tick < 90 ? [1, -1, 1] : [-1, 1, -1];
    const reference = adapter.advance(action, feedback, 0.02, tick), state = adapter.observe(1);
    for (let axis = 0; axis < 3; axis++) {
      const speed = axis === 2 ? LIMITS.offsetYawRate : LIMITS.offsetSpeedMetersPerSecond;
      const acceleration = axis === 2 ? LIMITS.offsetYawAcceleration : LIMITS.offsetAcceleration;
      assert(Math.abs(state.offsetVelocity[axis]) <= speed + 1e-12);
      assert(Math.abs(state.offsetVelocity[axis] - previous.offsetVelocity[axis]) <= acceleration * 0.02 + 1e-12);
      near(reference[axis + 3], state.offsetVelocity[axis]);
      near(state.appliedFieldOffset[axis] - previous.appliedFieldOffset[axis], state.offsetVelocity[axis] * 0.02);
    }
    const offset = state.appliedFieldOffset;
    assert(Math.abs((offset[0] - offset[1]) / Math.SQRT2) <= 0.08 + 1e-12);
    assert(Math.abs((offset[0] + offset[1]) / Math.SQRT2) <= 0.08 + 1e-12);
    assert(Math.abs(offset[2]) <= 0.1);
    previous = state;
  }
});

test('combined feedforward saturation prevents offset windup and allows immediate inward recovery', () => {
  const mock = nativeMock([0, 0, 0, 0.35, 0, 1.5]);
  const adapter = new GvfResidualAdapter(mock.control, sensors(), [0, 0, 0]);
  for (let tick = 1; tick <= 6; tick++) {
    const reference = adapter.advance([1, 0, 1], sensors(), 0.01, tick), state = adapter.observe(1);
    assert.deepEqual(reference, mock.fixture.reference);
    assert.deepEqual(state.appliedFieldOffset, [0, 0, 0]);
    assert.deepEqual(state.offsetVelocity, [0, 0, 0]);
  }
  const inward = adapter.advance([-1, 1, -1], sensors(), 0.01, 7);
  assert(inward[0] < 0 && inward[2] < 0);
  assert(Math.hypot(inward[3], inward[4]) <= 0.35 + 1e-12);
  assert(inward[5] < 1.5);
});

test('field projection back-calculates bounded applied offsets, clears windup and suppresses outward requests', () => {
  const mock = nativeMock([LIMITS.fieldMeters - 0.00001, LIMITS.fieldMeters - 0.00001, 0, 0.1, 0.1, 0]);
  const adapter = new GvfResidualAdapter(mock.control, sensors(), [0, 0, 0]);
  for (let tick = 1; tick <= 8; tick++) {
    const reference = adapter.advance([1, 1, 1], sensors(), 0.1, tick), state = adapter.observe(1);
    for (let axis = 0; axis < 2; axis++) {
      assert(Math.abs(reference[axis]) <= LIMITS.fieldMeters);
      near(state.appliedFieldOffset[axis], reference[axis] - state.baseReference[axis]);
      assert.equal(state.offsetVelocity[axis], 0);
      assert.equal(reference[axis + 3], 0);
    }
    assert(Math.abs(state.appliedFieldOffset[2]) <= 0.1);
  }
  adapter.advance([-1, -1, -1], sensors(), 0.01, 9);
  assert(adapter.observe(1).offsetVelocity.slice(0, 2).every(value => value < 0));
});

test('rotating measured body axes projects stored offsets without hidden displacement windup', () => {
  const mock = nativeMock([0, 0, 0, 0, 0, 0]);
  const adapter = new GvfResidualAdapter(mock.control, sensors(Math.PI / 4), [0, 0, 0]);
  for (let tick = 1; tick <= 60; tick++) adapter.advance([1, 1, 0], sensors(Math.PI / 4), 0.1, tick);
  assert(adapter.observe(1).appliedFieldOffset[0] > 0.08);
  adapter.advance([1, 1, 0], sensors(), 0.01, 61);
  const state = adapter.observe(1);
  assert(state.appliedFieldOffset.slice(0, 2).every(value => Math.abs(value) <= 0.08 + 1e-12));
  assert.deepEqual(state.offsetVelocity, [0, 0, 0]);
});

test('reset clears every residual/action/tick and reinitializes the native path, not the wheel controller', () => {
  const mock = nativeMock(), adapter = new GvfResidualAdapter(mock.control, sensors(), [0, 0.6, 0]);
  adapter.advance([1, 1, 1], sensors(), 0.1, 1);
  const feedback = sensors(0.4); feedback[0] = 0.2;
  assert.deepEqual(adapter.reset(feedback, [0.4, 0.6, 0.8]), mock.fixture.reference);
  const state = adapter.observe(3);
  assert.equal(adapter.tick, 0);
  assert.deepEqual(state.previousAction, ZERO_ACTION);
  assert.deepEqual(state.appliedFieldOffset, [0, 0, 0]);
  assert.deepEqual(state.offsetVelocity, [0, 0, 0]);
  assert.deepEqual(state.pose, feedback.slice(0, 3));
  assert.equal(mock.calls.initializations.length, 2);
  assert.equal(mock.calls.resets, 1);
  adapter.advance(ZERO_ACTION, feedback, 0.01, 1);
  assert.deepEqual(adapter.reference, mock.fixture.reference);
});

test('bad ticks/dt/sensors/actions do not sample or advance; bad native samples do not commit state', () => {
  const mock = nativeMock(), adapter = new GvfResidualAdapter(mock.control, sensors(), [0, 0.6, 0]);
  adapter.advance([1, 0, 0], sensors(), 0.01, 1);
  const before = adapter.observe(1), calls = mock.calls.feedback.length;
  for (const tick of [undefined, 0, 1, 3, 1.5, Infinity]) assert.throws(() => adapter.advance(ZERO_ACTION, sensors(), 0.01, tick), /once per sequential/);
  for (const dt of [0, -1, NaN, Infinity, 0.10001, '0.01']) assert.throws(() => adapter.advance(ZERO_ACTION, sensors(), dt, 2), /Physics dt/);
  assert.throws(() => adapter.advance([1, 0, 0, 0], sensors(), 0.01, 2));
  assert.throws(() => adapter.advance(ZERO_ACTION, Array(10), 0.01, 2));
  assert.equal(mock.calls.feedback.length, calls);
  for (const changes of [{parameter: NaN}, {parameter: 2}, {remainingMeters: -1}, {crossTrackMeters: Infinity},
    {reference: [0, 0, 0, 0.351, 0, 0]}, {reference: [0, 0, 0, 0, 0, -1.501]},
    {reference: [2, 0, 0, 0, 0, 0]}, {reference: [0, 0, NaN, 0, 0, 0]}]) {
    const saved = {...mock.fixture}; Object.assign(mock.fixture, changes);
    assert.throws(() => adapter.advance(ZERO_ACTION, sensors(), 0.01, 2));
    Object.assign(mock.fixture, saved);
    assert.deepEqual(adapter.observe(1), before);
  }
  mock.setValid(false);
  assert.throws(() => adapter.advance(ZERO_ACTION, sensors(), 0.01, 2), /unavailable/);
  assert.deepEqual(adapter.observe(1), before);
  for (const args of [[-1, 0], [NaN, 0], [1, -1], [1, 6001], [1, 0.5]]) assert.throws(() => adapter.observe(...args));
  assert.deepEqual(adapter.observe(1), before);
  assert.throws(() => adapter.reset(sensors(), [0, 0.6, 0]), /initialization/);
  assert.throws(() => adapter.observe(1), /Reset/);
  assert.throws(() => adapter.advance(ZERO_ACTION, sensors(), 0.01, 1), /Reset/);
  mock.setValid(true);
  adapter.reset(sensors(), [0, 0.6, 0]);
  assert.equal(adapter.tick, 0);
  assert.throws(() => new GvfResidualAdapter({}, sensors(), [0, 0, 0]), /Native gvf/);
  assert.throws(() => adapter.reset(sensors(), [2, 0, 0]), /bounds/);
});
