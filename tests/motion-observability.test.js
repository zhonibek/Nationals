'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {loadHeadless} = require('../simulator/headless');
const {DEFAULT_TASK, PARAMETERS} = require('../simulator/motion');
const Policy = require('../simulator/policy');
const runner = loadHeadless();

function sensorObservation() {
  return {
    pose: [0.11, -0.12, 0.13],
    velocity: [0.21, -0.22, 0.23],
    wheelRimSpeeds: [0.31, -0.32, 0.33, -0.34],
    targetError: [0.41, -0.42, Math.sin(0.43), Math.cos(0.43)],
    previousAction: {type: 'policy', values: [0.51, -0.52, 0.53, -0.54]},
    remainingSeconds: 6.1,
    policyState: {
      reference: [0.71, -0.72, 0.73, -0.74, 0.75, -0.76],
      referenceSeconds: 0.81, pace: 0.82,
      offset: [0.01, -0.02, 0.03],
      offsetVelocity: [0.04, -0.05, 0.06], actionAgeTicks: 7
    }
  };
}

function vectorArrays(observation) {
  return [
    ['pose', observation.pose], ['velocity', observation.velocity],
    ['wheelRimSpeeds', observation.wheelRimSpeeds], ['targetError', observation.targetError],
    ['previousAction.values', observation.previousAction.values],
    ['policyState.reference', observation.policyState.reference],
    ['policyState.offset', observation.policyState.offset],
    ['policyState.offsetVelocity', observation.policyState.offsetVelocity]
  ];
}

function vectorSlots(observation) {
  const slots = [];
  for (const [name, values] of vectorArrays(observation)) {
    for (let index = 0; index < values.length; index++) {
      slots.push({name: `${name}[${index}]`, container: values, key: index});
    }
  }
  for (const [key, container] of [
    ['remainingSeconds', observation], ['referenceSeconds', observation.policyState],
    ['pace', observation.policyState], ['actionAgeTicks', observation.policyState]
  ]) slots.push({name: key, container, key});
  return slots;
}

test('policy vector preserves the exact 34-slot sensor, target, action and reference order', () => {
  const expected = [
    0.11, -0.12, 0.13,
    0.21, -0.22, 0.23,
    0.31, -0.32, 0.33, -0.34,
    0.41, -0.42, Math.sin(0.43), Math.cos(0.43),
    0.51, -0.52, 0.53, -0.54,
    6.1,
    0.71, -0.72, 0.73, -0.74, 0.75, -0.76,
    0.81, 0.82,
    0.01, -0.02, 0.03,
    0.04, -0.05, 0.06,
    7
  ];
  assert.equal(Policy.OBSERVATION_SIZE, 34);
  assert.deepEqual(Policy.vector(sensorObservation()), expected);
});

test('absent and non-policy previous commands use only the nominal previous-action slots', () => {
  const expected = Policy.vector(sensorObservation());
  expected.splice(14, 4, 1, 0, 0, 0);
  for (const previousAction of [undefined, null, {type: 'stop', values: [NaN]},
    {type: 'pose', xIn: 12, yIn: 24, headingDeg: 90, values: [Infinity]}]) {
    const observation = sensorObservation();
    observation.previousAction = previousAction;
    assert.deepEqual(Policy.vector(observation), expected);
  }
});

test('policy encoding never reads evaluator traces, reward metadata or hidden configuration', () => {
  const observation = sensorObservation();
  const expected = Policy.vector(observation);
  for (const key of ['evaluator', 'info', 'metrics', 'configuration', 'seed', 'reward',
    'rewardComponents', 'terminated', 'truncated']) {
    Object.defineProperty(observation, key, {enumerable: true, get() {
      throw Error(`Evaluator-only field was read: ${key}`);
    }});
  }
  assert.deepEqual(Policy.vector(observation), expected);
});

test('fixed sensor readings isolate observations from all evaluator truth and supported plant parameters', () => {
  const episode = runner.createEpisode();
  episode.reset(13, {}, DEFAULT_TASK, {controlMode: 'policy', recordTransitions: false});
  episode.sim.odom = {x: -2, y: 3, theta: 45};
  episode.sim.odomVelocity = [0.17, -0.28, 0.39];
  episode.sim.wheelOmega = [3, -5, 7, -11];
  const observation = episode.observe();
  const expected = Policy.vector(observation);
  const before = episode.metrics();
  const truth = {
    x: 38, y: -27, theta: -135, Vx: 2.1, Vy: -3.2, w: 4.3,
    vx: -5.4, vy: 6.5, v: 7.6, Ax: 8.7, Ay: -9.8, ax: 10.9, ay: -11.1, alpha: 12.2,
    motorVolts: [1, -2, 3, -4], motorCurrents: [0.1, 0.2, 0.3, 0.4],
    motorTorques: [0.5, 0.6, 0.7, 0.8], wheelTractionForces: [9, 10, 11, 12],
    wheelSlips: [0.9, -1, 1.1, -1.2], batteryVoltage: 9.5, contactSeconds: 1.3,
    ekfPose: {x: 20, y: -30, theta: 150}
  };
  Object.assign(episode.sim, truth);
  assert.notDeepEqual(episode.metrics(), before);
  assert.deepEqual(episode.observe(), observation);
  assert.deepEqual(Policy.vector(episode.observe()), expected);
  for (const [key, limits] of Object.entries(PARAMETERS)) {
    for (const value of limits) {
      episode.sim[key] = value;
      assert.deepEqual(Policy.vector(episode.observe()), expected, `${key}=${value}`);
    }
  }
  for (const key of [...Object.keys(truth), ...Object.keys(PARAMETERS)]) {
    Object.defineProperty(episode.sim, key, {get() {
      throw Error(`Hidden simulator field was read: ${key}`);
    }});
  }
  Object.defineProperty(episode, 'configuration', {get() {
    throw Error('Resolved configuration was read');
  }});
  episode.metrics = () => { throw Error('Evaluator metrics were read'); };
  assert.deepEqual(episode.observe(), observation);
  assert.deepEqual(Policy.vector(episode.observe()), expected);
  assert.equal(episode.ticks, 0);
});

test('encoder pose and FL, BL, FR, BR wheel inputs retain SI units and body-relative target signs', () => {
  const episode = runner.createEpisode();
  episode.reset(14, {}, {...DEFAULT_TASK, goal: {xIn: 10, yIn: 20, headingDeg: 90}},
    {controlMode: 'policy', recordTransitions: false});
  episode.sim.odomVelocity = [0.17, -0.28, 0.39];
  episode.sim.wheelOmega = [3, -5, 7, -11];
  const radius = episode.sim.wheelRadiusM;
  const right = 12 * 0.0254;
  const forward = 17 * 0.0254;
  const cases = [
    {heading: 0, goalHeading: 90, target: [right, forward, 1, 0]},
    {heading: 90, goalHeading: -90, target: [-forward, right, 0, -1]},
    {heading: -90, goalHeading: 0, target: [forward, -right, 1, 0]},
    {heading: 180, goalHeading: 0, target: [-right, -forward, 0, -1]}
  ];
  for (const {heading, goalHeading, target} of cases) {
    episode.sim.odom = {x: -2, y: 3, theta: heading};
    episode.task.goal.headingDeg = goalHeading;
    const vector = Policy.vector(episode.observe());
    assert.deepEqual(vector.slice(0, 3), [-2 * 0.0254, 3 * 0.0254, heading * (Math.PI / 180)]);
    assert.deepEqual(vector.slice(3, 6), [0.17, -0.28, 0.39]);
    assert.deepEqual(vector.slice(6, 10), [3 * radius, -5 * radius, 7 * radius, -11 * radius]);
    for (let index = 0; index < target.length; index++) {
      assert(Math.abs(vector[10 + index] - target[index]) < 1e-12,
        `heading=${heading}, targetError[${index}]`);
    }
  }
  for (const direction of [-1, 1]) {
    episode.sim.odom.theta = direction * 179;
    episode.task.goal.headingDeg = -direction * 179;
    const vector = Policy.vector(episode.observe());
    const wrappedError = direction * 2 * Math.PI / 180;
    assert(Math.abs(vector[12] - Math.sin(wrappedError)) < 1e-12);
    assert(Math.abs(vector[13] - Math.cos(wrappedError)) < 1e-12);
  }
  assert.equal(episode.ticks, 0);
});

test('every policy input slot rejects nonfinite values and nonnumeric substitutes without coercion', () => {
  const observation = sensorObservation();
  const expected = Policy.vector(observation);
  const slots = vectorSlots(observation);
  assert.equal(slots.length, 34);
  const invalid = [
    ['NaN', NaN], ['Infinity', Infinity], ['-Infinity', -Infinity], ['undefined', undefined],
    ['null', null], ['numeric string', '0'], ['boolean', true], ['object', {}]
  ];
  for (const {name, container, key} of slots) {
    const original = container[key];
    for (const [label, value] of invalid) {
      container[key] = value;
      assert.throws(() => Policy.vector(observation), /Invalid fixed-length sensor policy observation/,
        `${name}: ${label}`);
    }
    container[key] = original;
  }
  assert.deepEqual(Policy.vector(observation), expected);
});

test('sparse sensor, target, action and reference arrays never produce a valid policy vector', () => {
  const observation = sensorObservation();
  const expected = Policy.vector(observation);
  for (const [name, values] of vectorArrays(observation)) {
    for (let index = 0; index < values.length; index++) {
      const original = values[index];
      delete values[index];
      assert.throws(() => Policy.vector(observation), /Invalid fixed-length sensor policy observation/,
        `${name}[${index}] is missing`);
      values[index] = original;
    }
  }
  assert.deepEqual(Policy.vector(observation), expected);
});

test('shortened and extended input groups fail rather than silently padding or truncating', () => {
  const observation = sensorObservation();
  const expected = Policy.vector(observation);
  for (const [name, values] of vectorArrays(observation)) {
    const original = values.pop();
    assert.throws(() => Policy.vector(observation), /Invalid fixed-length sensor policy observation/,
      `${name} is too short`);
    values.push(original, 0);
    assert.throws(() => Policy.vector(observation), /Invalid fixed-length sensor policy observation/,
      `${name} is too long`);
    values.pop();
  }
  assert.deepEqual(Policy.vector(observation), expected);
  observation.pose.push(0);
  observation.velocity.pop();
  assert.throws(() => Policy.vector(observation), /Invalid fixed-length sensor policy observation/);
});
