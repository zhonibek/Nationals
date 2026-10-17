'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {VexRobotSimulator} = require('../simulator/engine');

function fixture() {
  const calls = [];
  const sim = {simTime: 0.01, actionStartTime: 0, odom: {x: 0, y: 0, theta: 0},
    odomVelocity: [0, 0, 0], wheelOmega: [0, 0, 0, 0], wheelRadiusM: 0.03, routineQueue: [],
    productionControl: {reset() { calls.push('reset'); }, step(reference, feedback, dt, mode) {
      calls.push({reference: [...reference], feedback: [...feedback], dt, mode});
      return {valid: true, volts: [1, 2, 3, 4]};
    }}};
  return {sim, calls};
}

test('residual reference explicitly reaches the bounded GVF native mode and remains persistent', () => {
  const {sim, calls} = fixture();
  const action = {type: 'gvfResidualReference', referenceSI: [0, 0, 0, 0, 0, 0], timeout: 1000};
  VexRobotSimulator.prototype.initAction.call(sim, action);
  assert.equal(action.duration, 0);
  assert.equal(sim.lastMotionResult, 'Running');
  assert.equal(VexRobotSimulator.prototype.executeAction.call(sim, action, 0.01), false);
  assert.deepEqual(calls, ['reset', {reference: action.referenceSI, feedback: Array(10).fill(0), dt: 0.01, mode: 'gvf'}]);
  assert.deepEqual(sim.commandedWheelVoltages, [1, 2, 3, 4]);
  assert.equal(sim.lastMotionResult, 'Running');
});

test('legacy learned references retain the ordinary native control path without new GVF limits', () => {
  const {sim, calls} = fixture();
  const action = {type: 'policyReference', referenceSI: [0, 0, 0, 0.45, 0, 2], timeout: 1000};
  VexRobotSimulator.prototype.initAction.call(sim, action);
  assert.equal(VexRobotSimulator.prototype.executeAction.call(sim, action, 0.01), false);
  assert.equal(calls[1].mode, 'scripted');
  assert.deepEqual(calls[1].reference, action.referenceSI);
});

test('invalid or excessive residual requests neutralize without invoking the controller', () => {
  const references = [new Array(6), [0, 0, 0, 0, 0], [0, 0, NaN, 0, 0, 0], [1.525, 0, 0, 0, 0, 0],
    [0, -1.525, 0, 0, 0, 0], [0, 0, 0, 0.36, 0, 0], [0, 0, 0, 0.3, 0.3, 0], [0, 0, 0, 0, 0, -1.51]];
  for (const referenceSI of references) {
    const {sim, calls} = fixture();
    const action = {type: 'gvfResidualReference', referenceSI, duration: 0, timeout: 1000};
    assert.equal(VexRobotSimulator.prototype.executeAction.call(sim, action, 0.01), true);
    assert.equal(sim.lastMotionResult, 'InvalidInput');
    assert.deepEqual(sim.commandedWheelVoltages, [0, 0, 0, 0]);
    assert.deepEqual(calls, []);
  }
});

test('residual reference retains existing timeout and controller sensor-fault behavior', () => {
  const action = {type: 'gvfResidualReference', referenceSI: [0, 0, 0, 0, 0, 0], duration: 0, timeout: 10};
  const timed = fixture();
  assert.equal(VexRobotSimulator.prototype.executeAction.call(timed.sim, action, 0.01), true);
  assert.equal(timed.sim.lastMotionResult, 'TimedOut');
  assert.deepEqual(timed.calls, []);
  const failed = fixture();
  failed.sim.productionControl.step = () => ({valid: false, volts: [10, 10, 10, 10]});
  assert.equal(VexRobotSimulator.prototype.executeAction.call(failed.sim, {...action, timeout: 1000}, 0.01), true);
  assert.equal(failed.sim.lastMotionResult, 'SensorFault');
  assert.deepEqual(failed.sim.commandedWheelVoltages, [0, 0, 0, 0]);
});
