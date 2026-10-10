(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NationalsPolicy = factory();
})(globalThis, function() {
  'use strict';

  const CONTRACT_VERSION = 1;
  const ACTION_SIZE = 4;
  const OBSERVATION_SIZE = 34;
  const DEFAULT_ACTION = Object.freeze([1, 0, 0, 0]);
  const LIMITS = Object.freeze({minimumPace: 0.25, maximumPace: 1, paceRate: 0.5,
    translationOffsetMeters: 0.08, headingOffsetRadians: 0.1,
    offsetSpeedMetersPerSecond: 0.08, offsetAcceleration: 0.3,
    offsetYawRate: 0.25, offsetYawAcceleration: 1, fieldMeters: 60 * 0.0254});
  const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
  const copy = value => JSON.parse(JSON.stringify(value));

  function normalizeAction(values) {
    if (!Array.isArray(values) || values.length !== ACTION_SIZE ||
      Array.from(values).some(value => !Number.isFinite(value) || value < -1 || value > 1)) {
      throw Error('Policy action must contain four finite values in [-1, 1]');
    }
    return [...values];
  }

  function options(requested = {}) {
    const allowed = ['controlMode', 'policyIntervalTicks', 'actionTimeoutTicks', 'recordTransitions', 'policyIdentity'];
    if (!requested || typeof requested !== 'object' || Array.isArray(requested) ||
      Object.keys(requested).some(key => !allowed.includes(key))) throw Error('Unsupported episode options');
    const controlMode = requested.controlMode ?? 'scripted';
    if (!['scripted', 'policy'].includes(controlMode)) throw Error('Unknown controlMode');
    const policyIntervalTicks = requested.policyIntervalTicks ?? 5;
    const actionTimeoutTicks = requested.actionTimeoutTicks ?? policyIntervalTicks * 2;
    if (!Number.isInteger(policyIntervalTicks) || policyIntervalTicks < 1 || policyIntervalTicks > 20) throw Error('policyIntervalTicks must be 1..20');
    if (!Number.isInteger(actionTimeoutTicks) || actionTimeoutTicks < policyIntervalTicks || actionTimeoutTicks > 40) throw Error('actionTimeoutTicks must be between policyIntervalTicks and 40');
    const recordTransitions = requested.recordTransitions ?? controlMode === 'policy';
    if (typeof recordTransitions !== 'boolean') throw Error('recordTransitions must be boolean');
    const policyIdentity = requested.policyIdentity ?? {kind: 'untrained', id: 'external-action-stream'};
    if (!policyIdentity || typeof policyIdentity !== 'object' || Array.isArray(policyIdentity) ||
      Object.keys(policyIdentity).some(key => !['kind', 'id', 'sha256'].includes(key)) ||
      !['untrained', 'scripted', 'learned'].includes(policyIdentity.kind) ||
      typeof policyIdentity.id !== 'string' || !/^[a-zA-Z0-9_.-]{1,96}$/.test(policyIdentity.id) ||
      (policyIdentity.sha256 !== undefined && !/^[a-f0-9]{64}$/.test(policyIdentity.sha256)) ||
      (policyIdentity.kind === 'learned' && !policyIdentity.sha256)) throw Error('Invalid policyIdentity; learned policies require a checkpoint hash');
    return {controlMode, policyIntervalTicks, actionTimeoutTicks, recordTransitions, policyIdentity: copy(policyIdentity)};
  }

  class ReferenceAdapter {
    constructor(control, start, goal) {
      this.control = control;
      this.start = [...start];
      this.goal = [...goal];
      this.duration = control.duration(start, goal);
      if (!Number.isFinite(this.duration) || this.duration < 0) throw Error('Invalid policy reference duration');
      this.time = 0;
      this.pace = 1;
      this.offset = [0, 0, 0];
      this.offsetVelocity = [0, 0, 0];
      this.reference = [...start, 0, 0, 0];
    }

    advance(values, dt) {
      const action = normalizeAction(values);
      const requestedPace = LIMITS.minimumPace + (action[0] + 1) * (LIMITS.maximumPace - LIMITS.minimumPace) / 2;
      this.pace += clamp(requestedPace - this.pace, -LIMITS.paceRate * dt, LIMITS.paceRate * dt);
      const base = this.control.reference(this.start, this.goal, this.time, this.duration);
      const heading = base[2];
      const right = action[1] * LIMITS.translationOffsetMeters;
      const forward = action[2] * LIMITS.translationOffsetMeters;
      const target = [right * Math.cos(heading) + forward * Math.sin(heading),
        -right * Math.sin(heading) + forward * Math.cos(heading), action[3] * LIMITS.headingOffsetRadians];
      for (let axis = 0; axis < 3; axis++) {
        const accelerationLimit = axis === 2 ? LIMITS.offsetYawAcceleration : LIMITS.offsetAcceleration;
        const speedLimit = axis === 2 ? LIMITS.offsetYawRate : LIMITS.offsetSpeedMetersPerSecond;
        const displacementLimit = axis === 2 ? LIMITS.headingOffsetRadians : Math.SQRT2 * LIMITS.translationOffsetMeters;
        const acceleration = clamp(16 * (target[axis] - this.offset[axis]) - 8 * this.offsetVelocity[axis], -accelerationLimit, accelerationLimit);
        this.offsetVelocity[axis] = clamp(this.offsetVelocity[axis] + acceleration * dt, -speedLimit, speedLimit);
        const next = this.offset[axis] + this.offsetVelocity[axis] * dt;
        this.offset[axis] = clamp(next, -displacementLimit, displacementLimit);
        if (next !== this.offset[axis]) this.offsetVelocity[axis] = 0;
      }
      this.reference = [base[0] + this.offset[0], base[1] + this.offset[1], base[2] + this.offset[2],
        base[3] * this.pace + this.offsetVelocity[0], base[4] * this.pace + this.offsetVelocity[1], base[5] * this.pace + this.offsetVelocity[2]];
      for (let axis = 0; axis < 2; axis++) {
        const position = this.reference[axis];
        this.reference[axis] = clamp(position, -LIMITS.fieldMeters, LIMITS.fieldMeters);
        if (position !== this.reference[axis]) this.reference[axis + 3] = 0;
      }
      this.time = Math.min(this.duration, this.time + this.pace * dt);
      return [...this.reference];
    }

    observe(actionAgeTicks) {
      return {reference: [...this.reference], referenceSeconds: this.time, pace: this.pace,
        offset: [...this.offset], offsetVelocity: [...this.offsetVelocity], actionAgeTicks};
    }
  }

  function vector(observation) {
    if (!observation.policyState) throw Error('A policy-mode observation is required');
    const previous = observation.previousAction?.type === 'policy' ? observation.previousAction.values : DEFAULT_ACTION;
    const state = observation.policyState;
    const groups = [[observation.pose, 3], [observation.velocity, 3], [observation.wheelRimSpeeds, 4],
      [observation.targetError, 4], [previous, 4], [state.reference, 6], [state.offset, 3], [state.offsetVelocity, 3]];
    if (groups.some(([values, width]) => !Array.isArray(values) || values.length !== width)) throw Error('Invalid fixed-length sensor policy observation');
    const result = [...observation.pose, ...observation.velocity, ...observation.wheelRimSpeeds,
      ...observation.targetError, ...previous, observation.remainingSeconds, ...state.reference,
      state.referenceSeconds, state.pace, ...state.offset, ...state.offsetVelocity, state.actionAgeTicks];
    if (result.length !== OBSERVATION_SIZE || !result.every(Number.isFinite)) throw Error('Invalid fixed-length sensor policy observation');
    return result;
  }

  return {CONTRACT_VERSION, ACTION_SIZE, OBSERVATION_SIZE, DEFAULT_ACTION, LIMITS, normalizeAction, options, ReferenceAdapter, vector};
});
