(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NationalsGvfPolicy = factory();
})(globalThis, function() {
  'use strict';

  const CONTRACT_VERSION = 2;
  const CONTRACT_NAME = 'gvf-residual';
  const OBSERVATION_SCHEMA = 'gvf-residual-sensor34-v2';
  const ACTION_SIZE = 3;
  const OBSERVATION_SIZE = 34;
  const ZERO_ACTION = Object.freeze([0, 0, 0]);
  const DEFAULT_ACTION = ZERO_ACTION;
  const LIMITS = Object.freeze({translationOffsetMeters: 0.08, headingOffsetRadians: 0.1,
    offsetSpeedMetersPerSecond: 0.08, offsetAcceleration: 0.3,
    offsetYawRate: 0.25, offsetYawAcceleration: 1, fieldMeters: 60 * 0.0254,
    referenceSpeedMetersPerSecond: 0.35, referenceYawRate: 1.5});
  const OBSERVATION_FIELDS = Object.freeze([
    ['pose', 3], ['velocity', 3], ['wheelRimSpeeds', 4], ['targetError', 4],
    ['previousAction', 3], ['remainingSeconds', 1], ['baseReference', 6],
    ['projection', 1], ['remainingArcMeters', 1], ['crossTrackMeters', 1],
    ['appliedFieldOffset', 3], ['offsetVelocity', 3], ['actionAgeTicks', 1]
  ].map(field => Object.freeze(field)));
  const CONTRACT = Object.freeze({contractName: CONTRACT_NAME, contractVersion: CONTRACT_VERSION,
    observationSchema: OBSERVATION_SCHEMA, actionNames: Object.freeze(['dRight', 'dForward', 'dHeading']),
    notes: Object.freeze([
      'Experimental, untrained contract; request ceilings are not physical motion or stability guarantees.',
      'SI feedback is measured [x,y,heading,vx,vy,yawRate,FL,BL,FR,BR]; wheel values are rim speeds.',
      'Native gvf initializes one collinear cubic in inches from measured start to task goal; no controller reset is performed here.',
      'Actions target +/-0.08 m on each current measured body axis and +/-0.1 rad; heading is clockwise from field +Y.',
      'The field-frame filter uses acceleration clamp(16*error-8*velocity), component speed/acceleration ceilings 0.08/0.3 and yaw 0.25/1.',
      'Combined feedforward is capped at 0.35 m/s vector magnitude and 1.5 rad/s before offset integration, without rescaling the native base.',
      'Body/field displacement projection back-calculates applied offsets and zeros saturated filter velocities; field clipping suppresses outward feedforward.',
      'Safety projections override nominal filter acceleration smoothness; offsetVelocity is filter state, not the derivative of a changing projected base.',
      'Advance exactly once per sequential physics tick; observe/vector never sample native control or advance filters.',
      'Refresh sensors after the physics update before exposing the next observation; this samples only the native field, not the controller or residual filter.',
      'Zero is a neutral target, not an instantaneous filter reset; from reset it preserves the native base, including unwrapped heading.',
      'Deadline remaining is observation context only; there is no clock pace or time-parameterized reference.',
      'Later wiring must retain native step mode gvf for the downstream combined feedback/feedforward yaw request cap; this module does not actuate motors.'
    ])});
  const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
  const add = (value, offset) => offset === 0 ? value : value + offset;

  function finiteVector(values, width, label) {
    if (!Array.isArray(values) || values.length !== width || !Array.from(values).every(Number.isFinite)) {
      throw Error(`${label} must contain ${width} finite values`);
    }
    return [...values];
  }

  function normalizeAction(values) {
    const action = finiteVector(values, ACTION_SIZE, 'GVF residual action');
    if (action.some(value => value < -1 || value > 1)) throw Error('GVF residual action must be in [-1, 1]');
    return action;
  }

  function options(requested) {
    const allowed = ['controlMode', 'policyContractVersion', 'policyContractName', 'policyIntervalTicks',
      'actionTimeoutTicks', 'recordTransitions', 'policyIdentity'];
    if (!requested || typeof requested !== 'object' || Array.isArray(requested) ||
      Object.keys(requested).some(key => !allowed.includes(key)) || requested.controlMode !== CONTRACT_NAME ||
      requested.policyContractName !== CONTRACT_NAME || requested.policyContractVersion !== CONTRACT_VERSION) {
      throw Error('Explicit GVF residual mode/name/version and supported episode options required');
    }
    const policyIntervalTicks = requested.policyIntervalTicks ?? 5, actionTimeoutTicks = requested.actionTimeoutTicks ?? 10;
    if (policyIntervalTicks !== 5 || actionTimeoutTicks !== 10) throw Error('GVF residual v2 requires five-tick cadence and ten-tick watchdog');
    const recordTransitions = requested.recordTransitions ?? true;
    if (typeof recordTransitions !== 'boolean') throw Error('recordTransitions must be boolean');
    const identity = requested.policyIdentity ?? {kind: 'untrained', id: 'external-gvf-residual-stream'};
    if (!identity || typeof identity !== 'object' || Array.isArray(identity) ||
      Object.keys(identity).some(key => !['kind', 'id', 'sha256'].includes(key)) ||
      !['untrained', 'scripted', 'learned'].includes(identity.kind) || typeof identity.id !== 'string' ||
      !/^[a-zA-Z0-9_.-]{1,96}$/.test(identity.id) ||
      (identity.sha256 !== undefined && !/^[a-f0-9]{64}$/.test(identity.sha256)) ||
      (identity.kind === 'learned' && !identity.sha256)) throw Error('GVF residual learned identity requires a valid checkpoint hash');
    return {controlMode: CONTRACT_NAME, policyContractName: CONTRACT_NAME, policyContractVersion: CONTRACT_VERSION,
      policyIntervalTicks, actionTimeoutTicks, recordTransitions, policyIdentity: {...identity}};
  }

  function boundedPose(values) {
    const pose = finiteVector(values, 3, 'GVF pose');
    if (pose.slice(0, 2).some(value => Math.abs(value) > LIMITS.fieldMeters)) throw Error('GVF pose outside field request bounds');
    return pose;
  }

  function sampleReference(control, feedback) {
    const sample = control.gvfReference([...feedback]);
    if (!sample || typeof sample !== 'object') throw Error('Native GVF reference unavailable');
    const reference = finiteVector(sample.reference, 6, 'Native GVF reference');
    boundedPose(reference.slice(0, 3));
    if (Math.hypot(reference[3], reference[4]) > LIMITS.referenceSpeedMetersPerSecond + 1e-12 ||
      Math.abs(reference[5]) > LIMITS.referenceYawRate + 1e-12 ||
      !Number.isFinite(sample.parameter) || sample.parameter < 0 || sample.parameter > 1 ||
      !Number.isFinite(sample.remainingMeters) || sample.remainingMeters < 0 ||
      !Number.isFinite(sample.crossTrackMeters) || sample.crossTrackMeters < 0) throw Error('Invalid native GVF request/diagnostics');
    return {reference, parameter: sample.parameter, remainingMeters: sample.remainingMeters, crossTrackMeters: sample.crossTrackMeters};
  }

  function limitResidualVelocity(base, velocity) {
    const speed = LIMITS.referenceSpeedMetersPerSecond;
    if (Math.hypot(base[3] + velocity[0], base[4] + velocity[1]) > speed) {
      const squared = velocity[0] ** 2 + velocity[1] ** 2;
      const along = base[3] * velocity[0] + base[4] * velocity[1];
      const excess = base[3] ** 2 + base[4] ** 2 - speed ** 2;
      const scale = squared === 0 ? 0 : clamp((-along + Math.sqrt(Math.max(0, along ** 2 - squared * excess))) / squared, 0, 1);
      velocity[0] *= scale;
      velocity[1] *= scale;
    }
    velocity[2] = clamp(velocity[2], -LIMITS.referenceYawRate - base[5], LIMITS.referenceYawRate - base[5]);
  }

  function filteredReference(state, sample, action, feedback, dt) {
    const base = sample.reference;
    const cosine = Math.cos(feedback[2]), sine = Math.sin(feedback[2]);
    const right = action[0] * LIMITS.translationOffsetMeters, forward = action[1] * LIMITS.translationOffsetMeters;
    const target = [right * cosine + forward * sine, -right * sine + forward * cosine,
      action[2] * LIMITS.headingOffsetRadians];
    const velocity = state.offsetVelocity.map((value, axis) => {
      const acceleration = axis === 2 ? LIMITS.offsetYawAcceleration : LIMITS.offsetAcceleration;
      const speed = axis === 2 ? LIMITS.offsetYawRate : LIMITS.offsetSpeedMetersPerSecond;
      return clamp(value + clamp(16 * (target[axis] - state.offset[axis]) - 8 * value,
        -acceleration, acceleration) * dt, -speed, speed);
    });
    limitResidualVelocity(base, velocity);
    const offset = state.offset.map((value, axis) => value + velocity[axis] * dt);
    const bodyRight = offset[0] * cosine - offset[1] * sine;
    const bodyForward = offset[0] * sine + offset[1] * cosine;
    let scale = Math.min(1, LIMITS.translationOffsetMeters / Math.max(LIMITS.translationOffsetMeters, Math.abs(bodyRight)),
      LIMITS.translationOffsetMeters / Math.max(LIMITS.translationOffsetMeters, Math.abs(bodyForward)));
    const fieldClipped = [false, false];
    for (let axis = 0; axis < 2; axis++) {
      const position = base[axis] + offset[axis];
      fieldClipped[axis] = Math.abs(position) > LIMITS.fieldMeters;
      if (offset[axis] > 0) scale = Math.min(scale, (LIMITS.fieldMeters - base[axis]) / offset[axis]);
      if (offset[axis] < 0) scale = Math.min(scale, (-LIMITS.fieldMeters - base[axis]) / offset[axis]);
    }
    if (scale < 1) {
      offset[0] *= scale;
      offset[1] *= scale;
      velocity[0] = velocity[1] = 0;
    }
    const heading = clamp(offset[2], -LIMITS.headingOffsetRadians, LIMITS.headingOffsetRadians);
    if (heading !== offset[2]) velocity[2] = 0;
    offset[2] = heading;
    const reference = base.map((value, axis) => add(value, axis < 3 ? offset[axis] : velocity[axis - 3]));
    for (let axis = 0; axis < 2; axis++) {
      reference[axis] = clamp(reference[axis], -LIMITS.fieldMeters, LIMITS.fieldMeters);
      if (fieldClipped[axis] && reference[axis + 3] * offset[axis] > 0) reference[axis + 3] = 0;
    }
    return {reference, offset: reference.slice(0, 3).map((value, axis) => value - base[axis]), offsetVelocity: velocity};
  }

  class GvfResidualAdapter {
    #control;
    #state;

    constructor(control, feedback, goal) {
      if (!control || typeof control.gvf !== 'function' || typeof control.gvfReference !== 'function') {
        throw Error('Native gvf(pointsInches, h0, h1) and gvfReference(feedback10) are required');
      }
      this.#control = control;
      this.reset(feedback, goal);
    }

    reset(values, requestedGoal) {
      const feedback = finiteVector(values, 10, 'Sensor feedback');
      const start = boundedPose(feedback.slice(0, 3)), goal = boundedPose(requestedGoal);
      const points = Array.from({length: 4}, (_, index) => start.slice(0, 2).map((value, axis) =>
        clamp((value + (goal[axis] - value) * index / 3) / 0.0254, -60, 60)));
      this.#state = null;
      if (this.#control.gvf(points, start[2], goal[2]) !== true) throw Error('Native GVF initialization failed');
      const sample = sampleReference(this.#control, feedback);
      this.#state = {feedback, goal, sample, tick: 0, previousAction: [...ZERO_ACTION],
        offset: [0, 0, 0], offsetVelocity: [0, 0, 0], reference: [...sample.reference]};
      return this.reference;
    }

    get tick() { return this.#state?.tick ?? 0; }
    get reference() {
      if (!this.#state) throw Error('Reset the GVF residual adapter first');
      return [...this.#state.reference];
    }

    advance(values, sensors, dt, tick) {
      if (!this.#state) throw Error('Reset the GVF residual adapter first');
      const action = normalizeAction(values), feedback = finiteVector(sensors, 10, 'Sensor feedback');
      if (!Number.isFinite(dt) || dt <= 0 || dt > 0.1) throw Error('Physics dt must be finite and in (0, 0.1]');
      if (!Number.isSafeInteger(tick) || tick !== this.#state.tick + 1) throw Error('Advance once per sequential physics tick, starting at 1');
      const sample = sampleReference(this.#control, feedback);
      const filtered = filteredReference(this.#state, sample, action, feedback, dt);
      this.#state = {...this.#state, ...filtered, sample, feedback, previousAction: action, tick};
      return this.reference;
    }

    refreshSensors(values) {
      if (!this.#state) throw Error('Reset the GVF residual adapter first');
      const feedback = finiteVector(values, 10, 'Sensor feedback');
      const sample = sampleReference(this.#control, feedback);
      this.#state = {...this.#state, feedback, sample};
    }

    observe(remainingSeconds, actionAgeTicks = 0) {
      if (!this.#state) throw Error('Reset the GVF residual adapter first');
      if (!Number.isFinite(remainingSeconds) || remainingSeconds < 0 || !Number.isSafeInteger(actionAgeTicks) ||
        actionAgeTicks < 0 || actionAgeTicks > 6000) throw Error('Invalid deadline remaining/action age');
      const state = this.#state, feedback = state.feedback;
      const deltaX = state.goal[0] - feedback[0], deltaY = state.goal[1] - feedback[1];
      const cosine = Math.cos(feedback[2]), sine = Math.sin(feedback[2]);
      const goalCosine = Math.cos(state.goal[2]), goalSine = Math.sin(state.goal[2]);
      return {contractName: CONTRACT_NAME, contractVersion: CONTRACT_VERSION, observationSchema: OBSERVATION_SCHEMA,
        pose: feedback.slice(0, 3), velocity: feedback.slice(3, 6), wheelRimSpeeds: feedback.slice(6, 10),
        targetError: [deltaX * cosine - deltaY * sine, deltaX * sine + deltaY * cosine,
          goalSine * cosine - goalCosine * sine, goalCosine * cosine + goalSine * sine],
        previousAction: [...state.previousAction], remainingSeconds, baseReference: [...state.sample.reference],
        projection: state.sample.parameter, remainingArcMeters: state.sample.remainingMeters,
        crossTrackMeters: state.sample.crossTrackMeters, appliedFieldOffset: [...state.offset],
        offsetVelocity: [...state.offsetVelocity], actionAgeTicks};
    }
  }

  function vector(observation) {
    if (!observation || observation.contractVersion !== CONTRACT_VERSION || observation.contractName !== CONTRACT_NAME ||
      observation.observationSchema !== OBSERVATION_SCHEMA) throw Error('Explicit GVF residual contract version/name/schema required; length is not an identity');
    const keys = ['contractName', 'contractVersion', 'observationSchema', ...OBSERVATION_FIELDS.map(field => field[0])];
    if (Object.keys(observation).length !== keys.length || Object.keys(observation).some(key => !keys.includes(key))) {
      throw Error('Only the declared sensor observation fields are admitted');
    }
    const result = OBSERVATION_FIELDS.flatMap(([name, width]) => width === 1 ?
      finiteVector([observation[name]], 1, name) : finiteVector(observation[name], width, name));
    normalizeAction(observation.previousAction);
    if (observation.remainingSeconds < 0 || observation.projection < 0 || observation.projection > 1 ||
      observation.remainingArcMeters < 0 || observation.crossTrackMeters < 0 ||
      !Number.isSafeInteger(observation.actionAgeTicks) || observation.actionAgeTicks < 0 || observation.actionAgeTicks > 6000 || result.length !== OBSERVATION_SIZE) {
      throw Error('Invalid GVF residual sensor observation');
    }
    return result;
  }

  return {CONTRACT_VERSION, CONTRACT_NAME, OBSERVATION_SCHEMA, CONTRACT, ACTION_SIZE, OBSERVATION_SIZE,
    OBSERVATION_FIELDS, ZERO_ACTION, DEFAULT_ACTION, LIMITS, normalizeAction, options, GvfResidualAdapter, vector};
});
