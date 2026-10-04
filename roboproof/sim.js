'use strict';

const fs = require('node:fs');
const path = require('node:path');
const Control = require('../simulator/control-runtime');
const {ROOT, validate, clone, clamp, wrap, random, normal, hash, provenance} = require('./core');
let compiled;
let version;

function controller() {
  if (!compiled) {
    version = provenance();
    compiled = new WebAssembly.Module(fs.readFileSync(path.join(ROOT, 'simulator/control.wasm')));
  }
  return Control.fromModule(compiled);
}

function forward(wheel, radius) {
  const diagonal = Math.SQRT1_2 / 2;
  return [(wheel[0] - wheel[1] - wheel[2] + wheel[3]) * diagonal,
    (wheel[0] + wheel[1] + wheel[2] + wheel[3]) * diagonal,
    (wheel[0] + wheel[1] - wheel[2] - wheel[3]) / (4 * radius)];
}

function physics(state, voltage, environment, dt) {
  const substeps = Math.ceil(dt / 0.001);
  const interval = dt / substeps;
  const direction = [1, -1, -1, 1];
  const rotation = [1, 1, -1, -1];
  const effectiveWheelMass = 0.6;
  for (let substep = 0; substep < substeps; substep++) {
    const cosine = Math.cos(state.pose[2]), sine = Math.sin(state.pose[2]);
    const strafe = state.velocity[0] * cosine - state.velocity[1] * sine;
    const forwardSpeed = state.velocity[0] * sine + state.velocity[1] * cosine;
    const force = [];
    for (let wheel = 0; wheel < 4; wheel++) {
      const contact = (forwardSpeed + direction[wheel] * strafe) * Math.SQRT1_2 + rotation[wheel] * environment.radius * state.velocity[2];
      const efficiency = wheel < 2 ? environment.left_motor_efficiency : environment.right_motor_efficiency;
      const target = clamp(voltage[wheel], -environment.battery_voltage, environment.battery_voltage) / 12 * environment.max_wheel_speed * efficiency;
      const previous = state.wheel[wheel];
      const coupling = environment.traction_stiffness / effectiveWheelMass;
      let speed = (previous + interval * (target / environment.motor_tau + coupling * contact)) / (1 + interval / environment.motor_tau + interval * coupling);
      const requested = environment.traction_stiffness * (speed - contact);
      const limit = environment.friction * environment.mass * 9.81 / 4;
      force[wheel] = clamp(requested, -limit, limit);
      if (Math.abs(requested) > limit) speed = (previous + interval * (target / environment.motor_tau - force[wheel] / effectiveWheelMass)) / (1 + interval / environment.motor_tau);
      state.wheel[wheel] = speed;
    }
    const drag = environment.lateral_friction * environment.mass * 9.81;
    const lateral = (force[0] - force[1] - force[2] + force[3]) * Math.SQRT1_2 - drag * Math.tanh(strafe / 0.05);
    const longitudinal = (force[0] + force[1] + force[2] + force[3]) * Math.SQRT1_2 - drag * Math.tanh(forwardSpeed / 0.05);
    const torque = (force[0] + force[1] - force[2] - force[3]) * environment.radius - 0.05 * state.velocity[2];
    state.velocity[0] += (lateral * cosine + longitudinal * sine) / environment.mass * interval;
    state.velocity[1] += (-lateral * sine + longitudinal * cosine) / environment.mass * interval;
    state.velocity[2] += torque / environment.inertia * interval;
    state.pose = state.pose.map((value, axis) => value + state.velocity[axis] * interval);
    state.pose[2] = wrap(state.pose[2]);
  }
}

function measure(raw, prefix, environment, time, next) {
  const noisy = raw + environment[`${prefix}_bias`] + environment[`${prefix}_drift`] * time + environment[`${prefix}_noise`] * normal(next);
  const quantum = environment[`${prefix}_quantization`];
  return quantum > 0 ? Math.floor(noisy / quantum + 0.5) * quantum : noisy;
}

function pathDistance(pose, start, goal) {
  const dx = goal[0] - start[0], dy = goal[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  const progress = lengthSquared ? clamp(((pose[0] - start[0]) * dx + (pose[1] - start[1]) * dy) / lengthSquared, 0, 1) : 0;
  return Math.hypot(pose[0] - start[0] - progress * dx, pose[1] - start[1] - progress * dy);
}

function runScenario(input, options = {}) {
  const scenario = validate(clone(input));
  const {environment, task} = scenario;
  const thresholds = task.thresholds;
  const core = controller();
  core.configure(scenario.controller.radius, scenario.controller.max_wheel_speed);
  const referenceDuration = core.duration(task.start, task.goal) * scenario.controller.reference_time_scale;
  const state = {pose: task.start.map((value, axis) => value + environment[['initial_x_error', 'initial_y_error', 'initial_heading_error'][axis]]), velocity: [0, 0, 0], wheel: [0, 0, 0, 0]};
  const estimate = [...task.start];
  let estimatedVelocity = [0, 0, 0];
  const encoderRandom = random(scenario.random_seed ^ 0x85ebca6b);
  const imuRandom = random(scenario.random_seed ^ 0xc2b2ae35);
  let encoders = [0, 0, 0, 0], imu = task.start[2];
  const history = [{wheel: [...state.wheel], heading: state.pose[2]}];
  const commands = [];
  const telemetry = [];
  const encoderPeriod = Math.max(1, Math.ceil(environment.encoder_period / task.dt));
  const imuPeriod = Math.max(1, Math.ceil(environment.imu_period / task.dt));
  const encoderDelay = Math.ceil(environment.encoder_latency / task.dt);
  const imuDelay = Math.ceil(environment.imu_latency / task.dt);
  const controlDelay = Math.ceil(environment.control_latency / task.dt);
  let maxPath = 0, maxTracking = 0, maxHeading = 0, maxLocalization = 0, saturated = 0;
  let oscillations = 0, lastHeadingSign = 0, settleStreak = 0, completion = null;
  let invalid = false, numerical = false, firstDivergence = null;
  const steps = Math.ceil(task.duration / task.dt);
  let finalEndpoint = Infinity, finalHeading = Infinity, executed = 0;
  for (let step = 0; step < steps; step++) {
    const time = step * task.dt;
    const pastEncoder = history[Math.max(0, step - encoderDelay)];
    const pastImu = history[Math.max(0, step - imuDelay)];
    const reportedWheels = pastEncoder.wheel.map(value => measure(value, 'encoder', environment, time, encoderRandom));
    const reportedHeading = wrap(measure(pastImu.heading, 'imu', environment, time, imuRandom));
    const encoderAvailable = encoderRandom() >= environment.encoder_dropout;
    const imuAvailable = imuRandom() >= environment.imu_dropout;
    if (step % encoderPeriod === 0 && encoderAvailable) encoders = reportedWheels;
    const oldHeading = estimate[2];
    const body = forward(encoders, scenario.controller.radius);
    if (step > 0) {
      if (task.localization === 'encoders-imu') {
        if (step % imuPeriod === 0 && imuAvailable) imu = reportedHeading;
        estimate[2] = imu;
      } else estimate[2] = wrap(estimate[2] + body[2] * task.dt);
      const midpoint = oldHeading + wrap(estimate[2] - oldHeading) / 2;
      estimatedVelocity = [body[0] * Math.cos(midpoint) + body[1] * Math.sin(midpoint), -body[0] * Math.sin(midpoint) + body[1] * Math.cos(midpoint), wrap(estimate[2] - oldHeading) / task.dt];
      estimate[0] += estimatedVelocity[0] * task.dt;
      estimate[1] += estimatedVelocity[1] * task.dt;
    }
    if (task.localization === 'ground-truth-baseline') {
      estimate.splice(0, 3, ...state.pose);
      estimatedVelocity = [...state.velocity];
    }
    const reference = core.reference(task.start, task.goal, time, referenceDuration);
    const output = core.step(reference, [...estimate, ...estimatedVelocity, ...encoders], task.dt);
    if (!output.valid) invalid = true;
    commands.push(output.volts);
    const applied = step >= controlDelay ? commands[step - controlDelay] : [0, 0, 0, 0];
    const endpoint = Math.hypot(state.pose[0] - task.goal[0], state.pose[1] - task.goal[1]);
    const heading = wrap(task.goal[2] - state.pose[2]);
    const pathError = pathDistance(state.pose, task.start, task.goal);
    const tracking = Math.hypot(state.pose[0] - reference[0], state.pose[1] - reference[1]);
    const headingError = wrap(reference[2] - state.pose[2]);
    const localization = Math.hypot(state.pose[0] - estimate[0], state.pose[1] - estimate[1]);
    maxPath = Math.max(maxPath, pathError);
    maxTracking = Math.max(maxTracking, tracking);
    maxHeading = Math.max(maxHeading, Math.abs(headingError));
    maxLocalization = Math.max(maxLocalization, localization);
    if (applied.some(value => Math.abs(value) >= environment.battery_voltage - 0.01)) saturated++;
    if (time >= referenceDuration && Math.abs(heading) > 0.02) {
      const sign = Math.sign(heading);
      if (lastHeadingSign && sign !== lastHeadingSign) oscillations++;
      lastHeadingSign = sign;
    }
    const settled = time >= referenceDuration && endpoint <= thresholds.endpoint && Math.abs(heading) <= thresholds.heading && Math.hypot(...state.velocity.slice(0, 2)) <= thresholds.settle_speed && Math.abs(state.velocity[2]) <= thresholds.settle_omega;
    settleStreak = settled ? settleStreak + task.dt : 0;
    if (settleStreak + 1e-12 >= thresholds.settle_seconds && completion === null) completion = time;
    if (firstDivergence === null && (pathError > thresholds.path || localization > thresholds.localization)) firstDivergence = time;
    if (options.telemetry) telemetry.push({time, reference: reference.slice(0, 3), truth: [...state.pose], estimate: [...estimate], velocity: [...state.velocity], wheel_speed: [...state.wheel], measured_wheel: [...encoders], target_wheel: output.targets, command: output.volts, applied: [...applied], path_error: pathError, tracking_error: tracking, heading_error: headingError, localization_error: localization});
    executed++;
    physics(state, applied, environment, task.dt);
    if (![...state.pose, ...state.velocity, ...state.wheel].every(Number.isFinite)) { numerical = true; break; }
    history.push({wheel: [...state.wheel], heading: state.pose[2]});
  }
  finalEndpoint = numerical ? null : Math.hypot(state.pose[0] - task.goal[0], state.pose[1] - task.goal[1]);
  finalHeading = numerical ? null : Math.abs(wrap(task.goal[2] - state.pose[2]));
  const categories = [];
  if (numerical) categories.push('NUMERICAL_FAILURE');
  if (invalid) categories.push('CONTROLLER_DIVERGENCE');
  if (maxPath > thresholds.path) categories.push('PATH_DIVERGENCE');
  if (maxLocalization > thresholds.localization) categories.push('LOCALIZATION_DIVERGENCE');
  if (finalHeading > thresholds.heading) categories.push('HEADING_INSTABILITY');
  if (oscillations >= thresholds.oscillations) categories.push('OSCILLATION');
  if (saturated / executed > thresholds.saturation_fraction) categories.push('MOTOR_SATURATION');
  if (finalEndpoint > thresholds.endpoint) categories.push('ENDPOINT_FAILURE');
  if (completion === null) categories.push('TIMEOUT');
  const metrics = {endpoint_error: finalEndpoint, max_path_deviation: maxPath, max_tracking_error: maxTracking, final_heading_error: finalHeading, max_heading_error: maxHeading, max_localization_error: maxLocalization, saturation_fraction: saturated / executed, oscillations, completion_time: completion, first_divergence: firstDivergence, reference_duration: referenceDuration, steps: executed};
  const failureScore = numerical || invalid ? 1e6 : finalEndpoint / thresholds.endpoint + maxPath / thresholds.path + finalHeading / thresholds.heading + maxLocalization / thresholds.localization + 2 * saturated / executed + oscillations / thresholds.oscillations + (completion === null ? 2 : completion / task.duration);
  return {scenario, scenario_sha256: hash(JSON.stringify(scenario)), ...version, backend: 'cpu-wasm', categories: categories.length ? categories : ['SUCCESS'], passed: categories.length === 0, failure_score: failureScore, metrics, final_truth: numerical ? null : state.pose, final_estimate: estimate, ...(options.telemetry ? {telemetry} : {})};
}

module.exports = {runScenario, physics, forward, pathDistance};
