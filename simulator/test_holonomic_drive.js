// Deterministic integration tests for the X-drive / holonomic physics model.
// Run with: node simulator/test_holonomic_drive.js

const assert = require('assert');
const fs = require('fs');
const vm = require('vm');

global.window = {};
global.document = {
  getElementById: () => null,
  addEventListener: () => {}
};

const simCode = fs.readFileSync(__dirname + '/simulator.js', 'utf8');
vm.runInThisContext(simCode.split('document.addEventListener')[0]);

const DT = 0.01;

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function maxStep(samples, key) {
  let maximum = 0;
  for (let i = 1; i < samples.length; i++) {
    maximum = Math.max(maximum, Math.abs(samples[i][key] - samples[i - 1][key]));
  }
  return maximum;
}

function assertFiniteState(sim, label) {
  for (const value of [sim.x, sim.y, sim.theta, sim.vx, sim.vy, sim.v, sim.w]) {
    assert(Number.isFinite(value), `${label}: state contains a non-finite value`);
  }
}

function countDirectionChanges(samples, key) {
  let changes = 0;
  let previousSign = 0;
  for (const sample of samples) {
    const sign = Math.sign(sample[key]);
    if (sign !== 0 && previousSign !== 0 && sign !== previousSign) changes++;
    if (sign !== 0) previousSign = sign;
  }
  return changes;
}

function fieldVelocityAt(sim) {
  const heading = sim.theta * DEG_TO_RAD;
  return {
    x: sim.vx * Math.cos(heading) + sim.vy * Math.sin(heading),
    y: -sim.vx * Math.sin(heading) + sim.vy * Math.cos(heading)
  };
}

function instantaneousTurnCenter(sim) {
  assert(Math.abs(sim.w) > 1e-4, 'turn rate must be non-zero when measuring an arc');
  const heading = sim.theta * DEG_TO_RAD;
  const radiusScale = METER_TO_INCH / sim.w;

  // For a constant body-frame twist, the center of curvature is fixed in
  // field space. This remains valid even with a small lateral body velocity.
  return {
    x: sim.x + (sim.vy * Math.cos(heading) - sim.vx * Math.sin(heading)) * radiusScale,
    y: sim.y + (-sim.vy * Math.sin(heading) - sim.vx * Math.cos(heading)) * radiusScale
  };
}

function testStraightFieldLineWhileTurning() {
  const sim = new VexRobotSimulator(require('./control-runtime').fromModule(new WebAssembly.Module(fs.readFileSync(__dirname+'/control.wasm'))));
  sim.setPose(0, -45, 0);
  const samples = [];
  const fieldForwardVolts = 4.5;
  const turnVolts = 1.5;

  for (let step = 0; step < 450; step++) {
    // Convert a field-forward command into the robot frame on every step.
    // This is the defining holonomic case: translate along a straight field
    // line while the chassis rotates through more than 90 degrees.
    const heading = sim.theta * DEG_TO_RAD;
    const throttle = fieldForwardVolts * Math.cos(heading);
    const strafe = -fieldForwardVolts * Math.sin(heading);
    sim.stepHolonomicPhysics(throttle, strafe, turnVolts, DT);

    if (step >= 100) {
      const fieldVelocity = fieldVelocityAt(sim);
      samples.push({ x: sim.x, y: sim.y, theta: sim.theta, v: sim.v, w: sim.w, fieldVelocity });
    }
  }

  const forwardDistance = sim.y + 45;
  const maximumCrossTrackError = Math.max(...samples.map(sample => Math.abs(sample.x)));
  const maximumYawStep = maxStep(samples, 'w');

  assertFiniteState(sim, 'field-centric straight line');
  assert(forwardDistance > 65, 'field-centric drive did not make sufficient forward progress');
  assert(Math.abs(sim.theta) > 90, 'robot did not rotate through 90 degrees');
  assert(maximumCrossTrackError < 3.5,
    `field-centric path deviated ${maximumCrossTrackError.toFixed(2)} in from its straight line`);
  assert(maximumYawStep < 0.02, 'turn rate contains a physics-step discontinuity');
  assert(mean(samples.map(sample => sample.fieldVelocity.y)) > 0.35,
    'field-centric command did not retain positive field-forward velocity');

  console.log('[TEST 1] Field-centric straight line while rotating');
  console.log(`  forward distance: ${forwardDistance.toFixed(2)} in; max cross-track error: ${maximumCrossTrackError.toFixed(2)} in`);
  console.log(`  heading change: ${sim.theta.toFixed(1)} deg; max yaw-rate step: ${maximumYawStep.toFixed(5)} rad/s`);
}

function testConstantTwistIsSmoothCircularArc() {
  const sim = new VexRobotSimulator(require('./control-runtime').fromModule(new WebAssembly.Module(fs.readFileSync(__dirname+'/control.wasm'))));
  sim.setPose(-30, 0, 0);
  const samples = [];

  for (let step = 0; step < 600; step++) {
    // Constant robot-frame forward + turn is a constant twist, so its
    // field-space trajectory must be a circle after the motor transient.
    sim.stepHolonomicPhysics(6.0, 0.0, 2.0, DT);
    if (step >= 150) {
      const center = instantaneousTurnCenter(sim);
      samples.push({ x: sim.x, y: sim.y, v: sim.v, w: sim.w, center });
    }
  }

  const centerX = mean(samples.map(sample => sample.center.x));
  const centerY = mean(samples.map(sample => sample.center.y));
  const maximumCenterDrift = Math.max(...samples.map(sample =>
    Math.hypot(sample.center.x - centerX, sample.center.y - centerY)
  ));
  const averageRadius = mean(samples.map(sample => sample.v / sample.w * METER_TO_INCH));
  const maximumSpeedStep = maxStep(samples, 'v');
  const maximumYawStep = maxStep(samples, 'w');

  assertFiniteState(sim, 'constant-twist arc');
  assert(averageRadius > 20 && averageRadius < 60, 'arc radius is physically implausible');
  assert(maximumCenterDrift < 0.25,
    `constant-twist center drifted by ${maximumCenterDrift.toFixed(3)} in`);
  assert(maximumSpeedStep < 0.02, 'forward speed contains a contact-solver discontinuity');
  assert(maximumYawStep < 0.02, 'yaw rate contains a contact-solver discontinuity');

  console.log('[TEST 2] Constant forward + turn is a smooth circular arc');
  console.log(`  radius: ${averageRadius.toFixed(2)} in; center drift: ${maximumCenterDrift.toFixed(3)} in`);
  console.log(`  max speed step: ${maximumSpeedStep.toFixed(5)} m/s; max yaw-rate step: ${maximumYawStep.toFixed(5)} rad/s`);
}

function testSmoothHolonomicZigZag() {
  const sim = new VexRobotSimulator(require('./control-runtime').fromModule(new WebAssembly.Module(fs.readFileSync(__dirname+'/control.wasm'))));
  sim.setPose(0, -45, 0);
  const samples = [];

  for (let step = 0; step < 450; step++) {
    const time = step * DT;
    // A sinusoidal strafe produces a continuous S/zig-zag path. Forward and
    // lateral drive remain independent, which is the key X-drive behavior.
    const strafe = 3.0 * Math.sin((2.0 * Math.PI * time) / 1.2);
    sim.stepHolonomicPhysics(4.5, strafe, 0.0, DT);
    if (step >= 80) {
      samples.push({ x: sim.x, y: sim.y, theta: sim.theta, vx: sim.vx, vy: sim.vy, w: sim.w });
    }
  }

  const directionChanges = countDirectionChanges(samples, 'vx');
  const lateralSpan = Math.max(...samples.map(sample => sample.x)) - Math.min(...samples.map(sample => sample.x));
  const maximumLateralVelocityStep = maxStep(samples, 'vx');
  const headingDrift = Math.max(...samples.map(sample => Math.abs(sample.theta)));
  const forwardDistance = sim.y + 45;

  assertFiniteState(sim, 'smooth holonomic zig-zag');
  assert(forwardDistance > 60, 'zig-zag did not keep moving forward');
  assert(directionChanges >= 5, 'strafe velocity did not create repeated zig-zag direction changes');
  assert(lateralSpan > 2.5, 'zig-zag lateral excursion is too small');
  assert(maximumLateralVelocityStep < 0.02,
    'lateral velocity has an abrupt physics-step discontinuity');
  assert(headingDrift < 0.1, 'symmetric holonomic strafe introduced unintended yaw');

  console.log('[TEST 3] Smooth holonomic zig-zag');
  console.log(`  lateral direction changes: ${directionChanges}; lateral span: ${lateralSpan.toFixed(2)} in`);
  console.log(`  max lateral speed step: ${maximumLateralVelocityStep.toFixed(5)} m/s; heading drift: ${headingDrift.toFixed(4)} deg`);
}

console.log('================================================================================');
console.log('              HOLONOMIC X-DRIVE TRAJECTORY INTEGRATION TESTS');
console.log('================================================================================');

// Sensor noise is intentionally disabled only for geometric repeatability.
// The production simulator still retains its configured sensor noise and drift.
const savedRandom = Math.random;
Math.random = () => 0.5;
try {
  testStraightFieldLineWhileTurning();
  testConstantTwistIsSmoothCircularArc();
  testSmoothHolonomicZigZag();
} finally {
  Math.random = savedRandom;
}

console.log('================================================================================');
console.log('             >>> ALL HOLONOMIC TRAJECTORY TESTS PASSED <<<');
console.log('================================================================================');
