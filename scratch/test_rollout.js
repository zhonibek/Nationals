const fs = require('fs');
const vm = require('vm');

global.window = {};
global.document = {
  getElementById: () => null,
  addEventListener: () => {}
};

const simCode = fs.readFileSync(__dirname + '/../simulator/engine.js', 'utf8');
const headlessCode = simCode.split('document.addEventListener')[0];
vm.runInThisContext(headlessCode);

const ProductionControl = require('../simulator/control-runtime');
const wasmModule = new WebAssembly.Module(fs.readFileSync(__dirname + '/../simulator/control.wasm'));
const core = ProductionControl.fromModule(wasmModule);

function runSimRollout(actions, startPose = {x: 0, y: 0, theta: 0}) {
  const sim = new VexRobotSimulator(core.fork());
  sim.resetSimulation();
  sim.setPose(startPose.x, startPose.y, startPose.theta);
  sim.isRunning = true;
  for (const act of actions) {
    sim.queueAction({...act});
  }

  let totalEnergy = 0;
  let t = 0;
  const dt = 0.01;
  const maxSteps = 2500; // max 25s

  for (let step = 0; step < maxSteps && sim.isRunning; step++) {
    sim.update(dt);
    t += dt;
    if (sim.commandedWheelVoltages) {
      for (let v of sim.commandedWheelVoltages) {
        totalEnergy += (v * v) * dt;
      }
    }
  }

  return {
    time: +t.toFixed(2),
    energy: +totalEnergy.toFixed(1),
    finalX: +sim.x.toFixed(1),
    finalY: +sim.y.toFixed(1),
    finalTheta: +sim.theta.toFixed(1),
    settled: sim.lastMotionResult === 'Settled' || !sim.isRunning
  };
}

// Multi-step sequence:
// start at (0, 0, 0)
// go to (20, 20), heading 45
// go to (20, 40), heading 90
const start = {x: 0, y: 0, theta: 0};

// 1. Spline
const splineActs = [
  { type: 'spline', end: { x: 20, y: 20, theta: 45 }, maxVel: 0.45, maxAccel: 0.8 },
  { type: 'spline', end: { x: 20, y: 40, theta: 90 }, maxVel: 0.45, maxAccel: 0.8 }
];

// 2. GVF-TD3
const gvfActs = [
  { type: 'gvf', targetX: 20, targetY: 20, targetTheta: 45 },
  { type: 'gvf', targetX: 20, targetY: 40, targetTheta: 90 }
];

// 3. Strafe / Holonomic Pose
const strafeActs = [
  { type: 'pose', targetX: 20, targetY: 20, targetTheta: 45 },
  { type: 'pose', targetX: 20, targetY: 40, targetTheta: 90 }
];

// 4. Sprint PID
const sprintActs = [
  { type: 'turn', targetHeading: 45 },
  { type: 'drivePoint', targetX: 20, targetY: 20, targetHeading: 45 },
  { type: 'turn', targetHeading: 90 },
  { type: 'drivePoint', targetX: 20, targetY: 40, targetHeading: 90 }
];

console.log("Spline rollout:", runSimRollout(splineActs, start));
console.log("GVF rollout:   ", runSimRollout(gvfActs, start));
console.log("Strafe rollout:", runSimRollout(strafeActs, start));
console.log("Sprint rollout:", runSimRollout(sprintActs, start));
