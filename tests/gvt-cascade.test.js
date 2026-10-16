'use strict';
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
const root = path.resolve(__dirname, '..'), Control = require('../simulator/control-runtime');
const moduleCore = new WebAssembly.Module(fs.readFileSync(path.join(root, 'simulator/control.wasm')));

const context = { console, Math, setTimeout() {}, document: { addEventListener() {} }, window: {} };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'simulator/robot-config.js'), 'utf8'), context);
vm.runInContext(fs.readFileSync(path.join(root, 'simulator/simulator.js'), 'utf8') + '\nthis.Simulator=VexRobotSimulator;', context);

const make = () => new context.Simulator(Control.fromModule(moduleCore));

console.log('Testing GVT (Guiding Vector Field) + LTV-LQR + PID Cascade...');

// 1. Test nominal GVT path execution
{
  const sim = make();
  sim.setPose(0, 0, 0);
  sim.queueAction({
    type: 'gvt',
    points: [[0, 0], [0, 16], [24, 8], [24, 24]],
    startHeading: 0,
    endHeading: 90
  });
  sim.isRunning = true;
  let n = 0;
  while (sim.isRunning && n++ < 2000) sim.update(0.01);

  assert.equal(sim.lastMotionResult, 'Settled', `GVT should settle, got: ${sim.lastMotionResult}`);
  assert(Math.hypot(sim.x - 24, sim.y - 24) < 1.0, `End pose position error should be < 1 inch, got (${sim.x}, ${sim.y})`);
  assert(Math.abs(sim.theta - 90) < 3.0, `End pose heading error should be < 3 deg, got ${sim.theta}`);
  assert(sim.motorVolts.every(v => Number.isFinite(v) && Math.abs(v) <= 12), 'Motor voltages must be finite and <= 12V');
  console.log(`PASS: Nominal GVT settled at (${sim.x.toFixed(2)}, ${sim.y.toFixed(2)}, ${sim.theta.toFixed(1)}°) in ${sim.simTime.toFixed(2)}s`);
}

// 2. Test disturbance recovery (robot displaced laterally during motion)
{
  const sim = make();
  sim.setPose(0, 0, 0);
  sim.queueAction({
    type: 'gvt',
    points: [[0, 0], [0, 16], [24, 8], [24, 24]],
    startHeading: 0,
    endHeading: 90
  });
  sim.isRunning = true;
  for (let i = 0; i < 80; i++) sim.update(0.01); // Run for 0.8s
  
  // Apply a lateral bump / displacement of 8 inches (e.g. defense impact)
  sim.x += 8.0;
  sim.odom.x += 8.0;

  let n = 0;
  while (sim.isRunning && n++ < 2500) sim.update(0.01);

  assert.equal(sim.lastMotionResult, 'Settled', `GVT with bump should recover and settle, got: ${sim.lastMotionResult}`);
  assert(Math.hypot(sim.x - 24, sim.y - 24) < 1.0, `Recovered pose error should be < 1 inch, got (${sim.x}, ${sim.y})`);
  console.log(`PASS: GVT successfully recovered from 8" defense bump and settled at (${sim.x.toFixed(2)}, ${sim.y.toFixed(2)})`);
}

console.log('ALL GVT + LTV-LQR + PID Cascade tests passed successfully!');
