const fs = require('fs');
const vm = require('vm');

global.window = {};
global.document = {
  getElementById: () => null,
  addEventListener: () => {}
};

const simCode = fs.readFileSync(__dirname + '/simulator.js', 'utf8');
const headlessCode = simCode.split('document.addEventListener')[0];
vm.runInThisContext(headlessCode);

const sim = new VexRobotSimulator();
sim.setPose(0, 24, 0);

sim.queueAction({ type: "strafe", targetInches: 24, heading: 0, timeout: 3000, desc: "Strafe 24in" });
sim.isRunning = true;

console.log("=== TESTING PURE LATERAL STRAFE ===");
for (let step = 0; step < 300; step++) {
  sim.update(0.01);
  if (step % 20 === 0 || !sim.isRunning) {
    console.log(
      `t=${sim.simTime.toFixed(2)}s | ` +
      `X=${sim.odom.x.toFixed(1).padStart(5)}" Y=${sim.odom.y.toFixed(1).padStart(5)}" Th=${sim.odom.theta.toFixed(1).padStart(5)}° | ` +
      `v=${sim.v.toFixed(3)} w=${sim.w.toFixed(3)} | ` +
      `Vcmd=[${sim.motorVolts.map(v => v.toFixed(1)).join(', ')}] | action=${sim.currentAction ? sim.currentAction.type : 'none'}`
    );
  }
  if (!sim.isRunning && sim.routineQueue.length === 0) break;
}
