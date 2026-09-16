const fs = require('fs');
const vm = require('vm');

global.window = {};
global.document = { getElementById: () => null, addEventListener: () => {} };

const simCode = fs.readFileSync(__dirname + '/simulator.js', 'utf8');
vm.runInThisContext(simCode.split('document.addEventListener')[0]);

const sim = new VexRobotSimulator();
sim.setPose(24, 24, 0);

sim.queueAction({ type: "diagonal", forwardInches: 0, strafeInches: -24, endHeading: 0, timeout: 3000, desc: "Strafe Left" });
sim.isRunning = true;

console.log("=== TESTING STRAFE LEFT ===");
for (let step = 0; step < 250; step++) {
  sim.update(0.01);
  if (step % 10 === 0 || !sim.isRunning) {
    console.log(
      `t=${sim.simTime.toFixed(2)}s | ` +
      `X=${sim.odom.x.toFixed(2).padStart(6)}" | ` +
      `v=${sim.v.toFixed(3)} | ` +
      `strafeV=${sim.actionStrafe ? sim.actionStrafe.toFixed(2) : 0} | ` +
      `act=${sim.currentAction ? sim.currentAction.type : 'none'}`
    );
  }
  if (!sim.isRunning && sim.routineQueue.length === 0) break;
}
