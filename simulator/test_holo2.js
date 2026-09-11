const fs = require('fs');
const vm = require('vm');

global.window = {};
global.document = { getElementById: () => null, addEventListener: () => {} };

const simCode = fs.readFileSync(__dirname + '/simulator.js', 'utf8');
vm.runInThisContext(simCode.split('document.addEventListener')[0]);

const sim = new VexRobotSimulator();
sim.buildHolonomicSkillsRoutine();
sim.isRunning = true;

for (let s = 0; s < 600; s++) {
  sim.update(0.01);
  if (sim.currentAction && sim.currentAction.desc.includes('[Holo 2]')) {
    if (s % 10 === 0) {
      console.log(
        `t=${(s*0.01).toFixed(2)}s | ` +
        `X=${sim.x.toFixed(2).padStart(6)}" Y=${sim.y.toFixed(2).padStart(6)}" Th=${sim.theta.toFixed(2).padStart(6)}° | ` +
        `v=${sim.v.toFixed(3)} w=${sim.w.toFixed(3)} | ` +
        `actV=[thr=${sim.actionThrottle.toFixed(1)}, str=${sim.actionStrafe.toFixed(1)}, turn=${sim.actionTurn.toFixed(1)}] | ` +
        `motV=[${sim.motorVolts.map(v => v.toFixed(1)).join(', ')}]`
      );
    }
  }
  if (sim.currentAction && sim.currentAction.desc.includes('[Holo 3]')) {
    console.log(`Holo 2 complete! Started Holo 3 at t=${(s*0.01).toFixed(2)}s. Pose: (${sim.x.toFixed(2)}, ${sim.y.toFixed(2)}, ${sim.theta.toFixed(2)} deg)`);
    break;
  }
}
