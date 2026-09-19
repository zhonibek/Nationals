const fs = require('fs');
const vm = require('vm');

global.window = {};
global.document = { getElementById: () => null, addEventListener: () => {} };

const simCode = fs.readFileSync(__dirname + '/simulator.js', 'utf8');
vm.runInThisContext(simCode.split('document.addEventListener')[0]);

const sim = new VexRobotSimulator(require('./control-runtime').fromModule(new WebAssembly.Module(fs.readFileSync(__dirname+'/control.wasm'))));
sim.buildHolonomicSkillsRoutine();
sim.isRunning = true;

console.log('=== TESTING FULL HOLONOMIC SKILLS ROUTINE (extended) ===');
let lastActionDesc = '';
for (let s = 0; s < 2000; s++) {
  sim.update(0.01);
  let curDesc = sim.currentAction ? sim.currentAction.desc : 'IDLE/COMPLETE';
  if (curDesc !== lastActionDesc || s % 100 === 0) {
    console.log(`t=${(s*0.01).toFixed(2)}s | Action=${curDesc.padEnd(38, ' ')} | X=${sim.x.toFixed(1).padStart(5, ' ')}" | Y=${sim.y.toFixed(1).padStart(5, ' ')}" | Th=${sim.theta.toFixed(1).padStart(6, ' ')}° | v=${sim.v.toFixed(3)}m/s`);
    lastActionDesc = curDesc;
  }
  if (!sim.isRunning && sim.routineQueue.length === 0) {
    console.log(`>>> ALL STEPS FINISHED AT t=${(s*0.01).toFixed(2)}s! Final Pose: (${sim.x.toFixed(2)}", ${sim.y.toFixed(2)}", ${sim.theta.toFixed(1)}°) <<<`);
    break;
  }
}
