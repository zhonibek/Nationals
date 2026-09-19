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

const sim = new VexRobotSimulator(require('./control-runtime').fromModule(new WebAssembly.Module(fs.readFileSync(__dirname+'/control.wasm'))));
sim.buildLinearDriveTest(24.0);
sim.isRunning = true;

console.log("Simulating drive to 24 inches with current simulator.js code:");
for (let step = 0; step < 400; step++) {
  sim.update(0.01);
  if (step % 20 === 0) {
    console.log(`t=${(step*0.01).toFixed(2)}s | Y=${sim.y.toFixed(2)}" | X=${sim.x.toFixed(2)}" | Th=${sim.theta.toFixed(1)}° | v=${sim.v.toFixed(3)}m/s | isRunning=${sim.isRunning} | currentAction=${sim.currentAction ? sim.currentAction.type : 'null'}`);
  }
}
