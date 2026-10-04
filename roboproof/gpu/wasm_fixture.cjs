'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../..');
const Control = require(path.join(root, 'simulator/control-runtime'));
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'simulator/control-build.json'), 'utf8').replace(/^\uFEFF/, ''));
for (const [file, digest] of Object.entries(manifest.sources)) {
  if (hash(fs.readFileSync(path.join(root, file), 'utf8').replace(/\r\n/g, '\n')) !== digest) {
    throw Error(`Stale WASM source: ${file}; rebuild before claiming parity`);
  }
}
const binary = fs.readFileSync(path.join(root, 'simulator/control.wasm'));
if (hash(binary) !== manifest.wasm) throw Error('WASM digest mismatch');
const moduleCore = new WebAssembly.Module(binary);
const payload = JSON.parse(fs.readFileSync(0, 'utf8'), (_, value) =>
  value === 'NaN' ? NaN : value === 'Infinity' ? Infinity : value === '-Infinity' ? -Infinity : value);
if (payload.scenarios) {
  const {runScenario} = require(path.join(root, 'roboproof/sim.js'));
  process.stdout.write(JSON.stringify(payload.scenarios.map(scenario => runScenario(scenario, {telemetry: true}))));
} else {
  const results = payload.lanes.map(lane => {
    const core = Control.fromModule(moduleCore);
    if (lane.config) core.e.control_config(...lane.config);
    return lane.steps.map(step => {
      if (step.reset) core.reset();
      if (step.config) core.e.control_config(...step.config);
      core.buffer.set([...step.reference, ...step.feedback, step.dt, step.valid === false ? 0 : 1]);
      const valid = core.e.control_step() === 1;
      return {valid, volts: Array.from(core.buffer.slice(32, 36)), targets: Array.from(core.buffer.slice(36, 40))};
    });
  });
  process.stdout.write(JSON.stringify({wasm_sha256: manifest.wasm, results}));
}
