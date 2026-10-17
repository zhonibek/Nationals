'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const os = require('node:os');
const Control = require('../simulator/control-runtime');
const {loadHeadless} = require('../simulator/headless');
const Policy = require('../simulator/policy');
const {runGvf, replay} = require('../roboproof/motion');

const runner = loadHeadless();
const makeControl = () => Control.fromModule(new WebAssembly.Module(fs.readFileSync(path.join(__dirname, '../simulator/control.wasm'))));
const line = [[0, 0], [0, 8], [0, 16], [0, 24]];
const sensors = (x = 0, y = 0, heading = 0) => [x, y, heading, 0, 0, 0, 0, 0, 0, 0];
const near = (actual, expected, tolerance = 1e-6) => assert(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('GVF uses Pedro projection and bounded SI velocity without advancing a clock', () => {
  const control = makeControl();
  assert.equal(control.gvf(line, 0, Math.PI / 2), true);
  const feedback = sensors(0.1, 0.3048);
  const sample = control.gvfReference(feedback);
  near(sample.parameter, 0.5, 1e-4);
  near(sample.reference[0], 0);
  near(sample.reference[1], 0.3048, 1e-5);
  near(sample.reference[2], Math.PI / 4, 1e-4);
  assert(sample.reference[3] < 0);
  assert(sample.reference[4] > 0);
  assert(Math.hypot(...sample.reference.slice(3, 5)) <= 0.3500000001);
  assert.equal(sample.complete, false);
  for (let repeat = 0; repeat < 8; repeat++) assert.deepEqual(control.gvfReference(feedback), sample);
  assert(control.gvfReference(sensors(0, 0.45)).parameter > sample.parameter);
});

test('GVF completion checks the actual endpoint, heading and measured velocity', () => {
  const control = makeControl();
  control.gvf(line, 0, Math.PI / 2);
  assert.equal(control.gvfReference(sensors()).complete, false);
  assert.equal(control.gvfReference(sensors(0, 0.3048, Math.PI / 4)).complete, false);
  assert.equal(control.gvfReference(sensors(0, 0.6096, 0)).complete, false);
  const goal = sensors(0, 0.6096, Math.PI / 2);
  const sample = control.gvfReference(goal);
  assert.equal(sample.complete, true);
  near(Math.hypot(...sample.reference.slice(3, 5)), 0, 1e-6);
  goal[4] = 0.04;
  assert.equal(control.gvfReference(goal).complete, false);
});

test('GVF supports heading-only tasks with shortest-angle interpolation', () => {
  const control = makeControl();
  assert(control.gvf(Array.from({length: 4}, () => [0, 0]), 170 * Math.PI / 180, -170 * Math.PI / 180));
  const sample = control.gvfReference(sensors(0, 0, 170 * Math.PI / 180));
  near(sample.reference[2], 190 * Math.PI / 180);
  assert.equal(sample.complete, false);
  assert.equal(control.gvfReference(sensors(0, 0, -170 * Math.PI / 180)).complete, true);
});

test('GVF requires a precise sensor heading before releasing the controller', () => {
  const control = makeControl();
  assert(control.gvf(Array.from({length: 4}, () => [0, 0]), 0, Math.PI));
  assert.equal(control.gvfReference(sensors(0, 0, Math.PI - 0.02)).complete, false);
  assert.equal(control.gvfReference(sensors(0, 0, Math.PI - 0.004)).complete, true);
});

test('GVF limits the combined LQR and feedforward yaw request, not just its feedforward', () => {
  for (const direction of [-1, 1]) {
    const control = makeControl();
    assert(control.gvf(Array.from({length: 4}, () => [0, 0]), 0, direction * Math.PI));
    const feedback = sensors();
    const sample = control.gvfReference(feedback);
    sample.reference[5] = direction * 1.2;
    for (let tick = 0; tick < 80; tick++) {
      const output = control.step(sample.reference, feedback, 0.01, 'gvf');
      assert.equal(output.valid, true);
      const yaw = (output.targets[0] + output.targets[1] - output.targets[2] - output.targets[3]) / (4 * 0.1885853785424542);
      assert(Math.abs(yaw) <= 1.50000001);
      assert.equal(Math.sign(yaw), direction);
    }
  }
});

test('GVF stepping is explicit and cannot leak a yaw limit into the ordinary controller', () => {
  const fieldControl = makeControl(), ordinary = makeControl();
  const reference = [0, 0, Math.PI, 0, 0, 0], feedback = sensors();
  assert.equal(fieldControl.step(reference, feedback, 0.01, 'gvf').valid, false);
  assert(fieldControl.gvf(Array.from({length: 4}, () => [0, 0]), 0, Math.PI));
  for (let tick = 0; tick < 80; tick++) {
    assert.deepEqual(fieldControl.step(reference, feedback, 0.01), ordinary.step(reference, feedback, 0.01));
  }
  assert.equal(fieldControl.step(reference, feedback, 0.01, 'unknown').valid, false);
});

test('GVF cubic references are finite, cross-track attracting and curvature speed limited', () => {
  const control = makeControl();
  assert(control.gvf([[0, 0], [0, 16], [24, 8], [24, 24]], 0, Math.PI / 2));
  const sample = control.gvfReference(sensors(12 * 0.0254, 12 * 0.0254));
  assert(sample.reference.every(Number.isFinite));
  assert(sample.parameter > 0 && sample.parameter < 1);
  assert(Math.hypot(...sample.reference.slice(3, 5)) <= 0.3500000001);
  assert(Math.abs(sample.reference[5]) <= 1.5);
});

test('invalid GVF paths clear the previous path and malformed sensors cannot produce a reference', () => {
  const control = makeControl();
  assert(control.gvf(line, 0, 0));
  assert.equal(control.gvfReference(sensors(NaN)), null);
  assert.equal(control.gvfReference([0]), null);
  assert.equal(control.gvf([[0, 0], [0, 8], [0, 16], [61, 24]], 0, 0), false);
  assert.equal(control.gvfReference(sensors()), null);
  assert(control.gvf(line, 0, 0));
  assert.equal(control.gvf(line, Infinity, 0), false);
  assert.equal(control.gvfReference(sensors()), null);
});

test('existing four-action trained actors cannot be relabelled as GVF actors', () => {
  assert.throws(() => Policy.options({controlMode: 'gvf', policyIdentity: {kind: 'learned', id: 'old-actor', sha256: 'a'.repeat(64)}}), /not a GVF policy contract/);
  for (const sha256 of ['a'.repeat(64), 'invalid']) assert.throws(() => runGvf(42, undefined, {},
    {policyIdentity: {kind: 'learned', id: 'old-actor', sha256}}), /not a GVF policy contract|Invalid policyIdentity/);
  assert.equal(Policy.options().controlMode, 'scripted');
  assert.equal(Policy.ACTION_SIZE, 4);
  assert.equal(Policy.CONTRACT_VERSION, 1);
});

test('GVF rejects retracing and cusp paths rather than moving away from an ambiguous endpoint', () => {
  const control = makeControl();
  for (const points of [[[0, 0], [0, 16], [0, 12], [0, 12]], [[0, 0], [0, 24], [0, 24], [0, 0]]]) {
    assert.equal(control.gvf(points, 0, 0), false);
    assert.equal(control.gvfReference(sensors(0, 12 * 0.0254)), null);
  }
  assert.equal(control.gvf([[0, 0], [0, 24], [0, 24], [0, 24]], 0, 0), true);
  const endpoint = control.gvfReference(sensors(0, 24 * 0.0254));
  assert.equal(endpoint.complete, true);
  near(Math.hypot(...endpoint.reference.slice(3, 5)), 0);
  for (const origin of [0, 30]) {
    assert.equal(control.gvf([[origin, origin], [origin, origin], [origin, origin], [origin, origin + 24]], 0, 0), true);
    const start = control.gvfReference(sensors(origin * 0.0254, origin * 0.0254));
    assert(start.reference[4] > 0.34);
    assert.equal(start.complete, false);
  }
});

test('original Simulator GVF reads odometry rather than truth and preserves the persistent controller', () => {
  const episode = runner.createEpisode();
  const task = {start: {xIn: 0, yIn: 0, headingDeg: 0}, goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 0.1};
  episode.reset(42, {}, task, {controlMode: 'gvf'});
  episode.applyAction({type: 'pose', ...task.goal});
  const sim = episode.sim, control = sim.productionControl;
  let resets = 0;
  const reset = control.reset.bind(control);
  control.reset = () => { resets++; reset(); };
  const step = control.step.bind(control);
  let observedMode;
  control.step = (...arguments_) => { observedMode = arguments_[3]; return step(...arguments_); };
  episode.step();
  assert.equal(observedMode, 'gvf');
  const feedback = sensors(sim.odom.x * 0.0254, sim.odom.y * 0.0254, sim.odom.theta * Math.PI / 180);
  const reference = control.gvfReference(feedback);
  const position = [sim.x, sim.y, sim.theta];
  sim.x = 30;
  sim.y = -10;
  assert.deepEqual(control.gvfReference(feedback), reference);
  [sim.x, sim.y, sim.theta] = position;
  while (!episode.terminated && !episode.truncated) episode.step();
  assert.equal(resets, 1);
  assert.equal(control, sim.productionControl);
  assert.equal(episode.ticks, 10);
  assert.equal(episode.report().reason, 'time_limit');
});

test('bounded GVF evidence round-trips through the normal exact Simulator replay', () => {
  const task = {start: {xIn: 0, yIn: 0, headingDeg: 0}, goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 0.1};
  const report = runGvf(17, task, {}, {recordTransitions: true});
  assert.equal(report.options.controlMode, 'gvf');
  assert.equal(report.options.policyIdentity.kind, 'scripted');
  assert.equal(report.ticks, 10);
  assert.match(report.agent, /not a learned policy/);
  assert.deepEqual(replay(JSON.parse(JSON.stringify(report))), report);
});

test('GVF honors sensor faults and original Simulator explicit deadlines', () => {
  const episode = runner.createEpisode();
  const task = {start: {xIn: 0, yIn: 0, headingDeg: 0}, goal: {xIn: 24, yIn: 24, headingDeg: 90}, deadlineSeconds: 0.1};
  episode.reset(8, {}, task, {controlMode: 'gvf'});
  const sim = episode.sim;
  const action = {type: 'gvf', targetX: 24, targetY: 24, targetTheta: 90, timeout: 10};
  sim.initAction(action);
  sim.actionStartTime = sim.simTime;
  sim.odom.x = NaN;
  assert.equal(sim.executeAction(action, 0.01), true);
  assert.equal(sim.lastMotionResult, 'SensorFault');
  assert.deepEqual(sim.commandedWheelVoltages, [0, 0, 0, 0]);
});

test('default point and Bezier cascade numerics retain the pre-GVF WASM golden', () => {
  const control = makeControl(), rows = [];
  control.configure(0.1885853785424542, 0.864462);
  const feedback = [0.01, 0.02, 0.1, 0.01, -0.01, 0.02, 0, 0, 0, 0];
  for (let index = 0; index < 12; index++) {
    const reference = control.reference([0, 0, 0], [0.45, 0.61, Math.PI / 2], index * 0.01, 3);
    rows.push({reference, output: control.step(reference, feedback, 0.01)});
  }
  control.reset();
  rows.push(control.bezier([[0, 0], [0, 16], [24, 8], [24, 24]], 0, Math.PI / 2));
  for (let index = 0; index < 12; index++) {
    const reference = control.bezierReference(index * 0.01);
    rows.push({reference, output: control.step(reference, feedback, 0.01)});
  }
  assert.equal(createHash('sha256').update(JSON.stringify(rows)).digest('hex'), '295ff3aa2874e9e0eaa53f24ade3bedf9171fd166441a9378794470d01c80fa9');
});

test('Motion Lab API saves and replays a short GVF run with no learned inference', async context => {
  const {createServer} = require('../roboproof/server');
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'roboproof-gvf-test-'));
  const server = createServer({motionDirectory: directory, reportPath: path.join(directory, 'absent.json')});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise(resolve => {
    server.closeAllConnections();
    server.close(() => { fs.rmSync(directory, {recursive: true, force: true}); resolve(); });
  }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const response = await fetch(`${base}/api/motion/run`, {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: 'gvf', seed: 42, task: {start: {xIn: 0, yIn: 0, headingDeg: 0},
      goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 0.1}})});
  assert.equal(response.status, 200);
  const record = await response.json();
  assert.equal(record.report.options.controlMode, 'gvf');
  assert.equal(record.exactReplayVerified, true);
  assert.equal(record.inferencePerformed, false);
  assert.equal(record.motionPolicyTrained, false);
  assert.equal(fs.existsSync(path.join(directory, `${record.id}.json`)), true);
  const replayResponse = await fetch(`${base}/api/motion/replay`, {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: record.id})});
  assert.equal(replayResponse.status, 200);
  assert.equal((await replayResponse.json()).exactReplayVerified, true);
});
