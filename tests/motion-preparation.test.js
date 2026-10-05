'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const {createBridge, PROTOCOL_VERSION} = require('../roboproof/motion-bridge');
const {createMotionStore} = require('../roboproof/motion-store');
const {baseline, runPolicy, replay} = require('../roboproof/motion');
const {readiness} = require('../roboproof/motion-readiness');
const {suite, assertTrainingWorld, aggregate} = require('../roboproof/motion-evaluation');
const {DEFAULT_TASK} = require('../simulator/motion');
const {loadHeadless} = require('../simulator/headless');
const {Playback, matching} = require('../simulator/motion-replay');
const runner = loadHeadless();

test('local protocol exposes only bounded canonical operations and no file/shell/hardware tools', () => {
  const bridge = createBridge();
  let id = 0;
  const send = (op, fields = {}) => bridge.handle({protocolVersion: PROTOCOL_VERSION, id: ++id, op, ...fields}).result;
  const contract = send('contract');
  assert.equal(contract.identity.canonicalSimulator, true);
  assert.equal(contract.motionTrainingImplemented, true);
  assert.equal(contract.observationSize, 34);
  assert.equal(contract.actionSize, 4);
  assert.throws(() => send('exec', {command: 'anything'}), /Unsupported/);
  assert.throws(() => send('reset', {filename: '../anything'}), /Unsupported/);
  assert.throws(() => bridge.handle({protocolVersion: 99, id: 1, op: 'contract'}), /identity/);
  assert.equal(send('reset', {task: {...DEFAULT_TASK, deadlineSeconds: 0.03}}).vector.length, 34);
  const result = send('step', {action: [1, 0, 0, 0]});
  assert.equal(result.truncated, true);
  assert.equal(result.terminated, false);
  assert.equal(result.info.physicsTicks, 3);
  const report = send('report');
  assert.equal(report.transitions.length, 3);
  assert.deepEqual(replay(report), report);
  assert.throws(() => send('step', {action: [1, 0, 0, 0]}), /ended/);
});

test('frozen evaluation has expanded groups, immutable hashes and training exclusion', () => {
  const old = suite(1);
  const current = suite(2);
  assert.equal(old.cases.length, 12);
  assert.equal(current.cases.length, 24);
  assert.equal(new Set(current.cases.map(world => world.group)).size, 8);
  assert.equal(suite(2).corpusSha256, current.corpusSha256);
  assert.equal(current.acceptance.independentTrainingSeedsRequired, 3);
  assert.throws(() => assertTrainingWorld(current.cases[0]), /excluded/);
  assert.throws(() => assertTrainingWorld({...current.cases[0], id: 'training', seed: 12}), /excluded/);
  assert.throws(() => assertTrainingWorld({...current.cases[0], task: DEFAULT_TASK}), /excluded/);
  const training = {id: 'development-only', seed: 11, task: {...DEFAULT_TASK, goal: {xIn: 3, yIn: 11, headingDeg: 13}}};
  assert.equal(assertTrainingWorld(training), training);
  assert.throws(() => suite(3), /Unsupported/);
});

test('evaluation aggregates failed worlds instead of reporting only selected successes', () => {
  const report = (reason, elapsedSeconds, effort) => ({reason, metrics: {elapsedSeconds,
    effortProxyVAs: effort, contactSeconds: 0, positionErrorMeters: reason === 'success' ? 0 : 1}});
  const result = aggregate([report('success', 2, 10), report('time_limit', 10, 100)]);
  assert.equal(result.successRate, 0.5);
  assert.equal(result.meanElapsedSecondsAllWorlds, 6);
  assert.equal(result.meanEffortProxyVAsAllWorlds, 55);
  assert.equal(result.meanPositionErrorMetersAllWorlds, 0.5);
});

test('readiness remains false without current complete evidence, including stale or empty checks', context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-readiness-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const filename = path.join(directory, 'readiness.json');
  assert.equal(readiness(filename).readyForBoundedCpuExperiment, false);
  fs.writeFileSync(filename, JSON.stringify({schemaVersion: 1, verifiedAt: new Date().toISOString(),
    identity: runner.identity, readyForBoundedCpuExperiment: true, checks: {}}));
  assert.equal(readiness(filename).readyForBoundedCpuExperiment, false);
  assert.equal(readiness(filename).cloudExecutionEnabled, false);
  fs.writeFileSync(filename, '{}');
  assert.equal(readiness(filename).readyForBoundedCpuExperiment, false);
});

test('motion results survive store recreation and path traversal/oversized evidence is rejected', context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-store-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const store = createMotionStore(directory);
  assert.equal(store.latest(), null);
  const saved = store.create({report: baseline(1, {...DEFAULT_TASK, deadlineSeconds: 0.03}), exactReplayVerified: true});
  assert.deepEqual(createMotionStore(directory).latest(), saved);
  assert.deepEqual(createMotionStore(directory).read(saved.id), saved);
  assert.throws(() => store.read('../anything'), /Invalid/);
  assert.throws(() => store.create({text: 'a'.repeat(16 * 1024 * 1024)}), /exceeds/);
});

test('original Simulator playback recomputes recorded actions and rejects altered transitions', () => {
  const report = runPolicy({task: {...DEFAULT_TASK, deadlineSeconds: 0.3}});
  const player = new Playback(report, () => runner.createEpisode().controlFactory());
  const initial = player.sim;
  player.step();
  assert.equal(player.episode.ticks, 0);
  player.sim.isPaused = false;
  while (!player.finished) player.step();
  assert.equal(player.verified, true);
  player.sim.timeScale = 2;
  player.restart();
  assert.equal(player.sim, initial);
  assert.equal(player.sim.timeScale, 2);
  assert.equal(player.episode.sim, initial);
  assert.equal(player.episode.ticks, 0);
  const tampered = structuredClone(report);
  tampered.transitions[0].evaluator.poseInchesDegrees[0] += 1;
  const invalid = new Playback(tampered, () => runner.createEpisode().controlFactory());
  invalid.sim.isPaused = false;
  assert.throws(() => invalid.step(), /mismatch/);
  assert.equal(invalid.sim.isPaused, true);
  assert.equal(invalid.sim.isRunning, false);
  assert.equal(invalid.verified, false);
  assert.throws(() => new Playback(baseline(), () => runner.createEpisode().controlFactory()), /complete transitions/);
  assert.doesNotThrow(() => matching(1, 1 + 1e-10));
  assert.throws(() => matching(1, 1 + 1e-6), /mismatch/);
});

test('sensor faults return finite fallback policy vectors and explicit unsuccessful terminals', () => {
  const episode = runner.createEpisode();
  episode.reset(1, {}, DEFAULT_TASK, {controlMode: 'policy'});
  episode.sim.odom.x = NaN;
  const result = episode.policyStep([1, 0, 0, 0]);
  assert.equal(result.terminated, true);
  assert.equal(result.info.reason, 'controller_or_sensor_fault');
  assert.equal(result.info.observationValid, false);
  assert(result.vector.every(Number.isFinite));
  assert.equal(result.rewardComponents.fault, -1);
});

test('motion dashboard only reads status on load; inference, training and experiments are explicit', async () => {
  const ids = ['check', 'restore', 'evaluate', 'run', 'cancel', 'replay', 'download', 'form', 'status', 'readiness', 'gates',
    'learning-refresh', 'compare-learned', 'use-verified'];
  const elements = new Map(ids.map(id => [`motion-${id}`, {disabled: false, textContent: '', addEventListener() {}}]));
  const requests = [];
  const realm = vm.createContext({document: {getElementById: id => elements.get(id)},
    location: {protocol: 'http:', hostname: '127.0.0.1'}, console,
    fetch: async (url, options) => { requests.push({url, options}); return {ok: true,
      json: async () => ({readyForBoundedCpuExperiment: false})}; }});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../roboproof/dashboard/motion.js'), 'utf8'), realm);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(requests.map(entry => entry.url), ['/api/motion/status']);
  assert.equal(requests[0].options.method, undefined);
  assert.equal(elements.get('motion-readiness').textContent, 'READINESS NOT VERIFIED');
  assert.equal(elements.get('motion-compare-learned').disabled, true);
  assert.equal(elements.get('motion-use-verified').disabled, true);
});
