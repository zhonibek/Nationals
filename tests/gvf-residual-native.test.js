'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash, randomUUID} = require('node:crypto');
const {runGvfPolicy, replay} = require('../roboproof/motion');
const {loadHeadless} = require('../simulator/headless');
const {Playback} = require('../simulator/motion-replay');
const Gvf = require('../simulator/gvf-policy');

test('three-tick focused native contract call and action-only replay use the original Simulator/WASM', () => {
  const directory = path.join(__dirname, '../roboproof/runs/motion/research', `gvf-residual-contract-smoke-${randomUUID()}`);
  fs.mkdirSync(directory);
  const world = {seed: 1874401, task: {start: {xIn: 0, yIn: 0, headingDeg: 0},
    goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 0.03}, configuration: {}};
  const write = (name, value) => fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n', {flag: 'wx'});
  write('worlds.json', {schemaVersion: 1, cases: [world], maxPhysicsTicksPerExecution: 3, maxExecutions: 3,
    scope: 'Predeclared lightweight native contract test, not training, teacher data, performance evaluation or an independent final'});
  const receipt = {schemaVersion: 1, executionClass: 'lightweight focused native test', hostPlatform: process.platform,
    modelTrainingPerformed: false, teacherCollectionPerformed: false, campaignPerformed: false,
    learnedImprovementVerified: false, renderedUiVerified: false, executionsCompleted: 0};
  try {
    let calls = 0;
    const report = runGvfPolicy({...world, act(vector, observation) {
      calls++;
      assert.equal(vector.length, 34);
      assert.equal(observation.contractName, Gvf.CONTRACT_NAME);
      return [0.25, -0.25, 0.5];
    }});
    receipt.executionsCompleted++;
    write('original-report.json', report);
    assert.equal(calls, 1);
    assert.equal(report.ticks, 3);
    assert.equal(report.reason, 'time_limit');
    assert.equal(report.policyContractVersion, 2);
    assert.equal(report.policyContractName, Gvf.CONTRACT_NAME);
    assert.equal(report.identity.canonicalSimulator, true);
    assert.match(report.identity.sources['gvf-policy.js'], /^[a-f0-9]{64}$/);
    assert.equal(report.transitions.length, 3);
    assert(report.transitions.every(row => row.observation.contractName === Gvf.CONTRACT_NAME &&
      row.nextObservation.contractName === Gvf.CONTRACT_NAME && row.action.values.length === 3));
    replay(report);
    receipt.executionsCompleted++;
    const runner = loadHeadless();
    const playback = new Playback(report, runner.createEpisode().controlFactory);
    playback.sim.isPaused = false;
    for (let tick = 0; tick < 3; tick++) playback.step();
    receipt.executionsCompleted++;
    assert.equal(playback.verified, true);
    assert.equal(playback.episode.ticks, 3);
    receipt.passed = true;
    receipt.physicsTicksExecuted = 9;
    receipt.controllerWasmSha256 = report.identity.controllerWasmSha256;
    receipt.exactNodeReplayVerified = true;
    receipt.browserPlaybackLogicVerifiedInNode = true;
  } catch (error) {
    receipt.passed = false;
    receipt.error = error.stack;
    throw error;
  } finally {
    write('native-test-receipt.json', receipt);
    const files = fs.readdirSync(directory).map(filename => {
      const bytes = fs.readFileSync(path.join(directory, filename));
      return {path: filename, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex')};
    });
    write('preservation-manifest.json', {schemaVersion: 1, files});
    process.stdout.write(JSON.stringify({evidenceDirectory: directory, ...receipt}) + '\n');
  }
});
