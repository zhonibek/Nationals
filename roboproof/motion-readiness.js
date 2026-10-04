'use strict';

const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {baseline, runPolicy, replay} = require('./motion');
const {loadHeadless} = require('../simulator/headless');
const {DEFAULT_TASK, FIXED_DT} = require('../simulator/motion');
const Policy = require('../simulator/policy');
const {writeJson} = require('./core');
const reportPath = path.join(__dirname, 'runs/motion/readiness.json');
const checkNames = ['scriptedBaselineSuccess', 'boundedPolicyFixtureSuccess', 'randomEpisodeBounded',
  'transitionsRecorded', 'exactJsonReplay', 'staleActionStops', 'invalidActionStops'];

function verify() {
  const original = baseline(31415);
  assert.equal(original.reason, 'success');
  replay(original);
  const nominal = runPolicy({seed: 31415, options: {policyIdentity: {kind: 'scripted', id: 'readiness-fixture'}}});
  assert.equal(nominal.reason, 'success');
  assert.equal(nominal.transitions.length, nominal.ticks);
  replay(JSON.parse(JSON.stringify(nominal)));
  let state = 31415;
  const random = runPolicy({seed: 31416, task: {...DEFAULT_TASK, deadlineSeconds: 0.3},
    act: () => Array.from({length: Policy.ACTION_SIZE}, () => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 4294967296 * 2 - 1;
    })});
  assert.equal(random.truncated, true);
  assert.equal(random.ticks, 30);
  replay(JSON.parse(JSON.stringify(random)));
  const runner = loadHeadless();
  const episode = runner.createEpisode();
  episode.reset(31417, {}, DEFAULT_TASK, {controlMode: 'policy'});
  assert.equal(episode.step().info.reason, 'stale_policy_action');
  episode.reset(31417, {}, DEFAULT_TASK, {controlMode: 'policy'});
  assert.throws(() => episode.policyStep([NaN, 0, 0, 0]));
  assert.equal(episode.step().info.reason, 'stopped');
  const reports = {baseline: original, nominalPolicyFixture: nominal, boundedRandomFixture: random};
  for (const [name, report] of Object.entries(reports)) writeJson(path.join(__dirname, `runs/motion/readiness-${name}.json`), report);
  const evidence = {schemaVersion: 1, verifiedAt: new Date().toISOString(), identity: runner.identity,
    checks: {scriptedBaselineSuccess: true, boundedPolicyFixtureSuccess: true, randomEpisodeBounded: true,
      transitionsRecorded: true, exactJsonReplay: true, staleActionStops: true, invalidActionStops: true},
    readyForBoundedCpuExperiment: true, motionTrainingImplemented: false,
    scope: 'Original-Simulator reach-pose environment smoke, not full regression CI, learned improvement, game validation or physical calibration'};
  writeJson(reportPath, evidence);
  return evidence;
}

function readiness(filename = reportPath) {
  let evidence = null;
  let issue = null;
  try {
    const runner = loadHeadless();
    if (fs.existsSync(filename)) {
      evidence = JSON.parse(fs.readFileSync(filename, 'utf8'));
      assert.equal(evidence.schemaVersion, 1);
      assert.deepEqual(evidence.identity, runner.identity);
      assert(evidence.readyForBoundedCpuExperiment === true &&
        typeof evidence.verifiedAt === 'string' && Number.isFinite(Date.parse(evidence.verifiedAt)) &&
        evidence.checks && Object.keys(evidence.checks).length === checkNames.length &&
        checkNames.every(name => evidence.checks[name] === true));
    }
  } catch (error) { evidence = null; issue = `Readiness evidence unavailable or stale: ${error.message}`; }
  return {schemaVersion: 1, engine: 'original VexRobotSimulator', controller: 'iraLIB C++ LTV-LQR + wheel PI/feedforward',
    actionSize: Policy.ACTION_SIZE, observationSize: Policy.OBSERVATION_SIZE, fixedDt: FIXED_DT,
    defaultPolicyIntervalTicks: 5, readyForBoundedCpuExperiment: Boolean(evidence), evidence,
    motionTrainingImplemented: false, learnedImprovementVerified: false,
    issue: issue ?? (evidence ? null : 'Run the explicit motion-check command; GET status does not run experiments'),
    separateGates: ['Learned policy training and checkpoint/resume', 'Independent frozen before/after evaluation',
      'Full-game mechanics/scoring validation', 'Measured hardware calibration and timing', 'Authorized AMD provider execution',
      'Real camera/calibration and LocateAnything inference'], hardwareExecutionEnabled: false, cloudExecutionEnabled: false};
}

module.exports = {verify, readiness};
