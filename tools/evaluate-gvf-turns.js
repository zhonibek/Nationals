'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const root = path.resolve(__dirname, '..');
const {suite, aggregate} = require('../roboproof/motion-evaluation');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const write = (directory, name, value) => fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n', {flag: 'wx'});

function declare() {
  const conditions = [{id: 'nominal', configuration: {}}, {id: 'loaded', configuration: {
    massKg: 11.8, moiKgM2: 0.48, muLong: 0.4, muLat: 0.12, kSlip: 400, batteryInternalR: 0.1
  }}];
  const turns = [-180, -135, -90, -45, 45, 90, 135, 180].flatMap((headingDeg, index) => conditions.map((condition, variant) => ({
    id: `dev-heading-${headingDeg}-${condition.id}`, seed: 2026101601 + index * 2 + variant,
    configuration: condition.configuration, scope: 'Previously inspected development turn; not an independent final',
    task: {start: {xIn: 35, yIn: 5, headingDeg: 0}, goal: {xIn: 35, yIn: 5, headingDeg}, deadlineSeconds: 8}
  })));
  const historical = suite(2);
  const cases = [...turns, ...historical.cases.map(world => ({...world, scope: 'Previously consumed historical development; not an independent final'}))];
  assert.equal(cases.length, 40);
  return {schemaVersion: 1, name: 'gvf-v2-turn-repair-20261017', cases,
    sourceVariants: ['preserved-v1', 'working-v2'], modes: ['scripted', 'gvf'], casesPerVariant: 40,
    maximumEpisodes: 160, exactReplayEpisodes: 160, maximumWallSeconds: 240,
    maximumPhysicsTicksIncludingReplay: cases.reduce((sum, world) => sum + Math.ceil(world.task.deadlineSeconds / 0.01) * 8, 0),
    currentWasmSha256: '457f5fe1088cc325ac12b6909f88647c6401265a92c8e243ad4678d9a934b7a7',
    preservedWasmSha256: '0885404c31c17c3b14e579bfa8d443a4c8efc7e791d73d52f90dfdb8fbfcb1fa',
    corpusSha256: historical.corpusSha256, scriptSha256: hash(fs.readFileSync(__filename)),
    parameterSearch: false, training: false, independentQualification: false, promotion: false,
    scope: 'One fixed analytic repair comparison; preserve all failures, compare default numerics exactly, no learned-policy claim'};
}

function perform(project, world, mode, version, deadline) {
  const runner = require(path.join(project, 'simulator/headless')).loadHeadless();
  const episode = runner.createEpisode();
  episode.reset(world.seed, world.configuration, world.task, {controlMode: mode, recordTransitions: false,
    policyIdentity: {kind: 'scripted', id: mode === 'gvf' ? `pedro-gvf-v${version}-experimental` : 'controller-only-baseline'}});
  const control = episode.sim.productionControl, step = control.step.bind(control), reset = control.reset.bind(control);
  const observations = {controlCalls: 0, resets: 0, peakWheelTargetYaw: 0, peakVolts: 0, clippedVoltageTicks: 0};
  control.reset = () => { observations.resets++; return reset(); };
  control.step = (...arguments_) => {
    const output = step(...arguments_);
    observations.controlCalls++;
    const targets = output.targets;
    observations.peakWheelTargetYaw = Math.max(observations.peakWheelTargetYaw,
      Math.abs((targets[0] + targets[1] - targets[2] - targets[3]) / (4 * episode.sim.rEff)));
    observations.peakVolts = Math.max(observations.peakVolts, ...output.volts.map(Math.abs));
    observations.clippedVoltageTicks += Number(output.volts.some(value => Math.abs(value) >= 12 - 1e-8));
    return output;
  };
  episode.applyAction({type: 'pose', ...world.task.goal});
  while (!episode.terminated && !episode.truncated) {
    if (Date.now() > deadline) throw Error('Predeclared remote wall budget exceeded; preserve partial work without retry');
    episode.step();
  }
  const report = {...episode.report(), identity: runner.identity, agent: 'Fixed analytic controller comparison; not learned'};
  const feedbackHeading = episode.sim.odom.theta * Math.PI / 180;
  observations.finalInternalResult = episode.sim.lastMotionResult;
  observations.finalSensorHeadingError = Math.abs(Math.atan2(Math.sin(world.task.goal.headingDeg * Math.PI / 180 - feedbackHeading),
    Math.cos(world.task.goal.headingDeg * Math.PI / 180 - feedbackHeading)));
  observations.finalTruthHeadingError = report.metrics.headingErrorRadians;
  assert.equal(observations.resets, 1, 'Exactly one persistent-controller reset per episode required');
  if (version === 2 && mode === 'gvf') assert(observations.peakWheelTargetYaw <= 1.500001, 'GVF wheel-target yaw limit exceeded');
  if (Date.now() > deadline) throw Error('Wall budget exceeded before exact replay');
  require(path.join(project, 'roboproof/motion')).replay(structuredClone(report));
  return {report, observations, exactReplayVerified: true};
}

function evaluate(directory, preservedRoot) {
  if (process.platform !== 'linux' || process.env.ROBOPROOF_APPROVED_REMOTE !== '1') throw Error('Approved remote Linux host required; no laptop campaign');
  if (!directory || !preservedRoot || path.resolve(preservedRoot) === root) throw Error('Unused result directory and distinct preserved source required');
  const protocol = declare(), deadline = Date.now() + protocol.maximumWallSeconds * 1000;
  const previous = require(path.resolve(preservedRoot, 'simulator/headless')).loadHeadless();
  const current = require('../simulator/headless').loadHeadless();
  assert.equal(previous.identity.controllerWasmSha256, protocol.preservedWasmSha256);
  assert.equal(current.identity.controllerWasmSha256, protocol.currentWasmSha256);
  assert.deepEqual(current.identity.runtime, previous.identity.runtime);
  fs.mkdirSync(directory, {recursive: false});
  write(directory, 'worlds.json', {schemaVersion: 1, kind: protocol.name, cases: protocol.cases});
  write(directory, 'protocol.json', {...protocol, identities: {previous: previous.identity, current: current.identity}});
  const records = [];
  for (const [index, world] of protocol.cases.entries()) {
    const record = {world, previous: {}, current: {}};
    for (const mode of protocol.modes) {
      record.previous[mode] = perform(path.resolve(preservedRoot), world, mode, 1, deadline);
      record.current[mode] = perform(root, world, mode, 2, deadline);
    }
    const numeric = result => {
      const {identity, agent, ...rest} = result.report;
      return rest;
    };
    assert.deepEqual(numeric(record.previous.scripted), numeric(record.current.scripted), `Default numerical regression: ${world.id}`);
    record.defaultNumericsIdentical = true;
    write(directory, `case-${String(index).padStart(2, '0')}.json`, record);
    records.push(record);
    process.stdout.write(JSON.stringify({id: world.id, controller: record.current.scripted.report.reason,
      previousGvf: record.previous.gvf.report.reason, currentGvf: record.current.gvf.report.reason}) + '\n');
  }
  const groups = {};
  for (const [name, subset] of [['developmentTurns', records.slice(0, 16)], ['historicalAll', records.slice(16)], ['all', records]]) {
    const reports = selector => subset.map(selector);
    groups[name] = {controller: aggregate(reports(row => row.current.scripted.report)),
      previousGvf: aggregate(reports(row => row.previous.gvf.report)), currentGvf: aggregate(reports(row => row.current.gvf.report)),
      controllerSuccessRegressions: subset.filter(row => row.current.scripted.report.reason === 'success' && row.current.gvf.report.reason !== 'success').map(row => row.world.id),
      previousGvfSuccessRegressions: subset.filter(row => row.previous.gvf.report.reason === 'success' && row.current.gvf.report.reason !== 'success').map(row => row.world.id),
      contactRegressions: subset.filter(row => row.current.gvf.report.metrics.contactSeconds > row.current.scripted.report.metrics.contactSeconds).map(row => row.world.id)};
  }
  const summary = {schemaVersion: 1, groups, defaultNumericsIdenticalOnAll40: true, exactReplayEpisodes: 160,
    casesCompleted: records.length, learnedImprovementVerified: false, independentQualificationPerformed: false, promotion: false};
  write(directory, 'summary.json', summary);
  return summary;
}

if (require.main === module) {
  if (process.argv[2] === '--declare') process.stdout.write(JSON.stringify(declare()) + '\n');
  else process.stdout.write(JSON.stringify(evaluate(process.argv[2], process.argv[3])) + '\n');
}

module.exports = {declare, evaluate};
