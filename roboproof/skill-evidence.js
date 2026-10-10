'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {normalizeTask, PARAMETERS} = require('../simulator/motion');
const {suite} = require('./motion-evaluation');
const MotionLearner = require('./motion-learner');
const {sha256} = require('./agent-skills');
const {loadHeadless} = require('../simulator/headless');
const {isDeepStrictEqual} = require('node:util');
const MAX_EVIDENCE_BYTES = 10 * 1024;
const metricNames = ['positionErrorMeters', 'headingErrorRadians', 'speedMetersPerSecond',
  'yawRateRadiansPerSecond', 'elapsedSeconds', 'effortProxyVAs', 'contactSeconds', 'pathLengthMeters'];

function bounded(value) {
  if (Buffer.byteLength(JSON.stringify(value)) > MAX_EVIDENCE_BYTES) throw Error('Skill evidence exceeds context budget');
  return value;
}

function knownIdentity(value, template = loadHeadless().identity, trail = []) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !Object.hasOwn(template, key))) throw Error('Invalid saved motion identity');
  const result = {};
  for (const [key, expected] of Object.entries(template)) {
    const actual = value[key];
    if (expected && typeof expected === 'object') result[key] = knownIdentity(actual, expected, [...trail, key]);
    else {
      if (typeof actual !== typeof expected || (typeof actual === 'string' && (actual.length > 512 || /[\r\n]/.test(actual)))) throw Error('Invalid saved motion identity');
      if (typeof expected === 'string' && /^[a-f0-9]{64}$/.test(expected) && !/^[a-f0-9]{64}$/.test(actual)) throw Error('Invalid saved motion source hash');
      if (trail.length === 0 && !/^[a-f0-9]{64}$/.test(expected) && actual !== expected) throw Error('Invalid saved motion engine identity');
      if (trail.at(-1) === 'runtime' && (typeof actual !== 'string' || !/^[A-Za-z0-9._-]{1,100}$/.test(actual))) throw Error('Invalid saved motion runtime');
      result[key] = actual;
    }
  }
  return result;
}

function policyIdentity(value) {
  if (!value || typeof value !== 'object' || Object.keys(value).some(key => !['kind', 'id', 'sha256'].includes(key)) ||
      !['scripted', 'untrained', 'learned'].includes(value.kind) || typeof value.id !== 'string' ||
      !/^[A-Za-z0-9._:-]{1,200}$/.test(value.id) || (value.sha256 !== undefined && !/^[a-f0-9]{64}$/.test(value.sha256))) throw Error('Invalid policyIdentity');
  return {kind: value.kind, id: value.id, ...(value.sha256 === undefined ? {} : {sha256: value.sha256})};
}

function vector(value, width) {
  if (!Array.isArray(value) || value.length !== width || !Array.from(value).every(Number.isFinite)) throw Error('Invalid saved motion telemetry');
  return [...value];
}

function acceptanceSnapshot(value) {
  const keys = ['passed', 'regressions', 'contactRegressions', 'meanTimeImprovementFraction', 'meanEffortImprovementFraction', 'rule'];
  if (!value || typeof value !== 'object' || Object.keys(value).some(key => !keys.includes(key)) || typeof value.passed !== 'boolean' ||
      !['regressions', 'contactRegressions'].every(key => Array.isArray(value[key]) && value[key].length <= 1024 &&
        value[key].every(id => typeof id === 'string' && /^[A-Za-z0-9._:-]{1,200}$/.test(id))) ||
      !['meanTimeImprovementFraction', 'meanEffortImprovementFraction'].every(key => Number.isFinite(value[key])) ||
      typeof value.rule !== 'string' || value.rule.length > 500) throw Error('Invalid saved learning acceptance');
  return {...value, regressions: [...value.regressions], contactRegressions: [...value.contactRegressions]};
}

function robotProfile() {
  const filename = path.join(__dirname, '../config/robot.json');
  if (fs.statSync(filename).size > 16 * 1024) throw Error('Robot configuration exceeds budget');
  const bytes = fs.readFileSync(filename);
  const config = JSON.parse(bytes.toString('utf8'));
  const keys = ['ports', 'drive', 'cartridgeRpm', 'wheelDiameterIn', 'widthIn', 'wheelbaseIn',
    'externalGearRatio', 'imuPort', 'linear', 'angular'];
  return bounded({source: 'config/robot.json', sourceSha256: sha256(bytes),
    configuration: Object.fromEntries(keys.map(key => [key, config[key]])),
    controlPath: 'original Nationals Simulator -> iraLIB C++ LTV-LQR -> wheel PI/feedforward',
    interpretation: 'Nominal configuration snapshot; configured PID values are not proof of active LQR gains or hardware calibration',
    hardwareExecutionEnabled: false, cloudExecutionEnabled: false});
}

function motionEvidence(record, source) {
  const run = source === 'nemotron' ? record.run : record;
  const report = run?.report;
  if (!['motion', 'nemotron'].includes(source) || !report || report.taskType !== 'reach-pose-benchmark' ||
      !Number.isInteger(report.ticks) || report.ticks < 0 || report.ticks > 6000 || report.fixedDt !== 0.01 ||
      !['success', 'time_limit', 'controller_or_sensor_fault', 'stale_policy_action', 'stopped'].includes(report.reason)) {
    throw Error('Select a completed original-Simulator movement experiment');
  }
  const task = normalizeTask(report.task);
  const metrics = Object.fromEntries(metricNames.map(name => [name, report.metrics?.[name]]));
  if (!Object.values(metrics).every(value => Number.isFinite(value) && value >= 0)) throw Error('Invalid saved motion metrics');
  const configuration = Object.fromEntries(Object.keys(PARAMETERS).map(name => [name, report.configuration?.[name]]));
  if (!Object.values(configuration).every(Number.isFinite)) throw Error('Invalid saved motion configuration');
  const thresholds = {positionErrorMeters: 0.02032, headingErrorRadians: 0.035,
    speedMetersPerSecond: 0.0254, yawRateRadiansPerSecond: 0.0873};
  const finalStateChecks = Object.fromEntries(Object.entries(thresholds).map(([name, threshold]) =>
    [name, {value: metrics[name], threshold, belowThreshold: metrics[name] < threshold}]));
  const transitions = report.transitions ?? [];
  if (!Array.isArray(transitions) || transitions.length > 6000) throw Error('Invalid saved motion telemetry');
  const indices = [...new Set(Array.from({length: Math.min(5, transitions.length)}, (_, index) =>
    Math.round(index * (transitions.length - 1) / Math.max(1, Math.min(5, transitions.length) - 1))))];
  const telemetrySamples = indices.map(index => {
    const entry = transitions[index];
    if (!Number.isInteger(entry?.tick) || entry.tick < 0 || entry.tick >= report.ticks) throw Error('Invalid saved motion telemetry tick');
    return {tick: entry.tick, sensorPoseMetersRadians: vector(entry.observation?.pose, 3),
      sensorVelocity: vector(entry.observation?.velocity, 3), truthPoseInchesDegrees: vector(entry.evaluator?.poseInchesDegrees, 3),
      truthVelocity: vector(entry.evaluator?.velocity, 3), motorVolts: vector(entry.evaluator?.motorVolts, 4)};
  });
  return bounded({source, sessionId: record.id, reportSha256: sha256(JSON.stringify(report)),
    task, seed: report.seed, configuration, reason: report.reason, metrics, finalStateChecks,
    settlingRequiredConsecutiveTicks: 15, recordedExactReplayVerified: run.exactReplayVerified === true,
    identity: knownIdentity(report.identity), policyIdentity: policyIdentity(report.options?.policyIdentity),
    telemetry: {recordedTicks: transitions.length, sampledOnly: true, samples: telemetrySamples},
    interpretation: 'Saved simulator measurements, not a fresh replay, causal proof, physical validation or improvement benchmark'});
}

function learningSummary(directory) {
  const saved = MotionLearner.status(directory);
  const evaluation = saved.evaluation;
  const evaluationMatchesCurrentCheckpoints = Boolean(saved.available && saved.status !== 'running' && evaluation &&
    typeof saved.runId === 'string' && saved.identity?.canonicalSimulator === true &&
    evaluation.runId === saved.runId && isDeepStrictEqual(evaluation.identity, saved.identity) &&
    evaluation.corpusSha256 === suite(2).corpusSha256 &&
    JSON.stringify(evaluation.policyHashes) === JSON.stringify(saved.models.map(row => row.policySha256)));
  const comparison = summary => Object.fromEntries(['count', 'success',
    'meanElapsedSecondsAllWorlds', 'meanEffortProxyVAsAllWorlds']
    .filter(name => Object.hasOwn(summary, name)).map(name => [name, summary[name]]));
  return bounded({available: saved.available, status: saved.status, message: saved.message, runId: saved.runId,
    motionPolicyTrained: saved.motionPolicyTrained === true,
    learnedImprovementVerified: evaluationMatchesCurrentCheckpoints && saved.learnedImprovementVerified === true,
    evaluationMatchesCurrentCheckpoints,
    models: (saved.models ?? []).map(row => ({seed: row.seed, steps: row.steps, updates: row.updates,
      trained: row.trained, actorWeightsChanged: row.actorWeightsChanged, policySha256: row.policySha256})),
    evaluation: evaluationMatchesCurrentCheckpoints ? {corpusSha256: evaluation.corpusSha256,
      evaluatedAt: evaluation.evaluatedAt, independentTrainingSeedsRequirementMet: evaluation.independentTrainingSeedsRequirementMet,
      models: evaluation.models.map(row => ({seed: row.seed, baseline: comparison(row.baseline), untrained: comparison(row.untrained),
        learned: comparison(row.learned), acceptance: acceptanceSnapshot(row.acceptance),
        failedWorldIds: row.results.filter(world => world.learned.reason !== 'success').map(world => world.id)}))} : null,
    interpretation: 'Read-only saved checkpoint/evaluation summary. No training, inference, policy promotion, hardware or AMD workload executed'});
}

module.exports = {robotProfile, motionEvidence, learningSummary, MAX_EVIDENCE_BYTES};
