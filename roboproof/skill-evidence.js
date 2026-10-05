'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {normalizeTask, PARAMETERS} = require('../simulator/motion');
const {suite} = require('./motion-evaluation');
const MotionLearner = require('./motion-learner');
const {sha256} = require('./agent-skills');
const MAX_EVIDENCE_BYTES = 10 * 1024;
const metricNames = ['positionErrorMeters', 'headingErrorRadians', 'speedMetersPerSecond',
  'yawRateRadiansPerSecond', 'elapsedSeconds', 'effortProxyVAs', 'contactSeconds', 'pathLengthMeters'];

function bounded(value) {
  if (Buffer.byteLength(JSON.stringify(value)) > MAX_EVIDENCE_BYTES) throw Error('Skill evidence exceeds context budget');
  return value;
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
    return {tick: entry.tick, sensorPoseMetersRadians: entry.observation?.pose,
      sensorVelocity: entry.observation?.velocity, truthPoseInchesDegrees: entry.evaluator?.poseInchesDegrees,
      truthVelocity: entry.evaluator?.velocity, motorVolts: entry.evaluator?.motorVolts};
  });
  return bounded({source, sessionId: record.id, reportSha256: sha256(JSON.stringify(report)),
    task, seed: report.seed, configuration, reason: report.reason, metrics, finalStateChecks,
    settlingRequiredConsecutiveTicks: 15, recordedExactReplayVerified: run.exactReplayVerified === true,
    identity: report.identity, policyIdentity: report.options?.policyIdentity,
    telemetry: {recordedTicks: transitions.length, sampledOnly: true, samples: telemetrySamples},
    interpretation: 'Saved simulator measurements, not a fresh replay, causal proof, physical validation or improvement benchmark'});
}

function learningSummary(directory) {
  const saved = MotionLearner.status(directory);
  const evaluation = saved.evaluation;
  const evaluationMatchesCurrentCheckpoints = Boolean(saved.available && saved.status !== 'running' && evaluation &&
    evaluation.corpusSha256 === suite(2).corpusSha256 &&
    JSON.stringify(evaluation.policyHashes) === JSON.stringify(saved.models.map(row => row.policySha256)));
  const comparison = summary => Object.fromEntries(['count', 'success',
    'meanElapsedSecondsAllWorlds', 'meanEffortProxyVAsAllWorlds']
    .filter(name => Object.hasOwn(summary, name)).map(name => [name, summary[name]]));
  return bounded({available: saved.available, status: saved.status, message: saved.message, runId: saved.runId,
    motionPolicyTrained: saved.motionPolicyTrained === true,
    learnedImprovementVerified: saved.learnedImprovementVerified === true,
    evaluationMatchesCurrentCheckpoints,
    models: (saved.models ?? []).map(row => ({seed: row.seed, steps: row.steps, updates: row.updates,
      trained: row.trained, actorWeightsChanged: row.actorWeightsChanged, policySha256: row.policySha256})),
    evaluation: evaluationMatchesCurrentCheckpoints ? {corpusSha256: evaluation.corpusSha256,
      evaluatedAt: evaluation.evaluatedAt, independentTrainingSeedsRequirementMet: evaluation.independentTrainingSeedsRequirementMet,
      models: evaluation.models.map(row => ({seed: row.seed, baseline: comparison(row.baseline), untrained: comparison(row.untrained),
        learned: comparison(row.learned), acceptance: row.acceptance,
        failedWorldIds: row.results.filter(world => world.learned.reason !== 'success').map(world => world.id)}))} : null,
    interpretation: 'Read-only saved checkpoint/evaluation summary. No training, inference, policy promotion, hardware or AMD workload executed'});
}

module.exports = {robotProfile, motionEvidence, learningSummary, MAX_EVIDENCE_BYTES};
