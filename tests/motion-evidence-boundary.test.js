'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const Motion = require('../roboproof/motion');
const MotionLearner = require('../roboproof/motion-learner');
const {planTask} = require('../roboproof/nemotron');
const {suite} = require('../roboproof/motion-evaluation');
const {DEFAULT_TASK, PARAMETERS} = require('../simulator/motion');
const {robotProfile, motionEvidence, learningSummary, MAX_EVIDENCE_BYTES} = require('../roboproof/skill-evidence');

const sessionId = '11111111-1111-4111-8111-111111111111';
const runId = '22222222-2222-4222-8222-222222222222';
const secretMarker = 'fixture-private-token-not-model-evidence';
const sourceMarker = 'fixture-source-text-not-model-evidence';
const telemetryMarker = 'fixture-whole-telemetry-not-model-evidence';
const digest = value => createHash('sha256').update(value).digest('hex');
const metricNames = ['positionErrorMeters', 'headingErrorRadians', 'speedMetersPerSecond',
  'yawRateRadiansPerSecond', 'elapsedSeconds', 'effortProxyVAs', 'contactSeconds', 'pathLengthMeters'];
const thresholds = {positionErrorMeters: 0.02032, headingErrorRadians: 0.035,
  speedMetersPerSecond: 0.0254, yawRateRadiansPerSecond: 0.0873};
let savedReportFixture;

function selectedMotion() {
  savedReportFixture ??= Motion.baseline(42, {...DEFAULT_TASK, deadlineSeconds: 0.21}, {},
    {recordTransitions: true});
  return {schemaVersion: 1, id: sessionId, exactReplayVerified: true,
    report: structuredClone(savedReportFixture)};
}

function savedLearning(record = selectedMotion()) {
  const corpus = suite(2);
  const models = [42, 43, 44].map(seed => ({seed, steps: 40, updates: 1, trained: true,
    actorWeightsChanged: true, policySha256: digest(`fixture-checkpoint-${seed}`)}));
  const comparison = success => ({count: corpus.cases.length, success,
    meanElapsedSecondsAllWorlds: 8, meanEffortProxyVAsAllWorlds: 12});
  return {available: true, status: 'completed', runId, identity: structuredClone(record.report.identity),
    motionPolicyTrained: true, learnedImprovementVerified: false, models,
    evaluation: {runId, identity: structuredClone(record.report.identity),
      corpusSha256: corpus.corpusSha256, policyHashes: models.map(row => row.policySha256),
      evaluatedAt: '2026-10-06T00:00:00.000Z', independentTrainingSeedsRequirementMet: true,
      models: models.map(row => ({seed: row.seed, policySha256: row.policySha256,
        baseline: comparison(6), untrained: comparison(6),
        learned: {...comparison(6), meanElapsedSecondsAllWorlds: 8.8, meanEffortProxyVAsAllWorlds: 14.4},
        acceptance: {passed: false, regressions: [corpus.cases[0].id],
          contactRegressions: [corpus.cases[1].id], meanTimeImprovementFraction: -0.1,
          meanEffortImprovementFraction: -0.2,
          rule: 'Higher success than controller and untrained policy; zero success/contact regressions; >=5% mean time or effort gain. All frozen worlds included.'},
        results: corpus.cases.map((world, index) => ({id: world.id,
          baseline: {reason: index < 6 ? 'success' : 'time_limit', metrics: {contactSeconds: 0}},
          untrained: {reason: index < 6 ? 'success' : 'time_limit'},
          learned: {reason: index > 0 && index <= 6 ? 'success' : 'time_limit',
            metrics: {contactSeconds: index === 1 ? 0.01 : 0}}}))}))}};
}

function assertExcluded(value, markers = [secretMarker, sourceMarker, telemetryMarker]) {
  const encoded = JSON.stringify(value);
  for (const marker of markers) assert.equal(encoded.includes(marker), false, `Leaked ${marker}`);
}

function assertSafeBoundary(build, errorPattern) {
  let evidence;
  try { evidence = build(); }
  catch (error) {
    assert.match(error.message, errorPattern);
    return;
  }
  assertExcluded(evidence);
}

function forbidExecution(context) {
  const unexpected = () => { throw Error('Read-only evidence attempted execution or a filesystem write'); };
  for (const name of ['baseline', 'runPolicy', 'replay']) context.mock.method(Motion, name, unexpected);
  for (const name of ['runLearned', 'evaluateLearned']) context.mock.method(MotionLearner, name, unexpected);
  for (const name of ['writeFileSync', 'appendFileSync', 'renameSync', 'mkdirSync', 'unlinkSync']) {
    context.mock.method(fs, name, unexpected);
  }
  context.mock.method(globalThis, 'fetch', unexpected);
}

test('robot evidence hashes the actual configuration bytes and exposes only reviewed nominal keys', () => {
  const filename = path.join(__dirname, '../config/robot.json');
  const bytes = fs.readFileSync(filename);
  const configuration = JSON.parse(bytes.toString('utf8'));
  const keys = ['ports', 'drive', 'cartridgeRpm', 'wheelDiameterIn', 'widthIn', 'wheelbaseIn',
    'externalGearRatio', 'imuPort', 'linear', 'angular'];
  const evidence = robotProfile();
  assert.equal(evidence.sourceSha256, digest(bytes));
  assert.deepEqual(evidence.configuration, Object.fromEntries(keys.map(key => [key, configuration[key]])));
  assert.equal(evidence.configuration.note, undefined);
  assert.equal(evidence.hardwareExecutionEnabled, false);
  assert.equal(evidence.cloudExecutionEnabled, false);
  assert.match(evidence.interpretation, /not proof of active LQR gains or hardware calibration/);
  assert(Buffer.byteLength(JSON.stringify(evidence)) <= MAX_EVIDENCE_BYTES);
});

test('selected motion and nested Nemotron evidence bind the correct saved report, identity and replay flag', () => {
  const record = selectedMotion();
  const before = structuredClone(record);
  const evidence = motionEvidence(record, 'motion');
  assert.equal(evidence.source, 'motion');
  assert.equal(evidence.sessionId, sessionId);
  assert.equal(evidence.reportSha256, digest(JSON.stringify(record.report)));
  assert.deepEqual(evidence.identity, record.report.identity);
  assert.deepEqual(evidence.policyIdentity, record.report.options.policyIdentity);
  assert.deepEqual(evidence.task, record.report.task);
  assert.equal(evidence.seed, record.report.seed);
  assert.equal(evidence.reason, record.report.reason);
  assert.equal(evidence.recordedExactReplayVerified, true);
  assert.deepEqual(record, before);
  const nested = {id: runId, report: {...record.report, seed: 99}, exactReplayVerified: true,
    run: {...record, exactReplayVerified: false}};
  const nestedEvidence = motionEvidence(nested, 'nemotron');
  assert.equal(nestedEvidence.sessionId, runId);
  assert.equal(nestedEvidence.source, 'nemotron');
  assert.equal(nestedEvidence.seed, record.report.seed);
  assert.equal(nestedEvidence.reportSha256, evidence.reportSha256);
  assert.equal(nestedEvidence.recordedExactReplayVerified, false);
  for (const replayFlag of [undefined, false, 'true', 1, {}]) {
    record.exactReplayVerified = replayFlag;
    assert.equal(motionEvidence(record, 'motion').recordedExactReplayVerified, false);
  }
  assert.match(evidence.interpretation, /not a fresh replay/);
});

test('the report hash covers unsampled telemetry and policy/source identity, not outer session metadata', () => {
  const record = selectedMotion();
  const original = motionEvidence(record, 'motion');
  const changes = [
    report => { report.metrics.effortProxyVAs += 1; },
    report => { report.transitions[1].evaluator.motorVolts[0] += 0.01; },
    report => { report.identity.sources['engine.js'] = digest('different-saved-engine'); },
    report => { report.options.policyIdentity.id = 'different-saved-policy'; }
  ];
  for (const change of changes) {
    const changed = structuredClone(record);
    change(changed.report);
    const evidence = motionEvidence(changed, 'motion');
    assert.equal(evidence.reportSha256, digest(JSON.stringify(changed.report)));
    assert.notEqual(evidence.reportSha256, original.reportSha256);
  }
  record.savedAt = '2026-10-06T01:00:00.000Z';
  record.privateMetadata = secretMarker;
  assert.equal(motionEvidence(record, 'motion').reportSha256, original.reportSha256);
  const unsampled = structuredClone(record);
  unsampled.report.transitions[1].evaluator.motorVolts[0] += 0.01;
  assert.deepEqual(motionEvidence(unsampled, 'motion').telemetry.samples, original.telemetry.samples);
});

test('saved learned policy identity retains its checkpoint hash without claiming measured improvement', () => {
  const record = selectedMotion();
  record.report.options.policyIdentity = {kind: 'learned', id: 'fixture-saved-checkpoint',
    sha256: digest('fixture-saved-policy-not-inference')};
  const evidence = motionEvidence(record, 'motion');
  assert.deepEqual(evidence.policyIdentity, record.report.options.policyIdentity);
  assert.equal(evidence.reportSha256, digest(JSON.stringify(record.report)));
  assert.equal(Object.hasOwn(evidence, 'learnedImprovementVerified'), false);
  assert.match(evidence.interpretation, /not a fresh replay, causal proof, physical validation or improvement benchmark/);
});

test('compact selected-run evidence excludes source text, secrets, actions and whole transitions', () => {
  const record = selectedMotion();
  record.privateMetadata = secretMarker;
  record.report.sourceText = sourceMarker;
  record.report.metrics.privateToken = secretMarker;
  record.report.configuration.privateToken = secretMarker;
  record.report.options.privateToken = secretMarker;
  record.report.actions.push({tick: 0, action: {privateToken: secretMarker}});
  for (const entry of record.report.transitions) {
    entry.sourceText = sourceMarker;
    entry.observation.privateToken = secretMarker;
    entry.nextObservation.privateToken = secretMarker;
    entry.info.wholeTelemetry = telemetryMarker;
    entry.evaluator.wholeTelemetry = telemetryMarker;
  }
  const evidence = motionEvidence(record, 'motion');
  assertExcluded(evidence);
  assert.deepEqual(Object.keys(evidence.metrics), metricNames);
  assert.deepEqual(Object.keys(evidence.configuration), Object.keys(PARAMETERS));
  for (const name of ['actions', 'transitions', 'sourceText', 'options', 'privateMetadata']) {
    assert.equal(Object.hasOwn(evidence, name), false);
  }
  assert.equal(evidence.telemetry.recordedTicks, record.report.transitions.length);
  assert.equal(evidence.telemetry.sampledOnly, true);
  assert.equal(evidence.telemetry.samples.length, 5);
  assert.deepEqual(evidence.telemetry.samples.map(sample => sample.tick), [0, 5, 10, 15, 20]);
  for (const sample of evidence.telemetry.samples) {
    const entry = record.report.transitions[sample.tick];
    assert.deepEqual(sample, {tick: entry.tick, sensorPoseMetersRadians: entry.observation.pose,
      sensorVelocity: entry.observation.velocity, truthPoseInchesDegrees: entry.evaluator.poseInchesDegrees,
      truthVelocity: entry.evaluator.velocity, motorVolts: entry.evaluator.motorVolts});
  }
  assert(Buffer.byteLength(JSON.stringify(evidence)) <= MAX_EVIDENCE_BYTES);
});

test('the diagnosis fixture receives only compact selected evidence and cannot turn commentary into execution', async context => {
  const record = selectedMotion();
  record.privateToken = secretMarker;
  record.report.sourceText = sourceMarker;
  record.report.transitions[1].info.wholeTelemetry = telemetryMarker;
  const evidence = motionEvidence(record, 'motion');
  const response = (name, argumentsValue = {}) => ({choices: [{finish_reason: 'tool_calls',
    message: {role: 'assistant', tool_calls: [{id: name, type: 'function',
      function: {name, arguments: JSON.stringify(argumentsValue)}}]}}]});
  const responses = [response('load_skill', {name: 'diagnose-motion'}), response('get_saved_motion_evidence'),
    response('finish_analysis', {message: 'Fixture commentary only; recorded timeout, causality unverified.'})];
  const calls = [];
  const client = {metadata: {provider: 'fixture-not-live-inference', cloudEnabled: false},
    check: async () => ({}), complete: async (messages, signal, tools) => {
      calls.push({messages: structuredClone(messages), tools: structuredClone(tools)});
      assert(responses.length, 'Fixture response exhausted');
      return responses.shift();
    }};
  forbidExecution(context);
  const result = await planTask('Explain only this selected saved result', client, {evidence});
  assert.equal(calls.length, 3);
  assert.equal(responses.length, 0);
  assertExcluded(calls);
  const toolEvidence = calls[2].messages.find(message => message.role === 'tool' &&
    message.tool_call_id === 'get_saved_motion_evidence');
  assert(toolEvidence);
  assert.deepEqual(JSON.parse(toolEvidence.content), evidence);
  for (const call of calls) for (const entry of call.tools) {
    assert(['load_skill', 'ask_clarification', 'get_saved_motion_evidence', 'finish_analysis'].includes(entry.function.name));
  }
  assert.equal(result.status, 'analyzed');
  assert.equal(result.commentaryVerified, false);
  assert.deepEqual(result.analysisEvidence.motion, evidence);
  assert.equal(result.run, undefined);
  assert.equal(result.task, undefined);
  assert.match(result.execution, /not run/);
});

test('unreviewed nested engine and policy identity payloads cannot cross the model evidence boundary', () => {
  for (const inject of [
    report => { report.identity.sourceText = sourceMarker; },
    report => { report.identity.runtime.privateToken = secretMarker; },
    report => { report.identity.sources['engine.js'] = sourceMarker; },
    report => { report.options.policyIdentity.privateToken = secretMarker; }
  ]) {
    const record = selectedMotion();
    inject(record.report);
    assertSafeBoundary(() => motionEvidence(record, 'motion'), /(?:Invalid|Unsupported) (?:saved motion|policyIdentity)/i);
  }
});

test('sample vectors cannot smuggle arbitrary objects, source text or entire telemetry arrays', () => {
  for (const [container, key] of [['observation', 'pose'], ['observation', 'velocity'],
    ['evaluator', 'poseInchesDegrees'], ['evaluator', 'velocity'], ['evaluator', 'motorVolts']]) {
    for (const payload of [{sourceText: sourceMarker, privateToken: secretMarker},
      [0, 0, 0, {wholeTelemetry: telemetryMarker}]]) {
      const record = selectedMotion();
      record.report.transitions[0][container][key] = payload;
      assertSafeBoundary(() => motionEvidence(record, 'motion'), /(?:Invalid|Unsupported) saved motion telemetry/i);
    }
  }
});

test('sparse samples cannot prove consecutive settling or a causal explanation even with matching endpoints', () => {
  const record = selectedMotion();
  const alternate = structuredClone(record);
  const sampledTicks = new Set([0, 5, 10, 15, 20]);
  for (const report of [record.report, alternate.report]) {
    for (const name of Object.keys(thresholds)) report.metrics[name] = 0;
    for (const entry of report.transitions) {
      entry.observation.pose = [0, 24 * 0.0254, 0];
      entry.observation.velocity = [0, 0, 0];
      entry.evaluator.poseInchesDegrees = [0, 24, 0];
      entry.evaluator.velocity = [0, 0, 0];
      entry.evaluator.motorVolts = [0, 0, 0, 0];
    }
  }
  for (const entry of alternate.report.transitions) if (!sampledTicks.has(entry.tick)) {
    entry.evaluator.poseInchesDegrees = [0, 0, 0];
    entry.evaluator.velocity = [1, 0, 0];
    entry.evaluator.motorVolts = [12, -12, 12, -12];
  }
  const evidence = motionEvidence(record, 'motion');
  const alternateEvidence = motionEvidence(alternate, 'motion');
  assert.deepEqual(evidence.telemetry, alternateEvidence.telemetry);
  assert.deepEqual(evidence.finalStateChecks, alternateEvidence.finalStateChecks);
  assert.notEqual(evidence.reportSha256, alternateEvidence.reportSha256);
  assert(Object.values(evidence.finalStateChecks).every(check => check.belowThreshold));
  assert.equal(evidence.reason, 'time_limit');
  assert.equal(evidence.settlingRequiredConsecutiveTicks, 15);
  assert.equal(evidence.telemetry.sampledOnly, true);
  for (const snapshot of [evidence, alternateEvidence]) {
    assert.match(snapshot.interpretation, /not a fresh replay, causal proof, physical validation or improvement benchmark/);
    for (const name of ['settledTicks', 'settlingVerified', 'dwellVerified', 'causalityVerified',
      'oscillationVerified', 'saturationVerified', 'localizationCause', 'learnedImprovementVerified']) {
      assert.equal(Object.hasOwn(snapshot, name), false);
    }
  }
});

test('settling checks use all four strict final-state thresholds rather than position alone', () => {
  for (const [name, threshold] of Object.entries(thresholds)) {
    for (const value of [0, threshold, threshold * 2]) {
      const record = selectedMotion();
      for (const metric of Object.keys(thresholds)) record.report.metrics[metric] = 0;
      record.report.metrics[name] = value;
      const evidence = motionEvidence(record, 'motion');
      assert.deepEqual(evidence.finalStateChecks[name], {value, threshold, belowThreshold: value < threshold});
      assert.equal(evidence.reason, record.report.reason);
    }
  }
});

test('missing telemetry stays absent and large recorded telemetry remains a five-sample byte-bounded projection', () => {
  const record = selectedMotion();
  const absent = structuredClone(record);
  delete absent.report.transitions;
  assert.deepEqual(motionEvidence(absent, 'motion').telemetry, {recordedTicks: 0, sampledOnly: true, samples: []});
  for (const count of [0, 1, 2, 4, 5, 6000]) {
    const report = structuredClone(record.report);
    report.ticks = count;
    report.transitions = Array.from({length: count}, (_, tick) => ({...record.report.transitions[0], tick}));
    const evidence = motionEvidence({id: sessionId, report}, 'motion');
    assert.equal(evidence.telemetry.recordedTicks, count);
    assert.equal(evidence.telemetry.samples.length, Math.min(5, count));
    assert.equal(new Set(evidence.telemetry.samples.map(sample => sample.tick)).size, Math.min(5, count));
    if (count) {
      assert.equal(evidence.telemetry.samples[0].tick, 0);
      assert.equal(evidence.telemetry.samples.at(-1).tick, count - 1);
    }
    assert(Buffer.byteLength(JSON.stringify(evidence)) <= MAX_EVIDENCE_BYTES);
  }
});

test('unsupported selections, incomplete reports and invalid saved metrics fail closed', () => {
  const record = selectedMotion();
  for (const source of [undefined, 'file', 'learning', '../secret']) {
    assert.throws(() => motionEvidence(record, source), /Select a completed/);
  }
  assert.throws(() => motionEvidence({id: sessionId}, 'motion'), /Select a completed/);
  assert.throws(() => motionEvidence(record, 'nemotron'), /Select a completed/);
  for (const change of [report => { report.reason = 'running'; }, report => { report.fixedDt = 0.02; },
    report => { report.ticks = 6001; }, report => { report.taskType = 'game'; }]) {
    const report = structuredClone(record.report);
    change(report);
    assert.throws(() => motionEvidence({id: sessionId, report}, 'motion'), /Select a completed/);
  }
  for (const name of metricNames) for (const value of [undefined, null, -1, NaN, Infinity, '0']) {
    const report = structuredClone(record.report);
    report.metrics[name] = value;
    assert.throws(() => motionEvidence({id: sessionId, report}, 'motion'), /Invalid saved motion metrics/);
  }
  const oversized = structuredClone(record.report);
  oversized.identity.engine = 'Я'.repeat(MAX_EVIDENCE_BYTES);
  assert.throws(() => motionEvidence({id: sessionId, report: oversized}, 'motion'), /budget|Invalid|Unsupported/i);
});

test('current learning evidence retains every seed, negative all-world comparisons and the complete acceptance gate', context => {
  const saved = savedLearning();
  const before = structuredClone(saved);
  const directory = path.join(__dirname, 'fixtures', 'not-a-training-run');
  const status = context.mock.method(MotionLearner, 'status', requested => {
    assert.equal(requested, directory);
    return saved;
  });
  forbidExecution(context);
  const evidence = learningSummary(directory);
  assert.equal(status.mock.callCount(), 1);
  assert.equal(evidence.runId, runId);
  assert.equal(evidence.motionPolicyTrained, true);
  assert.equal(evidence.learnedImprovementVerified, false);
  assert.equal(evidence.evaluationMatchesCurrentCheckpoints, true);
  assert.equal(evidence.evaluation.corpusSha256, suite(2).corpusSha256);
  assert.equal(evidence.evaluation.independentTrainingSeedsRequirementMet, true);
  assert.deepEqual(evidence.models, saved.models);
  assert.deepEqual(evidence.evaluation.models.map(row => row.seed), [42, 43, 44]);
  for (const [index, row] of evidence.evaluation.models.entries()) {
    const original = saved.evaluation.models[index];
    for (const name of ['baseline', 'untrained', 'learned', 'acceptance']) assert.deepEqual(row[name], original[name]);
    assert.deepEqual(row.failedWorldIds, [suite(2).cases[0].id, ...suite(2).cases.slice(7).map(world => world.id)]);
    assert.equal(Object.hasOwn(row, 'results'), false);
  }
  assert.deepEqual(saved, before);
  assert.match(evidence.interpretation, /Read-only saved checkpoint\/evaluation summary/);
  assert.match(evidence.interpretation, /No training, inference, policy promotion, hardware or AMD workload executed/);
  assert(Buffer.byteLength(JSON.stringify(evidence)) <= MAX_EVIDENCE_BYTES);
});

test('stale checkpoint sets, corpus hashes and running or unavailable learners never expose comparison counts', context => {
  let saved = savedLearning();
  context.mock.method(MotionLearner, 'status', () => saved);
  for (const change of [
    snapshot => { snapshot.evaluation.policyHashes[0] = digest('stale-checkpoint'); },
    snapshot => { snapshot.evaluation.policyHashes.reverse(); },
    snapshot => { snapshot.evaluation.policyHashes.pop(); },
    snapshot => { snapshot.models.pop(); },
    snapshot => { snapshot.evaluation.corpusSha256 = suite(1).corpusSha256; },
    snapshot => { snapshot.status = 'running'; },
    snapshot => { snapshot.available = false; },
    snapshot => { snapshot.evaluation = null; }
  ]) {
    saved = savedLearning();
    change(saved);
    const evidence = learningSummary('fixture-directory-not-executed');
    assert.equal(evidence.evaluationMatchesCurrentCheckpoints, false);
    assert.equal(evidence.evaluation, null);
    assert.equal(evidence.learnedImprovementVerified, false);
    assert.equal(evidence.models.length, saved.models.length);
  }
});

test('matching checkpoint and corpus hashes do not make another run or engine identity a current comparison', context => {
  let saved = savedLearning();
  context.mock.method(MotionLearner, 'status', () => saved);
  for (const change of [
    snapshot => { snapshot.evaluation.runId = sessionId; },
    snapshot => { snapshot.evaluation.identity.sources['engine.js'] = digest('stale-evaluation-engine'); },
    snapshot => { snapshot.evaluation.identity.controllerWasmSha256 = digest('stale-evaluation-controller'); }
  ]) {
    saved = savedLearning();
    saved.learnedImprovementVerified = true;
    change(saved);
    const evidence = learningSummary('fixture-directory-not-executed');
    assert.equal(evidence.evaluationMatchesCurrentCheckpoints, false);
    assert.equal(evidence.evaluation, null);
    assert.equal(evidence.learnedImprovementVerified, false);
  }
});

test('learning projections exclude checkpoint paths, actor weights, raw results and unreviewed comparison fields', context => {
  const saved = savedLearning();
  saved.sourceText = sourceMarker;
  saved.privateToken = secretMarker;
  for (const model of saved.models) {
    model.policyFile = secretMarker;
    model.layers = [{weight: sourceMarker}];
  }
  for (const row of saved.evaluation.models) {
    row.sourceText = sourceMarker;
    for (const name of ['baseline', 'untrained', 'learned']) {
      row[name].wholeTelemetry = telemetryMarker;
      row[name].meanElapsedSecondsSuccessfulWorlds = secretMarker;
    }
    for (const world of row.results) {
      world.learned.report = {transitions: telemetryMarker, privateToken: secretMarker};
    }
  }
  context.mock.method(MotionLearner, 'status', () => saved);
  const evidence = learningSummary('fixture-directory-not-executed');
  assertExcluded(evidence);
  for (const row of evidence.evaluation.models) {
    assert.deepEqual(Object.keys(row.learned), ['count', 'success',
      'meanElapsedSecondsAllWorlds', 'meanEffortProxyVAsAllWorlds']);
  }
});

test('acceptance metadata cannot smuggle secrets or whole telemetry while keeping negative gates', context => {
  const saved = savedLearning();
  saved.evaluation.models[0].acceptance.privateToken = secretMarker;
  saved.evaluation.models[0].acceptance.sourceText = sourceMarker;
  saved.evaluation.models[0].acceptance.wholeTelemetry = telemetryMarker;
  context.mock.method(MotionLearner, 'status', () => saved);
  assertSafeBoundary(() => learningSummary('fixture-directory-not-executed'),
    /(?:Invalid|Unsupported) (?:saved learning|learning|evaluation|acceptance)/i);
});

test('reading robot, selected-run and saved learning evidence never runs replay, inference, evaluation or writes', context => {
  const record = selectedMotion();
  const saved = savedLearning(record);
  context.mock.method(MotionLearner, 'status', () => saved);
  forbidExecution(context);
  assert.equal(robotProfile().hardwareExecutionEnabled, false);
  assert.equal(motionEvidence(record, 'motion').sessionId, sessionId);
  assert.equal(motionEvidence({id: runId, run: record}, 'nemotron').sessionId, runId);
  assert.equal(learningSummary('fixture-directory-not-executed').motionPolicyTrained, true);
});
