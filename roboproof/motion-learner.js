'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {isDeepStrictEqual} = require('node:util');
const {loadHeadless} = require('../simulator/headless');
const Policy = require('../simulator/policy');
const {baseline, runPolicy, replay} = require('./motion');
const {suite, aggregate} = require('./motion-evaluation');
const {writeJson} = require('./core');
const DEFAULT_DIRECTORY = path.join(__dirname, 'runs/motion/learning');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const validRun = value => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(value);
const validInteger = value => Number.isInteger(value) && value >= 0 && value <= 0xffffffff;
const settlingNormalization = 'settling-normalized-deadline-v1';
const settlingFrameScales = {'3': 0.0254, '4': 0.0254, '5': 0.0873, '10': 0.02032, '11': 0.02032, '12': 0.035};
const coarseAugmentation = 'coarse-augmentation-deadline-v1';
const multiscaleFeatures = 'multiscale-deadline-v1';
const augmentationIndices = [3, 4, 5, 10, 11, 12];
const augmentationUnits = featureTransform => featureTransform === multiscaleFeatures ?
  [0.0254, 0.0254, 0.0873, 0.02032, 0.02032, 0.035] : [2, 2, 4, 1.524, 1.524, 1];
const augmentationDefinition = featureTransform => ({schemaVersion: 1, method: 'bounded-sensor-augmentation-v1', historyFrames: 4,
  baseFrameSize: 35, augmentedFrameSize: 41, sourceIndices: augmentationIndices, units: augmentationUnits(featureTransform),
  transform: 'tanh(reference-frame-feature/unit)', baseChannels: 'Unchanged field-scale reference-frame35', addedChannelScales: [1, 1, 1, 1, 1, 1],
  initialContext: 'Unchanged field-scale initial sensor goal context', clip: [-10, 10],
  scope: 'Derived public sensor/reference channels only; additional169-input capacity, no evaluator truth or fitted statistics'});
const settlingNormalizationDefinition = {schemaVersion: 1, method: 'sensor-settling-units-v1', historyFrames: 4,
  frameScales: settlingFrameScales,
  heading: 'Goal-heading sine divided by0.035; cosine retained at scale1, not a success classifier',
  initialContext: 'Unchanged field-scale initial sensor goal context', clip: [-10, 10],
  scope: 'Fixed public sensor/reference units only; no evaluator truth, fitted statistics or new input capacity'};

function readJson(filename, maximum = 1024 * 1024) {
  if (fs.statSync(filename).size > maximum) throw Error('Learning artifact exceeds size budget');
  const bytes = fs.readFileSync(filename);
  return {value: JSON.parse(bytes.toString('utf8')), sha256: hash(bytes)};
}

function policy(filename, expectedHash) {
  const {value: data, sha256} = readJson(filename);
  if (expectedHash && sha256 !== expectedHash) throw Error('Policy checkpoint hash mismatch');
  return policyFromSnapshot(data, sha256);
}

function policyFromSnapshot(value, sha256) {
  if (!/^[a-f0-9]{64}$/.test(sha256)) throw Error('Frozen policy artifact hash required');
  const data = structuredClone(value);
  const featureTransform = data.featureTransform === undefined ? 'identity-v1' : data.featureTransform;
  if (!['identity-v1', 'reference-frame-v1', 'history-context-v1', 'deadline-context-v1', 'long-history-deadline-v1', settlingNormalization, coarseAugmentation, multiscaleFeatures].includes(featureTransform)) throw Error('Unsupported motion feature transform');
  const augmented = featureTransform === coarseAugmentation || featureTransform === multiscaleFeatures;
  const deadline = featureTransform === 'deadline-context-v1' || featureTransform === 'long-history-deadline-v1' || featureTransform === settlingNormalization || augmented;
  const temporal = featureTransform === 'history-context-v1' || deadline;
  const historyFrames = featureTransform === 'long-history-deadline-v1' ? 16 : 4;
  const frameSize = augmented ? 41 : 35;
  const featureCount = temporal ? frameSize * historyFrames + 4 + (deadline ? 1 : 0) : featureTransform === 'reference-frame-v1' ? 35 : 34;
  if (temporal && (data.historyFrames !== historyFrames || data.initialContextSize !== 4)) throw Error('Temporal feature definition mismatch');
  if ((featureTransform !== 'identity-v1' || data.featureInputSize !== undefined) && data.featureInputSize !== featureCount) throw Error('Motion feature count mismatch');
  if (deadline && !isDeepStrictEqual(data.deadlineContext, {schemaVersion: 1, settlingReserveSeconds: 2,
    input: 'public-reference-duration-at-reset'})) throw Error('Deadline feature definition mismatch');
  if (data.schemaVersion !== 1 || !['ppo-beta-reference-v1', 'td3-beta-mean-reference-v1'].includes(data.algorithm) || data.policyContractVersion !== 1 ||
      data.observationSize !== 34 || data.actionSize !== 4 || data.width !== 32 ||
      !validInteger(data.trainingSeed) || !validInteger(data.steps) || !validInteger(data.updates) ||
      data.trained !== (data.updates > 0) || !Array.isArray(data.scales) || data.scales.length !== featureCount ||
      !data.scales.every(value => Number.isFinite(value) && value > 0 && value <= 100) ||
      !Array.isArray(data.layers) || data.layers.length !== 3) throw Error('Unsupported bounded motion model');
  if (deadline && data.scales[featureCount - 1] !== 1) throw Error('Deadline feature scale mismatch');
  if (featureTransform === settlingNormalization) {
    if (!isDeepStrictEqual(data.normalizationDefinition, settlingNormalizationDefinition)) throw Error('Sensor normalization definition mismatch');
    const frame = [1.524, 1.524, 3.142, 2, 2, 4, 3, 3, 3, 3,
      1.524, 1.524, 1, 1, 1, 1, 1, 1, 10, 1.524, 1.524, 1, 2, 2, 4, 10, 1,
      0.114, 0.114, 0.1, 0.08, 0.08, 0.25, 40, 1];
    for (const [index, scale] of Object.entries(settlingFrameScales)) frame[Number(index)] = scale;
    const expected = Array.from({length: 4}, () => frame).flat().concat([1.524, 1.524, 1, 1, 1]).map(Math.fround);
    if (!isDeepStrictEqual(data.scales, expected)) throw Error('Sensor normalization scale mismatch');
  }
  if (augmented) {
    if (data.frameInputSize !== 41 || !isDeepStrictEqual(data.normalizationDefinition, augmentationDefinition(featureTransform))) throw Error('Sensor augmentation definition mismatch');
    const frame = [1.524, 1.524, 3.142, 2, 2, 4, 3, 3, 3, 3,
      1.524, 1.524, 1, 1, 1, 1, 1, 1, 10, 1.524, 1.524, 1, 2, 2, 4, 10, 1,
      0.114, 0.114, 0.1, 0.08, 0.08, 0.25, 40, 1, 1, 1, 1, 1, 1, 1];
    const expected = Array.from({length: 4}, () => frame).flat().concat([1.524, 1.524, 1, 1, 1]).map(Math.fround);
    if (!isDeepStrictEqual(data.scales, expected)) throw Error('Sensor augmentation scale mismatch');
  }
  if (!isDeepStrictEqual(data.identity, loadHeadless().identity)) throw Error('Motion model engine/controller identity mismatch');
  const dimensions = [[featureCount, 32], [32, 32], [32, 8]];
  for (let index = 0; index < dimensions.length; index++) {
    const [inputs, outputs] = dimensions[index];
    const layer = data.layers[index];
    const finite = value => Number.isFinite(value) && Math.abs(value) <= 10000;
    if (!layer || !Array.isArray(layer.weight) || layer.weight.length !== outputs ||
        !Array.isArray(layer.bias) || layer.bias.length !== outputs || !layer.bias.every(finite) ||
        layer.weight.some(row => !Array.isArray(row) || row.length !== inputs || !row.every(finite))) throw Error('Malformed motion model weights');
  }
  let observationHistory = [], initialContext = null, referenceDurationSeconds = null;
  const act = observation => {
    if (!Array.isArray(observation) || observation.length !== 34 || !Array.from(observation).every(Number.isFinite)) throw Error('Invalid sensor model input');
    if (deadline && referenceDurationSeconds === null) throw Error('Public reference duration required at reset before deadline inference');
    const features = [...observation];
    if (featureTransform !== 'identity-v1') {
      const heading = observation[21], cosine = Math.cos(heading), sine = Math.sin(heading);
      for (const [first, second] of [[3, 4], [19, 20], [22, 23], [27, 28], [30, 31]]) {
        let right = observation[first], forward = observation[second];
        if (first === 19) { right -= observation[0]; forward -= observation[1]; }
        features[first] = right * cosine - forward * sine;
        features[second] = right * sine + forward * cosine;
      }
      const difference = heading - observation[2], deltaCosine = Math.cos(difference), deltaSine = Math.sin(difference);
      features[10] = observation[10] * deltaCosine - observation[11] * deltaSine;
      features[11] = observation[10] * deltaSine + observation[11] * deltaCosine;
      features[21] = deltaSine;
      features.push(deltaCosine);
    }
    if (augmented) {
      const units = augmentationUnits(featureTransform);
      features.push(...augmentationIndices.map((index, axis) => Math.tanh(features[index] / units[axis])));
    }
    let modelInput = features;
    if (temporal) {
      initialContext ??= observation.slice(10, 14);
      observationHistory = [...observationHistory, features].slice(-historyFrames);
      const padding = Array.from({length: historyFrames - observationHistory.length}, () => observationHistory[0]);
      modelInput = [...padding, ...observationHistory].flat().concat(initialContext);
    }
    if (deadline) modelInput.push(Math.max(0.25, Math.min(1,
      Math.max(0, referenceDurationSeconds - observation[25]) / Math.max(0.01, observation[18] - 2))));
    let hidden = modelInput.map((value, index) => Math.max(-10, Math.min(10, value / data.scales[index])));
    for (let index = 0; index < data.layers.length; index++) {
      const layer = data.layers[index];
      const output = layer.weight.map((row, neuron) => row.reduce((sum, weight, column) => sum + weight * hidden[column], layer.bias[neuron]));
      hidden = index < 2 ? output.map(Math.tanh) : output.map(value => Math.max(value, 0) + Math.log1p(Math.exp(-Math.abs(value))) + 1);
    }
    return hidden.slice(0, 4).map((alpha, index) => 2 * alpha / (alpha + hidden[index + 4]) - 1);
  };
  act.reset = context => {
    observationHistory = []; initialContext = null; referenceDurationSeconds = null;
    if (deadline && context !== undefined) {
      const duration = context?.referenceDurationSeconds;
      if (!Number.isFinite(duration) || duration < 0) throw Error('Invalid public reference duration at reset');
      referenceDurationSeconds = duration;
    }
  };
  return {data, sha256, act, identity: {kind: data.trained ? 'learned' : 'untrained',
    id: `${data.algorithm === 'td3-beta-mean-reference-v1' ? 'td3' : 'ppo'}-seed-${data.trainingSeed}-update-${data.updates}`, sha256}};
}

function latest(directory = DEFAULT_DIRECTORY) {
  if (!fs.existsSync(path.join(directory, 'latest.json'))) return null;
  const pointer = readJson(path.join(directory, 'latest.json'), 1024).value;
  if (pointer.schemaVersion !== 1 || !validRun(pointer.runId)) throw Error('Invalid local learning pointer');
  const runDirectory = path.join(directory, pointer.runId);
  const record = readJson(path.join(runDirectory, 'run.json'), 4 * 1024 * 1024).value;
  if (record.schemaVersion !== 1 || record.runId !== pointer.runId || record.device !== 'cpu' ||
      !Array.isArray(record.models) || record.models.length > 3 ||
      !['running', 'completed', 'budget-stopped', 'interrupted-or-failed'].includes(record.status)) throw Error('Invalid local learning record');
  if (record.identity && !isDeepStrictEqual(record.identity, loadHeadless().identity)) throw Error('Saved learning run is stale');
  for (const row of record.models) {
    if (!validInteger(row.seed) || !/^policy-[0-9]{4}\.json$/.test(row.policyFile) ||
        !/^[a-f0-9]{64}$/.test(row.policySha256) || !/^[a-f0-9]{64}$/.test(row.initialSha256)) throw Error('Invalid local learning checkpoint');
    const model = policy(path.join(runDirectory, `seed-${row.seed}`, row.policyFile), row.policySha256);
    if (model.data.trainingSeed !== row.seed || model.data.updates !== row.updates || model.data.steps !== row.steps ||
        model.data.trained !== row.trained) throw Error('Model summary does not match checkpoint');
  }
  return {record, runDirectory};
}

function status(directory = DEFAULT_DIRECTORY) {
  try {
    const saved = latest(directory);
    if (!saved) return {available: false, motionTrainingImplemented: true, learnedImprovementVerified: false,
      message: 'No saved motion learner. Run the explicit bounded CPU training command.'};
    const filename = path.join(saved.runDirectory, 'evaluation.json');
    const evaluation = fs.existsSync(filename) ? readJson(filename, 4 * 1024 * 1024).value : null;
    const currentHashes = saved.record.models.map(row => row.policySha256);
    const matched = evaluation?.runId === saved.record.runId && evaluation?.policyContractVersion === Policy.CONTRACT_VERSION &&
      evaluation?.suiteVersion === 2 && isDeepStrictEqual(evaluation?.identity, saved.record.identity) &&
      JSON.stringify(evaluation.policyHashes) === JSON.stringify(currentHashes) &&
      evaluation.corpusSha256 === suite(2).corpusSha256;
    const verified = saved.record.status !== 'running' && matched && evaluation.learnedImprovementVerified === true;
    return {available: true, ...saved.record, evaluation: matched ? evaluation : null, staleEvaluation: Boolean(evaluation && !matched), motionTrainingImplemented: true,
      motionPolicyTrained: saved.record.models.some(row => row.trained && row.actorWeightsChanged),
      learnedImprovementVerified: verified, hardwareExecutionEnabled: false, cloudExecutionEnabled: false};
  } catch (error) { return {available: false, motionTrainingImplemented: true,
    learnedImprovementVerified: false, message: error.message}; }
}

function runLearned({seed, task, configuration = {}, directory = DEFAULT_DIRECTORY}) {
  const saved = latest(directory);
  if (!saved) throw Error('Train a local motion policy first');
  if (saved.record.status === 'running') throw Error('Freeze training before experimental policy execution');
  const row = saved.record.models.find(entry => entry.trained && entry.actorWeightsChanged);
  if (!row) throw Error('No gradient-trained motion checkpoint is available');
  const model = policy(path.join(saved.runDirectory, `seed-${row.seed}`, row.policyFile), row.policySha256);
  const report = runPolicy({seed, task, configuration, act: model.act,
    options: {policyIdentity: model.identity}});
  replay(report);
  return {report, motionPolicyTrained: true, inferencePerformed: true, exactReplayVerified: true,
    learnedImprovementVerified: status(directory).learnedImprovementVerified,
    interpretation: 'Explicit experimental learned reference; production controller and default agent remain unchanged'};
}

function acceptance(results, original, learned, untrained, rules) {
  const regressions = results.filter(row => row.baseline.reason === 'success' && row.learned.reason !== 'success').map(row => row.id);
  const timeGain = 1 - learned.meanElapsedSecondsAllWorlds / original.meanElapsedSecondsAllWorlds;
  const effortGain = 1 - learned.meanEffortProxyVAsAllWorlds / original.meanEffortProxyVAsAllWorlds;
  const contactRegressions = results.filter(row => row.learned.metrics.contactSeconds > row.baseline.metrics.contactSeconds + 1e-9).map(row => row.id);
  const passed = learned.successRate > original.successRate && learned.successRate > untrained.successRate &&
    regressions.length <= rules.maximumSuccessRegressions && contactRegressions.length === 0 &&
    Math.max(timeGain, effortGain) >= rules.minimumMeanTimeOrEffortImprovementFraction;
  return {passed, regressions, contactRegressions, meanTimeImprovementFraction: timeGain,
    meanEffortImprovementFraction: effortGain, rule: 'Higher success than controller and untrained policy; zero success/contact regressions; >=5% mean time or effort gain. All frozen worlds included.'};
}

function evaluateLearned(directory = DEFAULT_DIRECTORY) {
  const saved = latest(directory);
  if (!saved || !saved.record.models.some(row => row.trained)) throw Error('No trained motion run available');
  if (saved.record.status === 'running') throw Error('Freeze training before evaluation');
  const corpus = suite(2);
  const baselines = corpus.cases.map(world => baseline(world.seed, world.task, world.configuration));
  const compact = report => ({reason: report.reason, metrics: report.metrics});
  const models = saved.record.models.filter(row => row.trained).map(row => {
    const directory = path.join(saved.runDirectory, `seed-${row.seed}`);
    const learned = policy(path.join(directory, row.policyFile), row.policySha256);
    const initial = policy(path.join(directory, 'initial-policy.json'), row.initialSha256);
    const run = (model, world) => runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
      options: {recordTransitions: false, policyIdentity: model.identity}, act: model.act});
    const learnedReports = corpus.cases.map(world => run(learned, world));
    const initialReports = corpus.cases.map(world => run(initial, world));
    const results = corpus.cases.map((world, index) => ({id: world.id, seed: world.seed,
      baseline: compact(baselines[index]), untrained: compact(initialReports[index]), learned: compact(learnedReports[index])}));
    const originalSummary = aggregate(baselines);
    const learnedSummary = aggregate(learnedReports);
    const initialSummary = aggregate(initialReports);
    return {seed: row.seed, policySha256: row.policySha256, actorWeightsChanged: row.actorWeightsChanged,
      baseline: originalSummary, untrained: initialSummary, learned: learnedSummary, results,
      acceptance: acceptance(results, originalSummary, learnedSummary, initialSummary, corpus.acceptance)};
  });
  const independentSeeds = new Set(models.map(row => row.seed)).size >= corpus.acceptance.independentTrainingSeedsRequired;
  const rates = models.map(row => row.learned.successRate);
  const meanSuccessRate = rates.reduce((sum, value) => sum + value, 0) / rates.length;
  const result = {schemaVersion: 1, runId: saved.record.runId, evaluatedAt: new Date().toISOString(),
    corpusSha256: corpus.corpusSha256, suiteVersion: 2, policyContractVersion: Policy.CONTRACT_VERSION, identity: loadHeadless().identity,
    policyHashes: saved.record.models.map(row => row.policySha256), models, acceptance: corpus.acceptance,
    independentTrainingSeedsRequirementMet: independentSeeds,
    independentSeedVariation: {count: rates.length, meanSuccessRate,
      populationStandardDeviation: Math.sqrt(rates.reduce((sum, value) => sum + (value - meanSuccessRate) ** 2, 0) / rates.length),
      minimumSuccessRate: Math.min(...rates), maximumSuccessRate: Math.max(...rates)},
    learnedImprovementVerified: independentSeeds && models.every(row => row.acceptance.passed && row.actorWeightsChanged),
    motionPolicyTrained: true, interpretation: 'Frozen software evaluation only. Negative results retained; no policy selected using this corpus; no physical/AMD claim.'};
  writeJson(path.join(saved.runDirectory, 'evaluation.json'), result);
  return result;
}

module.exports = {DEFAULT_DIRECTORY, readJson, policy, policyFromSnapshot, latest, status, runLearned, acceptance, evaluateLearned};
