'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const {loadHeadless} = require('../simulator/headless');
const {baseline, runPolicy, replay} = require('./motion');
const {suite, aggregate} = require('./motion-evaluation');
const {writeJson} = require('./core');
const DEFAULT_DIRECTORY = path.join(__dirname, 'runs/motion/learning');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const validRun = value => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(value);
const validInteger = value => Number.isInteger(value) && value >= 0 && value <= 0xffffffff;

function readJson(filename, maximum = 1024 * 1024) {
  if (fs.statSync(filename).size > maximum) throw Error('Learning artifact exceeds size budget');
  const bytes = fs.readFileSync(filename);
  return {value: JSON.parse(bytes.toString('utf8')), sha256: hash(bytes)};
}

function policy(filename, expectedHash) {
  const {value: data, sha256} = readJson(filename);
  if (expectedHash && sha256 !== expectedHash) throw Error('Policy checkpoint hash mismatch');
  if (data.schemaVersion !== 1 || data.algorithm !== 'ppo-beta-reference-v1' || data.policyContractVersion !== 1 ||
      data.observationSize !== 34 || data.actionSize !== 4 || data.width !== 32 ||
      !validInteger(data.trainingSeed) || !validInteger(data.steps) || !validInteger(data.updates) ||
      data.trained !== (data.updates > 0) || !Array.isArray(data.scales) || data.scales.length !== 34 ||
      !data.scales.every(value => Number.isFinite(value) && value > 0 && value <= 100) ||
      !Array.isArray(data.layers) || data.layers.length !== 3) throw Error('Unsupported bounded motion model');
  assert.deepEqual(data.identity, loadHeadless().identity, 'Motion model engine/controller identity mismatch');
  const dimensions = [[34, 32], [32, 32], [32, 8]];
  for (let index = 0; index < dimensions.length; index++) {
    const [inputs, outputs] = dimensions[index];
    const layer = data.layers[index];
    const finite = value => Number.isFinite(value) && Math.abs(value) <= 10000;
    if (!layer || !Array.isArray(layer.weight) || layer.weight.length !== outputs ||
        !Array.isArray(layer.bias) || layer.bias.length !== outputs || !layer.bias.every(finite) ||
        layer.weight.some(row => !Array.isArray(row) || row.length !== inputs || !row.every(finite))) throw Error('Malformed motion model weights');
  }
  const act = observation => {
    if (!Array.isArray(observation) || observation.length !== 34 || !observation.every(Number.isFinite)) throw Error('Invalid sensor model input');
    let hidden = observation.map((value, index) => Math.max(-10, Math.min(10, value / data.scales[index])));
    for (let index = 0; index < data.layers.length; index++) {
      const layer = data.layers[index];
      const output = layer.weight.map((row, neuron) => row.reduce((sum, weight, column) => sum + weight * hidden[column], layer.bias[neuron]));
      hidden = index < 2 ? output.map(Math.tanh) : output.map(value => Math.max(value, 0) + Math.log1p(Math.exp(-Math.abs(value))) + 1);
    }
    return hidden.slice(0, 4).map((alpha, index) => 2 * alpha / (alpha + hidden[index + 4]) - 1);
  };
  return {data, sha256, act, identity: {kind: data.trained ? 'learned' : 'untrained',
    id: `ppo-seed-${data.trainingSeed}-update-${data.updates}`, sha256}};
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
  if (record.identity) assert.deepEqual(record.identity, loadHeadless().identity, 'Saved learning run is stale');
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
    const verified = saved.record.status !== 'running' && evaluation?.learnedImprovementVerified === true &&
      JSON.stringify(evaluation.policyHashes) === JSON.stringify(currentHashes) &&
      evaluation.corpusSha256 === suite(2).corpusSha256;
    return {available: true, ...saved.record, evaluation, motionTrainingImplemented: true,
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
    corpusSha256: corpus.corpusSha256, suiteVersion: 2, identity: loadHeadless().identity,
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

module.exports = {DEFAULT_DIRECTORY, readJson, policy, latest, status, runLearned, acceptance, evaluateLearned};
