'use strict';

const fs = require('node:fs');
const {createHash} = require('node:crypto');
const {isDeepStrictEqual} = require('node:util');
const Gvf = require('../simulator/gvf-policy');
const {loadHeadless} = require('../simulator/headless');

const ALGORITHM = 'td3-tanh-gvf-residual-v1';
const FEATURE_TRANSFORM = 'gvf-body-reference36-v1';
const FEATURE_SIZE = 36;
const SCALES = Object.freeze([1.524, 1.524, 1, 2, 2, 4, 3, 3, 3, 3, 4.32, 4.32, 1, 1, 1, 1, 1, 60,
  4.32, 4.32, 1, 0.35, 0.35, 1.5, 1, 4.32, 4.32, 0.08, 0.08, 0.1, 0.08, 0.08, 0.25, 10, 1, 1].map(Math.fround));
const NORMALIZATION = Object.freeze({method: 'fixed-public-sensor-scales-v1', clip: Object.freeze([-10, 10])});
const KEYS = ['schemaVersion', 'algorithm', 'policyContractName', 'policyContractVersion', 'observationSchema',
  'observationSize', 'actionSize', 'featureTransform', 'featureInputSize', 'scales', 'normalization', 'width',
  'hiddenLayers', 'trainingSeed', 'steps', 'updates', 'trained', 'identity', 'layers'];
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const boundedInteger = (value, maximum) => Number.isSafeInteger(value) && value >= 0 && value <= maximum;
const finiteWeights = values => Array.isArray(values) && Array.from(values).every(value => Number.isFinite(value) && Math.abs(value) <= 10000);

function encodeObservation(observation) {
  const raw = Gvf.vector(observation), features = [...raw];
  const cosine = Math.cos(raw[2]), sine = Math.sin(raw[2]);
  for (const [rightIndex, forwardIndex] of [[3, 4], [18, 19], [21, 22], [27, 28], [30, 31]]) {
    let right = raw[rightIndex], forward = raw[forwardIndex];
    if (rightIndex === 18) { right -= raw[0]; forward -= raw[1]; }
    features[rightIndex] = right * cosine - forward * sine;
    features[forwardIndex] = right * sine + forward * cosine;
  }
  const difference = raw[20] - raw[2];
  features[2] = sine;
  features[20] = Math.sin(difference);
  features.push(cosine, Math.cos(difference));
  if (features.length !== FEATURE_SIZE || !features.every(Number.isFinite)) throw Error('Finite GVF model features required');
  return features;
}

function validateArtifact(data, expectedIdentity) {
  if (!data || typeof data !== 'object' || Array.isArray(data) || Object.keys(data).length !== KEYS.length ||
    Object.keys(data).some(name => !KEYS.includes(name)) || data.schemaVersion !== 1 || data.algorithm !== ALGORITHM ||
    data.policyContractName !== Gvf.CONTRACT_NAME || data.policyContractVersion !== Gvf.CONTRACT_VERSION ||
    data.observationSchema !== Gvf.OBSERVATION_SCHEMA || data.observationSize !== Gvf.OBSERVATION_SIZE ||
    data.actionSize !== Gvf.ACTION_SIZE || data.featureTransform !== FEATURE_TRANSFORM || data.featureInputSize !== FEATURE_SIZE ||
    ![32, 64, 128, 256, 512].includes(data.width) || ![2, 10].includes(data.hiddenLayers) ||
    !boundedInteger(data.trainingSeed, 0xffffffff) || !boundedInteger(data.steps, 10000000) ||
    !boundedInteger(data.updates, 100000) || data.trained !== (data.updates > 0) ||
    !isDeepStrictEqual(data.scales, [...SCALES]) || !isDeepStrictEqual(data.normalization, NORMALIZATION) ||
    !data.identity || typeof data.identity !== 'object' || Array.isArray(data.identity) || !Object.keys(data.identity).length ||
    Buffer.byteLength(JSON.stringify(data.identity)) > 256 * 1024 || !isDeepStrictEqual(data.identity, expectedIdentity)) {
    throw Error('Exact separate GVF actor contract, normalization and original engine identity required');
  }
  const sizes = [[FEATURE_SIZE, data.width], ...Array.from({length: data.hiddenLayers - 1}, () => [data.width, data.width]), [data.width, Gvf.ACTION_SIZE]];
  if (!Array.isArray(data.layers) || data.layers.length !== sizes.length) throw Error('Exact GVF model layer count required');
  for (const [index, [inputs, outputs]] of sizes.entries()) {
    const layer = data.layers[index];
    if (!layer || Object.keys(layer).length !== 2 || !Object.hasOwn(layer, 'weight') || !Object.hasOwn(layer, 'bias') ||
      !finiteWeights(layer.bias) || layer.bias.length !== outputs || !Array.isArray(layer.weight) || layer.weight.length !== outputs ||
      Array.from(layer.weight).some(row => !finiteWeights(row) || row.length !== inputs)) throw Error('Finite bounded GVF weight dimensions required');
  }
  return data;
}

function fromArtifact(value, expectedIdentity) {
  const data = validateArtifact(structuredClone(value), expectedIdentity);
  const act = observation => {
    let hidden = encodeObservation(observation).map((value, index) => Math.max(-10, Math.min(10, value / data.scales[index])));
    for (const layer of data.layers) {
      hidden = layer.weight.map((row, neuron) => Math.tanh(row.reduce((sum, weight, column) => sum + weight * hidden[column], layer.bias[neuron])));
    }
    return Gvf.normalizeAction(hidden);
  };
  return {data: structuredClone(data), act};
}

function policy(filename, expectedSha256) {
  if (expectedSha256 !== undefined && !/^[a-f0-9]{64}$/.test(expectedSha256)) throw Error('Exact GVF actor SHA256 required');
  const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
  let bytes;
  try {
    const before = fs.fstatSync(descriptor);
    if (!before.isFile() || before.size > 64 * 1024 * 1024) throw Error('Bounded regular GVF actor required');
    const buffer = Buffer.alloc(before.size + 1);
    const length = fs.readSync(descriptor, buffer, 0, buffer.length, 0);
    bytes = buffer.subarray(0, length);
    const after = fs.fstatSync(descriptor), current = fs.lstatSync(filename);
    const stamp = state => [state.dev, state.ino, state.mode, state.size, state.mtimeMs, state.ctimeMs];
    if (bytes.length !== before.size || current.isSymbolicLink() || !current.isFile() ||
      !isDeepStrictEqual(stamp(before), stamp(after)) || !isDeepStrictEqual(stamp(after), stamp(current))) throw Error('GVF actor changed while reading');
  } finally { fs.closeSync(descriptor); }
  const sha256 = hash(bytes);
  if (expectedSha256 !== undefined && expectedSha256 !== sha256) throw Error('GVF actor SHA256 mismatch');
  const data = JSON.parse(bytes.toString('utf8'));
  const architectureBudget = data.width === 512 ? 64 : data.hiddenLayers === 10 ? 24 : 8;
  if (bytes.length > architectureBudget * 1024 * 1024) throw Error('GVF actor exceeds architecture reader budget');
  const model = fromArtifact(data, loadHeadless().identity);
  return {...model, sha256, identity: {kind: data.trained ? 'learned' : 'untrained',
    id: `gvf-td3-seed-${data.trainingSeed}-update-${data.updates}`, sha256}};
}

function runActor({filename, sha256, seed, task, configuration = {}}) {
  const model = policy(filename, sha256);
  const {runGvfPolicy, replay} = require('./motion');
  const report = runGvfPolicy({seed, task, configuration, options: {policyIdentity: model.identity},
    act: (values, observation) => {
      if (!isDeepStrictEqual(values, Gvf.vector(observation))) throw Error('GVF vector/observation mismatch');
      return model.act(observation);
    }});
  replay(report);
  return {report, inferencePerformed: true, artifactDeclaresTraining: model.data.trained, gradientTrainingVerified: false,
    exactReplayVerified: true, learnedImprovementVerified: false, independentQualificationClaimed: false};
}

module.exports = {ALGORITHM, FEATURE_TRANSFORM, FEATURE_SIZE, SCALES, NORMALIZATION, encodeObservation, validateArtifact, fromArtifact, policy, runActor};
