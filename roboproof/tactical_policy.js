'use strict';

// Runs an exported tactical actor (roboproof/tactical_learning/export_policy.py) in plain JavaScript:
// each candidate's features go through tanh(W1 x + b1) -> tanh(W2 h + b2) -> w·h + b, the same per-candidate
// scoring as TacticalActorCritic's policy head. A v3 export (JSON manifest + float32 weights file) runs the deeper
// encoder instead: tanh(W0 x + b0), then h + tanh(W · LayerNorm(h) + b) per residual block. Inference only.

const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {FEATURE_COUNT, FEATURE_COUNT_V3} = require('./tactical_environment');
const {FEATURE_COUNT_V4} = require('./tactical_task_environment');

const POLICY_SCHEMA = 'roboproof-tactical-policy-v1';
const MODEL_ID = 'override-tactical-actor-critic-v2-loader';
const MAX_POLICY_BYTES = 8 * 1024 * 1024;
const BINARY_POLICY_SCHEMA = 'roboproof-tactical-policy-v3-binary';
const MODEL_ID_V3 = 'override-tactical-mlp-v3-preview';
// A multi-task policy: v3 features plus the task block (roboproof/tactical_task_environment.js).
const MODEL_ID_V4 = 'override-tactical-mlp-v4-task';
const MAX_WEIGHTS_BYTES = 64 * 1024 * 1024;
const LAYER_NORM_EPSILON = 1e-5;
// Training updates an exported policy may declare; an older export without the field is the original actor-critic.
const ALGORITHM_IDS = Object.freeze(['actor-critic', 'ppo-clip-v1']);

function vector(value, length, label) {
  if (!Array.isArray(value) || value.length !== length || !value.every(Number.isFinite)) {
    throw Error(`Tactical policy ${label} must be ${length} finite numbers`);
  }
  return Float64Array.from(value);
}

function matrix(value, rows, columns, label) {
  if (!Array.isArray(value) || value.length !== rows) throw Error(`Tactical policy ${label} must have ${rows} rows`);
  return value.map((row, index) => vector(row, columns, `${label}[${index}]`));
}

function layer(weights, bias, input) {
  const output = new Float64Array(bias.length);
  for (let row = 0; row < bias.length; row += 1) {
    let sum = bias[row];
    const line = weights[row];
    for (let column = 0; column < input.length; column += 1) sum += line[column] * input[column];
    output[row] = Math.tanh(sum);
  }
  return output;
}

function createPolicy(record) {
  if (!record || typeof record !== 'object' || record.schema !== POLICY_SCHEMA || record.model !== MODEL_ID) {
    throw Error(`Expected an exported ${MODEL_ID} policy (${POLICY_SCHEMA})`);
  }
  const {width} = record;
  if (record.featureCount !== FEATURE_COUNT || !Number.isInteger(width) || width < 16 || width > 512 || record.activation !== 'tanh') {
    throw Error('Unsupported tactical policy dimensions');
  }
  if (record.learnedImprovementVerified !== false || record.independentQualificationPerformed !== false) {
    throw Error('A tactical policy file must not claim verified improvement or qualification');
  }
  const algorithmId = record.algorithmId ?? 'actor-critic';
  if (!ALGORITHM_IDS.includes(algorithmId)) throw Error(`Unknown tactical policy algorithm ${algorithmId}`);
  const {hidden1, hidden2, policy} = record.layers || {};
  const w1 = matrix(hidden1?.weight, width, FEATURE_COUNT, 'hidden1.weight');
  const b1 = vector(hidden1?.bias, width, 'hidden1.bias');
  const w2 = matrix(hidden2?.weight, width, width, 'hidden2.weight');
  const b2 = vector(hidden2?.bias, width, 'hidden2.bias');
  const wp = vector(policy?.weight, width, 'policy.weight');
  if (!Number.isFinite(policy?.bias)) throw Error('Tactical policy policy.bias must be a finite number');
  const bp = policy.bias;

  // The actor's preference for one candidate action; higher is preferred.
  function score(features) {
    if (!Array.isArray(features) || features.length !== FEATURE_COUNT || !features.every(Number.isFinite)) {
      throw Error(`Tactical features must be ${FEATURE_COUNT} finite numbers`);
    }
    const hidden = layer(w2, b2, layer(w1, b1, features));
    let sum = bp;
    for (let index = 0; index < width; index += 1) sum += wp[index] * hidden[index];
    return sum;
  }

  // Deterministic choice, as in the trainer's held-out evaluation: the highest-scoring candidate (first on ties).
  function choose(actions) {
    let best = null;
    for (const action of actions) {
      const value = score(action.features);
      if (!best || value > best.score) best = {action, score: value};
    }
    return best;
  }

  return Object.freeze({score, choose, width, hiddenLayers: 2, featureSet: 'v2', featureCount: FEATURE_COUNT, algorithmId,
    ppo: algorithmId.startsWith('ppo'), algorithm: String(record.algorithm || algorithmId),
    provenance: record.provenance ?? null, evaluation: record.evaluation ?? null, independentHoldout: null,
    scope: record.scope ?? null});
}

// The tensors a v3 manifest must list, in the exporter's forward order.
function binaryLayout({featureCount, width, hiddenLayers, residual}) {
  const tensors = [['input.weight', [width, featureCount]], ['input.bias', [width]]];
  if (residual) {
    for (let block = 0; block < hiddenLayers - 1; block += 1) {
      tensors.push([`blocks.${block}.norm.weight`, [width]], [`blocks.${block}.norm.bias`, [width]],
        [`blocks.${block}.linear.weight`, [width, width]], [`blocks.${block}.linear.bias`, [width]]);
    }
  } else tensors.push(['hidden.weight', [width, width]], ['hidden.bias', [width]]);
  tensors.push(['policy.weight', [width]], ['policy.bias', [1]]);
  let offset = 0;
  return tensors.map(([name, shape]) => {
    const length = shape.reduce((product, size) => product * size, 1);
    const entry = {name, shape, offset, length};
    offset += length;
    return entry;
  });
}

function affine(weights, bias, input, output) {
  const columns = input.length;
  for (let row = 0; row < bias.length; row += 1) {
    let sum = bias[row];
    const base = row * columns;
    for (let column = 0; column < columns; column += 1) sum += weights[base + column] * input[column];
    output[row] = sum;
  }
  return output;
}

function createBinaryPolicy(record, weightsBuffer) {
  if (!record || typeof record !== 'object' || record.schema !== BINARY_POLICY_SCHEMA ||
      record.model !== (record.featureSet === 'v4' ? MODEL_ID_V4 : MODEL_ID_V3)) {
    throw Error(`Expected an exported ${MODEL_ID_V3} or ${MODEL_ID_V4} policy (${BINARY_POLICY_SCHEMA})`);
  }
  const {featureSet, featureCount, width, hiddenLayers, residual} = record;
  if (!((featureSet === 'v4' && featureCount === FEATURE_COUNT_V4) || (featureSet === 'v3' && featureCount === FEATURE_COUNT_V3) ||
        (featureSet === 'v2' && featureCount === FEATURE_COUNT))
      || !Number.isInteger(width) || width < 16 || width > 1024 || !Number.isInteger(hiddenLayers) || hiddenLayers < 2
      || hiddenLayers > 10 || residual !== hiddenLayers > 2 || record.activation !== 'tanh'
      || record.layerNormEpsilon !== LAYER_NORM_EPSILON || record.dtype !== 'float32-le') {
    throw Error('Unsupported tactical policy dimensions');
  }
  if (record.learnedImprovementVerified !== false || record.independentQualificationPerformed !== false
      || (record.independentHoldout && record.independentHoldout.learnedImprovementVerified !== false)) {
    throw Error('A tactical policy file must not claim verified improvement or qualification');
  }
  const algorithmId = record.algorithmId ?? 'actor-critic';
  if (!ALGORITHM_IDS.includes(algorithmId)) throw Error(`Unknown tactical policy algorithm ${algorithmId}`);
  const layout = binaryLayout(record);
  const total = layout.reduce((sum, entry) => sum + entry.length, 0);
  if (JSON.stringify(record.layout) !== JSON.stringify(layout) || record.actorParameters !== total) {
    throw Error('Tactical policy weight layout does not match its dimensions');
  }
  if (!Buffer.isBuffer(weightsBuffer) || weightsBuffer.length !== total * 4 || record.weightsBytes !== weightsBuffer.length) {
    throw Error('Tactical policy weights file has the wrong size');
  }
  if (crypto.createHash('sha256').update(weightsBuffer).digest('hex') !== record.weightsSha256) {
    throw Error('Tactical policy weights do not match the SHA-256 in the manifest');
  }
  if (os.endianness() !== 'LE') throw Error('Tactical policy weights are little-endian float32');
  const aligned = new Uint8Array(weightsBuffer.length);
  aligned.set(weightsBuffer);
  const values = new Float32Array(aligned.buffer);
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) throw Error('Tactical policy weights must be finite');
  }
  const tensor = Object.fromEntries(layout.map(entry => [entry.name, values.subarray(entry.offset, entry.offset + entry.length)]));
  const blocks = residual ? Array.from({length: hiddenLayers - 1}, (_, block) => ({
    gamma: tensor[`blocks.${block}.norm.weight`], beta: tensor[`blocks.${block}.norm.bias`],
    weight: tensor[`blocks.${block}.linear.weight`], bias: tensor[`blocks.${block}.linear.bias`]})) : [];
  const hidden = new Float64Array(width);
  const scratch = new Float64Array(width);
  const normalized = new Float64Array(width);

  // `onLayer(hidden)`, when given, sees the activations after the input layer and after every hidden block.
  function score(features, onLayer) {
    if (!Array.isArray(features) || features.length !== featureCount || !features.every(Number.isFinite)) {
      throw Error(`Tactical features must be ${featureCount} finite numbers`);
    }
    affine(tensor['input.weight'], tensor['input.bias'], features, hidden);
    for (let index = 0; index < width; index += 1) hidden[index] = Math.tanh(hidden[index]);
    onLayer?.(hidden);
    if (residual) {
      for (const {gamma, beta, weight, bias} of blocks) {
        let mean = 0;
        for (let index = 0; index < width; index += 1) mean += hidden[index];
        mean /= width;
        let variance = 0;
        for (let index = 0; index < width; index += 1) variance += (hidden[index] - mean) ** 2;
        const scale = 1 / Math.sqrt(variance / width + LAYER_NORM_EPSILON);
        for (let index = 0; index < width; index += 1) normalized[index] = (hidden[index] - mean) * scale * gamma[index] + beta[index];
        affine(weight, bias, normalized, scratch);
        for (let index = 0; index < width; index += 1) hidden[index] += Math.tanh(scratch[index]);
        onLayer?.(hidden);
      }
    } else {
      affine(tensor['hidden.weight'], tensor['hidden.bias'], hidden, scratch);
      for (let index = 0; index < width; index += 1) hidden[index] = Math.tanh(scratch[index]);
      onLayer?.(hidden);
    }
    let sum = tensor['policy.bias'][0];
    const head = tensor['policy.weight'];
    for (let index = 0; index < width; index += 1) sum += head[index] * hidden[index];
    return sum;
  }

  // The same forward pass for the training page's neuron view: every hidden layer summarized into `bins` averaged
  // cells (plus mean, spread and the most active neuron), and each cell's push towards the final score.
  function trace(features, bins = 64) {
    if (!Number.isInteger(bins) || bins < 1 || width % bins !== 0) throw Error(`Trace bins must divide the width ${width}`);
    const size = width / bins;
    const head = tensor['policy.weight'];
    const layers = [];
    const value = score(features, activations => {
      const cells = [];
      const contribution = [];
      let mean = 0;
      let peak = 0;
      for (let cell = 0; cell < bins; cell += 1) {
        let sum = 0;
        let push = 0;
        for (let index = cell * size; index < (cell + 1) * size; index += 1) {
          sum += activations[index];
          push += head[index] * activations[index];
          if (Math.abs(activations[index]) > Math.abs(activations[peak])) peak = index;
        }
        cells.push(sum / size);
        contribution.push(push);
        mean += sum;
      }
      mean /= width;
      let variance = 0;
      for (let index = 0; index < width; index += 1) variance += (activations[index] - mean) ** 2;
      layers.push({cells, contribution, mean, spread: Math.sqrt(variance / width), peakNeuron: peak, peakValue: activations[peak]});
    });
    return {score: value, layers, width, hiddenLayers, bins};
  }

  function choose(actions) {
    let best = null;
    for (const action of actions) {
      const value = score(action.features);
      if (!best || value > best.score) best = {action, score: value};
    }
    return best;
  }

  return Object.freeze({score: features => score(features), trace, choose, width, hiddenLayers, featureSet, featureCount,
    algorithmId, ppo: algorithmId.startsWith('ppo'), algorithm: String(record.algorithm || algorithmId),
    provenance: record.provenance ?? null, evaluation: record.evaluation ?? null,
    independentHoldout: record.independentHoldout ?? null, scope: record.scope ?? null});
}

function loadPolicy(file) {
  const stat = fs.statSync(file);
  if (!stat.isFile() || stat.size > MAX_POLICY_BYTES) throw Error('Tactical policy file is missing or too large');
  const record = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (record?.schema !== BINARY_POLICY_SCHEMA) return createPolicy(record);
  if (typeof record.weightsFile !== 'string' || !/^[\w.-]+\.bin$/.test(record.weightsFile)) {
    throw Error('Tactical policy weightsFile must be a plain .bin file name next to the manifest');
  }
  const weightsFile = path.join(path.dirname(file), record.weightsFile);
  const weightsStat = fs.statSync(weightsFile);
  if (!weightsStat.isFile() || weightsStat.size > MAX_WEIGHTS_BYTES) throw Error('Tactical policy weights file is missing or too large');
  return createBinaryPolicy(record, fs.readFileSync(weightsFile));
}

module.exports = {POLICY_SCHEMA, MODEL_ID, ALGORITHM_IDS, BINARY_POLICY_SCHEMA, MODEL_ID_V3, MODEL_ID_V4, binaryLayout, createPolicy,
  createBinaryPolicy, loadPolicy};
