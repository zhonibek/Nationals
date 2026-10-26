'use strict';

// Runs an exported tactical actor (roboproof/tactical_learning/export_policy.py) in plain JavaScript:
// each candidate's features go through tanh(W1 x + b1) -> tanh(W2 h + b2) -> w·h + b, the same per-candidate
// scoring as TacticalActorCritic's policy head. Inference only; nothing here trains.

const fs = require('node:fs');
const {FEATURE_COUNT} = require('./tactical_environment');

const POLICY_SCHEMA = 'roboproof-tactical-policy-v1';
const MODEL_ID = 'override-tactical-actor-critic-v2-loader';
const MAX_POLICY_BYTES = 8 * 1024 * 1024;
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

  return Object.freeze({score, choose, width, algorithmId, ppo: algorithmId.startsWith('ppo'), algorithm: String(record.algorithm || algorithmId),
    provenance: record.provenance ?? null, evaluation: record.evaluation ?? null, scope: record.scope ?? null});
}

function loadPolicy(file) {
  const stat = fs.statSync(file);
  if (!stat.isFile() || stat.size > MAX_POLICY_BYTES) throw Error('Tactical policy file is missing or too large');
  return createPolicy(JSON.parse(fs.readFileSync(file, 'utf8')));
}

module.exports = {POLICY_SCHEMA, MODEL_ID, ALGORITHM_IDS, createPolicy, loadPolicy};
