'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {FEATURE_COUNT, FEATURE_COUNT_V3, TacticalEnvironment} = require('../roboproof/tactical_environment');
const {BINARY_POLICY_SCHEMA, MODEL_ID_V3, MODEL_ID_V4, binaryLayout, createBinaryPolicy, loadPolicy} = require('../roboproof/tactical_policy');
const {parseTask, runTask} = require('../roboproof/tactical_task');
const {FEATURE_COUNT_V4} = require('../roboproof/tactical_task_environment');

// A small v3 export built in the test: deterministic pseudo-random float32 weights and the manifest the exporter writes.
function fixture({width = 16, hiddenLayers = 3, featureSet = 'v3', overrides = {}} = {}) {
  const featureCount = {v2: FEATURE_COUNT, v3: FEATURE_COUNT_V3, v4: FEATURE_COUNT_V4}[featureSet];
  const dimensions = {featureCount, width, hiddenLayers, residual: hiddenLayers > 2};
  const layout = binaryLayout(dimensions);
  const total = layout.reduce((sum, entry) => sum + entry.length, 0);
  const values = new Float32Array(total);
  let state = 12345;
  for (let index = 0; index < total; index += 1) {
    state = (state * 1103515245 + 12345) % 2147483648;
    values[index] = (state / 2147483648 - 0.5) * 0.6;
  }
  for (const entry of layout.filter(row => row.name.endsWith('norm.weight'))) values.fill(1.1, entry.offset, entry.offset + entry.length);
  const weights = Buffer.from(values.buffer);
  const record = {schema: BINARY_POLICY_SCHEMA, model: featureSet === 'v4' ? MODEL_ID_V4 : MODEL_ID_V3, featureSet, ...dimensions, activation: 'tanh',
    layerNormEpsilon: 1e-5, actorParameters: total, dtype: 'float32-le', weightsFile: 'policy.weights.bin',
    weightsBytes: weights.length, weightsSha256: crypto.createHash('sha256').update(weights).digest('hex'), layout,
    algorithmId: 'ppo-clip-v1', algorithm: 'PPO fixture', provenance: {runId: 'fixture-v3', trainingSeed: 1400001},
    evaluation: null, learnedImprovementVerified: false, independentQualificationPerformed: false, ...overrides};
  return {record, weights, values, layout};
}

// Written separately from the module: nested arrays and plain loops, the PyTorch forward pass of the actor.
function referenceScore({values, layout, record}, features) {
  const get = name => {
    const entry = layout.find(row => row.name === name);
    return Array.from(values.subarray(entry.offset, entry.offset + entry.length));
  };
  const dense = (weight, bias, input) => bias.map((b, row) => input.reduce((sum, x, column) => sum + weight[row * input.length + column] * x, b));
  let hidden = dense(get('input.weight'), get('input.bias'), features).map(Math.tanh);
  if (record.residual) {
    for (let block = 0; block < record.hiddenLayers - 1; block += 1) {
      const mean = hidden.reduce((a, b) => a + b, 0) / hidden.length;
      const variance = hidden.reduce((a, b) => a + (b - mean) ** 2, 0) / hidden.length;
      const gamma = get(`blocks.${block}.norm.weight`);
      const beta = get(`blocks.${block}.norm.bias`);
      const normalized = hidden.map((value, index) => (value - mean) / Math.sqrt(variance + 1e-5) * gamma[index] + beta[index]);
      const update = dense(get(`blocks.${block}.linear.weight`), get(`blocks.${block}.linear.bias`), normalized).map(Math.tanh);
      hidden = hidden.map((value, index) => value + update[index]);
    }
  } else hidden = dense(get('hidden.weight'), get('hidden.bias'), hidden).map(Math.tanh);
  const head = get('policy.weight');
  return hidden.reduce((sum, value, index) => sum + head[index] * value, get('policy.bias')[0]);
}

const featureRow = (count, shift) => Array.from({length: count}, (_, index) => Math.sin(index * 0.7 + shift));

test('a binary v3 policy scores candidates exactly like the reference residual forward pass', () => {
  for (const options of [{hiddenLayers: 3}, {hiddenLayers: 10, width: 32}, {hiddenLayers: 2}, {featureSet: 'v2', hiddenLayers: 4}]) {
    const built = fixture(options);
    const policy = createBinaryPolicy(built.record, built.weights);
    for (const shift of [0, 0.4, -1.3]) {
      const features = featureRow(built.record.featureCount, shift);
      assert.ok(Math.abs(policy.score(features) - referenceScore(built, features)) < 1e-9, JSON.stringify(options));
    }
    assert.deepEqual([policy.featureSet, policy.hiddenLayers, policy.width, policy.ppo],
      [built.record.featureSet, built.record.hiddenLayers, built.record.width, true]);
  }
  const built = fixture();
  const policy = createBinaryPolicy(built.record, built.weights);
  const actions = [0, 0.4, -1.3, 2].map(shift => ({id: `a${shift}`, features: featureRow(FEATURE_COUNT_V3, shift)}));
  const best = actions.reduce((top, action) => referenceScore(built, action.features) > referenceScore(built, top.features) ? action : top);
  assert.equal(policy.choose(actions).action.id, best.id);
  assert.throws(() => policy.score(featureRow(FEATURE_COUNT, 0)), /30 finite numbers/);
});

test('binary policies with a wrong hash, layout, size, value or claim are refused', () => {
  const {record, weights} = fixture();
  const flipped = Buffer.from(weights);
  flipped[10] ^= 1;
  assert.throws(() => createBinaryPolicy(record, flipped), /SHA-256/);
  assert.throws(() => createBinaryPolicy(record, weights.subarray(4)), /wrong size/);
  const nan = Buffer.from(weights);
  nan.writeFloatLE(Number.NaN, 0);
  assert.throws(() => createBinaryPolicy({...record, weightsSha256: crypto.createHash('sha256').update(nan).digest('hex')}, nan), /finite/);
  const swapped = record.layout.map(entry => ({...entry}));
  [swapped[0].name, swapped[1].name] = [swapped[1].name, swapped[0].name];
  const broken = {
    'layout order': {layout: swapped}, 'parameter count': {actorParameters: record.actorParameters + 1},
    'feature count': {featureCount: FEATURE_COUNT}, 'epsilon': {layerNormEpsilon: 1e-6}, 'dtype': {dtype: 'float16'},
    'residual flag': {residual: false}, 'depth': {hiddenLayers: 11}, 'schema': {schema: 'other'},
    'claims improvement': {learnedImprovementVerified: true},
    'holdout claims improvement': {independentHoldout: {learnedImprovementVerified: true}}
  };
  for (const [label, overrides] of Object.entries(broken)) {
    assert.throws(() => createBinaryPolicy({...record, ...overrides}, weights), Error, label);
  }
});

test('loadPolicy reads a manifest with its weights file and refuses paths outside its folder', t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tactical-policy-v3-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const {record, weights} = fixture();
  fs.writeFileSync(path.join(directory, 'policy.json'), JSON.stringify(record));
  fs.writeFileSync(path.join(directory, 'policy.weights.bin'), weights);
  assert.equal(loadPolicy(path.join(directory, 'policy.json')).hiddenLayers, 3);
  fs.writeFileSync(path.join(directory, 'escape.json'), JSON.stringify({...record, weightsFile: '../policy.weights.bin'}));
  assert.throws(() => loadPolicy(path.join(directory, 'escape.json')), /plain .bin file name/);
  fs.rmSync(path.join(directory, 'policy.weights.bin'));
  assert.throws(() => loadPolicy(path.join(directory, 'policy.json')), /ENOENT/);
});

test('a v3 policy plays a task on v3 features and refuses a v2 environment', () => {
  const {record, weights} = fixture();
  const policy = createBinaryPolicy(record, weights);
  const spec = parseTask('play the driver period for the best score');
  const played = runTask({spec, policy, seed: 1, alliance: 'red'});
  assert.ok(played.decisions > 0, JSON.stringify({stopReason: played.stopReason}));
  assert.throws(() => runTask({spec, policy, seed: 1, alliance: 'red', environment: new TacticalEnvironment()}),
    /needs v3 features, but the environment produces v2/);
});

test('a multi-task (v4) policy reads the task block and plays typed tasks through the same task runner', () => {
  const {record, weights} = fixture({featureSet: 'v4', hiddenLayers: 3});
  const policy = createBinaryPolicy(record, weights);
  assert.equal(policy.featureCount, FEATURE_COUNT_V4);
  for (const text of ['score 2 cups from the loader', 'park the robots', 'play the driver period for the best score']) {
    const played = runTask({spec: parseTask(text), policy, seed: 4, alliance: 'blue'});
    assert.ok(played.decisions > 0, text);
    assert.equal(typeof played.taskReturn, 'number');
    assert.ok(played.log.every(row => Number.isFinite(row.taskReward) && Number.isFinite(row.policyScore)));
  }
  assert.throws(() => createBinaryPolicy({...record, model: MODEL_ID_V3}, weights), /policy/);
  assert.throws(() => runTask({spec: parseTask('park the robots'), policy, seed: 4, environment: new TacticalEnvironment()}),
    /needs v4 features, but the environment produces v2/);
});
