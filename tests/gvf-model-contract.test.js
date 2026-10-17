'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {spawnSync} = require('node:child_process');
const Gvf = require('../simulator/gvf-policy');
const Model = require('../roboproof/motion-gvf-learner');

function observation() {
  return {contractName: Gvf.CONTRACT_NAME, contractVersion: Gvf.CONTRACT_VERSION, observationSchema: Gvf.OBSERVATION_SCHEMA,
    pose: [0.5, 0.8, Math.PI / 2], velocity: [1, 2, 0.3], wheelRimSpeeds: [0, 0, 0, 0], targetError: [0.1, 0.2, 0, 1],
    previousAction: [0, 0, 0], remainingSeconds: 4, baseReference: [0.8, 1.2, Math.PI, 0.1, 0.2, 0.4],
    projection: 0, remainingArcMeters: 0, crossTrackMeters: 0, appliedFieldOffset: [0.02, 0.03, 0.01],
    offsetVelocity: [0.01, 0.02, 0.03], actionAgeTicks: 0};
}

function artifact(depth = 2, identity = {engine: 'mock-original-identity', canonicalSimulator: true}, width = 32) {
  const sizes = [[36, width], ...Array.from({length: depth - 1}, () => [width, width]), [width, 3]];
  return {schemaVersion: 1, algorithm: Model.ALGORITHM, policyContractName: Gvf.CONTRACT_NAME,
    policyContractVersion: Gvf.CONTRACT_VERSION, observationSchema: Gvf.OBSERVATION_SCHEMA, observationSize: 34, actionSize: 3,
    featureTransform: Model.FEATURE_TRANSFORM, featureInputSize: 36, scales: [...Model.SCALES], normalization: structuredClone(Model.NORMALIZATION),
    width, hiddenLayers: depth, trainingSeed: 17, steps: 0, updates: 0, trained: false, identity,
    layers: sizes.map(([inputs, outputs]) => ({weight: Array.from({length: outputs}, () => Array(inputs).fill(0)), bias: Array(outputs).fill(0)}))};
}

test('separate shallow/deep synthetic GVF networks emit three exact zero actions', () => {
  for (const depth of [2, 10]) {
    const data = artifact(depth), model = Model.fromArtifact(data, data.identity);
    assert.deepEqual(model.act(observation()), [0, 0, 0]);
    data.layers.at(-1).bias[0] = 1;
    model.data.layers.at(-1).bias[1] = 1;
    assert.deepEqual(model.act(observation()), [0, 0, 0]);
  }
});

test('512-wide ten-layer GVF actors load and infer with exact dimensions', () => {
  const data = artifact(10, {engine: 'mock-original-identity', canonicalSimulator: true}, 512);
  const model = Model.fromArtifact(data, data.identity);
  assert.equal(model.data.width, 512);
  assert.equal(model.data.hiddenLayers, 10);
  assert.equal(model.data.layers.length, 11);
  assert.deepEqual(model.act(observation()), [0, 0, 0]);
});

test('sensor-relative feature transform matches declared body axes without mutating input', () => {
  const source = observation(), original = structuredClone(source), features = Model.encodeObservation(source);
  assert.equal(features.length, 36);
  for (const [index, expected] of [[2, 1], [3, -2], [4, 1], [18, -0.4], [19, 0.3], [20, 1], [27, -0.03], [28, 0.02], [34, 0], [35, 0]]) {
    assert(Math.abs(features[index] - expected) < 1e-12);
  }
  assert.deepEqual(source, original);
});

test('synthetic fixed output uses tanh rather than a legacy Beta or pace head', () => {
  const data = artifact();
  data.layers.at(-1).bias = [0.3, -0.7, 1];
  const model = Model.fromArtifact(data, data.identity);
  assert.deepEqual(model.act(observation()), data.layers.at(-1).bias.map(Math.tanh));
});

test('raw legacy vectors, truth additions, invalid tags and bad sensor values are refused', () => {
  for (const patch of [{truth: [0, 0, 0]}, {contractName: 'policy'}, {contractVersion: 1}, {previousAction: [1, 0, 0, 0]},
    {pose: [NaN, 0, 0]}, {pose: [false, 0, 0]}, {actionAgeTicks: 6001}]) assert.throws(() => Model.encodeObservation({...observation(), ...patch}));
  assert.throws(() => Model.encodeObservation(Array(34).fill(0)));
});

test('old algorithms, wrong dimensions, normalizations and metadata cannot become GVF weights', () => {
  const data = artifact();
  for (const patch of [{algorithm: 'td3-beta-mean-reference-v1'}, {policyContractVersion: 1}, {actionSize: 4},
    {featureInputSize: 34}, {featureTransform: 'deadline-context-v1'}, {width: true}, {hiddenLayers: 3},
    {trained: true}, {steps: -1}, {normalization: {method: 'fitted', clip: [-10, 10]}}, {extra: 1}]) {
    assert.throws(() => Model.fromArtifact({...data, ...patch}, data.identity));
  }
  const booleanScale = artifact();
  booleanScale.scales[2] = true;
  assert.throws(() => Model.fromArtifact(booleanScale, booleanScale.identity));
  assert.throws(() => Model.fromArtifact(data, {...data.identity, canonicalSimulator: 1}));
});

test('sparse, malformed, nonfinite and excessive synthetic layers fail closed', () => {
  for (const mutate of [value => value.layers.pop(), value => value.layers[0].weight[0].pop(),
    value => { value.layers[0].weight = new Array(32); }, value => { value.layers[0].bias[0] = Infinity; },
    value => { value.layers[0].bias[0] = 10001; }, value => { value.layers.at(-1).bias.push(0); }]) {
    const data = artifact(); mutate(data);
    assert.throws(() => Model.fromArtifact(data, data.identity));
  }
});

test('bounded file reader verifies exact artifact hash and current original engine identity without physics', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gvf-actor-contract-'));
  try {
    const identity = require('../simulator/headless').loadHeadless().identity;
    const filename = path.join(root, 'synthetic-unit-fixture.json'), data = artifact(2, identity);
    const bytes = JSON.stringify(data), sha256 = createHash('sha256').update(bytes).digest('hex');
    fs.writeFileSync(filename, bytes);
    const model = Model.policy(filename, sha256);
    assert.equal(model.identity.kind, 'untrained');
    assert.deepEqual(model.act(observation()), [0, 0, 0]);
    assert.throws(() => require('../roboproof/motion-learner').policy(filename));
    assert.throws(() => Model.policy(filename, '0'.repeat(64)), /SHA256 mismatch/);
    data.identity.engine = 'not the original'; fs.writeFileSync(filename, JSON.stringify(data));
    assert.throws(() => Model.policy(filename), /engine identity/);
  } finally {
    const target = path.resolve(root), parent = path.resolve(os.tmpdir());
    assert(target.startsWith(parent + path.sep) && path.basename(target).startsWith('gvf-actor-contract-'));
    fs.rmSync(target, {recursive: true, force: true});
  }
});

test('file growth cannot turn the actor reader into an unbounded read', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gvf-actor-contract-'));
  const filename = path.join(root, 'synthetic-growth-fixture.json');
  const data = artifact(2, require('../simulator/headless').loadHeadless().identity);
  fs.writeFileSync(filename, JSON.stringify(data));
  const size = fs.statSync(filename).size, originalRead = fs.readSync;
  let observedLimit;
  try {
    fs.readSync = function(descriptor, buffer, offset, length, position) {
      observedLimit = length;
      fs.appendFileSync(filename, ' '.repeat(1024));
      return originalRead.call(this, descriptor, buffer, offset, length, position);
    };
    assert.throws(() => Model.policy(filename), /changed while reading/);
    assert.equal(observedLimit, size + 1);
  } finally {
    fs.readSync = originalRead;
    const target = path.resolve(root), parent = path.resolve(os.tmpdir());
    assert(target.startsWith(parent + path.sep) && path.basename(target).startsWith('gvf-actor-contract-'));
    fs.rmSync(target, {recursive: true, force: true});
  }
});

test('stdlib Python and JavaScript agree on exact scales and actual tagged feature cases', () => {
  const cases = [observation(), {...observation(), pose: [0, 0, 0]}, {...observation(), pose: [-0.4, 0.1, -Math.PI + 1e-8]},
    {...observation(), pose: [0.8, -0.2, Math.PI - 1e-8]}];
  const python = process.env.ROBOPROOF_PYTHON || process.env.PYTHON || 'python';
  const program = 'import json,sys; from roboproof.motion_learning.gvf_contract import SCALES,encode_observation; print(json.dumps({"scales":SCALES,"features":[encode_observation(item) for item in json.load(sys.stdin)]}))';
  const result = spawnSync(python, ['-c', program], {cwd: path.resolve(__dirname, '..'), input: JSON.stringify(cases), encoding: 'utf8', timeout: 15000});
  assert.equal(result.status, 0, result.stderr || result.error?.message);
  const actual = JSON.parse(result.stdout);
  assert.deepEqual(actual.scales, [...Model.SCALES]);
  for (const [index, source] of cases.entries()) {
    const expected = Model.encodeObservation(source);
    for (const [column, value] of expected.entries()) assert(Math.abs(value - actual.features[index][column]) <= 1e-12);
  }
});
