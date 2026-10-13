'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const {readJson, policy, policyFromSnapshot} = require('../roboproof/motion-learner');

const widths = [32, 64, 128, 256];
const identity = loadHeadless().identity;
const transforms = ['identity-v1', 'reference-frame-v1', 'history-context-v1', 'deadline-context-v1',
  'long-history-deadline-v1', 'settling-normalized-deadline-v1', 'coarse-augmentation-deadline-v1', 'multiscale-deadline-v1'];
const frameScales = [1.524, 1.524, 3.142, 2, 2, 4, 3, 3, 3, 3,
  1.524, 1.524, 1, 1, 1, 1, 1, 1, 10, 1.524, 1.524, 1, 2, 2, 4, 10, 1,
  0.114, 0.114, 0.1, 0.08, 0.08, 0.25, 40, 1];
const settlingFrameScales = {'3': 0.0254, '4': 0.0254, '5': 0.0873, '10': 0.02032, '11': 0.02032, '12': 0.035};

function model(width, featureTransform = 'identity-v1') {
  const augmented = ['coarse-augmentation-deadline-v1', 'multiscale-deadline-v1'].includes(featureTransform);
  const settling = featureTransform === 'settling-normalized-deadline-v1';
  const deadline = transforms.indexOf(featureTransform) >= 3;
  const temporal = transforms.indexOf(featureTransform) >= 2;
  const historyFrames = featureTransform === 'long-history-deadline-v1' ? 16 : 4;
  const frameSize = augmented ? 41 : 35;
  const featureCount = temporal ? historyFrames * frameSize + 4 + Number(deadline) : frameSize - Number(featureTransform === 'identity-v1');
  let scales = Array.from({length: featureCount}, (_, index) => Math.fround(0.75 + index % 7 / 10));
  if (deadline) scales[featureCount - 1] = 1;
  if (settling || augmented) {
    const frame = [...frameScales];
    if (settling) for (const [index, scale] of Object.entries(settlingFrameScales)) frame[Number(index)] = scale;
    if (augmented) frame.push(1, 1, 1, 1, 1, 1);
    scales = Array.from({length: 4}, () => frame).flat().concat([1.524, 1.524, 1, 1, 1]).map(Math.fround);
  }
  const data = {schemaVersion: 1, algorithm: 'ppo-beta-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width, featureTransform, featureInputSize: featureCount, scales,
    layers: [[featureCount, width], [width, width], [width, 8]].map(([inputs, outputs], layerIndex) => ({
      weight: Array.from({length: outputs}, (_, neuron) => Array.from({length: inputs}, (_, column) =>
        Math.fround((((neuron + 1) * 17 + (column + 1) * 13 + layerIndex * 7) % 101 - 50) / (50 * Math.sqrt(inputs))))),
      bias: Array.from({length: outputs}, (_, neuron) => Math.fround(((neuron * 7 + layerIndex * 3) % 17 - 8) / 13))})),
    identity: structuredClone(identity), trainingSeed: 721, steps: 0, updates: 0, trained: false};
  if (temporal) Object.assign(data, {historyFrames, initialContextSize: 4});
  if (deadline) data.deadlineContext = {schemaVersion: 1, settlingReserveSeconds: 2, input: 'public-reference-duration-at-reset'};
  if (settling) data.normalizationDefinition = {schemaVersion: 1, method: 'sensor-settling-units-v1', historyFrames: 4,
    frameScales: {...settlingFrameScales}, heading: 'Goal-heading sine divided by0.035; cosine retained at scale1, not a success classifier',
    initialContext: 'Unchanged field-scale initial sensor goal context', clip: [-10, 10],
    scope: 'Fixed public sensor/reference units only; no evaluator truth, fitted statistics or new input capacity'};
  if (augmented) {
    data.frameInputSize = 41;
    data.normalizationDefinition = {schemaVersion: 1, method: 'bounded-sensor-augmentation-v1', historyFrames: 4,
      baseFrameSize: 35, augmentedFrameSize: 41, sourceIndices: [3, 4, 5, 10, 11, 12],
      units: featureTransform === 'multiscale-deadline-v1' ? [0.0254, 0.0254, 0.0873, 0.02032, 0.02032, 0.035] : [2, 2, 4, 1.524, 1.524, 1],
      transform: 'tanh(reference-frame-feature/unit)', baseChannels: 'Unchanged field-scale reference-frame35',
      addedChannelScales: [1, 1, 1, 1, 1, 1], initialContext: 'Unchanged field-scale initial sensor goal context', clip: [-10, 10],
      scope: 'Derived public sensor/reference channels only; additional169-input capacity, no evaluator truth or fitted statistics'};
  }
  return data;
}

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const decode = data => policyFromSnapshot(data, digest(JSON.stringify(data)));

function observation(tick = 0) {
  const raw = Array.from({length: 34}, (_, index) => Math.sin((index + 1) * 0.41 + tick * 0.13) * 0.35);
  raw[0] = 20; raw[1] = -20; raw[18] = 8; raw[21] = 0.6; raw[25] = tick * 0.17;
  return Object.freeze(raw);
}

function forward(data, features) {
  let activations = features.map((value, index) => Math.min(10, Math.max(-10, value / data.scales[index])));
  for (let layerIndex = 0; layerIndex < 3; layerIndex++) {
    const layer = data.layers[layerIndex];
    const following = [];
    for (let neuron = 0; neuron < layer.bias.length; neuron++) {
      let dotProduct = 0;
      for (let column = 0; column < activations.length; column++) dotProduct += layer.weight[neuron][column] * activations[column];
      const logit = dotProduct + layer.bias[neuron];
      following.push(layerIndex === 2 ? 1 + Math.log1p(Math.exp(logit)) : Math.tanh(logit));
    }
    activations = following;
  }
  return Array.from({length: 4}, (_, axis) => 2 * (activations[axis] / (activations[axis] + activations[axis + 4])) - 1);
}

function referenceFrame(raw) {
  const frame = [...raw];
  const rotate = (horizontal, vertical, angle) => [Math.cos(angle) * horizontal - Math.sin(angle) * vertical,
    Math.sin(angle) * horizontal + Math.cos(angle) * vertical];
  for (const [horizontal, vertical] of [[3, 4], [22, 23], [27, 28], [30, 31]]) {
    [frame[horizontal], frame[vertical]] = rotate(raw[horizontal], raw[vertical], raw[21]);
  }
  [frame[19], frame[20]] = rotate(raw[19] - raw[0], raw[20] - raw[1], raw[21]);
  [frame[10], frame[11]] = rotate(raw[10], raw[11], raw[21] - raw[2]);
  frame[21] = Math.sin(raw[21] - raw[2]);
  frame.push(Math.cos(raw[21] - raw[2]));
  return frame;
}

function assertActions(actual, expected) {
  assert.equal(actual.length, 4);
  actual.forEach((value, axis) => {
    assert.ok(Number.isFinite(value) && value >= -1 && value <= 1);
    assert.ok(Math.abs(value - expected[axis]) < 1e-12, `axis ${axis}: ${value} != ${expected[axis]}`);
  });
}

function freeze(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

for (const width of widths) {
  test(`width ${width} independently computes dense nonzero-weight PPO and TD3 Beta means`, () => {
    const data = model(width), raw = observation();
    assert.ok(data.layers.every(layer => layer.weight.some(row => row.some(value => value !== 0))));
    const expected = forward(data, raw);
    assert.ok(expected.some(value => Math.abs(value) > 0.01));
    const ppo = decode(data);
    assertActions(ppo.act(raw), expected);
    data.algorithm = 'td3-beta-mean-reference-v1';
    const td3 = decode(data);
    assert.deepEqual(td3.act(raw), ppo.act(raw));
    assert.equal(td3.identity.id, 'td3-seed-721-update-0');
    assert.equal(td3.identity.kind, 'untrained');
    const changed = structuredClone(data);
    changed.layers[1].weight[width - 1].fill(0);
    assert.notDeepEqual(decode(changed).act(raw), td3.act(raw));
  });

  test(`width ${width} independently retains sixteen-frame public history, deadline and reset`, () => {
    const data = model(width, 'long-history-deadline-v1'), actor = decode(data);
    const duration = 4, frames = [], first = observation();
    assert.throws(() => actor.act(first), /reference duration/);
    actor.act.reset({referenceDurationSeconds: duration});
    for (let tick = 0; tick < 18; tick++) {
      const raw = observation(tick);
      frames.push(referenceFrame(raw));
      const history = Array.from({length: 16}, (_, slot) => frames[Math.max(0, tick - 15 + slot)]).flat();
      const deadline = Math.min(1, Math.max(0.25, (duration - raw[25]) / 6));
      const features = history.concat(first.slice(10, 14), deadline);
      assertActions(actor.act(raw), forward(data, features));
    }
    actor.act.reset({referenceDurationSeconds: duration});
    const fresh = decode(data);
    fresh.act.reset({referenceDurationSeconds: duration});
    assert.deepEqual(actor.act(first), fresh.act(first));
    actor.act.reset();
    assert.throws(() => actor.act(first), /reference duration/);
  });

  test(`width ${width} rejects every layer shape mismatch and dimensions from another declared width`, () => {
    const original = model(width);
    for (let layerIndex = 0; layerIndex < 3; layerIndex++) {
      const mutations = [layer => { layer.weight.pop(); }, layer => { layer.weight.push([...layer.weight[0]]); },
        layer => { layer.weight[0].pop(); }, layer => { layer.weight[0].push(0); },
        layer => { layer.bias.pop(); }, layer => { layer.bias.push(0); },
        layer => { layer.weight = {}; }, layer => { layer.bias = null; }, layer => { layer.weight[0] = null; }];
      for (const mutate of mutations) {
        const altered = structuredClone(original);
        mutate(altered.layers[layerIndex]);
        assert.throws(() => decode(altered), /Malformed motion model weights/);
      }
    }
    for (const otherWidth of widths.filter(value => value !== width)) {
      const altered = structuredClone(original);
      altered.width = otherWidth;
      assert.throws(() => decode(altered), /Malformed motion model weights/);
    }
    for (const mutate of [data => { data.layers.pop(); }, data => { data.layers.push(data.layers[0]); },
      data => { data.scales.pop(); }, data => { data.scales.push(1); }, data => { data.featureInputSize++; },
      data => { data.observationSize = 35; }, data => { data.actionSize = 8; }]) {
      const altered = structuredClone(original); mutate(altered);
      assert.throws(() => decode(altered), /feature count|Unsupported/);
    }
  });

  test(`width ${width} rejects nonfinite, nonnumeric, excessive and sparse parameters or sensor inputs`, () => {
    const original = model(width);
    for (const invalid of [NaN, Infinity, -Infinity, true, false, '0', null, undefined, 10001, -10001]) {
      for (const location of ['weight', 'bias', 'scale']) {
        const altered = structuredClone(original);
        if (location === 'weight') altered.layers[0].weight[0][0] = invalid;
        if (location === 'bias') altered.layers[2].bias[7] = invalid;
        if (location === 'scale') altered.scales[0] = invalid;
        assert.throws(() => decode(altered), /Malformed|Unsupported/);
      }
    }
    for (const invalid of [0, -1, 101]) {
      const altered = structuredClone(original); altered.scales[0] = invalid;
      assert.throws(() => decode(altered), /Unsupported/);
    }
    for (const mutate of [data => { delete data.scales[0]; }, data => { delete data.layers[0]; },
      data => { delete data.layers[0].weight[0]; }, data => { delete data.layers[0].weight[0][0]; },
      data => { delete data.layers[2].bias[0]; }]) {
      const altered = structuredClone(original); mutate(altered);
      assert.throws(() => decode(altered), /Malformed|Unsupported/);
    }
    const actor = decode(original);
    for (const invalid of [NaN, Infinity, -Infinity, true, false, '0', null, undefined]) {
      const raw = [...observation()]; raw[0] = invalid;
      assert.throws(() => actor.act(raw), /sensor/);
    }
    for (const raw of [Array(34), Array(33).fill(0), Array(35).fill(0), new Float32Array(34)]) {
      assert.throws(() => actor.act(raw), /sensor/);
    }
  });

  test(`width ${width} binds engine/controller identity and does not mutate frozen snapshots or inputs`, () => {
    for (const mutate of [data => { data.identity.sources['engine.js'] = 'f'.repeat(64); },
      data => { data.identity.controllerWasmSha256 = 'f'.repeat(64); }, data => { data.identity.runtime.node = 'other'; }]) {
      const altered = model(width); mutate(altered);
      assert.throws(() => decode(altered), /identity mismatch/);
    }
    const original = freeze(model(width, 'long-history-deadline-v1')), before = structuredClone(original);
    const actor = decode(original), raw = observation(), rawBefore = [...raw];
    const publicContext = Object.freeze({referenceDurationSeconds: 4});
    actor.act.reset(publicContext);
    const first = actor.act(raw);
    actor.act(observation(1));
    actor.act.reset(publicContext);
    assert.deepEqual(actor.act(raw), first);
    assert.deepEqual(original, before);
    assert.deepEqual(actor.data, before);
    assert.deepEqual(raw, rawBefore);
    assert.deepEqual(publicContext, {referenceDurationSeconds: 4});
    const mutable = model(width), captured = structuredClone(mutable), isolated = decode(mutable);
    const expected = isolated.act(raw);
    mutable.layers[2].bias.fill(10000); mutable.layers[0].weight[0].fill(-10000);
    mutable.scales.fill(100); mutable.width = 16; mutable.identity.sources['engine.js'] = 'f'.repeat(64);
    assert.deepEqual(isolated.act(raw), expected);
    assert.deepEqual(isolated.data, captured);
    assert.equal(isolated.sha256, digest(JSON.stringify(captured)));
  });
}

test('actor width is a supported numeric integer, never coerced or inferred from weights', () => {
  for (const invalid of [true, false, '32', '64', '128', '256', null, undefined, NaN, Infinity, -Infinity,
    0, -32, 16, 31, 33, 63, 65, 127, 129, 255, 257, 512, 32.5, 64.25, 128.5, 256.5, new Number(32)]) {
    const data = model(32); data.width = invalid;
    assert.throws(() => decode(data), /Unsupported bounded motion model/);
  }
});

test('legacy width32 actions remain bit-for-bit identical when zero-padded to larger supported capacities for every transform', () => {
  for (const featureTransform of transforms) {
    const original = model(32, featureTransform);
    if (featureTransform === 'identity-v1') { delete original.featureTransform; delete original.featureInputSize; }
    const actors = [decode(original)];
    for (const width of [64, 128, 256]) {
      const expanded = structuredClone(original); expanded.width = width;
      expanded.layers = [[original.scales.length, width], [width, width], [width, 8]].map(([inputs, outputs], layerIndex) => {
        const previous = original.layers[layerIndex];
        return {weight: Array.from({length: outputs}, (_, neuron) => Array.from({length: inputs}, (_, column) => previous.weight[neuron]?.[column] ?? 0)),
          bias: Array.from({length: outputs}, (_, neuron) => previous.bias[neuron] ?? 0)};
      });
      actors.push(decode(expanded));
    }
    for (const actor of actors) actor.act.reset({referenceDurationSeconds: 4});
    for (let tick = 0; tick < 18; tick++) {
      const raw = observation(tick), expected = actors[0].act(raw);
      for (const actor of actors.slice(1)) assert.deepEqual(actor.act(raw), expected, featureTransform);
    }
    for (const actor of actors) actor.act.reset({referenceDurationSeconds: 4});
    const expected = actors[0].act(observation());
    for (const actor of actors.slice(1)) assert.deepEqual(actor.act(observation()), expected, featureTransform);
  }
});

test('only actor files get a bounded 8MiB budget for dense width256 long-history float32 exports', context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'motion-capacity-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const data = model(256, 'long-history-deadline-v1'), bytes = JSON.stringify(data);
  const size = Buffer.byteLength(bytes), filename = path.join(directory, 'actor.json');
  assert.ok(size > 4 * 1024 * 1024 && size < 8 * 1024 * 1024);
  assert.equal(data.layers.reduce((count, layer) => count + layer.bias.length +
    layer.weight.reduce((weights, row) => weights + row.length, 0), 0), 212744);
  context.diagnostic(`Dense width256/565-input actor fixture: ${size} bytes; 212,744 float32 weight/bias values.`);
  fs.writeFileSync(filename, bytes);
  assert.throws(() => readJson(filename), /size budget/);
  assert.throws(() => readJson(filename, 4 * 1024 * 1024), /size budget/);
  const actor = policy(filename, digest(bytes)), snapshot = decode(data), raw = observation();
  actor.act.reset({referenceDurationSeconds: 4}); snapshot.act.reset({referenceDurationSeconds: 4});
  assert.deepEqual(actor.act(raw), snapshot.act(raw));
  assert.throws(() => policy(filename, 'f'.repeat(64)), /hash mismatch/);
  assert.throws(() => policyFromSnapshot(data, 'invalid'), /artifact hash/);
  fs.appendFileSync(filename, ' '.repeat(8 * 1024 * 1024 - size));
  assert.equal(fs.statSync(filename).size, 8 * 1024 * 1024);
  const boundary = policy(filename);
  boundary.act.reset({referenceDurationSeconds: 4});
  assert.deepEqual(boundary.act(raw), actor.act(raw));
  fs.appendFileSync(filename, ' ');
  assert.equal(fs.statSync(filename).size, 8 * 1024 * 1024 + 1);
  assert.throws(() => policy(filename), /size budget/);
  const generic = path.join(directory, 'generic.json');
  fs.writeFileSync(generic, JSON.stringify({ok: true}));
  assert.deepEqual(readJson(generic).value, {ok: true});
  fs.appendFileSync(generic, ' '.repeat(1024 * 1024));
  assert.throws(() => readJson(generic), /size budget/);
  assert.deepEqual(readJson(generic, 4 * 1024 * 1024).value, {ok: true});
  fs.appendFileSync(generic, ' '.repeat(4 * 1024 * 1024 - fs.statSync(generic).size));
  assert.equal(fs.statSync(generic).size, 4 * 1024 * 1024);
  assert.deepEqual(readJson(generic, 4 * 1024 * 1024).value, {ok: true});
  fs.appendFileSync(generic, ' ');
  assert.throws(() => readJson(generic, 4 * 1024 * 1024), /size budget/);
});
