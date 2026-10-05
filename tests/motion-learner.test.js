'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {randomUUID, createHash} = require('node:crypto');
const {loadHeadless} = require('../simulator/headless');
const Learner = require('../roboproof/motion-learner');
const {createServer} = require('../roboproof/server');
const {DEFAULT_TASK} = require('../simulator/motion');

function fixture(directory, trained = true) {
  const runId = randomUUID();
  const root = path.join(directory, runId, 'seed-42');
  fs.mkdirSync(root, {recursive: true});
  const model = {schemaVersion: 1, algorithm: 'ppo-beta-reference-v1', policyContractVersion: 1,
    observationSize: 34, actionSize: 4, width: 32, scales: Array(34).fill(1),
    identity: loadHeadless().identity, trainingSeed: 42, steps: trained ? 4 : 0, updates: trained ? 1 : 0, trained,
    layers: [[34, 32], [32, 32], [32, 8]].map(([inputs, outputs]) =>
      ({weight: Array.from({length: outputs}, () => Array(inputs).fill(0)), bias: Array(outputs).fill(0)}))};
  const save = (filename, value) => {
    const bytes = JSON.stringify(value);
    fs.writeFileSync(path.join(root, filename), bytes);
    return createHash('sha256').update(bytes).digest('hex');
  };
  const policyFile = 'policy-0001.json';
  const policySha256 = save(policyFile, model);
  const initialSha256 = save('initial-policy.json', {...model, updates: 0, steps: 0, trained: false});
  fs.writeFileSync(path.join(directory, runId, 'run.json'), JSON.stringify({schemaVersion: 1, runId, device: 'cpu',
    status: 'completed', identity: model.identity, learnedImprovementVerified: false,
    models: [{seed: 42, steps: model.steps, updates: model.updates, trained, actorWeightsChanged: trained,
      policyFile, policySha256, initialSha256}]}));
  fs.writeFileSync(path.join(directory, 'latest.json'), JSON.stringify({schemaVersion: 1, runId}));
  return {model, filename: path.join(root, policyFile), policySha256};
}

test('bounded JSON inference rejects altered, nonfinite and stale policies', context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'motion-model-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const {model, filename, policySha256} = fixture(directory);
  const loaded = Learner.policy(filename, policySha256);
  assert.deepEqual(loaded.act(Array(34).fill(0)), [0, 0, 0, 0]);
  assert.equal(loaded.identity.kind, 'learned');
  assert.throws(() => loaded.act(Array(34).fill(Infinity)), /sensor/);
  assert.throws(() => Learner.policy(filename, 'f'.repeat(64)), /hash/);
  model.layers[2].bias[0] = 18;
  fs.writeFileSync(filename, JSON.stringify(model));
  assert(Learner.policy(filename).act(Array(34).fill(0))[0] > 0.8);
  model.layers[0].weight[0].pop();
  fs.writeFileSync(filename, JSON.stringify(model));
  assert.throws(() => Learner.policy(filename), /weights/);
  model.identity.sources['engine.js'] = 'f'.repeat(64);
  fs.writeFileSync(filename, JSON.stringify(model));
  assert.throws(() => Learner.policy(filename), /identity mismatch/);
});

test('acceptance never hides success, contact or efficiency regressions', () => {
  const summary = (successRate, time = 10, effort = 10) => ({successRate,
    meanElapsedSecondsAllWorlds: time, meanEffortProxyVAsAllWorlds: effort});
  const rules = {maximumSuccessRegressions: 0, minimumMeanTimeOrEffortImprovementFraction: 0.05};
  const rows = [{id: 'one', baseline: {reason: 'success', metrics: {contactSeconds: 0}},
    learned: {reason: 'success', metrics: {contactSeconds: 0}}}];
  const accepted = () => Learner.acceptance(rows, summary(0.25), summary(0.5, 9), summary(0.25), rules);
  assert.equal(accepted().passed, true);
  rows[0].learned.reason = 'time_limit';
  assert.equal(accepted().passed, false);
  rows[0].learned.reason = 'success';
  rows[0].learned.metrics.contactSeconds = 0.01;
  assert.equal(accepted().passed, false);
  rows[0].learned.metrics.contactSeconds = 0;
  assert.equal(Learner.acceptance(rows, summary(0.5), summary(0.5, 9), summary(0.25), rules).passed, false);
  assert.equal(Learner.acceptance(rows, summary(0.25), summary(0.5), summary(0.25), rules).passed, false);
});

test('explicit learned experiments share canonical transitions and replay; status never invents improvement', context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'motion-learned-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  assert.equal(Learner.status(directory).available, false);
  fixture(directory);
  assert.equal(Learner.status(directory).learnedImprovementVerified, false);
  const result = Learner.runLearned({seed: 12, task: {...DEFAULT_TASK, deadlineSeconds: 0.03}, directory});
  assert.equal(result.exactReplayVerified, true);
  assert.equal(result.inferencePerformed, true);
  assert.equal(result.report.transitions.length, 3);
  assert.equal(result.report.options.policyIdentity.kind, 'learned');
  assert.match(result.report.learningStatus, /requires independent/);
  fs.writeFileSync(path.join(directory, 'latest.json'), JSON.stringify({schemaVersion: 1, runId: '../escape'}));
  assert.equal(Learner.status(directory).available, false);
  assert.throws(() => Learner.runLearned({directory}), /pointer/);
});

test('learning API is read-only until explicit experiment and blocks paths, cloud and cross-origin requests', async context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'motion-learning-api-'));
  fixture(directory);
  const server = createServer({reportPath: path.join(directory, 'missing'), learningDirectory: directory,
    motionDirectory: path.join(directory, 'sessions')});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(async () => {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(directory, {recursive: true, force: true});
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  assert.equal((await (await fetch(`${base}/api/motion/learning`)).json()).learnedImprovementVerified, false);
  const post = (body, origin) => fetch(`${base}/api/motion/run`, {method: 'POST',
    headers: {'Content-Type': 'application/json', ...(origin ? {Origin: origin} : {})}, body: JSON.stringify(body)});
  const body = {mode: 'learned-experiment', seed: 11, task: {...DEFAULT_TASK, deadlineSeconds: 0.03}};
  assert.equal((await post({...body, policyFile: '../escape'})).status, 400);
  assert.equal((await post(body, 'https://evil.example')).status, 403);
  assert.equal((await post({...body, device: 'cloud'})).status, 400);
  const response = await post(body);
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.motionPolicyTrained, true);
  assert.equal(result.exactReplayVerified, true);
  assert.equal(result.learnedImprovementVerified, false);
});
