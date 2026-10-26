'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createServer} = require('../roboproof/server');
const {fixturePolicyRecord} = require('./helpers/tactical-policy-fixture');

async function start(t, tacticalPolicyPath) {
  const server = createServer({reportPath: 'does-not-exist.json', tacticalPolicyPath});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  return body => fetch(`${base}/api/tactics/task`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
}

test('POST /api/tactics/task runs a typed task with the exported policy and labels it honestly', async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tactical-task-server-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const file = path.join(directory, 'policy.json');
  fs.writeFileSync(file, JSON.stringify(fixturePolicyRecord()));
  const post = await start(t, file);
  const response = await post({task: 'score 2 match loads into the neutral goals, then park', seed: 101, alliance: 'red'});
  const result = await response.json();
  assert.equal(response.status, 200, JSON.stringify(result));
  assert.equal(result.summary, 'Place 2 match loads in neutral goals, then park');
  assert.equal(result.achieved, true);
  assert.equal(result.scoringActions, 2);
  assert(result.log.length >= 3);
  assert.equal(result.policy.provenance.runId, 'fixture-run');
  assert.match(result.policy.algorithm, /fixture/);
  assert.equal(result.learnedImprovementVerified, false);
  assert.equal(result.independentQualificationPerformed, false);
  assert.match(result.scope, /no robot/);
});

test('POST /api/tactics/task refuses bad input and explains a missing policy', async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tactical-task-server-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const post = await start(t, path.join(directory, 'missing-policy.json'));
  const missing = await post({task: 'park the robots', seed: 1});
  assert.equal(missing.status, 400);
  assert.match((await missing.json()).error, /No trained tactical policy is exported/);
  for (const [body, error] of [
    [{task: 'park', seed: 1, extra: true}, /Unsupported tactical task fields/],
    [{task: 'park', seed: -1}, /uint32/],
    [{task: 'park', seed: 1, alliance: 'green'}, /red or blue/],
    [{task: 'hello there', seed: 1}, /Task not understood/]
  ]) {
    const response = await post(body);
    assert.equal(response.status, 400);
    assert.match((await response.json()).error, error);
  }
});
