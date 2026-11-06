'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {createServer} = require('../roboproof/server');

test('the Tactical AI training page, its script and its saved data are served read-only', async context => {
  const server = createServer({reportPath: 'missing-report-fixture.json'});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(async () => { await new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }); });
  const base = `http://127.0.0.1:${server.address().port}`;
  const page = await fetch(`${base}/training.html`);
  assert.equal(page.status, 200);
  assert.match(page.headers.get('content-type'), /text\/html/);
  const text = await page.text();
  for (const id of ['best-cards', 'training-chart', 'match-chart', 'network-chart']) assert.match(text, new RegExp(`id="${id}"`));
  assert.match(await (await fetch(`${base}/training.js`)).text(), /training-data\.json/);
  const data = await (await fetch(`${base}/training-data.json`)).json();
  assert.equal(data.kind, 'tactical-training-dashboard');
  assert.equal(data.learnedImprovementVerified, false);
  const single = data.singleGoal;
  assert.equal(single.runs.length, 34);
  assert.ok(single.runs.every(run => run.curve.length > 1 && run.curve.every(([episode, value]) => Number.isInteger(episode) && Number.isFinite(value))));
  assert.equal(single.independentCheck.primary.index, single.best.index);
  assert.equal(single.independentCheck.primary.versusLoaderCycle.passes, true);
  const replay = await (await fetch(`${base}/training-replay.json`)).json();
  assert.equal(replay.kind, 'tactical-training-replay');
  assert.equal(replay.featureNames.length, replay.decisions[0].inputs.length);
  assert.ok(replay.decisions.every(row => row.layers.length === replay.policy.hiddenLayers && row.layers[0].cells.length === 64));
  assert.equal(replay.decisions.at(-1).scoreAfter.margin, replay.finalScore.margin);
  const task = await (await fetch(`${base}/training-replay-task.json`)).json();
  assert.equal(task.mode, 'task');
  assert.equal(task.policy.featureSet, 'v4');
  assert.equal(task.featureNames.length, 52);
  assert.ok(task.decisions.every(row => row.inputs.length === 52 && row.layers.length === task.policy.hiddenLayers));
  assert.equal(typeof task.result.achieved, 'boolean');
  assert.equal(task.baseline.policy, 'task-greedy');
  assert.equal((await fetch(`${base}/training-data.json`, {method: 'POST', body: '{}'})).status, 404);
});
