'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const http = require('node:http');
const {createServer} = require('../roboproof/server');
const {makeScenario} = require('../roboproof/core');
const {runScenario} = require('../roboproof/sim');
const {investigate, invoke} = require('../roboproof/agent');

test('loopback API runs actual production controller and rejects unsafe requests', async context => {
  const server = createServer({reportPath: 'does-not-exist.json'});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  assert.equal((await fetch(`${base}/api/report`)).status, 404);
  const hostStatus = await new Promise((resolve, reject) => {
    const request = http.get(`${base}/api/report`, {headers: {Host: 'evil.example'}}, response => { response.resume(); resolve(response.statusCode); });
    request.on('error', reject);
  });
  assert.equal(hostStatus, 403);
  const post = (endpoint, value, origin) => fetch(`${base}${endpoint}`, {method: 'POST', headers: {'Content-Type': 'application/json', ...(origin ? {Origin: origin} : {})}, body: JSON.stringify(value)});
  assert.equal((await post('/api/run', {count: 1, seed: 42}, 'https://evil.example')).status, 403);
  assert.equal((await post('/api/run', {count: 1001, seed: 42})).status, 400);
  assert.equal((await post('/api/replay', {scenario: {}})).status, 400);
  assert.equal((await fetch(`${base}/api/run`, {method: 'POST', body: '{}'})).status, 415);
  assert.equal((await fetch(`${base}/core.js`)).status, 404);
  const page = await fetch(`${base}/`);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /Watch in simulator/);
  const player = await fetch(`${base}/player.js`);
  assert.equal(player.status, 200);
  assert.match(player.headers.get('content-type'), /javascript/);
  assert.match(await player.text(), /RoboProofPlayer/);
  const response = await post('/api/run', {count: 2, seed: 42});
  assert.equal(response.status, 200);
  const report = await response.json();
  assert.equal(report.summary.count, 2);
  assert.equal(report.backend, 'cpu-wasm');
  assert.equal(report.nominal.passed, true);
  assert.equal(report.benchmark.gpu, 'NOT MEASURED');
  assert.deepEqual((await (await fetch(`${base}/api/report`)).json()).summary, report.summary);
  const replayResponse = await post('/api/replay', {scenario: report.counterexample.scenario});
  assert.equal(replayResponse.status, 200);
  const replay = await replayResponse.json();
  assert.deepEqual(replay.metrics, report.counterexample.metrics);
  assert.equal(replay.telemetry.length, 1000);
});

test('external-model protocol requires source, telemetry, counterfactual and valid evidence', async () => {
  const result = runScenario(makeScenario(9, {task: {duration: 0.1}}));
  const scripted = [
    {type: 'tool', name: 'inspect_controller', arguments: {}},
    {type: 'tool', name: 'find_first_divergence', arguments: {}},
    {type: 'tool', name: 'run_parameter_sweep', arguments: {parameter: 'friction', values: [0.1, 0.85]}},
    {type: 'final', observations: ['Test protocol fixture only'], hypotheses: [], limitations: ['No real LLM used in this unit test'], evidence: [0, 1, 2]}
  ];
  const diagnosis = await investigate(result, [], {provider: async () => scripted.shift()});
  assert.equal(diagnosis.tool_log.length, 3);
  assert.equal(diagnosis.mode, 'external-model-tool-agent');
  await assert.rejects(investigate(result, [], {provider: async () => ({type: 'tool', name: 'shell', arguments: {}})}), /Unknown/);
  await assert.rejects(investigate(result, [], {provider: async () => ({type: 'final', observations: [], hypotheses: [], limitations: [], evidence: [999]})}), /missing tool evidence/);
  await assert.rejects(investigate(result, [], {maxTurns: 1, provider: async () => ({type: 'tool', name: 'get_scenario', arguments: {}})}), /turn budget/);
});

test('model command transports JSON through stdin without shell interpolation', async () => {
  const context = {value: '$(do-not-run); | & "quoted"'};
  const response = await invoke([process.execPath, '-e', "let input='';process.stdin.on('data',data=>input+=data);process.stdin.on('end',()=>process.stdout.write(input));"], context);
  assert.deepEqual(response, context);
  await assert.rejects(invoke([], context), /argv/);
});
