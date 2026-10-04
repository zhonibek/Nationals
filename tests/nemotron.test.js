'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const {configuration, createClient, planTask, MAX_TURNS} = require('../roboproof/nemotron');
const {createStore} = require('../roboproof/nemotron-store');
const {createServer} = require('../roboproof/server');
const {replay} = require('../roboproof/motion');

const TASK = {start: {xIn: 0, yIn: 0, headingDeg: 0}, goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 10};
const metadata = {provider: 'test-fixture-not-real-inference', model: 'robotai-nemotron', baseUrl: 'http://127.0.0.1:8080/v1', cloudEnabled: false};
function tool(name, argumentsValue = {}, id = name) {
  return {choices: [{finish_reason: 'tool_calls', message: {role: 'assistant', content: null,
    tool_calls: [{id, type: 'function', function: {name, arguments: JSON.stringify(argumentsValue)}}]}}],
  usage: {prompt_tokens: 100, completion_tokens: 20, total_tokens: 120}};
}
function fixture(responses) {
  const requests = [];
  return {metadata, requests, check: async () => ({...metadata, available: true}),
    complete: async messages => { requests.push(structuredClone(messages)); return responses.shift(); }};
}
function preparedFixture(task = TASK) {
  return fixture([tool('get_motion_contract'), tool('prepare_reach_pose', {task, summary: 'Test fixture proposal; not live model evidence.'})]);
}
function tempDirectory(context) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-nemotron-test-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  return directory;
}
async function listen(server, context) {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  return `http://127.0.0.1:${server.address().port}`;
}

test('Nemotron configuration is local-only and never silently selects another model', () => {
  assert.equal(configuration({}).model, 'robotai-nemotron');
  for (const baseUrl of ['https://api.nebius.com/v1', 'http://192.168.1.1/v1', 'file:///v1',
    'http://user:secret@127.0.0.1/v1', 'http://127.0.0.1/v1?secret=yes', 'http://127.0.0.1/other']) {
    assert.throws(() => configuration({ROBOTAI_NEMOTRON_BASE_URL: baseUrl}), /loopback/);
  }
  assert.throws(() => configuration({ROBOTAI_NEMOTRON_MODEL: 'qwen'}), /Nemotron model/);
  assert.throws(() => configuration({ROBOTAI_NEMOTRON_TIMEOUT_MS: 'Infinity'}), /timeout/);
});

test('Nemotron native tools read the contract then prepare without executing any motion', async () => {
  const client = preparedFixture();
  const result = await planTask('Go forward 24 inches', client);
  assert.deepEqual(result.task, TASK);
  assert.equal(result.status, 'prepared');
  assert.equal(result.execution, 'not run; a separate user approval is required');
  assert.equal(result.motionLearning, 'not implemented');
  assert.equal(result.toolLog.length, 2);
  assert.equal(client.requests[1].at(-1).role, 'tool');
  assert.match(client.requests[1].at(-1).content, /inches/);
  assert.equal(result.run, undefined);
});

test('Nemotron rejects unsupported tools, uninspected tasks, malformed calls and invalid bounds', async () => {
  await assert.rejects(planTask('x', fixture([tool('shell', {command: 'do-not-run'})])), /Unsupported Nemotron tool/);
  await assert.rejects(planTask('x', fixture([tool('prepare_reach_pose', {task: TASK, summary: 'x'})])), /inspect the motion contract/);
  for (const task of [{...TASK, goal: {...TASK.goal, xIn: 61}}, {...TASK, deadlineSeconds: 0.015},
    {...TASK, motorVolts: [12, 12, 12, 12]}, {...TASK, goal: {...TASK.goal, xIn: '24'}}]) {
    await assert.rejects(planTask('x', preparedFixture(task)));
  }
  const malformed = tool('get_motion_contract');
  malformed.choices[0].message.tool_calls[0].function.arguments = '{';
  await assert.rejects(planTask('x', fixture([malformed])), /arguments JSON/);
  const parallel = tool('get_motion_contract');
  parallel.choices[0].message.tool_calls.push(parallel.choices[0].message.tool_calls[0]);
  await assert.rejects(planTask('x', fixture([parallel])), /one Nemotron tool/);
});

test('clarifications never create executable tasks; turn/token budgets fail closed', async () => {
  const result = await planTask('Go there', fixture([{choices: [{finish_reason: 'stop', message: {role: 'assistant', content: 'Which coordinates and units?'}}]}]));
  assert.equal(result.status, 'clarification');
  assert.equal(result.task, undefined);
  await assert.rejects(planTask('x', fixture(Array.from({length: MAX_TURNS}, (_, index) => tool('get_motion_contract', {}, String(index))))), /turn budget/);
  await assert.rejects(planTask('x', fixture([{choices: [{finish_reason: 'length', message: {role: 'assistant', content: 'Incomplete'}}]}])), /bounded response/);
  await assert.rejects(planTask(' '.repeat(10), preparedFixture()), /prompt/);
  await assert.rejects(planTask('x'.repeat(4001), preparedFixture()), /prompt/);
});

test('OpenAI-compatible local transport uses native tools, bounded output and private server-side auth', async context => {
  const requests = [];
  const provider = http.createServer(async (request, response) => {
    let body = '';
    for await (const chunk of request) body += chunk;
    requests.push({url: request.url, authorization: request.headers.authorization, body: body ? JSON.parse(body) : null});
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify(request.url.endsWith('/models') ? {data: [{id: metadata.model}]} : tool('get_motion_contract')));
  });
  const base = await listen(provider, context);
  const config = configuration({ROBOTAI_NEMOTRON_BASE_URL: `${base}/v1`, ROBOTAI_NEMOTRON_API_KEY: 'fixture-private-key'});
  const client = createClient(config);
  assert.equal((await client.check()).liveInferenceVerified, false);
  await client.complete([{role: 'user', content: 'fixture'}]);
  assert.equal(requests[1].body.max_tokens, 1024);
  assert.equal(requests[1].body.parallel_tool_calls, false);
  assert.equal(requests[1].body.chat_template_kwargs.enable_thinking, false);
  assert.deepEqual(requests[1].body.tools.map(row => row.function.name), ['get_motion_contract', 'prepare_reach_pose']);
  assert.equal(requests[1].authorization, 'Bearer fixture-private-key');
  assert.equal(JSON.stringify(client.metadata).includes('fixture-private-key'), false);
});

test('local transport rejects redirects, mismatched model IDs, oversized or malformed responses', async context => {
  let mode = 'redirect';
  const provider = http.createServer((request, response) => {
    if (mode === 'redirect') { response.writeHead(302, {Location: 'https://example.com/do-not-contact'}); response.end(); }
    else response.end(mode === 'huge' ? ' '.repeat(262145) : mode === 'bad' ? '{' : JSON.stringify({data: [{id: 'other'}]}));
  });
  const base = await listen(provider, context);
  const client = createClient(configuration({ROBOTAI_NEMOTRON_BASE_URL: `${base}/v1`}));
  await assert.rejects(client.check(), /Cannot connect/);
  mode = 'wrong-model'; await assert.rejects(client.check(), /not served/);
  mode = 'huge'; await assert.rejects(client.check(), /exceeds/);
  mode = 'bad'; await assert.rejects(client.check(), /invalid JSON/);
});

test('cancelled inference cannot execute or persist an approved task', async () => {
  const controller = new AbortController();
  controller.abort();
  const client = createClient(configuration({ROBOTAI_NEMOTRON_BASE_URL: 'http://127.0.0.1:1/v1'}));
  await assert.rejects(planTask('Go forward 24 inches', client, {signal: controller.signal}), /cancelled/);
});

test('session persistence survives store recreation and rejects path traversal', async context => {
  const directory = tempDirectory(context);
  const store = createStore(directory);
  const record = store.create('fixture prompt', await planTask('x', preparedFixture()));
  assert.deepEqual(createStore(directory).read(record.id), record);
  assert.deepEqual(createStore(directory).latest(), record);
  for (const id of ['../secret', '/absolute', 'arbitrary.json', null]) assert.throws(() => store.read(id), /ID/);
  assert.equal(fs.readdirSync(directory).filter(name => name.endsWith('.tmp')).length, 0);
});

test('dashboard API requires stored approval, executes original Simulator, replays exactly and retains evidence', async context => {
  const directory = tempDirectory(context);
  const server = createServer({reportPath: 'missing.json', nemotronDirectory: directory, nemotronClient: preparedFixture()});
  const base = await listen(server, context);
  const post = (route, body, origin) => fetch(`${base}/api/nemotron/${route}`, {method: 'POST',
    headers: {'Content-Type': 'application/json', ...(origin ? {Origin: origin} : {})}, body: JSON.stringify(body)});
  assert.equal((await fetch(`${base}/api/nemotron/latest`)).status, 404);
  assert.equal((await post('check', {}, 'https://evil.example')).status, 403);
  assert.equal((await post('plan', {prompt: 'x', baseUrl: 'https://evil.example'})).status, 400);
  assert.equal((await post('run', {task: TASK})).status, 400);
  const planResponse = await post('plan', {prompt: 'fixture: go 24 inches'});
  assert.equal(planResponse.status, 200);
  const plan = await planResponse.json();
  assert.equal(plan.run, undefined);
  assert.equal((await post('run', {id: plan.id, motorVolts: [12]})).status, 400);
  const runResponse = await post('run', {id: plan.id});
  assert.equal(runResponse.status, 200);
  const record = await runResponse.json();
  assert.equal(record.run.report.identity.canonicalSimulator, true);
  assert.equal(record.run.report.reason, 'success');
  assert.equal(record.run.exactReplayVerified, true);
  assert.deepEqual(replay(record.run.report), record.run.report);
  assert.deepEqual(createStore(directory).read(record.id), record);
  assert.deepEqual(await (await post('run', {id: record.id})).json(), record);
  const original = await fetch(`${base}/simulator/index.html?nemotron=${record.id}`);
  assert.equal(original.status, 200);
  assert.match(original.headers.get('content-security-policy'), /'wasm-unsafe-eval'/);
  assert.match(await original.text(), /Load Nemotron task/);
  assert.equal((await fetch(`${base}/simulator/control.wasm`)).headers.get('content-type'), 'application/wasm');
  assert.equal((await fetch(`${base}/simulator/native/control.cpp`)).status, 404);
  assert.equal((await fetch(`${base}/simulator/nemotron-task.js`)).status, 200);
});

test('API cannot approve a clarification and times out bounded agent requests', async context => {
  const directory = tempDirectory(context);
  const store = createStore(directory);
  const record = store.create('fixture', {schemaVersion: 1, status: 'clarification', message: 'Which target?'});
  const client = {...preparedFixture(), complete: async (messages, signal) => new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(Error('fixture cancelled')), {once: true});
  })};
  const server = createServer({reportPath: 'missing.json', nemotronDirectory: directory, nemotronClient: client, agentTimeout: 20});
  const base = await listen(server, context);
  const post = (route, body) => fetch(`${base}/api/nemotron/${route}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  assert.equal((await post('run', {id: record.id})).status, 400);
  const response = await post('plan', {prompt: 'fixture'});
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /cancelled/);
  assert.equal(fs.readdirSync(directory).length, 1);
});
