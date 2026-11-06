'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {INSTRUCTIONS, taskTool, parseTaskCall, parseTaskWithNemotron} = require('../roboproof/tactical-task-nemotron');
const {SPEC_VALUES} = require('../roboproof/tactical_task_environment');
const {parseTask, MAX_TASK_LENGTH} = require('../roboproof/tactical_task');
const {createServer} = require('../roboproof/server');
const {fixturePolicyRecord} = require('./helpers/tactical-policy-fixture');

const ARGS = {understood: true, mode: 'score', objects: 'any', source: 'loader', goals: 'neutral', count: 3, points: null,
  seconds: null, park: true, note: 'three match loads into the neutral goals'};

const toolReply = (calls, extra = {}) => ({choices: [{finish_reason: 'tool_calls', message: {tool_calls: calls}}],
  usage: {prompt_tokens: 410, completion_tokens: 38, total_tokens: 448}, ...extra});
const setTask = args => ({type: 'function', function: {name: 'set_task', arguments: JSON.stringify(args)}});

function fakeClient(response) {
  const calls = [];
  return {calls, metadata: {provider: 'fake', model: 'fake-nemotron'},
    complete: async (messages, signal, tools) => { calls.push({messages, signal, tools}); return response; }};
}

test('any wording becomes the same task spec the pattern parser gives', async () => {
  const client = fakeClient(toolReply([setTask(ARGS)]));
  const parsed = await parseTaskWithNemotron('  закинь три лоада в нейтральные гоулы и потом запаркуйся ', client);
  assert.deepEqual(parsed.spec, parseTask('score 3 match loads into the neutral goals, then park'));
  assert.equal(parsed.note, ARGS.note);
  assert.equal(parsed.parser, 'nemotron');
  assert.deepEqual(parsed.provider, {provider: 'fake', model: 'fake-nemotron'});
  assert.deepEqual(parsed.usage, {prompt_tokens: 410, completion_tokens: 38, total_tokens: 448});
  const [call] = client.calls;
  assert.deepEqual(call.messages, [{role: 'system', content: INSTRUCTIONS},
    {role: 'user', content: 'закинь три лоада в нейтральные гоулы и потом запаркуйся'}]);
  assert.deepEqual(call.tools, taskTool());
});

test('the set_task tool only offers the allowlisted spec values', () => {
  const [tool] = taskTool();
  assert.equal(tool.function.name, 'set_task');
  const {parameters} = tool.function;
  assert.equal(parameters.additionalProperties, false);
  assert.deepEqual([...parameters.required].sort(), Object.keys(ARGS).sort());
  for (const key of Object.keys(SPEC_VALUES)) assert.deepEqual(parameters.properties[key].enum, SPEC_VALUES[key]);
  assert.equal(parameters.properties.seconds.maximum, 105);
});

test('a request that is not a task is refused with the model\'s reason', () => {
  const reply = toolReply([setTask({...ARGS, understood: false, note: 'This is a question\nabout the weather.'})]);
  assert.throws(() => parseTaskCall(reply), /did not understand this as a task: This is a question about the weather\./);
});

test('non-scoring modes drop scoring targets, a park task always parks and play never parks', () => {
  const park = parseTaskCall(toolReply([setTask({...ARGS, mode: 'park', count: 3, park: false})])).spec;
  assert.deepEqual(park, parseTask('park the robots'));
  const play = parseTaskCall(toolReply([setTask({...ARGS, mode: 'play', park: true, seconds: 40})])).spec;
  assert.deepEqual([play.mode, play.park, play.count, play.seconds, play.goals], ['play', false, null, 40, 'any']);
});

test('anything but one valid set_task call fails closed', () => {
  for (const [reply, error] of [
    [{choices: []}, /bounded response/],
    [toolReply([setTask(ARGS)], {choices: [{finish_reason: 'length', message: {tool_calls: [setTask(ARGS)]}}]}), /bounded response/],
    [{choices: [{finish_reason: 'stop', message: {content: 'Sure, I will score three loads.'}}]}, /exactly once/],
    [toolReply([setTask(ARGS), setTask(ARGS)]), /exactly once/],
    [toolReply([{type: 'function', function: {name: 'run_shell', arguments: '{}'}}]), /Invalid Nemotron tool call/],
    [toolReply([{type: 'function', function: {name: 'set_task', arguments: '{"mode":'}}]), /not valid JSON/],
    [toolReply([{type: 'function', function: {name: 'set_task', arguments: `{"note":"${'x'.repeat(5000)}"}`}}]), /Invalid Nemotron tool call/],
    [toolReply([setTask({...ARGS, command: 'rm -rf /'})]), /Unsupported set_task arguments/],
    [toolReply([setTask({...ARGS, goals: 'opponent'})]), /Task goals must be one of/],
    [toolReply([setTask({...ARGS, mode: 'defend'})]), /Task mode must be one of/],
    [toolReply([setTask({...ARGS, seconds: 200})]), /time limit/],
    [toolReply([setTask({...ARGS, count: 2.5})]), /number of scoring actions/],
    [toolReply([setTask({...ARGS, points: 0})]), /points target/]
  ]) assert.throws(() => parseTaskCall(reply), error);
});

test('empty or over-long wording is refused before Nemotron is called', async () => {
  const client = fakeClient(toolReply([setTask(ARGS)]));
  for (const text of ['', '   ', 'x'.repeat(MAX_TASK_LENGTH + 1), 42]) {
    await assert.rejects(parseTaskWithNemotron(text, client), /Describe the task/);
  }
  assert.equal(client.calls.length, 0);
});

async function start(t, nemotronClient) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tactical-task-nemotron-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const tacticalPolicyPath = path.join(directory, 'policy.json');
  fs.writeFileSync(tacticalPolicyPath, JSON.stringify(fixturePolicyRecord()));
  const server = createServer({reportPath: 'does-not-exist.json', tacticalPolicyPath, nemotronClient});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  return body => fetch(`${base}/api/tactics/task`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
}

test('POST /api/tactics/task with parser nemotron: Nemotron reads the wording, the trained policy plays it', async t => {
  const client = fakeClient(toolReply([setTask(ARGS)]));
  const post = await start(t, client);
  const response = await post({task: 'put three loads in the middle goals and park after', seed: 101, alliance: 'red', parser: 'nemotron'});
  const result = await response.json();
  assert.equal(response.status, 200, JSON.stringify(result));
  assert.equal(result.summary, 'Place 3 match loads in neutral goals, then park');
  assert.deepEqual(result.understanding, {parser: 'nemotron', note: ARGS.note,
    provider: {provider: 'fake', model: 'fake-nemotron'}, usage: {prompt_tokens: 410, completion_tokens: 38, total_tokens: 448}});
  assert.equal(result.scoringActions, 3);
  assert.equal(result.learnedImprovementVerified, false);
  assert.match(result.scope, /no robot/);
  assert.equal(client.calls.length, 1);

  const patterns = await (await post({task: 'score 3 match loads into the neutral goals, then park', seed: 101})).json();
  assert.deepEqual(patterns.understanding, {parser: 'patterns'});
  assert.deepEqual(patterns.log, result.log, 'the same spec and seed play the same match whichever parser read it');
  assert.equal(client.calls.length, 1, 'the pattern parser never calls Nemotron');
});

test('POST /api/tactics/task refuses an unknown parser and passes on Nemotron refusals', async t => {
  const post = await start(t, fakeClient(toolReply([setTask({...ARGS, understood: false, note: 'Not a robot task.'})])));
  const unknown = await post({task: 'park', seed: 1, parser: 'regex'});
  assert.equal(unknown.status, 400);
  assert.match((await unknown.json()).error, /patterns or nemotron/);
  const refused = await post({task: 'what is the weather', seed: 1, parser: 'nemotron'});
  assert.equal(refused.status, 400);
  assert.match((await refused.json()).error, /did not understand this as a task: Not a robot task\./);
  const after = await post({task: 'park the robots', seed: 1});
  assert.equal(after.status, 200, 'a refused Nemotron request releases the server');
});
