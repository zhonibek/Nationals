'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const Game = require('../simulator/override');
const Snapshot = require('../simulator/tactics-snapshot');
const Tactics = require('../roboproof/tactics');
const {planTask} = require('../roboproof/nemotron');
const {createServer} = require('../roboproof/server');

function snapshot() { const game = new Game(); return Snapshot.capture(game.getState(), 'red-1', new Date().toISOString(), game.world); }
function tool(name, argumentsValue = {}, id = name) {
  return {choices: [{finish_reason: 'tool_calls', message: {role: 'assistant', tool_calls: [
    {type: 'function', id, function: {name, arguments: JSON.stringify(argumentsValue)}}]}}]};
}
function fixture(responses) {
  const calls = [];
  return {calls, metadata: {provider: 'tactics-fixture-not-real-inference', cloudEnabled: false}, check: async () => ({}),
    complete: async (messages, signal, tools) => {
      calls.push({messages: structuredClone(messages), tools: tools.map(tool => tool.function.name)});
      assert(responses.length, 'Fixture exhausted');
      return responses.shift();
    }};
}
const tacticalFixture = () => fixture([tool('load_skill', {name: 'plan-game-tactics'}), tool('get_game_rules'),
  tool('get_game_snapshot'), tool('finish_analysis', {message: 'Fixture advice only. No action, score or winning policy is demonstrated.'})]);

test('snapshot captures original game state without mutation and has bounded partial visibility', () => {
  const game = new Game();
  const state = game.getState();
  const before = JSON.stringify(state);
  const captured = Snapshot.capture(state, 'red-1', new Date().toISOString(), game.world);
  assert.equal(JSON.stringify(state), before);
  assert.equal(captured.selectedRobotId, 'red-1');
  assert.equal(captured.robots.length, 4);
  assert.equal(captured.goals.length, 9);
  assert.equal(captured.objects.filter(object => object.kind === 'pin').length, 6);
  assert.equal(captured.objects.filter(object => object.kind === 'cup').length, 4);
  assert(Buffer.byteLength(JSON.stringify(captured)) <= Snapshot.MAX_BYTES);
  assert.deepEqual(captured.score, state.score);
  assert.equal(Tactics.gameEvidence(captured).executableTask, false);
  assert.match(Tactics.gameEvidence(captured).provenance, /not independently/);
});

test('unknown games, instruction fields, bad IDs, inconsistent phases and stale snapshots fail closed', () => {
  const initial = snapshot();
  for (const change of [
    {manualVersion: '3.0'}, {season: 'other'}, {world: 'worlds'}, {instructions: 'execute shell'}, {selectedRobotId: '../secret'},
    {phase: 'driver', clock: 0}, {phase: 'autonomous', clock: 15},
    {capturedAt: new Date(Date.now() - 61000).toISOString()}, {capturedAt: new Date(Date.now() + 6000).toISOString()},
    {robots: initial.robots.map(robot => ({...robot, x: Infinity}))},
    {goals: initial.goals.map(goal => ({...goal, id: '../file'}))},
    {objects: [...initial.objects, initial.objects[0]]}, {score: {red: -1, blue: 0}},
    {resources: {fieldPins: 0, fieldCups: 0}}
  ]) assert.throws(() => Snapshot.validate({...initial, ...change}), /tactics|Tactics/);
});

test('rule evidence uses existing engine constants and hashes, not invented scoring or sub-AI capabilities', () => {
  const rules = Tactics.rules();
  assert.equal(rules.points.alliancePin, Game.constants.POINTS.alliancePin);
  assert.equal(rules.points.yellowPin, Game.constants.POINTS.yellowPin);
  assert.equal(rules.points.cupIndependentPoints, 0);
  assert.equal(rules.timing.endgameSeconds, Game.constants.ENDGAME_SECONDS);
  assert.equal(Object.keys(rules.engineHashes).length, 3);
  assert(Object.values(rules.engineHashes).every(hash => /^[a-f0-9]{64}$/.test(hash)));
  assert.match(rules.roles.manipulation, /not a trained/);
  assert.match(rules.roles.perception, /no live camera/);
  assert.match(rules.sources.scope, /normative PDF not fetched/);
});

test('tactical analysis reads both rules and selected state, retains provenance and cannot prepare motion', async () => {
  const client = tacticalFixture();
  const selected = snapshot();
  const result = await planTask('Explain tactics for this game', client, {gameSnapshot: selected});
  assert.equal(result.status, 'analyzed');
  assert.equal(result.task, undefined);
  assert.equal(result.analysisEvidence.game.state.selectedRobotId, 'red-1');
  assert.match(result.analysisEvidence.game.snapshotSha256, /^[a-f0-9]{64}$/);
  assert.equal(result.skills.loaded[0].name, 'plan-game-tactics');
  assert(client.calls.every(call => !call.tools.includes('prepare_reach_pose')));
  await assert.rejects(planTask('x', fixture([tool('load_skill', {name: 'prepare-motion-experiment'})]),
    {gameSnapshot: selected}), /read-only tactics/);
  await assert.rejects(planTask('x', fixture([tool('load_skill', {name: 'plan-game-tactics'}), tool('get_game_rules'),
    tool('finish_analysis', {message: 'Skip snapshot'})]), {gameSnapshot: selected}), /snapshot availability/);
});

test('without game state the adviser must read its absence and cannot fabricate a selected snapshot', async () => {
  const result = await planTask('Explain game rules', tacticalFixture());
  assert.equal(result.status, 'analyzed');
  assert.equal(result.analysisEvidence.game, undefined);
  assert.equal(result.toolLog.find(entry => entry.name === 'get_game_snapshot').result.available, false);
  assert.equal(result.task, undefined);
});

test('game API accepts explicit compact snapshot only and cannot approve tactical commentary', async context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-tactics-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const client = tacticalFixture();
  const server = createServer({reportPath: 'missing.json', nemotronDirectory: directory, nemotronClient: client});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const post = (route, body) => fetch(`${base}/api/nemotron/${route}`, {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  assert.equal((await post('plan', {prompt: 'x', gameSnapshot: {path: '/file'}})).status, 400);
  assert.equal((await post('plan', {prompt: 'x', gameSnapshot: snapshot(), evidence: {source: 'motion', id: 'x'}})).status, 400);
  assert.equal((await post('chat', {message: 'x', gameSnapshot: snapshot()})).status, 400);
  assert.equal(client.calls.length, 0);
  const response = await post('plan', {prompt: 'fixture tactics', gameSnapshot: snapshot()});
  assert.equal(response.status, 200);
  const record = await response.json();
  assert.equal(record.status, 'analyzed');
  assert.equal((await post('run', {id: record.id})).status, 400);
  assert.equal((await fetch(`${base}/simulator/tactics-snapshot.js`)).status, 200);
  assert.equal((await fetch(`${base}/simulator/tactics-ui.js`)).status, 200);
  assert.equal(record.run, undefined);
});
