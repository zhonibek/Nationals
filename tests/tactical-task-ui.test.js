'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const Game = require('../simulator/override');
const UI = require('../simulator/tactics-ui');

function fixture(fetch, location = {protocol: 'http:', hostname: '127.0.0.1', pathname: '/simulator/index.html'}) {
  const elements = new Map();
  const make = tag => ({tag, textContent: '', value: '', hidden: true, disabled: false, children: [], listeners: {},
    addEventListener(name, callback) { this.listeners[name] = callback; },
    appendChild(child) { this.children.push(child); return child; },
    replaceChildren() { this.children = []; }});
  const element = id => {
    if (!elements.has(id)) elements.set(id, make(id));
    return elements.get(id);
  };
  const game = new Game();
  game.startMatch('practice');
  UI.attach({fleet: {}, override: game, activeRobotId: 'red-1'},
    {document: {getElementById: element, createElement: make}, fetch, location});
  return {element};
}

const reply = {summary: 'Place 2 match loads in neutral goals, then park', achieved: true, stopReason: 'task-complete',
  scoringActions: 2, parkedRobots: ['red-2'], ownPointsGained: 36, clockStart: 15, clockEnd: 29.67,
  finalScore: {red: 40, blue: 4, margin: 36},
  log: [{decision: 1, clock: 15, id: '<b>load-score-cup:red-2:loader-red-n:g-neutral-e</b>', rejected: false, ownPoints: 0,
    seconds: 4.2, score: {margin: 0}, policyScore: 1.25}],
  policy: {algorithm: 'actor-critic (not PPO)', provenance: {runId: '157410e7-7c11', trainingSeed: 734001},
    evaluation: {meanScoreMargin: 191.5, baselineMeanScoreMargin: {'loader-cycle': 244.5}}}};

test('the task panel sends the typed task once and lists decisions as plain text', async () => {
  const calls = [];
  const ui = fixture(async (url, options) => { calls.push({url, options}); return {ok: true, json: async () => reply}; });
  ui.element('tacticalTaskText').value = '  score 2 match loads into the neutral goals, then park ';
  ui.element('tacticalTaskSeed').value = '101';
  ui.element('tacticalTaskAlliance').value = 'red';
  await ui.element('runTacticalTask').listeners.click();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/tactics/task');
  assert.deepEqual(JSON.parse(calls[0].options.body), {task: 'score 2 match loads into the neutral goals, then park', seed: 101, alliance: 'red'});
  const rows = ui.element('tacticalTaskLog').children;
  assert.equal(rows.length, 1);
  assert.equal(rows[0].children[2].textContent, '<b>load-score-cup:red-2:loader-red-n:g-neutral-e</b>');
  const status = ui.element('tacticalTaskStatus').textContent;
  assert.match(status, /^Understood: Place 2 match loads in neutral goals, then park\. Done \(task-complete\)/);
  assert.match(status, /actor-critic \(not PPO\) \(run 157410e7, seed 734001, held-out margin 191.5 vs loader-cycle 244.5\)/);
  assert.match(status, /no robot was driven/);
  assert.equal(ui.element('runTacticalTask').disabled, false);
});

const ppoPolicy = {algorithm: 'PPO: clipped surrogate objective', provenance: {runId: 'abcdef123456', trainingSeed: 735001},
  evaluation: {meanScoreMargin: 200, baselineMeanScoreMargin: {'loader-cycle': 244.5}}};

test('"Run with the Tactical PPO optimizer" runs the trained policy, and the panel names the loaded policy on first use', async () => {
  const calls = [];
  const ui = fixture(async (url, options = {}) => {
    calls.push({url, method: options.method || 'GET', body: options.body});
    return {ok: true, json: async () => url.endsWith('/api/tactics/policy') ? {available: true, ...ppoPolicy} : {...reply, policy: ppoPolicy}};
  }, {protocol: 'http:', hostname: '127.0.0.1', port: '8765', pathname: '/index.html'});
  assert.equal(calls.length, 0, 'nothing is fetched on page load');
  await ui.element('tacticalTaskText').listeners.focus();
  assert.deepEqual(calls.map(call => [call.method, call.url]), [['GET', 'http://127.0.0.1:8766/api/tactics/policy']]);
  assert.equal(ui.element('tacticalPolicyInfo').textContent,
    'Loaded policy: PPO: clipped surrogate objective (run abcdef12, seed 735001, held-out margin 200 vs loader-cycle 244.5).');
  await ui.element('tacticalTaskText').listeners.focus();
  assert.equal(calls.length, 1, 'policy info is read once');

  ui.element('tacticalTaskSeed').value = '101';
  ui.element('tacticalTaskAlliance').value = 'blue';
  await ui.element('btnTacticsToPpo').listeners.click();
  assert.equal(ui.element('tacticalTaskPanel').open, true);
  assert.deepEqual([calls[1].method, calls[1].url], ['POST', 'http://127.0.0.1:8766/api/tactics/task']);
  assert.deepEqual(JSON.parse(calls[1].body), {task: 'play the driver period for the best score', seed: 101, alliance: 'blue'});
  assert.match(ui.element('tacticalTaskStatus').textContent, /Policy: PPO: clipped surrogate objective \(run abcdef12/);
});

test('the tactical match panel can use the trained policy as its coordinator', async () => {
  const calls = [];
  const match = {coordinator: 'tactical-policy', alliance: 'red', finalScore: {red: 60, blue: 0, margin: 60}, decisions: 1,
    matchLoadsScored: 1, stopReason: 'match-complete', policy: ppoPolicy,
    log: [{decision: 1, clock: 15, id: 'load-score-pin:red-1:loader-red-s:g-red-s', points: 5, seconds: 2.5, score: {margin: 5}, reason: 'PPO score 1.2'}]};
  const ui = fixture(async (url, options) => { calls.push({url, body: options.body}); return {ok: true, json: async () => match}; });
  ui.element('tacticalCoordinator').value = 'tactical-policy';
  ui.element('tacticalSeed').value = '101';
  ui.element('tacticalAlliance').value = 'red';
  await ui.element('runTacticalMatch').listeners.click();
  assert.deepEqual(JSON.parse(calls[0].body), {coordinator: 'tactical-policy', seed: 101, alliance: 'red'});
  assert.equal(ui.element('tacticalMatchLog').children[0].children[6].textContent, 'PPO score 1.2');
  assert.match(ui.element('tacticalMatchStatus').textContent,
    /^tactical-policy, PPO: clipped surrogate objective \(run abcdef12, seed 735001, held-out margin 200 vs loader-cycle 244.5\): final margin 60/);
});

test('the task panel validates input, shows server errors and stays off outside RoboProof', async () => {
  const calls = [];
  const ui = fixture(async () => { calls.push(1); return {ok: false, status: 400, json: async () => ({error: 'Task not understood. Try …'})}; });
  ui.element('tacticalTaskSeed').value = '101';
  await ui.element('runTacticalTask').listeners.click();
  assert.equal(calls.length, 0);
  assert.match(ui.element('tacticalTaskStatus').textContent, /Type a task first/);
  ui.element('tacticalTaskText').value = 'hello';
  ui.element('tacticalTaskSeed').value = '-3';
  await ui.element('runTacticalTask').listeners.click();
  assert.match(ui.element('tacticalTaskStatus').textContent, /Seed must be/);
  ui.element('tacticalTaskSeed').value = '7';
  await ui.element('runTacticalTask').listeners.click();
  assert.equal(calls.length, 1);
  assert.equal(ui.element('tacticalTaskStatus').textContent, 'Tactical task failed: Task not understood. Try …');

  const remote = fixture(async () => { throw Error('must not be called'); }, {protocol: 'https:', hostname: 'example.com', pathname: '/'});
  assert.equal(remote.element('runTacticalTask').disabled, true);
  assert.match(remote.element('tacticalTaskStatus').textContent, /through RoboProof/);
});
