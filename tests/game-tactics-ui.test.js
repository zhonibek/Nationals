'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const Game = require('../simulator/override');
const UI = require('../simulator/tactics-ui');

function fixture(fetch, options = {}) {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {textContent: '', hidden: true, listeners: {},
      addEventListener(name, callback) { this.listeners[name] = callback; }});
    return elements.get(id);
  };
  const game = new Game();
  game.startMatch('practice');
  const sim = {fleet: {}, override: game, activeRobotId: 'red-1'};
  UI.attach(sim, {document: {getElementById: element}, fetch,
    location: {protocol: 'http:', hostname: '127.0.0.1', pathname: '/simulator/index.html'}, ...options});
  return {element, sim};
}

test('Simulator tactics shares a snapshot only after a click and displays advice as safe text without game mutation', async () => {
  const calls = [];
  const ui = fixture(async (url, options) => {
    calls.push({url, options});
    return {ok: true, json: async () => ({status: 'analyzed', message: '<script>inert fixture advice</script>'})};
  });
  const before = JSON.stringify(ui.sim.override.getState());
  assert.equal(calls.length, 0);
  await ui.element('askTactics').listeners.click();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/nemotron/plan');
  const body = JSON.parse(calls[0].options.body);
  assert.equal(body.gameSnapshot.selectedRobotId, 'red-1');
  assert.equal(body.task, undefined);
  assert.equal(JSON.stringify(ui.sim.override.getState()), before);
  assert.equal(ui.element('tacticsMessage').textContent, '<script>inert fixture advice</script>');
  assert.equal(ui.element('tacticsMessage').innerHTML, undefined);
  assert.equal(ui.element('tacticsReply').hidden, false);
  assert.match(ui.element('tacticsStatus').textContent, /никаких действий/);
});

test('replay and standalone Simulator cannot dispatch model requests, and movement results are rejected', async () => {
  let calls = 0;
  const fetch = async () => { calls++; return {ok: true, json: async () => ({status: 'prepared', message: 'Wrong mode'})}; };
  for (const options of [{replay: true}, {location: {protocol: 'http:', hostname: '127.0.0.1', pathname: '/'}}]) {
    const ui = fixture(fetch, options);
    assert.equal(ui.element('askTactics').disabled, true);
    await ui.element('askTactics').listeners.click();
  }
  assert.equal(calls, 0);
  const ui = fixture(fetch);
  await ui.element('askTactics').listeners.click();
  assert.equal(calls, 1);
  assert.match(ui.element('tacticsStatus').textContent, /только тактический совет/);
  assert.equal(ui.element('tacticsReply').hidden, true);
  const unsupported = fixture(fetch);
  unsupported.sim.override.world = 'worlds';
  await unsupported.element('askTactics').listeners.click();
  assert.equal(calls, 1);
  assert.match(unsupported.element('tacticsStatus').textContent, /Unsupported tactics/);
});

test('Stop cancels inference without changing game state', async () => {
  const ui = fixture(async (url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(Object.assign(Error('stop'), {name: 'AbortError'})), {once: true});
  }));
  const before = JSON.stringify(ui.sim.override.getState());
  const pending = ui.element('askTactics').listeners.click();
  assert.equal(ui.element('stopTactics').disabled, false);
  ui.element('stopTactics').listeners.click();
  await pending;
  assert.equal(JSON.stringify(ui.sim.override.getState()), before);
  assert.match(ui.element('tacticsStatus').textContent, /остановлен/);
  assert.equal(ui.element('askTactics').disabled, false);
});
