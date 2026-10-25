'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {playMatch, policyChooser} = require('../roboproof/tactical_baselines');

const BRIDGE = path.resolve(__dirname, '../roboproof/tactical_learning/bridge.js');

// Talks to the bridge the way train.py does: JSON lines over pipes, one request at a time.
function startBridge(t) {
  const child = spawn(process.execPath, [BRIDGE], {stdio: ['pipe', 'pipe', 'pipe']});
  t.after(() => child.kill());
  const lines = [];
  const waiting = [];
  let buffer = '';
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', chunk => {
    buffer += chunk;
    for (let index; (index = buffer.indexOf('\n')) >= 0;) {
      const response = JSON.parse(buffer.slice(0, index));
      buffer = buffer.slice(index + 1);
      lines.push(response);
      waiting.shift()?.(response);
    }
  });
  const exited = new Promise(resolve => child.on('exit', (code, signal) => resolve({code, signal})));
  const request = payload => new Promise(resolve => {
    waiting.push(resolve);
    child.stdin.write(JSON.stringify(payload) + '\n');
  });
  return {child, lines, request, exited};
}

function within(promise, seconds, message) {
  let timer;
  const timeout = new Promise((resolve, reject) => { timer = setTimeout(() => reject(Error(message)), seconds * 1000); });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

test('the tactical bridge exits with code 0 after answering close while its stdin stays open', async t => {
  const bridge = startBridge(t);
  assert.equal((await bridge.request({operation: 'reset', seed: 17, alliance: 'red'})).ok, true);
  assert.deepEqual(await bridge.request({operation: 'close'}), {ok: true});
  // stdin is deliberately left open: the trainer holds it while it waits for the exit.
  assert.deepEqual(await within(bridge.exited, 5, 'bridge still running 5 s after close'), {code: 0, signal: null});
  assert.equal(bridge.child.stdin.writableEnded, false);
});

test('the tactical bridge answers nothing after close', async t => {
  const bridge = startBridge(t);
  bridge.child.stdin.write('{"operation":"close"}\n{"operation":"reset","seed":17,"alliance":"red"}\n');
  assert.deepEqual(await within(bridge.exited, 5, 'bridge still running 5 s after close'), {code: 0, signal: null});
  assert.deepEqual(bridge.lines, [{ok: true}]);
});

test('the tactical bridge demonstrates the loader-cycle choices the baseline match runner makes', async t => {
  const decisions = 3;
  const match = await playMatch({seed: 17, alliance: 'red', chooser: policyChooser('loader-cycle', 17), maximumDecisions: decisions});
  const bridge = startBridge(t);
  let observation = (await bridge.request({operation: 'reset', seed: 17, alliance: 'red'})).observation;
  const demonstrated = [];
  for (let decision = 0; decision < decisions; decision += 1) {
    const answer = await bridge.request({operation: 'demonstrate'});
    assert.equal(answer.ok, true);
    assert.equal(answer.policy, 'loader-cycle');
    assert.equal(answer.candidate.id, answer.actionId);
    assert(observation.actions.some(action => action.id === answer.actionId), 'demonstrated ID must be an observed candidate');
    demonstrated.push(answer.actionId);
    observation = (await bridge.request({operation: 'step', actionId: answer.actionId})).observation;
  }
  assert.deepEqual(demonstrated, match.log.map(row => row.id));
  assert.deepEqual(await bridge.request({operation: 'demonstrate', policy: 'random'}),
    {ok: false, error: 'Unsupported demonstration policy random'});
  assert.deepEqual(await bridge.request({operation: 'close'}), {ok: true});
  assert.deepEqual(await within(bridge.exited, 5, 'bridge still running 5 s after close'), {code: 0, signal: null});
});

test('the tactical bridge also exits when its stdin reaches EOF without a close request', async t => {
  const bridge = startBridge(t);
  assert.equal((await bridge.request({operation: 'reset', seed: 17, alliance: 'red'})).ok, true);
  bridge.child.stdin.end();
  assert.deepEqual(await within(bridge.exited, 5, 'bridge still running 5 s after EOF'), {code: 0, signal: null});
});
