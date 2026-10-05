'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createServer} = require('../roboproof/server');
const {DEFAULT_TASK} = require('../simulator/motion');

test('motion API persists canonical transitions, exact replay and restore without any model call', async context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-api-'));
  const servers = [];
  const client = {metadata: {model: 'NO MODEL IN THIS TEST'},
    check: async () => { throw Error('Unexpected inference'); },
    complete: async () => { throw Error('Unexpected inference'); }};
  const start = async () => {
    const server = createServer({reportPath: path.join(directory, 'absent.json'),
      motionDirectory: path.join(directory, 'motion'), nemotronDirectory: path.join(directory, 'nemotron'), nemotronClient: client});
    servers.push(server);
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    return {server, base: `http://127.0.0.1:${server.address().port}`};
  };
  context.after(async () => {
    for (const server of servers) if (server.listening) {
      server.closeAllConnections();
      await new Promise(resolve => server.close(resolve));
    }
    fs.rmSync(directory, {recursive: true, force: true});
  });
  const {server, base} = await start();
  const post = (route, body, origin) => fetch(`${base}/api/motion/${route}`, {method: 'POST',
    headers: {'Content-Type': 'application/json', ...(origin ? {Origin: origin} : {})}, body: JSON.stringify(body)});
  const status = await (await fetch(`${base}/api/motion/status`)).json();
  assert.equal(status.engine, 'original VexRobotSimulator');
  assert.equal(status.motionTrainingImplemented, true);
  assert.equal(status.cloudExecutionEnabled, false);
  assert.equal((await fetch(`${base}/api/motion/latest`)).status, 404);
  assert.equal((await post('run', {}, 'https://evil.example')).status, 403);
  assert.equal((await post('run', {mode: 'trained', seed: 1, task: DEFAULT_TASK})).status, 400);
  assert.equal((await post('run', {mode: 'scripted', seed: 1, task: DEFAULT_TASK, executeHardware: true})).status, 400);
  assert.equal((await post('check', {out: '../other.json'})).status, 400);
  assert.equal((await post('run', {mode: 'policy-fixture', seed: 1, task: DEFAULT_TASK, configuration: {batteryVoltage: 10}})).status, 400);
  assert.equal((await fetch(`${base}/api/motion/session?id=../other`)).status, 400);
  const task = {...DEFAULT_TASK, deadlineSeconds: 0.03};
  const response = await post('run', {mode: 'policy-fixture', seed: 1, task});
  assert.equal(response.status, 200);
  const record = await response.json();
  assert.equal(record.report.schemaVersion, 2);
  assert.equal(record.report.identity.canonicalSimulator, true);
  assert.equal(record.report.transitions.length, 3);
  assert.equal(record.motionPolicyTrained, false);
  assert.equal(record.inferencePerformed, false);
  assert.equal(record.exactReplayVerified, true);
  assert.deepEqual((await (await fetch(`${base}/api/motion/latest`)).json()), record);
  const verified = await (await post('replay', {id: record.id})).json();
  assert.equal(verified.exactReplayVerified, true);
  assert.equal(verified.ticks, 3);
  const page = await (await fetch(`${base}/`)).text();
  assert.match(page, /Motion learning lab/);
  assert.match(page, /LEGACY RESEARCH/);
  assert.doesNotMatch(page, /Local AI root-cause diagnosis/);
  assert.equal((await fetch(`${base}/motion.js`)).status, 200);
  assert.equal((await fetch(`${base}/simulator/policy.js`)).status, 200);
  assert.equal((await fetch(`${base}/simulator/motion-replay.js`)).status, 200);
  assert.equal((await fetch(`${base}/simulator/headless.js`)).status, 404);
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  const restored = await start();
  assert.deepEqual(await (await fetch(`${restored.base}/api/motion/latest`)).json(), record);
});
