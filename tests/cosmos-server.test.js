'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {createServer} = require('../roboproof/server');

test('Cosmos API only reads saved results and exposes no inference or upload route', async context => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'cosmos-api-fixture-'));
  const directory = path.join(temporary, 'not-created');
  const server = createServer({reportPath: 'missing-report-fixture.json', cosmosDirectory: directory});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(async () => {
    await new Promise(resolve => { server.closeAllConnections(); server.close(resolve); });
    fs.rmSync(temporary, {recursive: true, force: true});
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  const response = await fetch(`${base}/api/cosmos/results`);
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.status, 'not_generated');
  assert.deepEqual(result.results, []);
  assert.equal(result.unverifiedModelCommentary, true);
  assert.equal(result.robotActionsExecuted, false);
  assert.equal(result.trainingPerformed, false);
  assert.equal(fs.existsSync(directory), false);
  assert.equal((await fetch(`${base}/api/cosmos/results`, {headers: {Origin: 'https://untrusted.invalid'}})).status, 403);
  assert.equal((await fetch(`${base}/api/cosmos/analyze`, {method: 'POST', body: '{}'})).status, 404);
  assert.equal((await fetch(`${base}/api/cosmos/results`, {method: 'POST', body: '{}'})).status, 404);
  const asset = await fetch(`${base}/cosmos.js`);
  assert.equal(asset.status, 200);
  assert.match(await asset.text(), /\/api\/cosmos\/results/);
});

test('Cosmos API rejects invalid saved records without exposing local paths', async context => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'cosmos-api-invalid-fixture-'));
  fs.writeFileSync(path.join(directory, 'invalid.json'), JSON.stringify({status: 'loaded_not_inference_verified'}));
  const server = createServer({reportPath: 'missing-report-fixture.json', cosmosDirectory: directory});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(async () => {
    await new Promise(resolve => { server.closeAllConnections(); server.close(resolve); });
    fs.rmSync(directory, {recursive: true, force: true});
  });
  const response = await fetch(`http://127.0.0.1:${server.address().port}/api/cosmos/results`);
  const raw = await response.text();
  const result = JSON.parse(raw);
  assert.equal(result.status, 'invalid');
  assert.deepEqual(result.results, []);
  assert.equal(raw.includes(directory), false);
  assert.equal(raw.includes('loaded_not_inference_verified'), false);
});

test('legacy Nemotron startup cannot silently launch the retired CPU runtime', () => {
  const source = fs.readFileSync(path.join(__dirname, '../tools/start-nemotron.ps1'), 'utf8');
  assert.match(source, /throw .*CPU launcher is disabled/);
  assert.match(source, /verified NVIDIA T4/);
  assert.doesNotMatch(source, /Start-Process|Get-FileHash|-ngl.*0/);
});
