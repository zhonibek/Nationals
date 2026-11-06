'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const {playMatch, policyChooser} = require('../roboproof/tactical_baselines');
const recording = require('../roboproof/tactical-recording');
const {createServer} = require('../roboproof/server');

// A completed match shaped like playNemotronMatch's result; the loader-cycle tactic stands in for the model.
async function fakeNemotronMatch(seed = 41, alliance = 'blue') {
  const match = await playMatch({seed, alliance, chooser: policyChooser('loader-cycle', seed)});
  match.log.forEach(row => { row.reason = `fixture reason ${row.decision}`; row.meta = {latencyMs: 900}; });
  return {coordinator: 'nemotron', provider: {provider: 'nebius-token-factory', model: 'fixture-model', baseUrl: 'https://example.invalid'},
    meanDecisionLatencyMs: 900, tokens: 1234, ...match};
}

test('a recorded Nemotron match replays exactly, and a changed score is caught', async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tactical-recordings-'));
  t.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const match = await fakeNemotronMatch();
  const name = recording.saveRecording(match, directory);
  assert.equal(name, 'nemotron-s41-blue');
  const [listed] = recording.describeRecordings(directory);
  assert.deepEqual([listed.seed, listed.alliance, listed.model, listed.finalMargin], [41, 'blue', 'fixture-model', match.finalScore.margin]);
  const record = recording.readRecording(name, directory);
  const replayed = await recording.replayRecording(record);
  assert.equal(replayed.replay.reproduced, true);
  assert.equal(replayed.replay.sameEngine, true);
  assert.equal(replayed.inferencePerformed, false);
  assert.equal(replayed.log[0].reason, 'fixture reason 1');
  const tampered = {...record, decisions: record.decisions.map((row, index) => (index === 2 ? {...row, margin: row.margin + 5} : row))};
  const caught = await recording.replayRecording(tampered);
  assert.equal(caught.replay.reproduced, false);
  assert.deepEqual(caught.replay.mismatches, [3]);
  assert.throws(() => recording.readRecording('../escape', directory), /Invalid recording name/);
  assert.throws(() => recording.recordingFromMatch({...match, coordinator: 'loader-cycle'}), /Only completed Nemotron matches/);
});

test('the server lists recordings and replays one without calling a model', async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'tactical-recordings-server-'));
  const name = recording.saveRecording(await fakeNemotronMatch(52, 'red'), directory);
  const server = createServer({reportPath: 'missing-report-fixture.json', tacticalRecordingsDirectory: directory});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => {
    await new Promise(resolve => { server.closeAllConnections(); server.close(resolve); });
    fs.rmSync(directory, {recursive: true, force: true});
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  const listing = await (await fetch(`${base}/api/tactics/recordings`)).json();
  assert.deepEqual(listing.recordings.map(row => row.name), [name]);
  const post = body => fetch(`${base}/api/tactics/match`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const replayed = await post({coordinator: 'nemotron-recording', recording: name, seed: 52, alliance: 'red'});
  assert.equal(replayed.status, 200);
  const result = await replayed.json();
  assert.equal(result.coordinator, 'nemotron-recording');
  assert.equal(result.replay.reproduced, true);
  assert.equal(result.inferencePerformed, false);
  assert.equal((await post({coordinator: 'nemotron-recording', recording: 'missing-recording', seed: 52})).status, 400);
  assert.equal((await post({coordinator: 'loader-cycle', recording: name, seed: 52})).status, 400);
});
