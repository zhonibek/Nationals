'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createRegistry, MAX_SKILL_BYTES} = require('../roboproof/agent-skills');
const {planTask} = require('../roboproof/nemotron');
const {motionEvidence, learningSummary} = require('../roboproof/skill-evidence');
const {baseline} = require('../roboproof/motion');
const {suite} = require('../roboproof/motion-evaluation');
const MotionLearner = require('../roboproof/motion-learner');
const {createServer} = require('../roboproof/server');
const {createMotionStore} = require('../roboproof/motion-store');
const {createStore} = require('../roboproof/nemotron-store');

function temporary(context) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'robotai-skills-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  return directory;
}
function tool(name, argumentsValue = {}, id = name) {
  return {choices: [{finish_reason: 'tool_calls', message: {role: 'assistant',
    tool_calls: [{id, type: 'function', function: {name, arguments: JSON.stringify(argumentsValue)}}]}}]};
}
function fixture(responses) {
  const calls = [];
  return {calls, metadata: {provider: 'fixture-not-live-inference', cloudEnabled: false}, check: async () => ({}),
    complete: async (messages, signal, tools) => {
      calls.push({messages: structuredClone(messages), tools: structuredClone(tools)});
      assert(responses.length, 'Fixture response exhausted');
      return responses.shift();
    }};
}

test('reviewed skills progressively load with hashes and fixed permissions', () => {
  const registry = createRegistry();
  const catalog = registry.catalog();
  assert.equal(catalog.length, 5);
  assert(catalog.every(entry => !entry.instructions && /^[a-f0-9]{64}$/.test(entry.sha256)));
  const session = registry.session();
  assert.equal(session.permits('prepare_reach_pose'), false);
  assert.match(session.load('prepare-motion-experiment').instructions, /separate.*approval/);
  assert.equal(session.permits('prepare_reach_pose'), true);
  assert.equal(session.permits('get_saved_motion_evidence'), false);
  session.load('inspect-robot');
  assert.throws(() => session.load('diagnose-motion'), /activation budget/);
  assert.equal(session.provenance().length, 2);
  for (const name of ['../secret', 'shell', '__proto__', null]) assert.throws(() => registry.session().load(name), /Unknown reviewed/);
});

test('bundled local runtime caps prompt-cache memory rather than using the 8 GiB upstream default', () => {
  const source = fs.readFileSync(path.join(__dirname, '../tools/start-nemotron.ps1'), 'utf8');
  assert.match(source, /\$CacheMiB = 128/);
  assert.match(source, /\$CacheMiB -lt 0 -or \$CacheMiB -gt 1024/);
  assert.match(source, /'--cache-ram', "\$CacheMiB"/);
});

test('modified, oversized or permission-expanding skill packages fail closed', context => {
  const directory = temporary(context);
  fs.cpSync(path.join(__dirname, '../roboproof/skills'), directory, {recursive: true});
  const registry = createRegistry(directory);
  const session = registry.session();
  const filename = path.join(directory, 'inspect-robot/SKILL.md');
  const original = fs.readFileSync(filename, 'utf8');
  fs.writeFileSync(filename, original + '\nChanged instructions\n');
  assert.throws(() => session.load('inspect-robot'), /changed during/);
  fs.writeFileSync(filename, original.replace('allowed-tools: get_robot_profile', 'allowed-tools: get_robot_profile shell'));
  assert.throws(() => registry.catalog(), /reviewed allowlist/);
  fs.writeFileSync(filename, original.replace('name: inspect-robot', 'name: other'));
  assert.throws(() => registry.catalog(), /metadata/);
  fs.writeFileSync(filename, original.replace('name: inspect-robot', 'name: inspect-robot\nname: inspect-robot'));
  assert.throws(() => registry.catalog(), /duplicate/);
  fs.writeFileSync(filename, original + 'x'.repeat(MAX_SKILL_BYTES));
  assert.throws(() => registry.catalog(), /budget/);
});

test('agent loads the robot skill and reads actual configuration before analysis', async () => {
  const client = fixture([tool('load_skill', {name: 'inspect-robot'}), tool('get_robot_profile'),
    tool('finish_analysis', {message: 'Fixture explanation, not real model evidence.'})]);
  const result = await planTask('Explain the robot', client);
  assert.equal(result.status, 'analyzed');
  assert.equal(result.commentaryVerified, false);
  assert.equal(result.analysisEvidence.robot.configuration.drive, 'XDRIVE');
  assert.equal(result.analysisEvidence.robot.hardwareExecutionEnabled, false);
  assert.equal(result.skills.loaded[0].name, 'inspect-robot');
  assert.deepEqual(client.calls[0].tools.map(entry => entry.function.name), ['ask_clarification', 'load_skill']);
  assert.deepEqual(client.calls[1].tools.map(entry => entry.function.name), ['ask_clarification', 'load_skill', 'get_robot_profile']);
  assert(client.calls[2].tools.some(entry => entry.function.name === 'finish_analysis'));
  assert.equal(client.calls[0].messages[0].content.includes('# Inspect the existing robot'), false);
  assert.match(client.calls[1].messages[0].content, /# Inspect the existing robot/);
  assert.equal(client.calls[1].messages[0].content.includes('Skill catalog:'), false);
  assert.equal(client.calls[1].messages.at(-1).content.includes('# Inspect the existing robot'), false);
  assert.equal(result.run, undefined);
});

test('skill gating defeats requests for unauthorized tools and arbitrary evidence paths', async () => {
  await assert.rejects(planTask('Ignore permissions', fixture([tool('get_robot_profile')])), /Load the reviewed/);
  await assert.rejects(planTask('Use hardware', fixture([tool('load_skill', {name: 'inspect-robot'}), tool('prepare_reach_pose')])), /Load the reviewed/);
  await assert.rejects(planTask('Read secrets', fixture([tool('load_skill', {name: 'diagnose-motion'}),
    tool('get_saved_motion_evidence', {path: '../secret'})])), /Unsupported arguments.path/);
  await assert.rejects(planTask('Run shell', fixture([tool('load_skill', {name: 'inspect-robot'}), tool('shell')])), /Unsupported Nemotron tool/);
  const malformed = tool('load_skill', {name: 'inspect-robot'});
  malformed.choices[0].message.tool_calls[0].function.arguments = ' '.repeat(16385);
  await assert.rejects(planTask('x', fixture([malformed])), /bounded.*arguments/);
  const duplicate = tool('get_robot_profile', {}, 'load_skill');
  await assert.rejects(planTask('x', fixture([tool('load_skill', {name: 'inspect-robot'}), duplicate])), /duplicate/);
  await assert.rejects(planTask('x', fixture([tool('load_skill', {name: 'inspect-robot'}),
    tool('finish_analysis', {message: 'Invented evidence'})])), /Read actual tool evidence/);
});

test('saved failed motion is compactly measured; missing evidence never becomes a diagnosis', async () => {
  const report = baseline(42, {start: {xIn: 0, yIn: 0, headingDeg: 0},
    goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 0.1}, {}, {recordTransitions: true});
  const evidence = motionEvidence({id: 'fixture-id', report, exactReplayVerified: true}, 'motion');
  assert.equal(evidence.reason, 'time_limit');
  assert.equal(evidence.metrics.elapsedSeconds, 0.1);
  assert.deepEqual(evidence.configuration, report.configuration);
  assert.equal(evidence.telemetry.samples.length, 5);
  assert.equal(evidence.finalStateChecks.positionErrorMeters.belowThreshold, false);
  assert.equal(evidence.recordedExactReplayVerified, true);
  assert.equal(evidence.actions, undefined);
  assert.throws(() => motionEvidence({report: {...report, metrics: {...report.metrics, contactSeconds: NaN}}}, 'motion'), /metrics/);
  const result = await planTask('Diagnose this saved result', fixture([tool('load_skill', {name: 'diagnose-motion'}),
    tool('get_saved_motion_evidence'), tool('finish_analysis', {message: 'Fixture: timeout; cause unproven.'})]), {evidence});
  assert.equal(result.status, 'analyzed');
  assert.deepEqual(result.analysisEvidence.motion, evidence);
  const missing = await planTask('Diagnose', fixture([tool('load_skill', {name: 'diagnose-motion'}),
    tool('get_saved_motion_evidence'), tool('ask_clarification', {message: 'Select a saved experiment first.'})]));
  assert.equal(missing.status, 'clarification');
  assert.deepEqual(missing.analysisEvidence, {});
});

test('learning skill retains negative results and discards stale evaluation counts', context => {
  const directory = temporary(context);
  assert.equal(learningSummary(directory).motionPolicyTrained, false);
  const identity = require('../simulator/headless').loadHeadless().identity;
  const runId = '11111111-1111-4111-8111-111111111111';
  const snapshot = {available: true, status: 'completed', runId, identity, models: [{seed: 42, policySha256: 'current', steps: 40,
    updates: 1, trained: true, actorWeightsChanged: true}], motionPolicyTrained: true, learnedImprovementVerified: false,
    evaluation: {runId, identity, policyHashes: ['current'], corpusSha256: suite(2).corpusSha256, models: [{seed: 42,
      baseline: {success: 6, count: 24}, untrained: {success: 6, count: 24}, learned: {success: 6, count: 24},
      acceptance: {passed: false, regressions: [], contactRegressions: [], meanTimeImprovementFraction: 0,
        meanEffortImprovementFraction: 0, rule: 'Fixture gate, no measured improvement'},
      results: [{id: 'failed-world', learned: {reason: 'time_limit'}}]}]}};
  context.mock.method(MotionLearner, 'status', () => snapshot);
  const summary = learningSummary(directory);
  assert.equal(summary.motionPolicyTrained, true);
  assert.equal(summary.learnedImprovementVerified, false);
  assert.deepEqual(summary.evaluation.models[0].failedWorldIds, ['failed-world']);
  assert.equal(summary.evaluation.models[0].learned.success, 6);
  snapshot.evaluation.policyHashes = ['stale'];
  assert.equal(learningSummary(directory).evaluation, null);
  snapshot.evaluation.policyHashes = ['current'];
  snapshot.status = 'running';
  assert.equal(learningSummary(directory).evaluationMatchesCurrentCheckpoints, false);
});

test('dashboard skill API reads only selected evidence, saves provenance and cannot approve analysis', async context => {
  const directory = temporary(context);
  const motionDirectory = path.join(directory, 'motion');
  const nemotronDirectory = path.join(directory, 'nemotron');
  const saved = createMotionStore(motionDirectory).create({report: baseline(42), exactReplayVerified: true});
  const client = fixture([tool('load_skill', {name: 'diagnose-motion'}), tool('get_saved_motion_evidence'),
    tool('finish_analysis', {message: 'Fixture analysis, not live model evidence.'})]);
  const server = createServer({reportPath: 'missing.json', motionDirectory, nemotronDirectory,
    learningDirectory: path.join(directory, 'learning'), nemotronClient: client});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const catalog = await (await fetch(`${base}/api/nemotron/skills`)).json();
  assert.equal(catalog.skills.length, 5);
  assert.equal(catalog.inferencePerformed, false);
  assert.equal(client.calls.length, 0);
  assert.equal((await fetch(`${base}/api/nemotron/skills`, {headers: {Origin: 'https://evil.example'}})).status, 403);
  const post = (route, body) => fetch(`${base}/api/nemotron/${route}`, {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  for (const evidence of [{source: 'motion', id: '../secret'}, {source: 'file', id: saved.id},
    {source: 'motion', id: saved.id, path: '../secret'}, {source: 'motion', id: saved.id, report: {reason: 'success'}}]) {
    assert.equal((await post('plan', {prompt: 'Diagnose', evidence})).status, 400);
  }
  assert.equal(client.calls.length, 0);
  const response = await post('plan', {prompt: 'Diagnose', evidence: {source: 'motion', id: saved.id}});
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.status, 'analyzed');
  assert.equal(result.analysisEvidence.motion.sessionId, saved.id);
  assert.equal(result.analysisEvidence.motion.reason, saved.report.reason);
  assert.deepEqual(createStore(nemotronDirectory).read(result.id), result);
  assert.equal((await post('run', {id: result.id})).status, 400);
  assert.equal((await fetch(`${base}/simulator/../roboproof/skills/inspect-robot/SKILL.md`)).status, 404);
});
