'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {createServer} = require('../roboproof/server');
const {baseline, replay} = require('../roboproof/motion');
const {parseGrounding, prepareProposal, readiness, MAX_FRAME_AGE_MS, MAX_ANSWER_BYTES, MAX_DETECTIONS} = require('../roboproof/perception');
const inputFixture = require('../roboproof/perception/fixtures/reach-point.json');
const NOW = Date.parse(inputFixture.grounding.frame.capturedAt);
const fixture = () => structuredClone(inputFixture);
const prepare = input => prepareProposal(input, {now: NOW});

test('LocateAnything preparation never claims installed runtime, inference or learned movement', () => {
  const status = readiness();
  for (const key of ['runtimeEnabled', 'cloudEnabled', 'automaticExecutionEnabled', 'inferenceAvailable',
    'liveInferenceVerified', 'cameraConnected', 'calibrationInstalled', 'movementPolicyTrained']) assert.equal(status[key], false);
  assert.equal(status.model, 'nvidia/LocateAnything-3B');
  status.nextGates.length = 0;
  assert.equal(readiness().nextGates.length, 5);
});

test('native LocateAnything token parser preserves point/box order and original image scale', () => {
  const input = fixture().grounding;
  input.frame.width = 640;
  input.frame.height = 480;
  input.answer = 'target <box><250><500><1000><1000></box> <box><500><250></box>';
  const parsed = parseGrounding(input);
  assert.equal(parsed.status, 'grounded');
  assert.deepEqual(parsed.detections.map(detection => detection.pixels), [[160, 240, 640, 480], [320, 120]]);
  assert.deepEqual(parsed.detections.map(detection => detection.type), ['box', 'point']);
  assert.deepEqual(parsed.detections.map(detection => detection.confidence), [null, null]);
  assert.equal(parsed.sourceVerified, false);
  assert.equal(parsed.inferencePerformedByRoboProof, false);
  assert.match(parsed.answerSha256, /^[a-f0-9]{64}$/);
});

test('malformed, truncated, inverted, nonfinite and oversized grounding fails closed', () => {
  for (const answer of ['<box><500><300>', '<box><-1><300></box>', '<box><1001><300></box>',
    '<box><NaN><300></box>', '<box><0><0><0><10></box>', '<box><10><0><0><10></box>',
    '<box><1><2><3></box>', '<box>1,2</box>', '<BOX><1><2></BOX>',
    '<box><1><2></box><box><3>', '<box><1><2></box></boxoops>', 'x'.repeat(MAX_ANSWER_BYTES + 1),
    '<box><1><2></box>'.repeat(MAX_DETECTIONS + 1)]) {
    assert.throws(() => parseGrounding({...fixture().grounding, answer}));
  }
  for (const truncated of [true, undefined, 'false']) assert.throws(() => parseGrounding({...fixture().grounding, truncated}), /Truncated/);
  assert.throws(() => parseGrounding({...fixture().grounding, schemaVersion: 2}), /schema/);
  assert.throws(() => parseGrounding({...fixture().grounding, execute: true}), /Unsupported/);
});

test('no recognized geometry does not become a goal or prove object absence', () => {
  const input = fixture();
  input.grounding.answer = 'No confidently grounded target.';
  assert.equal(parseGrounding(input.grounding).status, 'no_geometry');
  assert.throws(() => prepare(input), /absence is not proven/);
});

test('frame contract binds image resolution, identities, hash and canonical capture time', () => {
  for (const [key, value] of [['width', 0], ['height', 2561], ['width', 640.5], ['height', NaN],
    ['cameraId', ''], ['cameraPoseId', '../unknown'], ['coordinateSpaceId', null], ['imageSha256', 'not-a-hash'],
    ['capturedAt', 'yesterday'], ['capturedAt', '2026-10-02']]) {
    const input = fixture().grounding;
    input.frame[key] = value;
    assert.throws(() => parseGrounding(input));
  }
  const input = fixture().grounding;
  input.frame.crop = [1, 2];
  assert.throws(() => parseGrounding(input), /Unsupported frame/);
});

test('reviewed fixture prepares the existing Simulator task without executing it', () => {
  const input = fixture();
  const original = structuredClone(input);
  const result = prepare(input);
  assert.deepEqual(result.task, {start: {xIn: 0, yIn: 0, headingDeg: 0}, goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 10});
  assert.equal(result.status, 'proposal_only');
  assert.equal(result.simulationOnly, true);
  assert.equal(result.requiresSeparateSimulatorApproval, true);
  assert.equal(result.inferencePerformedByRoboProof, false);
  assert.match(result.execution, /not run/);
  assert.deepEqual(input, original);
  result.selection.pixelPoint.x = 123;
  assert.equal(input.selection.pixelPoint.x, 500);
});

test('frame freshness is bounded and future captures cannot pass', () => {
  assert.throws(() => prepareProposal(fixture(), {now: NOW + MAX_FRAME_AGE_MS + 1}), /stale/);
  assert.throws(() => prepareProposal(fixture(), {now: NOW - 1}), /future/);
  assert.throws(() => prepareProposal(fixture(), {now: Infinity}), /clock/);
  assert.equal(prepareProposal(fixture(), {now: NOW + MAX_FRAME_AGE_MS}).status, 'proposal_only');
});

test('selection must explicitly review a floor point on the same frame and geometry', () => {
  for (const [key, value] of [['detectionIndex', -1], ['detectionIndex', 1], ['detectionIndex', '0'],
    ['frameId', 'another-frame'], ['imageSha256', '1'.repeat(64)], ['plane', 'object-center'], ['reviewedBy', ''],
    ['reviewedAt', '2026-10-01T23:59:59.999Z'], ['reviewedAt', '2026-10-02T00:00:00.001Z'],
    ['pixelPoint', {x: 501, y: 300}], ['pixelPoint', {x: 500, y: -1}]]) {
    const input = fixture();
    input.selection[key] = value;
    assert.throws(() => prepare(input));
  }
  const input = fixture();
  delete input.selection.pixelPoint;
  assert.throws(() => prepare(input));
  delete input.selection;
  assert.throws(() => prepare(input), /selection/);
});

test('bounding boxes need an explicitly reviewed floor point, never an inferred center', () => {
  const input = fixture();
  input.grounding.answer = '<box><490><290><510><310></box>';
  assert.equal(prepare(input).task.goal.yIn, 24);
  input.selection.pixelPoint = {x: 700, y: 500};
  assert.throws(() => prepare(input), /selected geometry/);
});

test('calibration cannot survive pose/profile/dimension changes, expiry or missing review', () => {
  for (const [key, value] of [['schemaVersion', 2], ['kind', 'unrectified-perspective'], ['fieldFrame', 'pixels'],
    ['cameraId', 'another'], ['cameraPoseId', 'moved'], ['coordinateSpaceId', 'cropped'], ['width', 640],
    ['height', 480], ['expiresAt', inputFixture.grounding.frame.capturedAt],
    ['validFrom', '2026-10-02T00:00:00.001Z'], ['reviewedAt', '2026-10-02T00:00:00.001Z'], ['reviewedBy', '']]) {
    const input = fixture();
    input.calibration[key] = value;
    assert.throws(() => prepare(input));
  }
  const input = fixture();
  delete input.calibration;
  assert.throws(() => prepare(input), /calibration/);
});

test('singular mappings, extrapolation and unvalidated field coverage are rejected', () => {
  for (const homography of [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0.12, 0, -60, 0, -0.12, 60, 0.002, 0, -1], [1, 0, 0, 0, 1, 0, 0, 0, 1], [NaN], [Infinity]]) {
    const input = fixture();
    input.calibration.homography = homography;
    assert.throws(() => prepare(input));
  }
  for (const pixelCoverage of [[0, 0, 400, 400], [0, 0, 0, 100], [-1, 0, 1000, 1000], [0, 0, 1001, 1000], []]) {
    const input = fixture();
    input.calibration.pixelCoverage = pixelCoverage;
    assert.throws(() => prepare(input));
  }
  const input = fixture();
  input.calibration.homography = input.calibration.homography.map(value => value * 100);
  assert.deepEqual(prepare(input).task, prepare(fixture()).task);
});

test('calibration validation metrics are bounded; declared reports are not independently verified', () => {
  for (const [key, value] of [['sampleCount', 3], ['sampleCount', Infinity], ['rmsErrorIn', 0.51],
    ['maxErrorIn', 1.01], ['rmsErrorIn', -1], ['kind', 'training-fit']]) {
    const input = fixture();
    input.calibration.validation[key] = value;
    assert.throws(() => prepare(input));
  }
  assert.match(prepare(fixture()).provenanceVerification, /not independently verified/);
});

test('real imported outputs require a pinned revision and non-fixture calibration evidence', () => {
  const input = fixture();
  input.grounding.source.kind = 'imported-model-output';
  assert.throws(() => prepare(input), /revision/);
  input.grounding.source.revision = 'a'.repeat(40);
  assert.throws(() => prepare(input), /Fixture calibration/);
  input.calibration.validation.kind = 'held-out-physical-measurements';
  assert.equal(prepare(input).grounding.sourceVerified, false);
  input.grounding.source.model = 'another-model';
  assert.throws(() => prepare(input), /LocateAnything/);
});

test('proposal shares original field/heading/deadline bounds instead of inventing motion semantics', () => {
  for (const [key, value] of [['start', {xIn: 61, yIn: 0, headingDeg: 0}], ['goalHeadingDeg', 181],
    ['deadlineSeconds', 10.005], ['deadlineSeconds', 0], ['deadlineSeconds', 61]]) {
    assert.throws(() => prepare({...fixture(), [key]: value}));
  }
});

test('explicit test execution of synthetic proposal reaches goal in original Simulator and replays exactly', () => {
  const proposal = prepare(fixture());
  const report = baseline(42, proposal.task);
  assert.equal(report.reason, 'success');
  assert.equal(report.learningStatus, 'scripted benchmark only; motion-policy training is not implemented');
  assert.ok(report.metrics.positionErrorMeters < 0.02032);
  assert.deepEqual(replay(report), report);
});

test('offline CLI requires explicit fixture mode and never runs models or simulation', () => {
  const script = path.join(__dirname, '../tools/prepare-perception.js');
  const filename = path.join(__dirname, '../roboproof/perception/fixtures/reach-point.json');
  const rejected = spawnSync(process.execPath, [script, filename], {encoding: 'utf8'});
  assert.equal(rejected.status, 1);
  assert.match(rejected.stderr, /require --fixture/);
  const accepted = spawnSync(process.execPath, [script, '--fixture', filename], {encoding: 'utf8'});
  assert.equal(accepted.status, 0, accepted.stderr);
  const proposal = JSON.parse(accepted.stdout);
  assert.equal(proposal.grounding.source.kind, 'fixture');
  assert.match(proposal.validationClock, /not a live/);
  assert.match(proposal.execution, /not run/);
});

test('read-only perception status and dashboard never invoke models or expose execution endpoints', async context => {
  const forbiddenInference = async () => { throw Error('Status must not invoke a model'); };
  const server = createServer({reportPath: 'does-not-exist.json', nemotronClient: {metadata: {}, check: forbiddenInference, complete: forbiddenInference}});
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const status = await fetch(`${base}/api/perception/status`);
  assert.equal(status.status, 200);
  assert.deepEqual(await status.json(), readiness());
  assert.equal((await fetch(`${base}/api/perception/status`, {headers: {Origin: 'https://evil.example'}})).status, 403);
  for (const endpoint of ['/api/perception/run', '/api/perception/infer', '/api/perception/status']) {
    assert.equal((await fetch(`${base}${endpoint}`, {method: 'POST', body: '{}'})).status, 404);
  }
  assert.equal((await fetch(`${base}/perception/index.js`)).status, 404);
  const page = await fetch(`${base}/`);
  assert.match(await page.text(), /id="perception"/);
  assert.match(page.headers.get('content-security-policy'), /script-src 'self'/);
  const script = await fetch(`${base}/perception.js`);
  assert.equal(script.status, 200);
  assert.match(script.headers.get('content-type'), /javascript/);
  assert.match(await script.text(), /api\/perception\/status/);
});
