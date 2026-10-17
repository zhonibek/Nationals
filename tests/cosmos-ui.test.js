'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../roboproof/dashboard/cosmos.js'), 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
const MODEL_ID = 'nvidia/Cosmos-Reason2-2B';
const REVISION = '9ce19a195e423419c349abfc86fd07178b230561';
const identifier = index => `00000000-0000-4000-8000-${String(index).padStart(12, '0')}`;

function syntheticRecord(index = 1) {
  return {
    schemaVersion: 1, id: identifier(index), createdAt: '2026-10-16T08:00:00.000Z', provider: 'huggingface-zerogpu',
    source: {kind: 'original-simulator-pose-visualization', sessionId: identifier(100), sessionSha256: 'a'.repeat(64), inputSha256: 'b'.repeat(64)},
    prompt: 'SYNTHETIC CONTRACT FIXTURE: <img src=x onerror=alert(1)>',
    result: {
      status: 'generated', commentary: 'Synthetic fixture, NOT real inference. <script>alert(1)</script>\nSecond line.',
      evidence_kind: 'unverified_model_commentary', model_id: MODEL_ID, requested_revision: REVISION, resolved_revision: REVISION,
      device: 'cuda:0', dtype: 'bfloat16',
      hardware: {gpu_name: 'Synthetic GPU (contract fixture only)', device: 'cuda:0', dtype: 'bfloat16', total_gpu_bytes: 48 * 1024 ** 3,
        free_gpu_bytes_at_admission: 40 * 1024 ** 3},
      media: {input_sha256: 'b'.repeat(64), kind: 'video', sampled_frames: 2, sampled_timestamps_seconds: [0, 1.5],
        duration_seconds: 2, source_fps: 30, processor_timestamp_basis: 'Synthetic basis <script>text only</script>'},
      robot_actions_executed: false, human_approval_required: true, training_performed: false, simulator_measurements_modified: false,
      possibly_truncated: false, input_tokens: 32, generated_tokens: 16, output_token_limit: 256, elapsed_seconds: 0.5
    }
  };
}

function response(results = [], status = results.length ? 'available' : 'not_generated') {
  return {schemaVersion: 1, status, modelId: MODEL_ID, revision: REVISION, results,
    unverifiedModelCommentary: true, robotActionsExecuted: false, trainingPerformed: false};
}

const ok = value => ({ok: true, json: async () => value});

function dashboard(handler = async () => ok(response()), protocol = 'http:', missingElement = null) {
  const elements = new Map();
  const calls = [];
  const timeouts = [];
  const created = [];
  function makeElement(tag = 'div') {
    let content = '';
    const node = {tagName: tag, children: [], listeners: {}, disabled: false, href: '',
      addEventListener(name, callback) { this.listeners[name] = callback; },
      replaceChildren() { content = ''; this.children = []; },
      appendChild(child) { this.children.push(child); return child; }};
    Object.defineProperty(node, 'textContent', {
      get() { return content + this.children.map(child => child.textContent).join('\n'); },
      set(value) { content = String(value); this.children = []; }
    });
    Object.defineProperty(node, 'innerHTML', {get() { throw Error('HTML parsing is prohibited'); }, set() { throw Error('HTML parsing is prohibited'); }});
    return node;
  }
  for (const id of ['refresh', 'status', 'results', 'hardware-policy']) elements.set(`cosmos-${id}`, makeElement());
  elements.get('cosmos-status').textContent = 'RUNNING';
  elements.get('cosmos-results').appendChild(makeElement('stale-fixture'));
  const realm = vm.createContext({document: {
    getElementById: id => id === missingElement ? null : elements.get(id) || null,
    createElement: tag => { const node = makeElement(tag); created.push(node); return node; }
  }, location: {protocol, hostname: '127.0.0.1'},
  AbortSignal: {timeout(milliseconds) { timeouts.push(milliseconds); return AbortSignal.timeout(milliseconds); }},
  fetch: async (url, options = {}) => { calls.push({url, options}); return handler(url, options); }});
  vm.runInContext(source, realm);
  return {element: id => elements.get(`cosmos-${id}`), calls, timeouts, created,
    refresh: () => elements.get('cosmos-refresh').listeners.click()};
}

test('load and refresh GET only the saved-results endpoint with a timeout, never cloud inference or training', async () => {
  const ui = dashboard();
  await settle();
  assert.match(ui.element('status').textContent, /No real Cosmos answers are saved/);
  assert.equal(ui.element('results').children.length, 0);
  await ui.refresh();
  assert.equal(ui.calls.length, 2);
  for (const call of ui.calls) {
    assert.equal(call.url, '/api/cosmos/results');
    assert.equal(call.options.method, 'GET');
    assert.equal(call.options.body, undefined);
    assert.equal(call.options.cache, 'no-store');
    assert.equal(call.options.redirect, 'error');
    assert(call.options.signal instanceof AbortSignal);
  }
  assert.deepEqual(ui.timeouts, [5000, 5000]);
  assert.match(ui.element('hardware-policy').textContent, /separately approved NVIDIA CUDA GPU/);
  assert.match(ui.element('hardware-policy').textContent, /never calls a cloud model/);
  assert.equal(ui.element('refresh').disabled, false);
});

test('synthetic saved prompt/commentary are text, with reported GPU/provider, revisions, media and only a validated local source link', async () => {
  const record = syntheticRecord();
  record.metadata = {url: 'https://example.invalid/ignore', privatePath: '/private-secret'};
  record.source.media_path = 'javascript:alert(1)';
  const ui = dashboard(async () => ok(response([record])));
  await settle();
  const rendered = ui.element('results').textContent;
  assert.equal(ui.element('results').children.length, 1);
  assert.match(rendered, /<img src=x onerror=alert\(1\)>/);
  assert.match(rendered, /<script>alert\(1\)<\/script>\nSecond line/);
  assert.match(rendered, /Unverified model commentary/);
  assert(rendered.includes(MODEL_ID));
  assert(rendered.includes(`requested revision: ${REVISION}`));
  assert(rendered.includes(`reported resolved revision: ${REVISION}`));
  assert.match(rendered, /Reported provider: huggingface-zerogpu/);
  assert.match(rendered, /reported GPU: Synthetic GPU \(contract fixture only\).*cuda:0.*bfloat16/);
  assert(rendered.includes(String(record.result.hardware.total_gpu_bytes)));
  assert.match(rendered, /video · 2 frame\(s\) · timestamps \(seconds\): 0, 1.5/);
  assert.match(rendered, /duration \(seconds\): 2 · source FPS: 30/);
  assert.match(rendered, /Synthetic basis <script>text only<\/script>/);
  assert(rendered.includes(record.source.sessionSha256));
  assert(rendered.includes(record.source.inputSha256));
  assert.match(rendered, /not camera footage/);
  assert.match(rendered, /not independent proof of GPU authenticity or commentary correctness/);
  assert.match(rendered, /Human approval is required/);
  assert.doesNotMatch(rendered, /private-secret|example\.invalid|javascript:/);
  const anchors = ui.created.filter(node => node.tagName === 'a');
  assert.deepEqual(anchors.map(node => node.href), [`/simulator/index.html?motion=${identifier(100)}`]);
  assert.equal(ui.created.some(node => ['img', 'script', 'video', 'iframe'].includes(node.tagName)), false);
});

test('both reported providers, half precision, sampled images and possible output truncation are represented honestly', async () => {
  const first = syntheticRecord(1);
  first.result.possibly_truncated = true;
  const second = syntheticRecord(2);
  second.provider = 'colab-cuda';
  second.result.device = second.result.hardware.device = 'cuda:2';
  second.result.dtype = second.result.hardware.dtype = 'float16';
  second.result.hardware.gpu_name = 'Other synthetic GPU';
  second.result.generated_tokens = second.result.output_token_limit;
  second.result.media = {input_sha256: second.source.inputSha256, kind: 'image', sampled_frames: 1,
    sampled_timestamps_seconds: [], duration_seconds: null, source_fps: null, processor_timestamp_basis: null};
  const ui = dashboard(async () => ok(response([first, second])));
  await settle();
  assert.equal(ui.element('results').children.length, 2);
  assert.match(ui.element('results').children[0].textContent, /Possibly truncated/);
  assert.match(ui.element('results').children[1].textContent, /Possibly truncated/);
  assert.match(ui.element('results').children[1].textContent, /colab-cuda.*Other synthetic GPU.*cuda:2.*float16/);
  assert.match(ui.element('results').children[1].textContent, /image · 1 frame\(s\).*timestamps \(seconds\): not reported/);
  assert.match(ui.element('status').textContent, /2 saved Cosmos answer/);
});

test('startup never shows RUNNING as a real answer, and concurrent refreshes do not duplicate requests', async () => {
  let complete;
  const ui = dashboard(() => new Promise(resolve => { complete = resolve; }));
  assert.equal(ui.element('results').children.length, 0);
  assert.doesNotMatch(ui.element('status').textContent, /RUNNING|saved Cosmos answer\(s\)/);
  assert.equal(ui.element('refresh').disabled, true);
  await ui.refresh();
  assert.equal(ui.calls.length, 1);
  complete(ok(response([syntheticRecord()])));
  await settle();
  assert.equal(ui.element('results').children.length, 1);
  assert.equal(ui.element('refresh').disabled, false);
});

test('refresh clears stale answers immediately and keeps them cleared after network, HTTP, parsing or timeout failure', async () => {
  const failures = [
    async () => { throw Error('private-secret C:\\private\\receipt.json'); },
    async () => ({ok: false, json: async () => response([syntheticRecord()])}),
    async () => ({ok: true, json: async () => { throw Error('private-secret malformed JSON'); }}),
    async () => { throw Object.assign(Error('private-secret timeout'), {name: 'TimeoutError'}); },
    async () => { throw Object.assign(Error('private-secret abort'), {name: 'AbortError'}); },
    async () => ok(response([], 'invalid'))
  ];
  for (const failure of failures) {
    let mode = 'success';
    let rejectRefresh;
    const ui = dashboard(async () => {
      if (mode === 'success') return ok(response([syntheticRecord()]));
      await new Promise(resolve => { rejectRefresh = resolve; });
      return failure();
    });
    await settle();
    assert.equal(ui.element('results').children.length, 1);
    mode = 'failure';
    const refreshing = ui.refresh();
    assert.equal(ui.element('results').children.length, 0);
    assert.equal(ui.element('refresh').disabled, true);
    rejectRefresh();
    await refreshing;
    assert.equal(ui.element('results').children.length, 0);
    assert.match(ui.element('status').textContent, /unavailable or invalid/);
    assert.doesNotMatch(ui.element('status').textContent, /private-secret|receipt\.json/);
    assert.equal(ui.element('refresh').disabled, false);
    mode = 'success';
    await ui.refresh();
    assert.equal(ui.element('results').children.length, 1);
  }
});

test('a successful empty refresh removes old commentary and explicitly says no real answers are saved', async () => {
  let saved = response([syntheticRecord()]);
  const ui = dashboard(async () => ok(saved));
  await settle();
  saved = response();
  await ui.refresh();
  assert.equal(ui.element('results').children.length, 0);
  assert.match(ui.element('status').textContent, /No real Cosmos answers are saved/);
  assert.match(ui.element('status').textContent, /loading is not a generated answer/);
});

test('invalid envelopes and startup/model-loaded states never render purported answers', async () => {
  const cases = [null, {}, [], {...response(), schemaVersion: 2}, {...response(), modelId: 'other'},
    {...response(), revision: 'main'}, {...response(), trainingPerformed: true}, {...response(), robotActionsExecuted: true},
    {...response(), unverifiedModelCommentary: false}, response([], 'RUNNING'), response([], 'running'),
    response([], 'loaded'), response([], 'not_loaded'), response([], 'available'), response([syntheticRecord()], 'not_generated'),
    response([syntheticRecord()], 'invalid'), {...response(), results: 'not an array'},
    response(Array.from({length: 17}, (_, index) => syntheticRecord(index + 1))), response([syntheticRecord(), syntheticRecord()])];
  for (const saved of cases) {
    const ui = dashboard(async () => ok(saved));
    await settle();
    assert.equal(ui.element('results').children.length, 0);
    assert.match(ui.element('status').textContent, /unavailable or invalid/);
    assert.equal(ui.created.filter(node => node.tagName === 'a').length, 0);
  }
});

test('bad records, traversing source IDs, CPU/model mismatches and unsafe claims fail closed before rendering any result', async () => {
  const cases = [
    record => { record.schemaVersion = 2; }, record => { record.id = '../../secret'; },
    record => { record.createdAt = '2026-02-30T00:00:00Z'; }, record => { record.provider = 'https://example.invalid'; },
    record => { record.source.sessionId = '../../secret'; }, record => { record.source.sessionId = 'javascript:alert(1)'; },
    record => { record.source.kind = 'camera'; }, record => { record.source.inputSha256 = 'b'.repeat(63); },
    record => { record.result.status = 'RUNNING'; }, record => { record.result.status = 'loaded'; },
    record => { record.result.model_id = 'wrong-model'; }, record => { record.result.resolved_revision = null; },
    record => { record.result.requested_revision = 'main'; }, record => { record.result.device = 'cpu'; },
    record => { record.result.hardware.device = 'cpu'; }, record => { record.result.dtype = 'float32'; },
    record => { record.result.hardware.gpu_name = ''; }, record => { record.result.hardware.total_gpu_bytes = Infinity; },
    record => { record.result.hardware.free_gpu_bytes_at_admission = -1; }, record => { record.result.media.input_sha256 = 'c'.repeat(64); },
    record => { record.result.media.sampled_frames = 65; }, record => { record.result.media.sampled_timestamps_seconds = [NaN]; },
    record => { record.result.media.sampled_timestamps_seconds = Array(65).fill(0); }, record => { record.result.media.source_fps = 0; },
    record => { record.result.training_performed = true; }, record => { record.result.robot_actions_executed = true; },
    record => { record.result.simulator_measurements_modified = true; }, record => { record.result.human_approval_required = false; },
    record => { record.result.evidence_kind = 'verified_measurement'; }, record => { record.result.input_tokens = 8193; },
    record => { record.result.generated_tokens = 4097; }, record => { record.result.output_token_limit = 0; },
    record => { record.result.commentary = ''; }, record => { record.result.commentary = 'x'.repeat(100001); },
    record => { record.prompt = 'x'.repeat(4001); }, record => { record.result.possibly_truncated = 'true'; },
    record => { record.result.elapsed_seconds = NaN; }, record => { record.metadata = {trainingPerformed: true}; },
    record => { record.result.hardware.hardwareVerified = true; }
  ];
  for (const mutate of cases) {
    const record = syntheticRecord();
    mutate(record);
    const ui = dashboard(async () => ok(response([syntheticRecord(2), record])));
    await settle();
    assert.equal(ui.element('results').children.length, 0, mutate.toString());
    assert.equal(ui.created.length, 0, mutate.toString());
    assert.match(ui.element('status').textContent, /unavailable or invalid/);
  }
});

test('file mode and unsupported protocols never fetch, and missing panel elements safely disable initialization', async () => {
  for (const protocol of ['file:', 'about:', 'ftp:']) {
    const ui = dashboard(async () => { throw Error('Offline mode must not fetch'); }, protocol);
    await settle();
    await ui.refresh();
    assert.equal(ui.calls.length, 0);
    assert.equal(ui.element('refresh').disabled, true);
    assert.equal(ui.element('results').children.length, 0);
    assert.match(ui.element('status').textContent, /Offline view.*no real Cosmos answers/);
  }
  for (const id of ['refresh', 'status', 'results', 'hardware-policy']) {
    const ui = dashboard(undefined, 'http:', `cosmos-${id}`);
    await settle();
    assert.equal(ui.calls.length, 0);
  }
});
