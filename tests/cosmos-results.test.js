'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {readResults} = require('../roboproof/cosmos-results');

const MODEL_ID = 'nvidia/Cosmos-Reason2-2B';
const REVISION = '9ce19a195e423419c349abfc86fd07178b230561';
const MAX_BYTES = 256 * 1024;
const identifier = index => `00000000-0000-4000-8000-${String(index).padStart(12, '0')}`;
const empty = status => ({schemaVersion: 1, status, modelId: MODEL_ID, revision: REVISION, results: [],
  unverifiedModelCommentary: true, robotActionsExecuted: false, trainingPerformed: false});

function syntheticRecord(index = 1) {
  return {
    schemaVersion: 1, id: identifier(index), createdAt: '2026-10-16T08:00:00.000Z', provider: 'huggingface-zerogpu',
    source: {kind: 'original-simulator-pose-visualization', sessionId: identifier(100),
      sessionSha256: 'a'.repeat(64), inputSha256: 'b'.repeat(64)},
    prompt: 'SYNTHETIC CONTRACT FIXTURE ONLY: describe this pose.',
    result: {
      status: 'generated', commentary: 'Synthetic contract commentary only, NOT real Cosmos inference.',
      evidence_kind: 'unverified_model_commentary', model_id: MODEL_ID, requested_revision: REVISION, resolved_revision: REVISION,
      device: 'cuda:0', dtype: 'bfloat16',
      hardware: {gpu_name: 'Synthetic GPU (contract fixture only)', device: 'cuda:0', dtype: 'bfloat16',
        total_gpu_bytes: 48 * 1024 ** 3, free_gpu_bytes_at_admission: 40 * 1024 ** 3},
      media: {input_sha256: 'b'.repeat(64), kind: 'image', sampled_frames: 1, sampled_timestamps_seconds: [],
        duration_seconds: null, source_fps: null, processor_timestamp_basis: null},
      robot_actions_executed: false, human_approval_required: true, training_performed: false, simulator_measurements_modified: false,
      possibly_truncated: false, input_tokens: 32, generated_tokens: 16, output_token_limit: 256, elapsed_seconds: 0.5
    }
  };
}

function fixture(context) {
  const temporary = fs.realpathSync(os.tmpdir());
  const root = fs.mkdtempSync(path.join(temporary, 'cosmos-results-test-'));
  context.after(() => {
    assert.equal(path.dirname(path.resolve(root)), temporary);
    assert(path.basename(root).startsWith('cosmos-results-test-'));
    fs.rmSync(root, {recursive: true, force: true});
  });
  const directory = path.join(root, 'results');
  fs.mkdirSync(directory);
  return {root, directory, write(record, name = 'synthetic.json') {
    const filename = path.join(directory, name);
    fs.writeFileSync(filename, JSON.stringify(record));
    return filename;
  }};
}

function assign(record, field, value) {
  const segments = field.split('.');
  let target = record;
  for (const segment of segments.slice(0, -1)) target = target[segment];
  target[segments.at(-1)] = value;
}

test('missing and empty directories are honest, read-only not_generated results; default is roboproof/runs/cosmos', context => {
  const saved = fixture(context);
  const missing = path.join(saved.root, 'missing');
  for (const name of ['mkdirSync', 'writeFileSync', 'appendFileSync', 'renameSync']) {
    context.mock.method(fs, name, () => { throw Error('Reader must not write'); });
  }
  assert.deepEqual(readResults(missing), empty('not_generated'));
  assert.equal(fs.existsSync(missing), false);
  assert.deepEqual(readResults(saved.directory), empty('not_generated'));
  const defaultDirectory = path.resolve(__dirname, '../roboproof/runs/cosmos');
  const original = fs.lstatSync;
  let inspected = false;
  context.mock.method(fs, 'lstatSync', filename => {
    if (filename === defaultDirectory) {
      inspected = true;
      throw Object.assign(Error('Synthetic missing directory'), {code: 'ENOENT'});
    }
    return original(filename);
  });
  assert.deepEqual(readResults(), empty('not_generated'));
  assert.equal(inspected, true);
});

test('synthetic receipts retain only known evidence and reported provenance, without mutating saved files', context => {
  const saved = fixture(context);
  const record = syntheticRecord();
  record.operator_metadata = {privatePath: path.join(saved.root, 'private-secret'), url: 'https://example.invalid/not-fetched'};
  record.source.media_path = '../private-secret';
  record.result.hardware.untrusted = 'private-secret';
  record.result.media.filename = 'private-secret';
  record.result.debug = {access_token: 'private-secret'};
  const filename = saved.write(record);
  const before = fs.readFileSync(filename, 'utf8');
  const response = readResults(saved.directory);
  assert.equal(response.status, 'available');
  assert.deepEqual(response.results, [syntheticRecord()]);
  assert.equal(response.unverifiedModelCommentary, true);
  assert.equal(response.robotActionsExecuted, false);
  assert.equal(response.trainingPerformed, false);
  assert.equal(fs.readFileSync(filename, 'utf8'), before);
  assert.doesNotMatch(JSON.stringify(response), /private-secret|example\.invalid|access_token|hardwareVerified/);
});

test('both approved providers, CUDA indices, half precisions, Python ISO timestamps and sampled video are accepted', context => {
  const saved = fixture(context);
  const first = syntheticRecord(1);
  first.result.device = first.result.hardware.device = 'cuda';
  first.result.dtype = first.result.hardware.dtype = 'float16';
  first.result.hardware.free_gpu_bytes_at_admission = 0;
  first.createdAt = '2026-10-16T08:00:00.123456+00:00';
  const second = syntheticRecord(2);
  second.provider = 'colab-cuda';
  second.createdAt = '2026-10-16T14:00:01+05:00';
  second.result.device = second.result.hardware.device = 'cuda:12';
  second.result.media = {input_sha256: second.source.inputSha256, kind: 'video', sampled_frames: 64,
    sampled_timestamps_seconds: Array.from({length: 64}, (_, index) => index / 4), duration_seconds: 16,
    source_fps: 30, processor_timestamp_basis: 'Synthetic timestamp basis'};
  second.result.input_tokens = 8192;
  second.result.generated_tokens = second.result.output_token_limit = 4096;
  second.result.possibly_truncated = true;
  second.result.elapsed_seconds = 0;
  saved.write(first, 'first.json');
  saved.write(second, 'second.json');
  const response = readResults(saved.directory);
  assert.equal(response.status, 'available');
  assert.deepEqual(response.results, [second, first]);
});

test('at most sixteen direct files are read, deterministically sorted, never nested or unrelated files', context => {
  const saved = fixture(context);
  fs.mkdirSync(path.join(saved.directory, 'nested'));
  fs.writeFileSync(path.join(saved.directory, 'nested', 'ignored.json'), 'not a receipt');
  fs.writeFileSync(path.join(saved.directory, 'ignored.txt'), 'not a receipt');
  for (let index = 16; index >= 1; index--) saved.write(syntheticRecord(index), `${index}.json`);
  const response = readResults(saved.directory);
  assert.equal(response.status, 'available');
  assert.deepEqual(response.results.map(record => record.id), Array.from({length: 16}, (_, index) => identifier(index + 1)));
  saved.write(syntheticRecord(17), '17.json');
  let opened = 0;
  context.mock.method(fs, 'openSync', () => { opened++; throw Error('No result should be opened past the file budget'); });
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
  assert.equal(opened, 0);
});

test('exactly 256 KiB is admitted, oversized files are never opened, and reads stay within the byte budget', context => {
  const saved = fixture(context);
  const record = syntheticRecord();
  record.padding = '';
  record.padding = 'x'.repeat(MAX_BYTES - Buffer.byteLength(JSON.stringify(record)));
  const filename = saved.write(record);
  assert.equal(fs.statSync(filename).size, MAX_BYTES);
  const originalRead = fs.readSync;
  let readBytes = 0;
  context.mock.method(fs, 'readSync', (descriptor, buffer, offset, length, position) => {
    assert(length <= MAX_BYTES);
    const count = originalRead(descriptor, buffer, offset, length, position);
    readBytes += count;
    return count;
  });
  assert.equal(readResults(saved.directory).status, 'available');
  assert.equal(readBytes, MAX_BYTES);
  fs.appendFileSync(filename, ' ');
  context.mock.method(fs, 'openSync', () => { throw Error('Oversized files must not be opened'); });
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
  assert.equal(readBytes, MAX_BYTES);
});

test('invalid records fail closed as a whole, including versions, hashes, limits, CPU and action/training claims', context => {
  const saved = fixture(context);
  saved.write(syntheticRecord(2), 'also-valid.json');
  const cases = [
    ['schemaVersion', 2], ['schemaVersion', '1'], ['id', '../secret'], ['id', identifier(1).toUpperCase().replace('0000-', 'ABCD-')],
    ['id', [identifier(1)]], ['createdAt', '2026-02-30T08:00:00Z'], ['createdAt', '2026-10-16'],
    ['createdAt', '2026-10-16T24:00:00Z'], ['createdAt', '2026-10-16T08:00:00+25:00'],
    ['provider', 'local-cpu'], ['provider', 'https://example.invalid'], ['prompt', ' '], ['prompt', 'x'.repeat(4001)],
    ['source.kind', 'camera'], ['source.sessionId', '../secret'], ['source.sessionId', [identifier(100)]],
    ['source.sessionSha256', 'A'.repeat(64)], ['source.sessionSha256', ['a'.repeat(64)]], ['source.inputSha256', 'b'.repeat(63)],
    ['result.status', 'RUNNING'], ['result.status', 'loaded'], ['result.commentary', ''], ['result.commentary', 'x'.repeat(100001)],
    ['result.evidence_kind', 'verified_measurement'], ['result.model_id', 'nvidia/other-model'],
    ['result.requested_revision', 'main'], ['result.resolved_revision', null], ['result.resolved_revision', 'c'.repeat(40)],
    ['result.device', 'cpu'], ['result.device', 'cuda:../0'], ['result.dtype', 'float32'],
    ['result.hardware.gpu_name', ' '], ['result.hardware.gpu_name', 'x'.repeat(129)], ['result.hardware.device', 'cpu'],
    ['result.hardware.dtype', 'float16'], ['result.hardware.total_gpu_bytes', 0], ['result.hardware.total_gpu_bytes', true],
    ['result.hardware.free_gpu_bytes_at_admission', -1], ['result.hardware.free_gpu_bytes_at_admission', 49 * 1024 ** 3],
    ['result.media.input_sha256', 'c'.repeat(64)], ['result.media.kind', 'audio'], ['result.media.sampled_frames', 0],
    ['result.media.sampled_frames', 65], ['result.media.sampled_frames', 1.5], ['result.media.sampled_frames', true],
    ['result.media.sampled_timestamps_seconds', Array(65).fill(0)], ['result.media.sampled_timestamps_seconds', [-1]],
    ['result.media.sampled_timestamps_seconds', ['0']], ['result.media.sampled_timestamps_seconds', null],
    ['result.media.duration_seconds', -1], ['result.media.source_fps', 0], ['result.media.processor_timestamp_basis', {}],
    ['result.robot_actions_executed', true], ['result.human_approval_required', false], ['result.training_performed', true],
    ['result.simulator_measurements_modified', true], ['result.possibly_truncated', 0],
    ['result.input_tokens', 0], ['result.input_tokens', 8193], ['result.input_tokens', true],
    ['result.generated_tokens', 0], ['result.generated_tokens', 4097], ['result.generated_tokens', 257],
    ['result.output_token_limit', 0], ['result.output_token_limit', 4097], ['result.elapsed_seconds', -1],
    ['trainingPerformed', true], ['robot_actions_executed', true], ['humanApprovalRequired', false],
    ['result.hardware.hardwareVerified', true], ['result.commentaryVerified', true],
    ['result.metadata', {nested: {training_performed: true}}]
  ];
  for (const [field, value] of cases) {
    const record = syntheticRecord();
    assign(record, field, value);
    saved.write(record);
    assert.deepEqual(readResults(saved.directory), empty('invalid'), field);
  }
});

test('missing required fields, malformed JSON/UTF-8, nonfinite numbers and empty result files fail closed', context => {
  const saved = fixture(context);
  const filename = saved.write(syntheticRecord());
  const fields = ['schemaVersion', 'id', 'createdAt', 'provider', 'source', 'prompt', 'result'];
  for (const field of fields) {
    const record = syntheticRecord();
    delete record[field];
    saved.write(record);
    assert.deepEqual(readResults(saved.directory), empty('invalid'), field);
  }
  for (const field of Object.keys(syntheticRecord().result)) {
    const record = syntheticRecord();
    delete record.result[field];
    saved.write(record);
    assert.deepEqual(readResults(saved.directory), empty('invalid'), `result.${field}`);
  }
  for (const content of ['', '{private-secret', 'null', '[]', JSON.stringify(syntheticRecord()).replace('"elapsed_seconds":0.5', '"elapsed_seconds":1e999'),
    Buffer.concat([Buffer.from('{"bad":"'), Buffer.from([0xff]), Buffer.from('"}')])]) {
    fs.writeFileSync(filename, content);
    assert.deepEqual(readResults(saved.directory), empty('invalid'));
  }
});

test('duplicate receipt IDs, nonregular JSON entries and overly deep metadata are rejected', context => {
  const saved = fixture(context);
  saved.write(syntheticRecord(), 'first.json');
  saved.write(syntheticRecord(), 'second.json');
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
  fs.unlinkSync(path.join(saved.directory, 'second.json'));
  fs.mkdirSync(path.join(saved.directory, 'directory.json'));
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
  fs.rmdirSync(path.join(saved.directory, 'directory.json'));
  const record = syntheticRecord();
  let nested = {};
  for (let index = 0; index < 35; index++) nested = {nested};
  record.metadata = nested;
  saved.write(record, 'first.json');
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
});

test('directory traversal, network/alternate-stream paths and filesystem error text never escape the generic status', context => {
  const saved = fixture(context);
  for (const value of [null, {}, '', 'C:', 'C:relative', 'https://example.invalid/results', '\\\\server\\share',
    '//server/share', `${saved.directory}${path.sep}..${path.sep}results`, `${saved.directory}:private-secret`]) {
    assert.deepEqual(readResults(value), empty('invalid'));
  }
  const original = fs.lstatSync;
  context.mock.method(fs, 'lstatSync', filename => {
    if (filename === saved.directory) throw Object.assign(Error(`private-secret at ${filename}`), {code: 'EACCES'});
    return original(filename);
  });
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
  assert.doesNotMatch(JSON.stringify(readResults(saved.directory)), /private-secret|EACCES|results-test/);
});

test('symlink result files are rejected before opening even on hosts without symlink privileges', context => {
  const saved = fixture(context);
  const filename = saved.write(syntheticRecord());
  const original = fs.lstatSync;
  context.mock.method(fs, 'lstatSync', target => target === filename ? {isSymbolicLink: () => true, isFile: () => true} : original(target));
  let opened = false;
  context.mock.method(fs, 'openSync', () => { opened = true; throw Error('No following symlinks'); });
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
  assert.equal(opened, false);
});

test('symlink directories and linked ancestors of missing directories are rejected', context => {
  const saved = fixture(context);
  saved.write(syntheticRecord());
  const linked = path.join(saved.root, 'linked');
  try { fs.symlinkSync(saved.directory, linked, process.platform === 'win32' ? 'junction' : 'dir'); }
  catch (error) {
    if (['EPERM', 'EACCES', 'ENOTSUP'].includes(error.code)) return context.skip('Host does not permit synthetic directory links');
    throw error;
  }
  assert.deepEqual(readResults(linked), empty('invalid'));
  assert.deepEqual(readResults(path.join(linked, 'missing')), empty('invalid'));
});

test('file identity/mutation checks reject changed descriptors and bounded short reads complete safely', context => {
  const saved = fixture(context);
  saved.write(syntheticRecord());
  const originalRead = fs.readSync;
  context.mock.method(fs, 'readSync', (descriptor, buffer, offset, length, position) => originalRead(descriptor, buffer, offset, Math.min(length, 31), position));
  assert.equal(readResults(saved.directory).status, 'available');
  const originalStat = fs.fstatSync;
  let checks = 0;
  context.mock.method(fs, 'fstatSync', descriptor => {
    const stat = originalStat(descriptor);
    return ++checks === 2 ? {...stat, mtimeMs: stat.mtimeMs + 1} : stat;
  });
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
});

test('directory enumeration is bounded and closes its handle without reading result contents', context => {
  const saved = fixture(context);
  let entries = 0;
  let closed = false;
  context.mock.method(fs, 'opendirSync', () => ({
    readSync: () => ++entries <= 1025 ? {name: `unrelated-${entries}.txt`, isSymbolicLink: () => false} : null,
    closeSync() { closed = true; }
  }));
  assert.deepEqual(readResults(saved.directory), empty('invalid'));
  assert.equal(entries, 1025);
  assert.equal(closed, true);
});
