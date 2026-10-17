'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {TextDecoder} = require('node:util');

const MODEL_ID = 'nvidia/Cosmos-Reason2-2B';
const REVISION = '9ce19a195e423419c349abfc86fd07178b230561';
const MAX_FILES = 16;
const MAX_BYTES = 256 * 1024;
const MAX_ENTRIES = 1024;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const uuid = value => typeof value === 'string' && UUID.test(value);
const hash = value => typeof value === 'string' && SHA256.test(value);
const text = (value, maximum) => typeof value === 'string' && value.trim().length > 0 && value.length <= maximum;
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const integer = (value, maximum) => Number.isInteger(value) && value >= 1 && value <= maximum;
const nonnegative = value => Number.isFinite(value) && value >= 0;
const positive = value => Number.isFinite(value) && value > 0;

function timestamp(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/.test(value)) return false;
  const calendar = new Date(`${value.slice(0, 19)}Z`);
  return Number.isFinite(Date.parse(value)) && Number.isFinite(calendar.getTime()) && calendar.toISOString().slice(0, 19) === value.slice(0, 19);
}

function safeClaims(record) {
  const negative = new Set(['robotactionsexecuted', 'trainingperformed', 'simulatormeasurementsmodified',
    'hardwareverified', 'hardwareauthenticityverified', 'commentaryverified', 'livemodelverified', 'liveinferenceverified']);
  const affirmative = new Set(['humanapprovalrequired', 'unverifiedmodelcommentary']);
  const pending = [{value: record, depth: 0}];
  let visited = 0;
  while (pending.length) {
    const {value, depth} = pending.pop();
    if (++visited > 20000 || depth > 32) return false;
    if (value === null || typeof value !== 'object') continue;
    for (const [key, child] of Object.entries(value)) {
      const flag = key.toLowerCase().replace(/[^a-z0-9]/g, '');
      if ((negative.has(flag) && child !== false) || (affirmative.has(flag) && child !== true)) return false;
      pending.push({value: child, depth: depth + 1});
    }
  }
  return true;
}

function project(record) {
  if (!object(record) || record.schemaVersion !== 1 || !uuid(record.id) || !timestamp(record.createdAt)
    || !['huggingface-zerogpu', 'colab-cuda'].includes(record.provider) || !text(record.prompt, 4000) || !safeClaims(record)) throw Error();
  const {source, result} = record;
  if (!object(source) || source.kind !== 'original-simulator-pose-visualization' || !uuid(source.sessionId)
    || !hash(source.sessionSha256) || !hash(source.inputSha256) || !object(result)) throw Error();
  const {hardware, media} = result;
  if (result.status !== 'generated' || !text(result.commentary, 100000) || result.evidence_kind !== 'unverified_model_commentary'
    || result.model_id !== MODEL_ID || result.requested_revision !== REVISION || result.resolved_revision !== REVISION
    || typeof result.device !== 'string' || !/^cuda(?::(?:0|[1-9]\d*))?$/.test(result.device)
    || !['float16', 'bfloat16'].includes(result.dtype) || !object(hardware) || !text(hardware.gpu_name, 128)
    || hardware.device !== result.device || hardware.dtype !== result.dtype || !positive(hardware.total_gpu_bytes)
    || !nonnegative(hardware.free_gpu_bytes_at_admission) || hardware.free_gpu_bytes_at_admission > hardware.total_gpu_bytes) throw Error();
  if (!object(media) || media.input_sha256 !== source.inputSha256 || !['image', 'video'].includes(media.kind)
    || !integer(media.sampled_frames, 64) || !Array.isArray(media.sampled_timestamps_seconds)
    || media.sampled_timestamps_seconds.length > 64 || !media.sampled_timestamps_seconds.every(nonnegative)
    || !(media.duration_seconds === null || nonnegative(media.duration_seconds))
    || !(media.source_fps === null || positive(media.source_fps))
    || !(media.processor_timestamp_basis === null || typeof media.processor_timestamp_basis === 'string')
    || result.robot_actions_executed !== false || result.human_approval_required !== true || result.training_performed !== false
    || result.simulator_measurements_modified !== false || typeof result.possibly_truncated !== 'boolean'
    || !integer(result.input_tokens, 8192) || !integer(result.generated_tokens, 4096) || !integer(result.output_token_limit, 4096)
    || result.generated_tokens > result.output_token_limit || !nonnegative(result.elapsed_seconds)) throw Error();
  return {
    schemaVersion: 1, id: record.id, createdAt: record.createdAt, provider: record.provider,
    source: {kind: source.kind, sessionId: source.sessionId, sessionSha256: source.sessionSha256, inputSha256: source.inputSha256},
    prompt: record.prompt,
    result: {
      status: 'generated', commentary: result.commentary, evidence_kind: 'unverified_model_commentary', model_id: MODEL_ID,
      requested_revision: REVISION, resolved_revision: REVISION, device: result.device, dtype: result.dtype,
      hardware: {gpu_name: hardware.gpu_name, device: hardware.device, dtype: hardware.dtype,
        total_gpu_bytes: hardware.total_gpu_bytes, free_gpu_bytes_at_admission: hardware.free_gpu_bytes_at_admission},
      media: {input_sha256: media.input_sha256, kind: media.kind, sampled_frames: media.sampled_frames,
        sampled_timestamps_seconds: [...media.sampled_timestamps_seconds], duration_seconds: media.duration_seconds,
        source_fps: media.source_fps, processor_timestamp_basis: media.processor_timestamp_basis},
      robot_actions_executed: false, human_approval_required: true, training_performed: false, simulator_measurements_modified: false,
      possibly_truncated: result.possibly_truncated, input_tokens: result.input_tokens, generated_tokens: result.generated_tokens,
      output_token_limit: result.output_token_limit, elapsed_seconds: result.elapsed_seconds
    }
  };
}

function sameFile(before, after) {
  return before.dev === after.dev && before.ino === after.ino && before.size === after.size
    && before.mtimeMs === after.mtimeMs && before.ctimeMs === after.ctimeMs;
}

function directoryPath(directory) {
  if (typeof directory !== 'string' || !directory || /\0/.test(directory) || /^[/\\]{2}/.test(directory)
    || directory.split(/[/\\]/).includes('..') || /^[a-z]:(?:$|[^/\\])/i.test(directory)) throw Error();
  const resolved = path.resolve(directory);
  if (resolved.slice(path.parse(resolved).root.length).includes(':')) throw Error();
  return resolved;
}

function checkDirectory(directory) {
  const ancestors = [];
  let current = directory;
  while (true) {
    ancestors.push(current);
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  let result;
  for (const ancestor of ancestors.reverse()) {
    result = fs.lstatSync(ancestor);
    if (result.isSymbolicLink() || !result.isDirectory()) throw Error();
  }
  return result;
}

function readRecord(directory, name, directoryStat) {
  if (!name.endsWith('.json') || name.includes('/') || name.includes('\\') || name.includes(':') || name.includes('\0')) throw Error();
  const filename = path.resolve(directory, name);
  if (path.dirname(filename) !== directory || !sameFile(directoryStat, checkDirectory(directory))) throw Error();
  const stat = fs.lstatSync(filename);
  if (stat.isSymbolicLink() || !stat.isFile() || stat.size < 1 || stat.size > MAX_BYTES) throw Error();
  const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0) | (fs.constants.O_NONBLOCK ?? 0));
  try {
    const before = fs.fstatSync(descriptor);
    if (!before.isFile() || !sameFile(stat, before)) throw Error();
    const buffer = Buffer.alloc(before.size);
    let length = 0;
    while (length < buffer.length) {
      const count = fs.readSync(descriptor, buffer, length, buffer.length - length, length);
      if (!count) throw Error();
      length += count;
    }
    const after = fs.lstatSync(filename);
    if (!sameFile(before, fs.fstatSync(descriptor)) || !sameFile(before, after) || after.isSymbolicLink() || !after.isFile()
      || !sameFile(directoryStat, checkDirectory(directory))) throw Error();
    return project(JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(buffer)));
  } finally { fs.closeSync(descriptor); }
}

function readResults(directory = path.join(__dirname, 'runs', 'cosmos')) {
  const response = {schemaVersion: 1, status: 'not_generated', modelId: MODEL_ID, revision: REVISION, results: [],
    unverifiedModelCommentary: true, robotActionsExecuted: false, trainingPerformed: false};
  try {
    const resolved = directoryPath(directory);
    let directoryStat;
    try { directoryStat = checkDirectory(resolved); }
    catch (error) {
      if (error.code === 'ENOENT') return response;
      throw error;
    }
    const files = [];
    const handle = fs.opendirSync(resolved);
    try {
      let entries = 0;
      for (let entry = handle.readSync(); entry; entry = handle.readSync()) {
        if (++entries > MAX_ENTRIES || entry.isSymbolicLink()) throw Error();
        if (!entry.name.endsWith('.json')) continue;
        if (!entry.isFile() || files.length >= MAX_FILES) throw Error();
        files.push(entry.name);
      }
    } finally { handle.closeSync(); }
    if (!sameFile(directoryStat, checkDirectory(resolved))) throw Error();
    const results = files.sort().map(name => readRecord(resolved, name, directoryStat));
    if (new Set(results.map(record => record.id)).size !== results.length || !sameFile(directoryStat, checkDirectory(resolved))) throw Error();
    results.sort((left, right) => Date.parse(right.createdAt) - Date.parse(left.createdAt) || left.id.localeCompare(right.id));
    return {...response, status: results.length ? 'available' : 'not_generated', results};
  } catch { return {...response, status: 'invalid'}; }
}

module.exports = {readResults};
