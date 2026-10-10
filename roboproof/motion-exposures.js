'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {normalizeTask} = require('../simulator/motion');
const {boundedAuditJsonSha256} = require('./motion-holdout');

const ROOT = path.resolve(__dirname, '..');
const LIMITS = {files: 2048, entries: 16000, bytes: 128 * 1024 * 1024, fileBytes: 4 * 1024 * 1024, worlds: 200000, depth: 7};
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const seedValid = seed => Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff;

function geometryFingerprint(task) {
  const normalized = normalizeTask(task);
  const angle = normalized.start.headingDeg * Math.PI / 180;
  const deltaX = normalized.goal.xIn - normalized.start.xIn, deltaY = normalized.goal.yIn - normalized.start.yIn;
  const heading = ((normalized.goal.headingDeg - normalized.start.headingDeg + 540) % 360) - 180;
  return hash([Math.round((deltaX * Math.cos(angle) - deltaY * Math.sin(angle)) * 10),
    Math.round((deltaX * Math.sin(angle) + deltaY * Math.cos(angle)) * 10), Math.round(heading)].join(':'));
}

function collectExposureLedger({root = ROOT} = {}) {
  root = path.resolve(root);
  const files = new Map(), seeds = new Set(), geometries = new Set();
  const roleRecordCounts = {};
  let entries = 0, bytesRead = 0, worlds = 0, seedOnlyRecords = 0;
  const relative = filename => path.relative(root, filename).split(path.sep).join('/');
  function read(filename, role) {
    if (!fs.lstatSync(filename).isFile()) throw Error('Exposure evidence must be a regular file');
    const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
    try {
      const before = fs.fstatSync(descriptor);
      if (before.size > LIMITS.fileBytes || bytesRead + before.size > LIMITS.bytes || files.size >= LIMITS.files) throw Error('Exposure reader budget exhausted');
      const buffer = Buffer.alloc(before.size + 1);
      const length = fs.readSync(descriptor, buffer, 0, buffer.length, 0);
      const after = fs.fstatSync(descriptor);
      if (length !== before.size || after.size !== before.size || after.mtimeMs !== before.mtimeMs) throw Error('Exposure file changed while reading');
      const content = buffer.subarray(0, length);
      bytesRead += length;
      const value = JSON.parse(content.toString('utf8'));
      files.set(relative(filename), {path: relative(filename), sha256: hash(content), role});
      return {value, sha256: hash(content)};
    } finally { fs.closeSync(descriptor); }
  }
  function add(world, role) {
    if (++worlds > LIMITS.worlds || !seedValid(world?.seed)) throw Error('Bounded uint32 exposure seed required');
    seeds.add(world.seed);
    roleRecordCounts[role] = (roleRecordCounts[role] || 0) + 1;
    if (world.task === undefined) seedOnlyRecords++;
    else geometries.add(geometryFingerprint(world.task));
  }
  function extract(value, role) {
    for (const cases of [value.cases, value.development?.cases]) if (Array.isArray(cases)) for (const world of cases) add(world, role);
    const histories = [value.history, ...(Array.isArray(value.models) ? value.models.map(model => model.history) : [])];
    for (const history of histories) if (Array.isArray(history)) for (const row of history) {
      if (!Array.isArray(row.worlds)) throw Error('Training history requires retained exposure rows');
      for (const world of row.worlds) add(world, 'training');
    }
  }
  function scan(directory, depth = 0) {
    if (!fs.existsSync(directory)) return;
    if (!fs.lstatSync(directory).isDirectory() || depth > LIMITS.depth) throw Error('Regular bounded exposure directories required');
    for (const item of fs.readdirSync(directory, {withFileTypes: true}).sort((first, second) => first.name.localeCompare(second.name, 'en'))) {
      if (++entries > LIMITS.entries || item.isSymbolicLink()) throw Error('Exposure entry budget or symlink guard failed');
      const filename = path.join(directory, item.name);
      if (item.isDirectory()) scan(filename, depth + 1);
      else if (item.isFile() && ['run.json', 'summary.json', 'trial-protocol.json', 'worlds.json', 'exposure.json'].includes(item.name)) {
        extract(read(filename, item.name === 'trial-protocol.json' || item.name === 'worlds.json' ? 'development' : 'training').value,
          item.name === 'trial-protocol.json' || item.name === 'worlds.json' ? 'development' : 'training');
      }
    }
  }
  for (const filename of ['motion-evaluation-worlds.json', 'motion-evaluation-worlds-v2.json']) {
    const loaded = read(path.join(root, 'tests/fixtures', filename), 'historical-frozen-test').value;
    if (!Array.isArray(loaded.cases) || !loaded.cases.length) throw Error('Historical frozen cases required');
    extract(loaded, 'historical-frozen-test');
  }
  for (const directory of ['learning', 'research', 'development', 'td3']) scan(path.join(root, 'roboproof/runs/motion', directory));
  const holdouts = path.join(root, 'roboproof/runs/motion/holdout');
  if (fs.existsSync(holdouts)) {
    if (!fs.lstatSync(holdouts).isDirectory()) throw Error('Regular holdout directory required');
    for (const item of fs.readdirSync(holdouts, {withFileTypes: true}).sort((first, second) => first.name.localeCompare(second.name, 'en'))) {
      if (++entries > LIMITS.entries || item.isSymbolicLink()) throw Error('Holdout entry budget or symlink guard failed');
      if (!item.isDirectory()) continue;
      const directory = path.join(holdouts, item.name), consumedPath = path.join(directory, 'consumed.json');
      if (!fs.existsSync(consumedPath)) continue;
      const consumed = read(consumedPath, 'consumed-test-marker').value;
      if (consumed.schemaVersion !== 1 || consumed.kind !== 'local-filesystem-holdout-audit-consumed') throw Error('Valid consumed marker required');
      const commitment = read(path.join(directory, 'commitment.json'), 'consumed-test');
      if (consumed.commitmentSha256 !== commitment.sha256 || commitment.value.casesSha256 !== boundedAuditJsonSha256(commitment.value.corpus?.cases)) throw Error('Consumed exposure commitment mismatch');
      extract(commitment.value.corpus, 'consumed-test');
    }
  }
  return {schemaVersion: 1, specification: 'known-local-motion-exposure-v1',
    scope: 'Known saved training/development, historical fixtures and already-consumed tests only; not external custody, all historical human memory or arbitrary OOD separation',
    geometryDefinition: 'Hashed body-relative displacement quantized to 0.1 inch and heading to 1 degree; deadlines and physical variations ignored',
    sourceFiles: [...files.values()].sort((first, second) => first.path.localeCompare(second.path, 'en')),
    worldSeeds: [...seeds].sort((first, second) => first - second), bodyGeometries: [...geometries].sort(),
    observedWorldRecords: worlds, seedOnlyRecords, roleRecordCounts,
    legacyGeometryFullyKnown: seedOnlyRecords === 0};
}

function assertFreshCases(cases, ledger) {
  if (ledger?.specification !== 'known-local-motion-exposure-v1' || !Array.isArray(cases) || !cases.length || cases.length > 256 ||
      !Array.isArray(ledger.worldSeeds) || !ledger.worldSeeds.every(seedValid) || !Array.isArray(ledger.bodyGeometries) ||
      !ledger.bodyGeometries.every(value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value)) ||
      !Number.isSafeInteger(ledger.seedOnlyRecords) || ledger.seedOnlyRecords < 0 ||
      ledger.legacyGeometryFullyKnown !== (ledger.seedOnlyRecords === 0)) throw Error('Bounded cases and valid exposure ledger required');
  const knownSeeds = new Set(ledger.worldSeeds), knownGeometry = new Set(ledger.bodyGeometries);
  for (const world of cases) {
    if (!seedValid(world?.seed)) throw Error('Final requires a uint32 seed');
    if (knownSeeds.has(world.seed) || knownGeometry.has(geometryFingerprint(world.task))) throw Error('Final overlaps known training/development/consumed exposure');
  }
  return {verified: true, ledgerSha256: boundedAuditJsonSha256(ledger), finalCount: cases.length,
    knownUniqueSeeds: knownSeeds.size, knownUniqueGeometries: knownGeometry.size,
    seedOnlyRecords: ledger.seedOnlyRecords, legacyGeometryFullyKnown: ledger.legacyGeometryFullyKnown, scope: ledger.scope};
}

function snapshotExposureLedger(directory, options = {}) {
  const ledger = collectExposureLedger(options), bytes = JSON.stringify(ledger) + '\n';
  if (Buffer.byteLength(bytes) > LIMITS.fileBytes) throw Error('Exposure snapshot exceeds byte budget');
  fs.mkdirSync(directory, {recursive: true});
  if (!fs.lstatSync(directory).isDirectory()) throw Error('Regular exposure snapshot directory required');
  const descriptor = fs.openSync(path.join(directory, 'exposure-ledger.json'), 'wx', 0o600);
  try { fs.writeFileSync(descriptor, bytes); fs.fsyncSync(descriptor); }
  finally { fs.closeSync(descriptor); }
  return {ledger, ledgerSha256: boundedAuditJsonSha256(ledger)};
}

function readExposureSnapshot(directory) {
  const filename = path.join(directory, 'exposure-ledger.json');
  if (!fs.lstatSync(filename).isFile() || fs.statSync(filename).size > LIMITS.fileBytes) throw Error('Bounded regular exposure snapshot required');
  const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
  try {
    const before = fs.fstatSync(descriptor);
    const bytes = Buffer.alloc(before.size + 1);
    const length = fs.readSync(descriptor, bytes, 0, bytes.length, 0);
    const after = fs.fstatSync(descriptor);
    if (length !== before.size || after.size !== before.size || after.mtimeMs !== before.mtimeMs) throw Error('Exposure snapshot changed while reading');
    const ledger = JSON.parse(bytes.subarray(0, length).toString('utf8'));
    if (ledger.specification !== 'known-local-motion-exposure-v1') throw Error('Unsupported exposure snapshot');
    return {ledger, ledgerSha256: boundedAuditJsonSha256(ledger)};
  } finally { fs.closeSync(descriptor); }
}

function assertExposureSnapshotCurrent(directory, options = {}) {
  const saved = readExposureSnapshot(directory);
  if (boundedAuditJsonSha256(collectExposureLedger(options)) !== saved.ledgerSha256) throw Error('Known exposures changed; prepare a new commitment before reveal');
  return saved;
}

module.exports = {collectExposureLedger, assertFreshCases, geometryFingerprint, snapshotExposureLedger,
  readExposureSnapshot, assertExposureSnapshotCurrent};
