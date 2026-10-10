'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {createHash, randomUUID} = require('node:crypto');

const LOCAL_FILESYSTEM_AUDIT_SCOPE = 'Local filesystem audit guards only, not tamper-proof secrecy. Stored cases are plaintext. Freshness/training exclusion and evaluation are caller responsibilities. The identity reader must return current source hashes and runtime identity on every call. Evaluate returned candidate JSON snapshots, not reloaded mutable files. Consumption burns the holdout before return, even if evaluation fails. Exclusive publication requires local filesystem hard-link support.';
const AUDIT_LIMITS = Object.freeze({jsonBytes: 4 * 1024 * 1024, identityBytes: 64 * 1024,
  cases: 4096, candidates: 32, depth: 32, nodes: 100000});
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const validHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const plainObject = value => value !== null && typeof value === 'object' &&
  [Object.prototype, null].includes(Object.getPrototypeOf(value));
const validLabel = value => typeof value === 'string' && value.trim().length > 0 && Buffer.byteLength(value) <= 256;

function boundedAuditJson(value, maximumBytes = AUDIT_LIMITS.jsonBytes) {
  const chunks = [], ancestors = new Set();
  let bytes = 0, nodes = 0;
  function append(chunk) {
    bytes += Buffer.byteLength(chunk);
    if (bytes > maximumBytes) throw Error('Audit JSON exceeds byte limit');
    chunks.push(chunk);
  }
  function visit(entry, depth) {
    if (++nodes > AUDIT_LIMITS.nodes || depth > AUDIT_LIMITS.depth) throw Error('Audit JSON exceeds structural limit');
    if (entry === null || typeof entry === 'boolean') return append(JSON.stringify(entry));
    if (typeof entry === 'string') {
      if (entry.length > maximumBytes) throw Error('Audit JSON exceeds byte limit');
      return append(JSON.stringify(entry));
    }
    if (typeof entry === 'number' && Number.isFinite(entry)) return append(JSON.stringify(entry));
    if (!Array.isArray(entry) && !plainObject(entry)) throw Error('Audit evidence must be plain JSON');
    if (ancestors.has(entry)) throw Error('Cyclic audit JSON');
    if (Object.getOwnPropertySymbols(entry).length) throw Error('Audit evidence must be plain JSON');
    ancestors.add(entry);
    const keys = Object.keys(entry);
    if (keys.length > AUDIT_LIMITS.nodes || (Array.isArray(entry) && keys.length !== entry.length)) throw Error('Audit evidence must be bounded dense JSON');
    append(Array.isArray(entry) ? '[' : '{');
    const ordered = Array.isArray(entry) ? Array.from({length: entry.length}, (_, index) => String(index)) : keys.sort();
    for (const [index, key] of ordered.entries()) {
      const descriptor = Object.getOwnPropertyDescriptor(entry, key);
      if (!descriptor || !Object.hasOwn(descriptor, 'value')) throw Error('Audit JSON accessors are not supported');
      if (index) append(',');
      if (!Array.isArray(entry)) {
        if (key.length > maximumBytes) throw Error('Audit JSON exceeds byte limit');
        append(JSON.stringify(key));
        append(':');
      }
      visit(descriptor.value, depth + 1);
    }
    append(Array.isArray(entry) ? ']' : '}');
    ancestors.delete(entry);
  }
  visit(value, 0);
  return chunks.join('');
}

function boundedAuditJsonSha256(value) {
  return digest(boundedAuditJson(value));
}

function readBoundedJson(filename) {
  if (!fs.lstatSync(filename).isFile()) throw Error('Audit evidence must be a regular file, not a symlink');
  const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
  try {
    const before = fs.fstatSync(descriptor);
    if (!before.isFile() || before.size > AUDIT_LIMITS.jsonBytes) throw Error('Audit JSON file exceeds byte limit');
    const buffer = Buffer.alloc(before.size + 1);
    let count = 0, read = 0;
    do {
      read = fs.readSync(descriptor, buffer, count, buffer.length - count, count);
      count += read;
    } while (read && count < buffer.length);
    const after = fs.fstatSync(descriptor);
    if (count !== before.size || after.size !== before.size || after.mtimeMs !== before.mtimeMs) throw Error('Audit evidence changed while reading');
    const bytes = buffer.subarray(0, count), text = bytes.toString('utf8');
    if (!Buffer.from(text, 'utf8').equals(bytes)) throw Error('Audit JSON must be UTF-8');
    const value = JSON.parse(text);
    const canonical = boundedAuditJson(value);
    return {value, bytes, sha256: digest(bytes), canonical};
  } finally {
    fs.closeSync(descriptor);
  }
}

function publishExclusiveJson(filename, value) {
  const bytes = boundedAuditJson(value) + '\n';
  if (Buffer.byteLength(bytes) > AUDIT_LIMITS.jsonBytes) throw Error('Audit JSON exceeds byte limit');
  fs.mkdirSync(path.dirname(filename), {recursive: true, mode: 0o700});
  const temporary = `${filename}.${randomUUID()}.tmp`;
  const descriptor = fs.openSync(temporary, 'wx', 0o600);
  try {
    try {
      fs.writeFileSync(descriptor, bytes);
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    fs.linkSync(temporary, filename);
  } finally {
    fs.unlinkSync(temporary);
  }
  return digest(bytes);
}

function corpusSnapshot(corpus) {
  const snapshot = JSON.parse(boundedAuditJson(corpus));
  if (!plainObject(snapshot) || !Array.isArray(snapshot.cases) || !snapshot.cases.length ||
      snapshot.cases.length > AUDIT_LIMITS.cases) throw Error('Bounded nonempty corpus.cases required');
  const ids = new Set();
  for (const entry of snapshot.cases) {
    if (!plainObject(entry) || !validLabel(entry.id) || ids.has(entry.id)) throw Error('Unique bounded corpus case IDs required');
    ids.add(entry.id);
  }
  return snapshot;
}

function identitySnapshot(identity) {
  const snapshot = JSON.parse(boundedAuditJson(identity, AUDIT_LIMITS.identityBytes));
  if (!plainObject(snapshot) || !plainObject(snapshot.sources) || !Object.keys(snapshot.sources).length ||
      Object.entries(snapshot.sources).some(([name, sha256]) => !validLabel(name) || !validHash(sha256)) ||
      !plainObject(snapshot.runtime) || !Object.keys(snapshot.runtime).length) throw Error('Current source/runtime audit identity required');
  return snapshot;
}

function createLocalFilesystemHoldoutAudit(directory, {readCurrentSourceRuntimeIdentity} = {}) {
  if (typeof directory !== 'string' || !directory || typeof readCurrentSourceRuntimeIdentity !== 'function') throw Error('Audit directory and current source/runtime identity reader required');
  const root = path.resolve(directory);
  const filenames = Object.fromEntries(['commitment', 'candidates', 'consumed'].map(name => [name, path.join(root, `${name}.json`)]));
  const exists = filename => {
    try { fs.lstatSync(filename); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; }
  };
  const currentIdentity = () => identitySnapshot(readCurrentSourceRuntimeIdentity());
  function readRecord(name) {
    const record = readBoundedJson(filenames[name]);
    if (!plainObject(record.value) || record.value.schemaVersion !== 1 || record.value.kind !== `local-filesystem-holdout-audit-${name}` ||
        record.value.localFilesystemAuditOnlyNotTamperProofSecrecy !== true ||
        !record.bytes.equals(Buffer.from(record.canonical + '\n'))) throw Error(`Invalid ${name} audit record`);
    return record;
  }
  function baseRecord(name) {
    return {schemaVersion: 1, kind: `local-filesystem-holdout-audit-${name}`, localFilesystemAuditOnlyNotTamperProofSecrecy: true};
  }
  function assertCurrentIdentity(commitment) {
    if (boundedAuditJsonSha256(currentIdentity()) !== commitment.identitySha256) throw Error('Holdout source/runtime identity mismatch');
  }
  function readCommitment() {
    const record = readRecord('commitment'), commitment = record.value;
    const corpus = corpusSnapshot(commitment.corpus), identity = identitySnapshot(commitment.identity);
    if (commitment.auditScope !== LOCAL_FILESYSTEM_AUDIT_SCOPE ||
        typeof commitment.commitmentId !== 'string' || !/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/.test(commitment.commitmentId) ||
        commitment.caseCount !== corpus.cases.length || commitment.casesSha256 !== boundedAuditJsonSha256(corpus.cases) ||
        commitment.corpusJsonSha256 !== boundedAuditJsonSha256(corpus) ||
        commitment.identitySha256 !== boundedAuditJsonSha256(identity)) throw Error('Holdout commitment hash/identity mismatch');
    assertCurrentIdentity(commitment);
    return record;
  }
  function readCandidateFreeze(commitment) {
    const record = readRecord('candidates'), freeze = record.value;
    if (freeze.commitmentSha256 !== commitment.sha256 || freeze.identitySha256 !== commitment.value.identitySha256 ||
        !Array.isArray(freeze.candidates) || !freeze.candidates.length || freeze.candidates.length > AUDIT_LIMITS.candidates) throw Error('Candidate freeze commitment/identity mismatch');
    const ids = new Set(), paths = new Set();
    for (const candidate of freeze.candidates) {
      const filename = typeof candidate?.path === 'string' ? candidate.path : '';
      const pathKey = process.platform === 'win32' ? filename.toLowerCase() : filename;
      if (!plainObject(candidate) || !validLabel(candidate.id) || ids.has(candidate.id) || !path.isAbsolute(filename) ||
          Buffer.byteLength(filename) > 4096 || paths.has(pathKey) || !validHash(candidate.sha256) ||
          !Number.isSafeInteger(candidate.sizeBytes) || candidate.sizeBytes < 1 || candidate.sizeBytes > AUDIT_LIMITS.jsonBytes) throw Error('Invalid frozen candidate identity');
      ids.add(candidate.id);
      paths.add(pathKey);
    }
    return record;
  }
  function summary(commitment) {
    const {corpus, identity, ...metadata} = commitment.value;
    return {...metadata, commitmentSha256: commitment.sha256};
  }
  function assertNotConsumed() {
    if (exists(filenames.consumed)) throw Error('Local holdout audit already consumed; no second reveal or evaluation');
  }
  return {
    localFilesystemAuditOnlyNotTamperProofSecrecy: true,
    auditScope: LOCAL_FILESYSTEM_AUDIT_SCOPE,
    commitFreshCorpus(corpus) {
      if (exists(filenames.candidates) || exists(filenames.consumed)) throw Error('Existing holdout audit state cannot be recommitted');
      const snapshot = corpusSnapshot(corpus), identity = currentIdentity();
      const commitment = {...baseRecord('commitment'), auditScope: LOCAL_FILESYSTEM_AUDIT_SCOPE,
        commitmentId: randomUUID(), committedAt: new Date().toISOString(), caseCount: snapshot.cases.length,
        casesSha256: boundedAuditJsonSha256(snapshot.cases), corpusJsonSha256: boundedAuditJsonSha256(snapshot),
        identitySha256: boundedAuditJsonSha256(identity), identity, corpus: snapshot};
      const sha256 = publishExclusiveJson(filenames.commitment, commitment);
      return summary({value: commitment, sha256});
    },
    freezeCandidateFilesBeforeReveal(candidateFiles) {
      assertNotConsumed();
      const commitment = readCommitment();
      if (!Array.isArray(candidateFiles) || !candidateFiles.length || candidateFiles.length > AUDIT_LIMITS.candidates) throw Error('Bounded nonempty candidate files required');
      const candidates = candidateFiles.map(candidate => {
        if (!plainObject(candidate) || !validLabel(candidate.id) || typeof candidate.path !== 'string' ||
            !path.isAbsolute(candidate.path) || Buffer.byteLength(candidate.path) > 4096) throw Error('Candidate ID and absolute JSON file path required');
        const filename = path.resolve(candidate.path), artifact = readBoundedJson(filename);
        return {id: candidate.id, path: filename, sha256: artifact.sha256, sizeBytes: artifact.bytes.length};
      });
      if (new Set(candidates.map(candidate => candidate.id)).size !== candidates.length ||
          new Set(candidates.map(candidate => process.platform === 'win32' ? candidate.path.toLowerCase() : candidate.path)).size !== candidates.length) throw Error('Unique candidate IDs and paths required');
      const freeze = {...baseRecord('candidates'), frozenAt: new Date().toISOString(),
        commitmentSha256: commitment.sha256, identitySha256: commitment.value.identitySha256, candidates};
      assertCurrentIdentity(commitment.value);
      const freezeSha256 = publishExclusiveJson(filenames.candidates, freeze);
      return {...freeze, freezeSha256};
    },
    consumeAndRevealOnce() {
      assertNotConsumed();
      if (!exists(filenames.candidates)) throw Error('Candidate files must be frozen before holdout reveal');
      const commitment = readCommitment(), freeze = readCandidateFreeze(commitment);
      const candidates = freeze.value.candidates.map(candidate => {
        const artifact = readBoundedJson(candidate.path);
        if (artifact.sha256 !== candidate.sha256 || artifact.bytes.length !== candidate.sizeBytes) throw Error('Frozen candidate hash mismatch');
        return {...candidate, json: artifact.value};
      });
      const consumed = {...baseRecord('consumed'), consumedAt: new Date().toISOString(),
        commitmentSha256: commitment.sha256, freezeSha256: freeze.sha256, identitySha256: commitment.value.identitySha256};
      assertCurrentIdentity(commitment.value);
      publishExclusiveJson(filenames.consumed, consumed);
      return {commitment: summary(commitment), candidateFreeze: {...freeze.value, freezeSha256: freeze.sha256},
        consumed, corpus: commitment.value.corpus, candidates};
    },
    readAuditStatus() {
      const commitment = readCommitment();
      const freeze = exists(filenames.candidates) ? readCandidateFreeze(commitment) : null;
      const consumed = exists(filenames.consumed) ? readRecord('consumed').value : null;
      if (consumed && (!freeze || consumed.commitmentSha256 !== commitment.sha256 || consumed.freezeSha256 !== freeze.sha256 ||
          consumed.identitySha256 !== commitment.value.identitySha256)) throw Error('Consumed marker commitment/identity mismatch');
      return {commitment: summary(commitment), candidateFreeze: freeze ? {...freeze.value, freezeSha256: freeze.sha256} : null, consumed};
    }
  };
}

module.exports = {createLocalFilesystemHoldoutAudit, boundedAuditJsonSha256, AUDIT_LIMITS, LOCAL_FILESYSTEM_AUDIT_SCOPE};
