'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {packExposureLedger, unpackExposureLedger} = require('./motion-exposure-codec');
const {boundedAuditJsonSha256} = require('./motion-holdout');
const {RECOVERY_DESCRIPTOR} = require('./motion-exposure-recovery');

const SPECIFICATION = 'audited-potential-exclusion-capsule-v1';
const COUNT = 43136;
const hash = value => createHash('sha256').update(value).digest('hex');
const ORIGINAL_MAPPING = Object.freeze({specification: 'original-body-relative-0.1inch-1degree-v1', fixedDt: 0.01,
  geometryFunctionSha256: '0f4f734ecb77fbe4ff27d8f25ead0ea45a0d7d30eb7d2fea30be3b62e6b97d56',
  normalizationHelpersSha256: '788906db3e85a50a48dddec1bf2c62e5395232cbcf6843fffdf99a759f224cf1'});
const PARENTS = Object.freeze({envelopeArchiveSha256: 'd15aff07bb92d55e09f9da0924e182249fcd3fe43528da3fe869a81e45b76168',
  envelopeArchiveBytes: 7583049, envelopeManifestSha256: RECOVERY_DESCRIPTOR.envelopeManifestSha256,
  independentAuditSha256: RECOVERY_DESCRIPTOR.auditSha256, sourcePackageSha256: RECOVERY_DESCRIPTOR.sourceArchiveSha256,
  fullAdmissionProofSha256: 'f1ce3e978c9624be35c8a24871859fb4e83062fac59065c9d36577359771715a',
  fullAdmissionLedgerSha256: '6e27a99f17140c20c8ef91175c52b8760c1445f434f637fac5608724a9d3f2ab'});
const SCOPE = 'Lossless ordered exclusion-key projection only. Original full task/configuration bytes remain in the pinned archive. Not measured queries, recovered models/outcomes, complete historical geometry, an independent final or learned improvement.';

function encodeExclusionPairs(worldSeeds, bodyGeometries) {
  const encoded = packExposureLedger({schemaVersion: 1, specification: 'known-local-motion-exposure-v1', worldSeeds, bodyGeometries}).packed;
  if (encoded.worldSeedCount !== encoded.bodyGeometryCount) throw Error('One geometry hash for every ordered exclusion seed required');
  return encoded;
}

function decodeExclusionPairs(vectors) {
  const decoded = unpackExposureLedger({schemaVersion: 2, specification: 'known-local-motion-exposure-packed-v1', packed: vectors});
  if (decoded.worldSeeds.length !== decoded.bodyGeometries.length) throw Error('Complete paired ordered exclusion vectors required');
  return {worldSeeds: decoded.worldSeeds, bodyGeometries: decoded.bodyGeometries};
}

function mappingIdentity() {
  const motion = require('../simulator/motion');
  const {geometryFingerprint} = require('./motion-exposures');
  const source = fs.readFileSync(path.join(__dirname, '../simulator/motion.js'), 'utf8').replace(/\r\n/g, '\n');
  const first = source.indexOf('  function keys('), last = source.indexOf('  function configuration(');
  assert(first >= 0 && last > first, 'Explicit original normalization helper region required');
  const normalization = source.slice(first, last);
  assert(normalization.includes(motion.normalizeTask.toString().replace(/\r\n/g, '\n')));
  assert.equal(motion.FIXED_DT, 0.01);
  const actual = {specification: 'original-body-relative-0.1inch-1degree-v1', fixedDt: motion.FIXED_DT,
    geometryFunctionSha256: hash(geometryFingerprint.toString().replace(/\r\n/g, '\n')),
    normalizationHelpersSha256: hash(normalization)};
  assert.deepEqual(actual, ORIGINAL_MAPPING, 'Immutable original geometry and normalization mapping required');
  return actual;
}

function sourceList(manifest) {
  return [...manifest.shards.map(shard => ({...shard, role: 'potential'})), {...manifest.offlineCorpus, role: 'offline'}];
}

function capsuleHeader(manifest) {
  return {schemaVersion: 1, specification: SPECIFICATION, recordCount: COUNT, parents: {...PARENTS}, mapping: mappingIdentity(),
    trainingSeeds: [136001, 136002, 136003], sources: sourceList(manifest), potentialRecords: 43008, offlineRecords: 128,
    actualQueriesInferred: false, modelOrOutcomeRecoveryClaimed: false, qualificationClaimed: false, newPhysicsQueries: 0,
    eventState: 'unresolved', scope: SCOPE};
}

function validateCapsuleHeader(capsule, manifest) {
  boundedAuditJsonSha256(capsule);
  const {vectors, ...header} = capsule;
  assert.deepEqual(header, capsuleHeader(manifest), 'Exact immutable parents, source order, mapping and metadata-only scope required');
  assert.equal(vectors?.worldSeedCount, COUNT);
  assert.equal(vectors?.bodyGeometryCount, COUNT);
}

function createCapsule({manifestBytes, worldSeeds, bodyGeometries}) {
  if (process.platform !== 'linux' || process.env.ROBOPROOF_APPROVED_REMOTE !== '1') throw Error('Generate the complete capsule only on an approved remote Linux host');
  assert(Buffer.isBuffer(manifestBytes) && manifestBytes.length === 21219);
  assert.equal(hash(manifestBytes), PARENTS.envelopeManifestSha256);
  const manifest = JSON.parse(manifestBytes);
  const capsule = {...capsuleHeader(manifest), vectors: encodeExclusionPairs(worldSeeds, bodyGeometries)};
  validateCapsuleHeader(capsule, manifest);
  const bytes = JSON.stringify(capsule) + '\n';
  assert(Buffer.byteLength(bytes) <= 4 * 1024 * 1024);
  return {capsule, bytes, sha256: hash(bytes)};
}

function decodeCapsule(capsule, manifest) {
  validateCapsuleHeader(capsule, manifest);
  const pairs = decodeExclusionPairs(capsule.vectors);
  assert.equal(pairs.worldSeeds.length, COUNT);
  return {specification: SPECIFICATION, ...pairs, potentialRecords: 43008, offlineRecords: 128};
}

module.exports = {encodeExclusionPairs, decodeExclusionPairs, mappingIdentity, capsuleHeader, validateCapsuleHeader, createCapsule, decodeCapsule};
