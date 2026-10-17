'use strict';

const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const BASE = 'roboproof/runs/motion/recovery/colab-expiry-20261016';
const DESCRIPTOR = Object.freeze({schemaVersion: 1, specification: 'audited-td3-prefix-exclusion-v1',
  envelopeDirectory: `${BASE}/envelope`, sourceArchive: `${BASE}/source.zip`, auditorSource: `${BASE}/auditor.py`,
  historicalInspectionFile: `${BASE}/retained-history-inspection.json`,
  auditFile: 'roboproof/runs/motion/recovery/continuation-audit-20261017/exposure-audit.json',
  auditSha256: '4513a338848ad79cbef65744053833ad69abcc25793ed7ed287edea95189413c',
  envelopeManifestSha256: '9e600ecea59aa2b9e42e0cf3ad9d0e65f038bb35ff52d9e219b016fc9e33b3a6',
  sourceArchiveSha256: 'c437521e2b95934baf6b2d8cf36f6089685428904844ef1f123e59b4e67949f4',
  auditorSha256: 'a86608be23a3f999e50bbc0f14321b6030c35b28f7ded7ed0dc2cc5b3804159b',
  historicalInspectionSha256: '40427436d6912e09ae8c81d25a87e2074f2eaa8daed46ba5b3df17c39edda603'});
const TRAINING_SEEDS = [136001, 136002, 136003];
const RUNS = ['95663b0d-135b-4bc0-b36d-57d1699296f7', 'cb350f34-c592-4703-a2b9-73ec08a0b8c1'];
const SPECIFICATION = 'td3-full-corpus-prefix-envelope-v1';
const fail = detail => { throw Error(`Retained training exposure is unavailable; verified conservative exclusion required: ${detail}`); };

function knownEvent(marker) {
  return marker?.schemaVersion === 1 && marker.specification === 'unavailable-motion-exposure-v1' && marker.state === 'unresolved' &&
    JSON.stringify(marker.trainingSeeds) === JSON.stringify(TRAINING_SEEDS) && JSON.stringify(marker.observedRunIds) === JSON.stringify(RUNS) &&
    marker.lastVerifiedObservation === '2026-10-16T10:17:44.382861+00:00' && marker.lossObservedAt === '2026-10-16T16:35:58.364892+00:00' &&
    (marker.study === undefined || marker.study === 'episode-balanced-remote-study-20261016');
}

function pathState(root, relative) {
  if (typeof relative !== 'string' || path.posix.isAbsolute(relative) || relative.includes('\\') || relative.split('/').some(part => !part || part === '.' || part === '..')) fail('canonical owned evidence paths required');
  const names = [root];
  for (const part of relative.split('/')) names.push(path.join(names.at(-1), part));
  return names.map((filename, index) => {
    const state = fs.lstatSync(filename);
    if (state.isSymbolicLink() || (index < names.length - 1 ? !state.isDirectory() : !state.isFile())) fail('regular evidence and non-symlink ancestors required');
    return [filename, state.dev, state.ino, state.mode, state.size, state.mtimeMs, state.ctimeMs];
  });
}

function evidenceReader(context) {
  const stamps = [];
  const read = (relative, digest, role, parseJson = true, bytes) => {
    const before = pathState(context.root, relative);
    const record = context.read(path.join(context.root, ...relative.split('/')), role, parseJson);
    assert.deepEqual(pathState(context.root, relative), before, 'Recovery evidence path changed during read');
    if (record.sha256 !== digest || (bytes !== undefined && before.at(-1)[4] !== bytes)) fail(`changed evidence: ${relative}`);
    stamps.push({relative, state: before});
    return record.value;
  };
  read.verifyStable = () => {
    for (const stamp of stamps) assert.deepEqual(pathState(context.root, stamp.relative), stamp.state, 'Recovery evidence changed before admission');
  };
  return read;
}

function validateEnvelopeCatalog(manifest, audit) {
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.specification, SPECIFICATION);
  assert.deepEqual(manifest.trainingSeeds, TRAINING_SEEDS);
  assert.equal(manifest.totalPotentialWorlds, 43008);
  assert.equal(manifest.budget.nativeCollections, 56);
  assert.equal(manifest.budget.maximumGeneratedWorlds, 14336);
  assert.equal(manifest.budget.nativeCurriculumOffset, 128);
  for (const flag of ['actualQueriesInferred', 'modelOrResultRecoveryClaimed']) assert.equal(manifest[flag], false);
  assert.equal(manifest.newPhysicsQueries, 0);
  assert.equal(audit.schemaVersion, 1);
  assert.equal(audit.specification, 'independent-td3-prefix-envelope-audit-v1');
  assert.equal(audit.verified, true);
  assert.deepEqual(audit.trainingSeeds, TRAINING_SEEDS);
  assert.equal(audit.verifiedPotentialWorlds, 43008);
  assert.equal(audit.verifiedOfflineWorlds, 128);
  assert.equal(audit.verifiedExclusionWorldRecords, 43136);
  assert.equal(audit.counters.worldRowsCompared, 43909);
  assert.equal(audit.lossStateRemains, 'unresolved');
  for (const flag of ['actualQueriesInferred', 'modelOrResultRecoveryClaimed', 'qualificationClaimed', 'lossRegistryModified']) assert.equal(audit[flag], false);
  assert.equal(audit.newPhysicsQueries, 0);
  assert.equal(audit.auditor.sha256, DESCRIPTOR.auditorSha256);
  assert.equal(audit.sourcePackage.sha256, DESCRIPTOR.sourceArchiveSha256);
  assert.equal(audit.envelopeManifest.sha256, DESCRIPTOR.envelopeManifestSha256);
  assert.deepEqual(audit.runtime.platform, 'Linux');
  assert.deepEqual(audit.runtime.python, '3.12.3');
  assert.equal(manifest.shards.length, 45);
  const expected = ['envelope-manifest.json', 'offline-corpus/exposure.json'];
  let index = 0;
  for (const trainingSeed of TRAINING_SEEDS) for (let part = 0; part < 15; part++) {
    const shard = manifest.shards[index++];
    assert.equal(shard.path, `seed-${trainingSeed}-part-${String(part).padStart(2, '0')}/exposure.json`);
    assert.equal(shard.trainingSeed, trainingSeed);
    assert.equal(shard.firstOrdinal, part * 1000);
    assert.equal(shard.count, part === 14 ? 336 : 1000);
    assert.equal(shard.sha256, audit.envelopeFiles[shard.path]?.sha256);
    assert.equal(shard.bytes, audit.envelopeFiles[shard.path]?.bytes);
    expected.push(shard.path);
  }
  assert.equal(manifest.offlineCorpus.path, 'offline-corpus/exposure.json');
  assert.equal(manifest.offlineCorpus.count, 128);
  assert.equal(manifest.offlineCorpus.sha256, audit.envelopeFiles[manifest.offlineCorpus.path]?.sha256);
  assert.equal(manifest.offlineCorpus.bytes, audit.envelopeFiles[manifest.offlineCorpus.path]?.bytes);
  assert.deepEqual(Object.keys(audit.envelopeFiles).sort(), expected.sort());
  return expected;
}

function envelopeInventory(root, directory) {
  const files = [];
  let entries = 0;
  const visit = (relative, depth) => {
    if (depth > 2) fail('bounded envelope depth required');
    const filename = path.join(root, ...relative.split('/'));
    if (!fs.lstatSync(filename).isDirectory() || fs.lstatSync(filename).isSymbolicLink()) fail('regular envelope directories required');
    for (const entry of fs.readdirSync(filename, {withFileTypes: true})) {
      if (++entries > 128 || entry.isSymbolicLink()) fail('bounded non-symlink envelope inventory required');
      const child = `${relative}/${entry.name}`;
      if (entry.isDirectory()) visit(child, depth + 1);
      else if (entry.isFile()) files.push(child.slice(directory.length + 1));
      else fail('regular envelope members required');
    }
  };
  visit(directory, 0);
  return files.sort();
}

function admitAuditedRecovery(context) {
  const {markers, registry, ledger} = context;
  if (!Array.isArray(markers) || !markers.length || !markers.every(knownEvent) || registry?.losses?.length !== 1 ||
      !knownEvent(registry.losses[0]) || registry.losses[0].study !== 'episode-balanced-remote-study-20261016') fail('unsupported or incomplete unavailable event');
  try { assert.deepEqual(registry.losses[0].recovery, DESCRIPTOR); }
  catch { fail('versioned exact recovery descriptor required'); }
  const read = evidenceReader(context);
  const historical = read(DESCRIPTOR.historicalInspectionFile, DESCRIPTOR.historicalInspectionSha256, 'retained-history-inspection', true, 1592136);
  assert.equal(historical.specification, 'incomplete-retained-motion-history-inspection-v1');
  assert.equal(historical.completeForIndependentFinal, false);
  assert.equal(historical.sourceFiles.length, 818);
  assert.equal(historical.worldSeeds.length, 19477);
  assert.equal(historical.bodyGeometries.length, 17920);
  assert.equal(historical.seedOnlyRecords, 830);
  assert.equal(historical.unavailableLossMarkers.length, 2);
  assert(historical.unavailableLossMarkers.every(knownEvent));
  const currentSources = new Map(ledger.sourceFiles.map(row => [row.path, row]));
  for (const retained of historical.sourceFiles) {
    if (retained.role === 'unavailable-training-exposure-registry') continue;
    if (currentSources.get(retained.path)?.sha256 !== retained.sha256 || currentSources.get(retained.path)?.role !== retained.role) fail(`missing historical exclusion source: ${retained.path}`);
  }
  const seeds = new Set(ledger.worldSeeds), geometries = new Set(ledger.bodyGeometries);
  if (!historical.worldSeeds.every(seed => seeds.has(seed)) || !historical.bodyGeometries.every(geometry => geometries.has(geometry)) || ledger.seedOnlyRecords < historical.seedOnlyRecords) fail('complete retained historical exclusions required');
  read(DESCRIPTOR.sourceArchive, DESCRIPTOR.sourceArchiveSha256, 'recovery-original-source-archive', false, 205248);
  read(DESCRIPTOR.auditorSource, DESCRIPTOR.auditorSha256, 'recovery-independent-auditor-source', false, 37746);
  const audit = read(DESCRIPTOR.auditFile, DESCRIPTOR.auditSha256, 'recovery-independent-audit', true, 20368);
  const manifestPath = `${DESCRIPTOR.envelopeDirectory}/envelope-manifest.json`;
  const manifest = read(manifestPath, DESCRIPTOR.envelopeManifestSha256, 'recovery-envelope-manifest', true, 21219);
  const inventory = validateEnvelopeCatalog(manifest, audit);
  assert.deepEqual(envelopeInventory(context.root, DESCRIPTOR.envelopeDirectory), inventory);
  for (const shard of manifest.shards) {
    const value = read(`${DESCRIPTOR.envelopeDirectory}/${shard.path}`, shard.sha256, 'conservative-unavailable-training-exclusion', true, shard.bytes);
    assert.equal(value.schemaVersion, 1);
    assert.equal(value.specification, SPECIFICATION);
    assert.equal(value.trainingSeed, shard.trainingSeed);
    assert.equal(value.conservativeSupersetNotMeasuredQueries, true);
    assert.equal(value.cases.length, shard.count);
    for (const [offset, world] of value.cases.entries()) {
      assert.equal(world.ordinal, shard.firstOrdinal + offset);
      assert.equal(world.trainingSeed, shard.trainingSeed);
      assert.equal(world.id, `possible-${shard.trainingSeed}-${world.ordinal}`);
      assert.equal(world.kind, 'conservative-potential-training');
      context.add(world, 'conservative-unavailable-training-exclusion');
    }
  }
  const offline = manifest.offlineCorpus;
  const value = read(`${DESCRIPTOR.envelopeDirectory}/${offline.path}`, offline.sha256, 'retained-offline-training-exclusion', true, offline.bytes);
  assert.equal(value.schemaVersion, 1);
  assert.equal(value.specification, SPECIFICATION);
  assert.equal(value.declaredOriginalCorpusExposure, true);
  assert.equal(value.cases.length, 128);
  for (const world of value.cases) context.add(world, 'retained-offline-training-exclusion');
  for (const seed of TRAINING_SEEDS) context.add({seed}, 'reserved-unavailable-training-seed');
  assert.deepEqual(envelopeInventory(context.root, DESCRIPTOR.envelopeDirectory), inventory);
  read.verifyStable();
  return {...context.finish(), scope: 'Known saved history plus the exact independently audited conservative possible-training exclusion. This is not recovered model/outcome evidence, an actual query count, external custody or arbitrary OOD separation.',
    unavailableExposureExclusions: [{specification: DESCRIPTOR.specification,
    trainingSeeds: [...TRAINING_SEEDS], auditSha256: DESCRIPTOR.auditSha256, envelopeManifestSha256: DESCRIPTOR.envelopeManifestSha256,
    historicalInspectionSha256: DESCRIPTOR.historicalInspectionSha256, conservativeWorldRecords: 43136, reservedSeedRecords: 3,
    eventState: 'unresolved', modelOrOutcomeRecoveryClaimed: false, independentQualificationClaimed: false}]};
}

module.exports = {RECOVERY_DESCRIPTOR: DESCRIPTOR, validateEnvelopeCatalog, admitAuditedRecovery};
