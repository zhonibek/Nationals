'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {boundedAuditJsonSha256} = require('../roboproof/motion-holdout');
const {packExposureLedger, unpackExposureLedger, encodeExposureSnapshot, decodeExposureSnapshot,
  exposureLedgerSha256} = require('../roboproof/motion-exposure-codec');

const fixture = () => ({schemaVersion: 1, specification: 'known-local-motion-exposure-v1',
  worldSeeds: [0, 4294967295, 71, 71], bodyGeometries: ['0'.repeat(64), 'f'.repeat(64), '0'.repeat(64)],
  sourceFiles: [{path: 'synthetic/exposure.json', sha256: 'a'.repeat(64), role: 'training'}],
  observedWorldRecords: 7, seedOnlyRecords: 0, legacyGeometryFullyKnown: true});

test('lossless packed exposure preserves all values, ordering, duplicates and provenance', () => {
  const ledger = fixture(), packed = packExposureLedger(ledger);
  assert.equal(packed.schemaVersion, 2);
  assert.equal(packed.worldSeeds, undefined);
  assert.deepEqual(unpackExposureLedger(packed), ledger);
  assert.equal(decodeExposureSnapshot(packed).ledgerSha256, boundedAuditJsonSha256(packed));
  assert.equal(packed.packed.worldSeedCount, 4);
  assert.equal(packed.packed.bodyGeometryCount, 3);
});

test('small historical snapshots retain their original logical format and canonical hash', () => {
  const ledger = fixture(), encoded = encodeExposureSnapshot(ledger);
  assert.equal(encoded.payload.schemaVersion, 1);
  assert.equal(encoded.ledgerSha256, boundedAuditJsonSha256(ledger));
  assert.deepEqual(decodeExposureSnapshot(JSON.parse(encoded.bytes)), {ledger, ledgerSha256: encoded.ledgerSha256});
  assert.equal(exposureLedgerSha256(ledger), encoded.ledgerSha256);
});

test('empty sets remain exact and a changed packed seed cannot keep the original hash', () => {
  const ledger = fixture();
  ledger.worldSeeds = []; ledger.bodyGeometries = [];
  assert.deepEqual(unpackExposureLedger(packExposureLedger(ledger)), ledger);
  const first = packExposureLedger(fixture()), second = structuredClone(first);
  const bytes = Buffer.from(second.packed.worldSeeds, 'base64');
  bytes.writeUInt32BE(1, 0); second.packed.worldSeeds = bytes.toString('base64');
  assert.notEqual(decodeExposureSnapshot(first).ledgerSha256, decodeExposureSnapshot(second).ledgerSha256);
});

test('truncated, noncanonical or ambiguous binary records fail closed', () => {
  for (const mutate of [packed => packed.packed.worldSeedCount++, packed => packed.packed.worldSeedCount = true,
    packed => packed.packed.bodyGeometries = packed.packed.bodyGeometries.slice(0, -4),
    packed => packed.packed.worldSeeds += '\n', packed => packed.packed.encoding = 'lossy-v1',
    packed => packed.packed.extra = true, packed => packed.worldSeeds = [], packed => packed.packed.bodyGeometryCount = 200001]) {
    const packed = packExposureLedger(fixture()); mutate(packed);
    assert.throws(() => decodeExposureSnapshot(packed));
  }
});

test('invalid logical values, sparse arrays and accessors are never encoded', () => {
  for (const value of [-1, true, 4294967296, NaN, '71']) {
    const ledger = fixture(); ledger.worldSeeds[0] = value;
    assert.throws(() => encodeExposureSnapshot(ledger));
  }
  const sparse = fixture(); delete sparse.worldSeeds[0];
  assert.throws(() => encodeExposureSnapshot(sparse));
  const getter = fixture(); let reads = 0;
  Object.defineProperty(getter, 'worldSeeds', {get() { reads++; return []; }});
  assert.throws(() => encodeExposureSnapshot(getter), /accessors/);
  assert.equal(reads, 0);
  const element = fixture(); Object.defineProperty(element.worldSeeds, '0', {get() { reads++; return 0; }});
  assert.throws(() => encodeExposureSnapshot(element));
  assert.equal(reads, 0);
});

test('packing keeps unknown provenance rather than silently dropping over-budget data', () => {
  const ledger = fixture(); ledger.extra = {retained: 'original additional provenance'};
  assert.deepEqual(unpackExposureLedger(packExposureLedger(ledger)), ledger);
  assert.equal(decodeExposureSnapshot(packExposureLedger(ledger)).ledger.extra.retained, ledger.extra.retained);
});
