'use strict';

const {boundedAuditJsonSha256} = require('./motion-holdout');

const MAX_BYTES = 4 * 1024 * 1024;
const MAX_WORLDS = 200000;
const LOGICAL_SPECIFICATION = 'known-local-motion-exposure-v1';
const PACKED_SPECIFICATION = 'known-local-motion-exposure-packed-v1';
const ENCODING = 'uint32be-sha256-base64-v1';
const plain = value => value !== null && typeof value === 'object' &&
  [Object.prototype, null].includes(Object.getPrototypeOf(value));

function properties(value) {
  if (!plain(value) || Object.getOwnPropertySymbols(value).length) throw Error('Plain exposure metadata required');
  const descriptors = Object.getOwnPropertyDescriptors(value);
  if (Object.values(descriptors).some(entry => !Object.hasOwn(entry, 'value'))) throw Error('Exposure metadata accessors are forbidden');
  return descriptors;
}

function arrayValues(value, valid) {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype || value.length > MAX_WORLDS ||
      Object.getOwnPropertySymbols(value).length || Object.keys(value).length !== value.length) throw Error('Bounded dense exposure arrays required');
  const result = [];
  for (let index = 0; index < value.length; index++) {
    const entry = Object.getOwnPropertyDescriptor(value, String(index));
    if (!entry || !Object.hasOwn(entry, 'value') || !valid(entry.value)) throw Error('Valid dense exposure values required');
    result.push(entry.value);
  }
  return result;
}

function logicalValues(ledger) {
  const entries = properties(ledger);
  if (entries.schemaVersion?.value !== 1 || entries.specification?.value !== LOGICAL_SPECIFICATION || Object.hasOwn(entries, 'packed')) {
    throw Error('Versioned logical exposure ledger required');
  }
  return {seeds: arrayValues(entries.worldSeeds?.value, value => Number.isInteger(value) && value >= 0 && value <= 0xffffffff),
    geometries: arrayValues(entries.bodyGeometries?.value, value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value))};
}

function packExposureLedger(ledger) {
  const {seeds, geometries} = logicalValues(ledger);
  const seedBytes = Buffer.alloc(seeds.length * 4);
  seeds.forEach((seed, index) => seedBytes.writeUInt32BE(seed, index * 4));
  const geometryBytes = Buffer.alloc(geometries.length * 32);
  geometries.forEach((fingerprint, index) => Buffer.from(fingerprint, 'hex').copy(geometryBytes, index * 32));
  const {worldSeeds, bodyGeometries, ...metadata} = ledger;
  return {...metadata, schemaVersion: 2, specification: PACKED_SPECIFICATION,
    packed: {encoding: ENCODING, worldSeedCount: seeds.length, bodyGeometryCount: geometries.length,
      worldSeeds: seedBytes.toString('base64'), bodyGeometries: geometryBytes.toString('base64')}};
}

function decodedBytes(encoded, count, width) {
  if (!Number.isInteger(count) || count < 0 || count > MAX_WORLDS || typeof encoded !== 'string' ||
      encoded.length !== 4 * Math.ceil(count * width / 3) || encoded.length > MAX_BYTES) throw Error('Bounded exact packed exposure length required');
  const bytes = Buffer.from(encoded, 'base64');
  if (bytes.length !== count * width || bytes.toString('base64') !== encoded) throw Error('Canonical complete packed exposure bytes required');
  return bytes;
}

function unpackExposureLedger(payload) {
  const entries = properties(payload);
  if (entries.schemaVersion?.value !== 2 || entries.specification?.value !== PACKED_SPECIFICATION ||
      Object.hasOwn(entries, 'worldSeeds') || Object.hasOwn(entries, 'bodyGeometries')) throw Error('Unambiguous packed exposure ledger required');
  const packed = entries.packed?.value, fields = properties(packed);
  if (Object.keys(fields).sort().join(',') !== 'bodyGeometries,bodyGeometryCount,encoding,worldSeedCount,worldSeeds' ||
      fields.encoding.value !== ENCODING) throw Error('Exact packed exposure encoding required');
  const seeds = decodedBytes(fields.worldSeeds.value, fields.worldSeedCount.value, 4);
  const geometries = decodedBytes(fields.bodyGeometries.value, fields.bodyGeometryCount.value, 32);
  const {packed: ignored, ...metadata} = payload;
  const ledger = {...metadata, schemaVersion: 1, specification: LOGICAL_SPECIFICATION,
    worldSeeds: Array.from({length: seeds.length / 4}, (_, index) => seeds.readUInt32BE(index * 4)),
    bodyGeometries: Array.from({length: geometries.length / 32}, (_, index) => geometries.subarray(index * 32, (index + 1) * 32).toString('hex'))};
  logicalValues(ledger);
  return ledger;
}

function boundedSnapshot(payload) {
  const ledgerSha256 = boundedAuditJsonSha256(payload);
  const bytes = JSON.stringify(payload) + '\n';
  if (Buffer.byteLength(bytes) > MAX_BYTES) throw Error('Exposure snapshot exceeds byte budget');
  return {payload, bytes, ledgerSha256};
}

function encodeExposureSnapshot(ledger) {
  logicalValues(ledger);
  try { return boundedSnapshot(ledger); }
  catch (error) {
    if (!/^Audit JSON exceeds (byte|structural) limit$|^Audit evidence must be bounded dense JSON$|^Exposure snapshot exceeds byte budget$/.test(error.message)) throw error;
  }
  return boundedSnapshot(packExposureLedger(ledger));
}

function decodeExposureSnapshot(payload) {
  const entry = boundedSnapshot(payload);
  const ledger = payload?.specification === PACKED_SPECIFICATION ? unpackExposureLedger(payload) : payload;
  logicalValues(ledger);
  return {ledger, ledgerSha256: entry.ledgerSha256};
}

const exposureLedgerSha256 = ledger => encodeExposureSnapshot(ledger).ledgerSha256;

module.exports = {packExposureLedger, unpackExposureLedger, encodeExposureSnapshot, decodeExposureSnapshot, exposureLedgerSha256};
