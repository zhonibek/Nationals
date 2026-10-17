'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {encodeExclusionPairs, decodeExclusionPairs, mappingIdentity, capsuleHeader, validateCapsuleHeader, createCapsule} = require('../roboproof/motion-exposure-capsule');
const {manifest} = require('./fixtures/motion-exposure-recovery-catalog.json');

test('ordered exclusion encoding preserves duplicate seeds, duplicate geometry and pair order', () => {
  const seeds = [7, 2, 7, 0xffffffff], geometries = ['b'.repeat(64), 'a'.repeat(64), 'b'.repeat(64), 'c'.repeat(64)];
  assert.deepEqual(decodeExclusionPairs(encodeExclusionPairs(seeds, geometries)), {worldSeeds: seeds, bodyGeometries: geometries});
  assert.deepEqual(decodeExclusionPairs(encodeExclusionPairs([], [])), {worldSeeds: [], bodyGeometries: []});
});

test('paired encoding rejects sparse, unpaired, non-uint32 and malformed hash data', () => {
  for (const [seeds, geometries] of [[[1], []], [[-1], ['a'.repeat(64)]], [[true], ['a'.repeat(64)]],
    [new Array(1), ['a'.repeat(64)]], [[1], ['not-a-hash']], [[1], new Array(1)]]) assert.throws(() => encodeExclusionPairs(seeds, geometries));
  const getter = [1];
  Object.defineProperty(getter, '0', {get() { throw Error('Accessor must not execute'); }});
  assert.throws(() => encodeExclusionPairs(getter, ['a'.repeat(64)]), /dense/);
});

test('packed exclusion bytes cannot be truncated, rebound, padded or mispaired', () => {
  const vectors = encodeExclusionPairs([4, 5], ['a'.repeat(64), 'b'.repeat(64)]);
  for (const patch of [{worldSeeds: vectors.worldSeeds.slice(0, -4)}, {worldSeeds: vectors.worldSeeds + '='},
    {worldSeedCount: 1}, {bodyGeometryCount: 3}, {encoding: 'loose'}, {ignored: true}]) assert.throws(() => decodeExclusionPairs({...vectors, ...patch}));
});

function headerFixture() {
  return {...capsuleHeader(manifest), vectors: {worldSeedCount: 43136, bodyGeometryCount: 43136}};
}

test('capsule metadata pins original audit, full proof, raw archive, every ordered source and mapper', () => {
  const capsule = headerFixture();
  validateCapsuleHeader(capsule, manifest);
  assert.equal(capsule.sources.length, 46);
  assert.equal(capsule.parents.envelopeArchiveBytes, 7583049);
  assert.equal(capsule.eventState, 'unresolved');
  assert.equal(capsule.actualQueriesInferred, false);
  assert.deepEqual(capsule.mapping, mappingIdentity());
  const mutations = [
    value => { value.parents.independentAuditSha256 = '0'.repeat(64); },
    value => { value.parents.envelopeArchiveSha256 = '0'.repeat(64); },
    value => { value.mapping.geometryFunctionSha256 = '0'.repeat(64); },
    value => { value.mapping.normalizationHelpersSha256 = '0'.repeat(64); },
    value => { value.sources.reverse(); }, value => { value.sources.pop(); },
    value => { value.recordCount--; }, value => { value.eventState = 'resolved'; },
    value => { value.qualificationClaimed = true; }, value => { value.modelOrOutcomeRecoveryClaimed = true; },
    value => { value.extra = 'ignore'; }, value => { value.vectors.bodyGeometryCount--; }
  ];
  for (const mutate of mutations) {
    const value = headerFixture();
    mutate(value);
    assert.throws(() => validateCapsuleHeader(value, manifest));
  }
});

test('complete capsule creation is remote-only and is not a route around final admission', () => {
  const previous = process.env.ROBOPROOF_APPROVED_REMOTE;
  delete process.env.ROBOPROOF_APPROVED_REMOTE;
  try { assert.throws(() => createCapsule({}), /approved remote Linux/); }
  finally { if (previous !== undefined) process.env.ROBOPROOF_APPROVED_REMOTE = previous; }
  assert.equal(headerFixture().specification, 'audited-potential-exclusion-capsule-v1');
  const {assertFreshCases} = require('../roboproof/motion-exposures');
  assert.throws(() => assertFreshCases([{seed: 1}], headerFixture()), /valid exposure ledger/);
});

test('current source cannot self-certify changed original normalization', () => {
  const fs = require('node:fs');
  const read = fs.readFileSync;
  fs.readFileSync = function(filename, ...options) {
    const content = read.call(this, filename, ...options);
    return String(filename).replace(/\\/g, '/').endsWith('/simulator/motion.js') && typeof content === 'string' ?
      content.replace('Unsupported ${label}.${key}', 'Changed ${label}.${key}') : content;
  };
  try { assert.throws(mappingIdentity, /Immutable original/); }
  finally { fs.readFileSync = read; }
});
