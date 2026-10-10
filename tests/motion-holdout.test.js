'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {createLocalFilesystemHoldoutAudit, boundedAuditJsonSha256, AUDIT_LIMITS, LOCAL_FILESYSTEM_AUDIT_SCOPE} = require('../roboproof/motion-holdout');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

function fixture(context) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'motion-holdout-audit-'));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const auditDirectory = path.join(directory, 'holdout');
  const sourcePath = path.join(directory, 'source.json'), candidatePath = path.join(directory, 'policy.json');
  fs.writeFileSync(sourcePath, '{"contract":1}');
  fs.writeFileSync(candidatePath, '{"weights":[0.1,0.2]}\n');
  const runtime = {node: process.version, v8: process.versions.v8, platform: process.platform, arch: process.arch};
  const create = () => createLocalFilesystemHoldoutAudit(auditDirectory, {
    readCurrentSourceRuntimeIdentity: () => ({sources: {'source.json': hash(fs.readFileSync(sourcePath))}, runtime})
  });
  const corpus = {specification: 'new-independent-corpus', cases: [{id: 'fresh-1', seed: 123, task: {goal: {x: 1, y: 2}}}]};
  const candidateFiles = [{id: 'candidate-1', path: candidatePath}];
  return {directory, auditDirectory, sourcePath, candidatePath, runtime, create, corpus, candidateFiles};
}

test('local audit API explicitly disclaims secrecy, gates reveal on freeze and consumes exactly once', context => {
  const setup = fixture(context), audit = setup.create();
  assert.equal(audit.localFilesystemAuditOnlyNotTamperProofSecrecy, true);
  assert.match(LOCAL_FILESYSTEM_AUDIT_SCOPE, /not tamper-proof secrecy.*plaintext.*caller responsibilities/);
  const commitment = audit.commitFreshCorpus(setup.corpus);
  assert.equal(commitment.caseCount, 1);
  assert.equal(commitment.casesSha256, boundedAuditJsonSha256(setup.corpus.cases));
  assert.equal(commitment.corpusJsonSha256, boundedAuditJsonSha256(setup.corpus));
  assert.equal(commitment.commitmentSha256, hash(fs.readFileSync(path.join(setup.auditDirectory, 'commitment.json'))));
  assert.equal(Object.hasOwn(commitment, 'corpus'), false);
  assert.equal(Object.hasOwn(audit.readAuditStatus().commitment, 'corpus'), false);
  assert.throws(() => audit.consumeAndRevealOnce(), /must be frozen before/);
  assert.equal(fs.existsSync(path.join(setup.auditDirectory, 'consumed.json')), false);
  const freeze = audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  assert.equal(freeze.candidates[0].sha256, hash(fs.readFileSync(setup.candidatePath)));
  assert.equal(Object.hasOwn(freeze.candidates[0], 'json'), false);
  const revealed = setup.create().consumeAndRevealOnce();
  assert.deepEqual(revealed.corpus, setup.corpus);
  assert.deepEqual(revealed.candidates[0].json, {weights: [0.1, 0.2]});
  assert.equal(revealed.consumed.freezeSha256, freeze.freezeSha256);
  assert.equal(revealed.consumed.commitmentSha256, commitment.commitmentSha256);
  assert.deepEqual(setup.create().readAuditStatus().consumed, revealed.consumed);
  assert.throws(() => setup.create().consumeAndRevealOnce(), /already consumed/);
  assert.throws(() => audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles), /already consumed/);
  assert.throws(() => audit.commitFreshCorpus(setup.corpus), /cannot be recommitted/);
  assert.deepEqual(fs.readdirSync(setup.auditDirectory).sort(), ['candidates.json', 'commitment.json', 'consumed.json']);
});

test('commitment and candidate freeze cannot overwrite an existing record', context => {
  const setup = fixture(context), audit = setup.create();
  audit.commitFreshCorpus(setup.corpus);
  const commitmentPath = path.join(setup.auditDirectory, 'commitment.json'), committed = fs.readFileSync(commitmentPath);
  assert.throws(() => setup.create().commitFreshCorpus({...setup.corpus, specification: 'replacement'}), /EEXIST/);
  assert.deepEqual(fs.readFileSync(commitmentPath), committed);
  audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  const freezePath = path.join(setup.auditDirectory, 'candidates.json'), frozen = fs.readFileSync(freezePath);
  assert.throws(() => setup.create().freezeCandidateFilesBeforeReveal(setup.candidateFiles), /EEXIST/);
  assert.deepEqual(fs.readFileSync(freezePath), frozen);
  assert.equal(fs.readdirSync(setup.auditDirectory).some(name => name.endsWith('.tmp')), false);
});

test('candidate changes after freeze fail closed without exposing or consuming cases', context => {
  const setup = fixture(context), audit = setup.create();
  audit.commitFreshCorpus(setup.corpus);
  audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  const original = fs.readFileSync(setup.candidatePath);
  fs.writeFileSync(setup.candidatePath, '{"weights":[0.2,0.1]}\n');
  assert.throws(() => audit.consumeAndRevealOnce(), /Frozen candidate hash mismatch/);
  assert.equal(fs.existsSync(path.join(setup.auditDirectory, 'consumed.json')), false);
  fs.writeFileSync(setup.candidatePath, original);
  assert.deepEqual(audit.consumeAndRevealOnce().corpus, setup.corpus);
});

test('current source and runtime identities are checked before freeze, reveal and status', context => {
  const setup = fixture(context), audit = setup.create();
  audit.commitFreshCorpus(setup.corpus);
  const original = fs.readFileSync(setup.sourcePath);
  fs.writeFileSync(setup.sourcePath, '{"contract":2}');
  assert.throws(() => audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles), /source\/runtime identity mismatch/);
  assert.throws(() => audit.readAuditStatus(), /source\/runtime identity mismatch/);
  fs.writeFileSync(setup.sourcePath, original);
  audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  setup.runtime.node = 'different-runtime';
  assert.throws(() => setup.create().consumeAndRevealOnce(), /source\/runtime identity mismatch/);
  assert.equal(fs.existsSync(path.join(setup.auditDirectory, 'consumed.json')), false);
  setup.runtime.node = process.version;
  fs.writeFileSync(setup.sourcePath, '{"contract":3}');
  assert.throws(() => setup.create().consumeAndRevealOnce(), /source\/runtime identity mismatch/);
});

test('source identity drift during candidate reads blocks both freeze and consumption', context => {
  const setup = fixture(context), audit = setup.create(), open = fs.openSync;
  audit.commitFreshCorpus(setup.corpus);
  const original = fs.readFileSync(setup.sourcePath);
  let changeOnRead = true;
  context.mock.method(fs, 'openSync', (filename, ...argumentsRest) => {
    if (filename === setup.candidatePath && changeOnRead) {
      changeOnRead = false;
      fs.writeFileSync(setup.sourcePath, '{"contract":2}');
    }
    return open(filename, ...argumentsRest);
  });
  assert.throws(() => audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles), /source\/runtime identity mismatch/);
  assert.equal(fs.existsSync(path.join(setup.auditDirectory, 'candidates.json')), false);
  fs.writeFileSync(setup.sourcePath, original);
  audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  changeOnRead = true;
  assert.throws(() => audit.consumeAndRevealOnce(), /source\/runtime identity mismatch/);
  assert.equal(fs.existsSync(path.join(setup.auditDirectory, 'consumed.json')), false);
});

test('JSON hash identity is deterministic, bounded and does not silently coerce non-JSON evidence', () => {
  assert.equal(boundedAuditJsonSha256({second: {value: 2}, first: 1}), boundedAuditJsonSha256({first: 1, second: {value: 2}}));
  assert.notEqual(boundedAuditJsonSha256([1, 2]), boundedAuditJsonSha256([2, 1]));
  const cyclic = {}; cyclic.value = cyclic;
  let deep = null;
  for (let index = 0; index <= AUDIT_LIMITS.depth; index++) deep = {value: deep};
  const accessor = Object.defineProperty({}, 'value', {enumerable: true, get() { throw Error('Must not invoke getter'); }});
  for (const invalid of [undefined, NaN, Infinity, 1n, () => 1, new Date(), {value: undefined}, Array(2), {[Symbol('value')]: 1}]) {
    assert.throws(() => boundedAuditJsonSha256(invalid), /plain JSON|bounded dense JSON/);
  }
  assert.throws(() => boundedAuditJsonSha256(cyclic), /Cyclic/);
  assert.throws(() => boundedAuditJsonSha256(deep), /structural limit/);
  assert.throws(() => boundedAuditJsonSha256(accessor), /accessors/);
  assert.throws(() => boundedAuditJsonSha256('x'.repeat(AUDIT_LIMITS.jsonBytes)), /byte limit/);
  assert.throws(() => boundedAuditJsonSha256(Array.from({length: AUDIT_LIMITS.nodes}, () => 0)), /structural limit/);
});

test('invalid or oversized corpus and identity fail before publication', context => {
  const setup = fixture(context), audit = setup.create();
  for (const corpus of [null, {}, {cases: []}, {cases: [null]}, {cases: [{id: ''}]},
    {cases: [{id: 'same'}, {id: 'same'}]}, {cases: Array.from({length: AUDIT_LIMITS.cases + 1}, (_, index) => ({id: `case-${index}`}))},
    {...setup.corpus, text: 'x'.repeat(AUDIT_LIMITS.jsonBytes)}]) {
    assert.throws(() => audit.commitFreshCorpus(corpus), /corpus|case IDs|byte limit/);
  }
  assert.equal(fs.existsSync(setup.auditDirectory), false);
  assert.throws(() => createLocalFilesystemHoldoutAudit(setup.auditDirectory), /identity reader required/);
  for (const identity of [{sources: {}, runtime: {node: process.version}},
    {sources: {source: 'bad'}, runtime: {node: process.version}}, {sources: {source: 'a'.repeat(64)}, runtime: {}},
    {sources: {source: 'a'.repeat(64)}, runtime: {value: 'x'.repeat(AUDIT_LIMITS.identityBytes)}}]) {
    const invalid = createLocalFilesystemHoldoutAudit(setup.auditDirectory, {readCurrentSourceRuntimeIdentity: () => identity});
    assert.throws(() => invalid.commitFreshCorpus(setup.corpus), /source\/runtime.*required|byte limit/);
  }
  assert.equal(fs.existsSync(setup.auditDirectory), false);
});

test('candidate freeze rejects duplicate, missing, relative, non-JSON and oversized files', context => {
  const setup = fixture(context), audit = setup.create();
  audit.commitFreshCorpus(setup.corpus);
  for (const candidates of [[], Array(AUDIT_LIMITS.candidates + 1), [{id: 'bad', path: '../policy.json'}],
    [{id: '', path: setup.candidatePath}], [{id: 'missing', path: path.join(setup.directory, 'missing.json')}],
    [...setup.candidateFiles, ...setup.candidateFiles], [...setup.candidateFiles, {id: 'second', path: setup.candidatePath}]]) {
    assert.throws(() => audit.freezeCandidateFilesBeforeReveal(candidates), /candidate|Candidate|ENOENT/);
  }
  fs.writeFileSync(setup.candidatePath, '{broken');
  assert.throws(() => audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles), SyntaxError);
  fs.writeFileSync(setup.candidatePath, Buffer.from([0x22, 0xff, 0x22]));
  assert.throws(() => audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles), /UTF-8/);
  fs.writeFileSync(setup.candidatePath, 'x'.repeat(AUDIT_LIMITS.jsonBytes + 1));
  assert.throws(() => audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles), /byte limit/);
  assert.equal(fs.existsSync(path.join(setup.auditDirectory, 'candidates.json')), false);
});

test('publication exposes only complete JSON and an exclusive competing commitment wins without overwrite', context => {
  const setup = fixture(context), audit = setup.create(), link = fs.linkSync;
  let competing = false;
  let winning;
  context.mock.method(fs, 'linkSync', (temporary, filename) => {
    assert.doesNotThrow(() => JSON.parse(fs.readFileSync(temporary, 'utf8')));
    if (!competing) {
      assert.equal(fs.existsSync(filename), false);
      competing = true;
      winning = setup.create().commitFreshCorpus({...setup.corpus, specification: 'winning-commitment'});
    }
    return link(temporary, filename);
  });
  assert.throws(() => audit.commitFreshCorpus(setup.corpus), /EEXIST/);
  assert.equal(audit.readAuditStatus().commitment.commitmentSha256, winning.commitmentSha256);
  assert.deepEqual(fs.readdirSync(setup.auditDirectory), ['commitment.json']);
});

test('publication failure leaves no partial commitment or premature consumed marker', context => {
  const setup = fixture(context), audit = setup.create(), link = fs.linkSync;
  const publishFailure = () => { throw Object.assign(Error('Publication failed'), {code: 'EIO'}); };
  const mocked = context.mock.method(fs, 'linkSync', publishFailure);
  assert.throws(() => audit.commitFreshCorpus(setup.corpus), /Publication failed/);
  assert.deepEqual(fs.readdirSync(setup.auditDirectory), []);
  mocked.mock.mockImplementation(link);
  audit.commitFreshCorpus(setup.corpus);
  audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  mocked.mock.mockImplementation(publishFailure);
  assert.throws(() => audit.consumeAndRevealOnce(), /Publication failed/);
  assert.deepEqual(fs.readdirSync(setup.auditDirectory).sort(), ['candidates.json', 'commitment.json']);
  mocked.mock.mockImplementation(link);
  assert.deepEqual(audit.consumeAndRevealOnce().corpus, setup.corpus);
});

test('exclusive consumed publication permits only one competing reveal', context => {
  const setup = fixture(context), audit = setup.create(), link = fs.linkSync;
  audit.commitFreshCorpus(setup.corpus);
  audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  let competing = false;
  let revealed;
  context.mock.method(fs, 'linkSync', (temporary, filename) => {
    if (!competing && filename === path.join(setup.auditDirectory, 'consumed.json')) {
      competing = true;
      revealed = setup.create().consumeAndRevealOnce();
    }
    return link(temporary, filename);
  });
  assert.throws(() => audit.consumeAndRevealOnce(), /EEXIST/);
  assert.deepEqual(revealed.corpus, setup.corpus);
  assert.deepEqual(audit.readAuditStatus().consumed, revealed.consumed);
  assert.throws(() => setup.create().consumeAndRevealOnce(), /already consumed/);
  assert.equal(fs.readdirSync(setup.auditDirectory).filter(name => name === 'consumed.json').length, 1);
});

test('failure after marker publication still burns the holdout without permitting a retry', context => {
  const setup = fixture(context), audit = setup.create(), link = fs.linkSync;
  audit.commitFreshCorpus(setup.corpus);
  audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  context.mock.method(fs, 'linkSync', (temporary, filename) => {
    link(temporary, filename);
    throw Error('Interrupted after publication');
  });
  assert.throws(() => audit.consumeAndRevealOnce(), /Interrupted after publication/);
  assert.notEqual(audit.readAuditStatus().consumed, null);
  assert.throws(() => setup.create().consumeAndRevealOnce(), /already consumed/);
});

test('tampered commitment, freeze and consumed records fail closed', context => {
  const setup = fixture(context), audit = setup.create();
  audit.commitFreshCorpus(setup.corpus);
  audit.freezeCandidateFilesBeforeReveal(setup.candidateFiles);
  const commitmentPath = path.join(setup.auditDirectory, 'commitment.json'), original = fs.readFileSync(commitmentPath, 'utf8');
  fs.writeFileSync(commitmentPath, original.replace('new-independent-corpus', 'old-independent-corpus'));
  assert.throws(() => audit.consumeAndRevealOnce(), /commitment hash\/identity mismatch/);
  fs.writeFileSync(commitmentPath, original.replace(/"committedAt":"[^"]+"/, '"committedAt":"changed"'));
  assert.throws(() => audit.consumeAndRevealOnce(), /freeze commitment\/identity mismatch/);
  fs.writeFileSync(commitmentPath, original);
  const freezePath = path.join(setup.auditDirectory, 'candidates.json'), frozen = fs.readFileSync(freezePath, 'utf8');
  fs.writeFileSync(freezePath, frozen.replace('candidate-1', ''));
  assert.throws(() => audit.consumeAndRevealOnce(), /Invalid frozen candidate identity/);
  fs.writeFileSync(freezePath, frozen);
  audit.consumeAndRevealOnce();
  const consumedPath = path.join(setup.auditDirectory, 'consumed.json'), marker = fs.readFileSync(consumedPath, 'utf8');
  fs.writeFileSync(consumedPath, marker.replace(/"freezeSha256":"[a-f0-9]+"/, `"freezeSha256":"${'0'.repeat(64)}"`));
  assert.throws(() => audit.readAuditStatus(), /Consumed marker commitment\/identity mismatch/);
  assert.throws(() => audit.consumeAndRevealOnce(), /already consumed/);
});
