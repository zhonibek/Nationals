'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {isDeepStrictEqual} = require('node:util');
const {boundedAuditJsonSha256} = require('./motion-holdout');
const {aggregate, suite} = require('./motion-evaluation');
const Learner = require('./motion-learner');
const {readExposureSnapshot, assertFreshCases} = require('./motion-exposures');

const DEFAULT_HOLDOUT = path.join(__dirname, 'runs/motion/holdout');
const uuid = value => typeof value === 'string' && /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/.test(value);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');

function read(filename, maximum = 4 * 1024 * 1024) {
  if (!fs.lstatSync(filename).isFile()) throw Error('Regular evidence files required');
  const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
  try {
    const before = fs.fstatSync(descriptor);
    if (!before.isFile() || before.size > maximum) throw Error('Evidence exceeds reader budget');
    const bytes = Buffer.alloc(before.size + 1);
    const length = fs.readSync(descriptor, bytes, 0, bytes.length, 0);
    const after = fs.fstatSync(descriptor);
    if (length !== before.size || after.size !== before.size || before.mtimeMs !== after.mtimeMs) throw Error('Evidence changed while reading');
    const content = bytes.subarray(0, length);
    return {value: JSON.parse(content.toString('utf8')), sha256: hash(content)};
  } finally { fs.closeSync(descriptor); }
}

function readIndependentEvaluation(status, learningDirectory = path.join(__dirname, 'runs/motion/learning'), {holdoutDirectory = DEFAULT_HOLDOUT,
  readCurrentIdentity = () => require('./motion-experiment').identity()} = {}) {
  const unavailable = message => ({available: false, policyPromotionAllowed: false, message});
  if (!status?.available || !uuid(status.runId)) return unavailable('No compatible saved learner');
  try {
    const runDirectory = path.join(learningDirectory, status.runId);
    const indexPath = path.join(runDirectory, 'independent-final.json');
    if (!fs.existsSync(indexPath)) return unavailable('No independent final linked to these checkpoints');
    if (!fs.lstatSync(runDirectory).isDirectory()) throw Error('Regular run directory required');
    const index = read(indexPath, 4096).value;
    if (index.schemaVersion !== 1 || index.runId !== status.runId || !uuid(index.holdoutId) ||
        !/^[a-f0-9]{64}$/.test(index.evaluationSha256)) throw Error('Invalid independent index');
    const directory = path.join(holdoutDirectory, index.holdoutId);
    if (!fs.lstatSync(directory).isDirectory()) throw Error('Regular holdout directory required');
    const consumed = read(path.join(directory, 'consumed.json'), 4096).value;
    if (consumed.kind !== 'local-filesystem-holdout-audit-consumed' || consumed.schemaVersion !== 1) throw Error('Consumed holdout required');
    const commitment = read(path.join(directory, 'commitment.json'));
    const freeze = read(path.join(directory, 'candidates.json'));
    const evaluated = read(path.join(directory, 'evaluation.json'));
    const result = evaluated.value, committed = commitment.value;
    if (evaluated.sha256 !== index.evaluationSha256 || consumed.commitmentSha256 !== commitment.sha256 ||
        consumed.freezeSha256 !== freeze.sha256 || freeze.value.commitmentSha256 !== commitment.sha256 ||
        consumed.identitySha256 !== committed.identitySha256 || freeze.value.identitySha256 !== committed.identitySha256 ||
        boundedAuditJsonSha256(committed.identity) !== committed.identitySha256 ||
        boundedAuditJsonSha256(committed.corpus) !== committed.corpusJsonSha256 ||
        boundedAuditJsonSha256(committed.corpus.cases) !== committed.casesSha256 ||
        !isDeepStrictEqual(result.holdoutAudit?.consumed, consumed) ||
        result.holdoutAudit?.commitment?.commitmentSha256 !== commitment.sha256 ||
        result.caseCount !== committed.caseCount || result.caseCount !== committed.corpus.cases.length ||
        !Number.isInteger(result.caseCount) || result.caseCount < 1 || result.caseCount > 256 ||
        !isDeepStrictEqual(result.identity, status.identity) || !Array.isArray(status.models) || status.models.length !== 3 ||
        !Array.isArray(result.models) || result.models.length !== 3 ||
        new Set(status.models.map(model => model.seed)).size !== 3 ||
        !isDeepStrictEqual(committed.corpus.acceptance, suite(2).acceptance)) throw Error('Independent evidence binding mismatch');
    if (committed.corpus.knownExposure) {
      const exposure = readExposureSnapshot(directory);
      if (exposure.ledgerSha256 !== committed.corpus.knownExposure.ledgerSha256 ||
          !isDeepStrictEqual(committed.corpus.knownExposure, result.knownExposureExclusion) ||
          !isDeepStrictEqual(assertFreshCases(committed.corpus.cases, exposure.ledger), result.knownExposureExclusion)) throw Error('Independent known-exposure binding mismatch');
    }
    const candidates = freeze.value.candidates;
    if (!Array.isArray(candidates) || candidates.length !== 6 || new Set(candidates.map(row => row.id)).size !== 6) throw Error('Six unique frozen candidates required');
    const models = status.models.map(saved => {
      const matching = result.models.filter(model => model.seed === saved.seed);
      if (matching.length !== 1) throw Error('Unique matching independent seed required');
      const model = matching[0];
      if (model.policySha256 !== saved.policySha256 || model.initialSha256 !== saved.initialSha256 ||
          candidates.find(candidate => candidate.id === `learned-${saved.seed}`)?.sha256 !== saved.policySha256 ||
          candidates.find(candidate => candidate.id === `initial-${saved.seed}`)?.sha256 !== saved.initialSha256 ||
          !Array.isArray(model.results) || model.results.length !== result.caseCount) throw Error('Independent checkpoint mismatch');
      for (const [position, row] of model.results.entries()) {
        if (row.id !== committed.corpus.cases[position].id || row.seed !== committed.corpus.cases[position].seed) throw Error('Independent cases incomplete or reordered');
        for (const kind of ['baseline', 'neutral', 'untrained', 'learned']) {
          if (typeof row[kind]?.reason !== 'string' || !['elapsedSeconds', 'effortProxyVAs', 'contactSeconds', 'positionErrorMeters']
            .every(key => Number.isFinite(row[kind].metrics?.[key]) && row[kind].metrics[key] >= 0)) throw Error('Finite complete measured results required');
        }
      }
      const summary = kind => aggregate(model.results.map(row => row[kind]));
      const baseline = summary('baseline'), neutral = summary('neutral'), untrained = summary('untrained'), learned = summary('learned');
      if (![['baseline', baseline], ['neutral', neutral], ['untrained', untrained], ['learned', learned]]
        .every(([kind, calculated]) => isDeepStrictEqual(calculated, model[kind]))) throw Error('Independent aggregate mismatch');
      const acceptance = Learner.acceptance(model.results, baseline, learned, untrained, suite(2).acceptance);
      if (!isDeepStrictEqual(acceptance, model.acceptance)) throw Error('Independent acceptance mismatch');
      return {seed: saved.seed, policySha256: saved.policySha256, baseline, neutral, untrained, learned,
        gateReportedPassed: acceptance.passed, successRegressions: acceptance.regressions.length,
        contactRegressions: acceptance.contactRegressions.length,
        meanEffortImprovementFraction: acceptance.meanEffortImprovementFraction};
    });
    return {available: true, holdoutId: index.holdoutId, caseCount: result.caseCount,
      sourceRuntimeMatches: boundedAuditJsonSha256(readCurrentIdentity()) === committed.identitySha256,
      gateReportedPassed: result.learnedImprovementVerified === true && models.every(model => model.gateReportedPassed),
      policyPromotionAllowed: false, models,
      scope: 'Consumed independent reach test; local file audit, not external custody, full-game, physical or AMD proof'};
  } catch { return unavailable('Independent evidence is missing, stale or inconsistent; no policy promotion'); }
}

module.exports = {readIndependentEvaluation};
