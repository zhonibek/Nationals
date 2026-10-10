'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {createHash, randomInt, randomUUID} = require('node:crypto');
const {worlds} = require('./motion-development');
const {createLocalFilesystemHoldoutAudit} = require('./motion-holdout');
const {assertFreshCases, snapshotExposureLedger, assertExposureSnapshotCurrent, geometryFingerprint} = require('./motion-exposures');
const {loadHeadless} = require('../simulator/headless');
const {baseline, runPolicy} = require('./motion');
const {aggregate, suite} = require('./motion-evaluation');
const Learner = require('./motion-learner');
const {writeJson} = require('./core');

const ROOT = path.resolve(__dirname, '..');
const SOURCES = ['roboproof/motion_learning/model.py', 'roboproof/motion_learning/train.py', 'roboproof/motion_learning/objective.py',
  'roboproof/motion_learning/td3.py', 'roboproof/motion_learning/td3_train.py',
  'roboproof/motion_learning/imitation.py',
  'roboproof/motion_env.py', 'roboproof/motion-bridge.js', 'roboproof/motion-development.js', 'roboproof/motion-experiment.js',
  'roboproof/motion-learner.js', 'roboproof/motion-evaluation.js', 'roboproof/motion-exposures.js', 'roboproof/motion.js'];
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const textHash = filename => hash(fs.readFileSync(path.join(ROOT, filename), 'utf8').replace(/\r\n/g, '\n'));

function identity() {
  return {sources: {...Object.fromEntries(SOURCES.map(filename => [filename, textHash(filename)])),
    originalEngine: hash(JSON.stringify(loadHeadless().identity))},
    runtime: {node: process.version, v8: process.versions.v8, platform: process.platform, architecture: process.arch,
      requiredTorch: '2.8.0+cpu'}};
}
function audit(directory) { return createLocalFilesystemHoldoutAudit(directory, {readCurrentSourceRuntimeIdentity: identity}); }

function prepare(directory, {shortFamilies = 8, fieldFamilies = 16, fullFamilies = 8} = {}) {
  if (![shortFamilies, fieldFamilies, fullFamilies].every(value => Number.isInteger(value) && value >= 1 && value <= 32)) throw Error('Bounded final families required');
  if (4 * (shortFamilies + fieldFamilies + fullFamilies) > 256) throw Error('Final corpus exceeds evaluator budget');
  const exposure = snapshotExposureLedger(directory);
  let cases, exclusion, attempts = 0;
  for (; attempts < 20; attempts++) {
    cases = [...worlds(randomInt(0, 0xffffffff), shortFamilies, 'short-reach').cases,
      ...worlds(randomInt(0, 0xffffffff), fieldFamilies, 'field-reach').cases,
      ...worlds(randomInt(0, 0xffffffff), fullFamilies, 'full-contract').cases];
    try { exclusion = assertFreshCases(cases, exposure.ledger); break; }
    catch (error) { if (!error.message.includes('Final overlaps known')) throw error; }
  }
  if (!exclusion) throw Error('Fresh-corpus exclusion draw budget exhausted; no commitment created');
  const corpus = {schemaVersion: 1, specification: 'independent-full-reach-final-v3',
    scope: 'Fresh mixed short/field/boundary reach tasks and all six declared physical-parameter ranges; not a physical or full-game benchmark',
    grouping: 'Four physical variations per independently sampled task family',
    split: 'Final only. Cases must not enter training/development/model selection', cases,
    knownExposure: exclusion, exclusionDraws: attempts + 1,
    acceptance: {...suite(2).acceptance},
    secondaryEfficiencyEvidence: 'May report non-inferior success and measured proxy-effort improvement, but this does not relax the strict verified-policy gate',
    candidateProtocol: 'Freeze three independently trained actors and their untrained initial actors, same source/config and common update selection. No best-seed selection; select only on separate development worlds.'};
  const commitment = audit(directory).commitFreshCorpus(corpus);
  writeJson(path.join(directory, 'protocol.json'), {schemaVersion: 1, caseCount: corpus.cases.length,
    specification: corpus.specification, scope: corpus.scope, shortFamilies, fieldFamilies, fullFamilies,
    acceptance: corpus.acceptance, candidateProtocol: corpus.candidateProtocol, knownExposure: exclusion, commitment});
  return commitment;
}

function compareCorpus(corpus, models) {
  if (!Array.isArray(corpus?.cases) || !corpus.cases.length || corpus.cases.length > 256 || !Array.isArray(models) || models.length < 1 || models.length > 3) throw Error('Bounded corpus and models required');
  const originalReports = [], neutralReports = [], slowReports = [];
  const seedReports = models.map(() => ({initial: [], learned: [], results: []}));
  for (const world of corpus.cases) {
    const original = baseline(world.seed, world.task, world.configuration);
    const neutral = runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
      options: {recordTransitions: false, policyIdentity: {kind: 'scripted', id: 'neutral-persistent-reference'}}});
    const slow = runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
      options: {recordTransitions: false, policyIdentity: {kind: 'scripted', id: 'fixed-pace-0.7'}}, act: () => [0.2, 0, 0, 0]});
    originalReports.push(original); neutralReports.push(neutral); slowReports.push(slow);
    for (const [index, model] of models.entries()) {
      const initial = runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
        options: {recordTransitions: false, policyIdentity: model.initial.identity}, act: model.initial.act});
      const learned = runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
        options: {recordTransitions: false, policyIdentity: model.learned.identity}, act: model.learned.act});
      const summary = report => ({reason: report.reason, metrics: report.metrics});
      seedReports[index].initial.push(initial); seedReports[index].learned.push(learned);
      seedReports[index].results.push({id: world.id, group: world.group, seed: world.seed,
        baseline: summary(original), neutral: summary(neutral), fixedPace: summary(slow), initial: summary(initial),
        untrained: summary(initial), learned: summary(learned)});
    }
  }
  const original = aggregate(originalReports), neutral = aggregate(neutralReports), fixedPace = aggregate(slowReports);
  const rules = corpus.acceptance || suite(2).acceptance;
  const resultModels = models.map((model, index) => {
    const records = seedReports[index];
    const initial = aggregate(records.initial), learned = aggregate(records.learned);
    const accepted = Learner.acceptance(records.results, original, learned, initial, rules);
    const initialRegressions = records.results.filter(row => row.initial.reason === 'success' && row.learned.reason !== 'success').map(row => row.id);
    const effortImprovementAgainstInitial = 1 - learned.meanEffortProxyVAsAllWorlds / initial.meanEffortProxyVAsAllWorlds;
    return {seed: model.seed, policySha256: model.learned.sha256, initialSha256: model.initial.sha256,
      baseline: original, neutral, fixedPace, untrained: initial, learned, results: records.results, acceptance: accepted,
      initialSuccessRegressions: initialRegressions,
      measuredEfficiencyEvidence: learned.successRate >= original.successRate && learned.successRate >= initial.successRate &&
        accepted.regressions.length === 0 && initialRegressions.length === 0 && accepted.contactRegressions.length === 0 &&
        accepted.meanEffortImprovementFraction >= 0.05 && effortImprovementAgainstInitial >= 0.05,
      meanEffortImprovementAgainstInitial: effortImprovementAgainstInitial};
  });
  return {schemaVersion: 1, evaluatedAt: new Date().toISOString(), identity: loadHeadless().identity,
    specification: corpus.specification, scope: corpus.scope, caseCount: corpus.cases.length, baseline: original, neutral, fixedPace,
    models: resultModels, learnedImprovementVerified: models.length === 3 && resultModels.every(model => model.acceptance.passed),
    measuredEfficiencyEvidence: models.length === 3 && resultModels.every(model => model.measuredEfficiencyEvidence),
    interpretation: 'All cases, failures and independent seeds retained. Secondary efficiency evidence is not permission to bypass the strict verified-policy gate.'};
}

function modelsForRun(directory, update) {
  directory = path.resolve(directory);
  const saved = Learner.latest(directory);
  if (!saved || saved.record.status === 'running' || saved.record.torch !== '2.8.0+cpu') throw Error('A completed compatible CPU training run is required');
  for (const [filename, digest] of Object.entries(saved.record.sources)) {
    if (textHash(filename) !== digest) throw Error('Training source differs from the declared evaluation method');
  }
  return saved.record.models.map(row => {
    const folder = path.join(saved.runDirectory, `seed-${row.seed}`);
    const filename = update === undefined ? row.policyFile : `policy-${String(update).padStart(4, '0')}.json`;
    return {seed: row.seed, trainingWorlds: row.history.flatMap(entry => entry.worlds), initialFile: path.join(folder, 'initial-policy.json'), learnedFile: path.join(folder, filename),
      initial: Learner.policy(path.join(folder, 'initial-policy.json')), learned: Learner.policy(path.join(folder, filename))};
  });
}

function evaluateFinal(directory, learningDirectory, update) {
  const models = modelsForRun(learningDirectory, update);
  if (models.length !== 3 || !models.every(model => model.learned.data.trained && !model.initial.data.trained)) throw Error('Three trained and initial candidates required before final reveal');
  const holdout = audit(directory);
  const exposure = assertExposureSnapshotCurrent(directory);
  const protocol = JSON.parse(fs.readFileSync(path.join(directory, 'protocol.json'), 'utf8'));
  if (protocol.knownExposure?.ledgerSha256 !== exposure.ledgerSha256) throw Error('Known exposure protocol mismatch before reveal');
  holdout.freezeCandidateFilesBeforeReveal(models.flatMap(model => [
    {id: `initial-${model.seed}`, path: model.initialFile}, {id: `learned-${model.seed}`, path: model.learnedFile}]));
  const revealed = holdout.consumeAndRevealOnce();
  if (revealed.corpus.knownExposure?.ledgerSha256 !== exposure.ledgerSha256) throw Error('Known exposure commitment mismatch; final consumed without evaluation');
  const knownExposureExclusion = assertFreshCases(revealed.corpus.cases, exposure.ledger);
  const finalSeeds = new Set(revealed.corpus.cases.map(world => world.seed));
  const finalGeometry = new Set(revealed.corpus.cases.map(world => geometryFingerprint(world.task)));
  for (const model of models) for (const world of model.trainingWorlds) {
    if (!world.task) throw Error('Training exposure not retained; final already consumed without evaluating');
    if (finalSeeds.has(world.seed) || finalGeometry.has(geometryFingerprint(world.task))) throw Error('Final overlaps training exposure; entire corpus consumed without evaluating');
  }
  const frozenModels = models.map(model => {
    const initial = revealed.candidates.find(candidate => candidate.id === `initial-${model.seed}`);
    const learned = revealed.candidates.find(candidate => candidate.id === `learned-${model.seed}`);
    return {seed: model.seed, initial: Learner.policyFromSnapshot(initial.json, initial.sha256),
      learned: Learner.policyFromSnapshot(learned.json, learned.sha256)};
  });
  const result = compareCorpus(revealed.corpus, frozenModels);
  result.holdoutAudit = {commitment: revealed.commitment, consumed: revealed.consumed};
  result.knownExposureExclusion = knownExposureExclusion;
  result.trainingExclusion = {verified: true, identity: 'world seeds and body-frame geometry quantized to 0.1 inch / 1 degree, ignoring deadlines',
    trainingEpisodesChecked: models.map(model => ({seed: model.seed, count: model.trainingWorlds.length})), finalCount: revealed.corpus.cases.length,
    scope: 'Local deterministic exposure audit, not proof of arbitrary region/OOD separation or tamper-proof custody'};
  writeJson(path.join(directory, 'evaluation.json'), result);
  return result;
}

if (require.main === module) {
  const operation = process.argv[2];
  if (operation === 'prepare') {
    const directory = path.join(__dirname, 'runs/motion/holdout', randomUUID());
    const commitment = prepare(directory);
    console.log(JSON.stringify({directory, commitment}, null, 2));
  } else if (operation === 'development') {
    const models = modelsForRun(process.argv[3]);
    const corpus = worlds(913571, 4, 'field-reach');
    const result = compareCorpus(corpus, models);
    writeJson(path.join(path.resolve(process.argv[3]), `development-${Date.now()}.json`), result);
    console.log(JSON.stringify({scope: 'Development only', baseline: result.baseline,
      models: result.models.map(({results, ...summary}) => summary)}, null, 2));
  } else throw Error('Use prepare or development; final evaluation requires an explicitly frozen protocol');
}

module.exports = {identity, audit, prepare, compareCorpus, modelsForRun, evaluateFinal};
