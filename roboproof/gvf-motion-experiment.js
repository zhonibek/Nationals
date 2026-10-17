'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {createHash, randomInt} = require('node:crypto');
const {isDeepStrictEqual} = require('node:util');
const {worlds} = require('./motion-development');
const {createLocalFilesystemHoldoutAudit} = require('./motion-holdout');
const {assertFreshCases, snapshotExposureLedger, assertExposureSnapshotCurrent} = require('./motion-exposures');
const {loadHeadless} = require('../simulator/headless');
const {baseline, runGvfPolicy, runGvfZeroResidual, replay} = require('./motion');
const {suite, aggregate} = require('./motion-evaluation');
const Learner = require('./motion-learner');
const GvfLearner = require('./motion-gvf-learner');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_HOLDOUT = path.join(__dirname, 'runs/motion/holdout');
const REQUIRED_SEEDS = Object.freeze([50512, 50513, 50514]);
const COHORTS = Object.freeze([
  Object.freeze({id: 'primary-gvf512x10-v1', width: 512, hiddenLayers: 10, seeds: REQUIRED_SEEDS}),
  Object.freeze({id: 'historical-gvf64x2-v1', width: 64, hiddenLayers: 2, seeds: Object.freeze([50101, 50102, 50103])})
]);
const MANIFEST_PURPOSE = 'remote-only experimental GVF residual TD3; original Simulator/iraLIB';
const SOURCE_FILES = [
  'roboproof/core.js', 'roboproof/scenario-spec.json', 'roboproof/gvf-motion-experiment.js',
  'roboproof/motion.js', 'roboproof/motion-evaluation.js',
  'roboproof/motion-development.js', 'roboproof/motion-gvf-learner.js', 'roboproof/motion-learner.js',
  'roboproof/motion-exposures.js', 'roboproof/motion-exposure-codec.js', 'roboproof/motion-holdout.js',
  'roboproof/motion-exposure-recovery.js'
];
const TRAINING_SOURCE_FILES = [
  'config/robot.json', 'roboproof/motion_env.py', 'roboproof/motion-bridge.js', ...SOURCE_FILES,
  'roboproof/motion_learning/gvf_contract.py', 'roboproof/motion_learning/gvf_model.py',
  'roboproof/motion_learning/gvf_td3.py', 'roboproof/motion_learning/gvf_train.py',
  'roboproof/motion_learning/train.py', 'roboproof/motion_learning/evaluation_exclusion.py',
  'simulator/control-build.json', 'simulator/control.wasm', 'simulator/headless.js',
  'simulator/engine.js', 'simulator/motion.js', 'simulator/policy.js', 'simulator/gvf-policy.js',
  'simulator/control-runtime.js', 'simulator/robot-config.js', 'simulator/override.js',
  'simulator/override-geometry.js', 'simulator/override-dynamics.js', 'simulator/override-autonomy.js',
  'tests/fixtures/motion-evaluation-worlds.json', 'tests/fixtures/motion-evaluation-worlds-v2.json'
];
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

function requireApprovedRemote() {
  if (process.platform !== 'linux' || process.env.ROBOPROOF_APPROVED_REMOTE !== '1') {
    throw Error('Approved Linux remote required; mass simulation has no local fallback');
  }
}

function readBytes(filename, maximum = 8 * 1024 * 1024) {
  const beforePath = fs.lstatSync(filename);
  if (!beforePath.isFile() || beforePath.isSymbolicLink() || beforePath.size > maximum) throw Error('Bounded regular evidence file required');
  const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
  try {
    const before = fs.fstatSync(descriptor), buffer = Buffer.alloc(before.size + 1);
    let offset = 0, length;
    do { length = fs.readSync(descriptor, buffer, offset, buffer.length - offset, offset); offset += length; } while (length && offset < buffer.length);
    const after = fs.fstatSync(descriptor), current = fs.lstatSync(filename);
    const state = value => [value.dev, value.ino, value.mode, value.size, value.mtimeMs, value.ctimeMs];
    if (offset !== before.size || !isDeepStrictEqual(state(before), state(after)) ||
        current.isSymbolicLink() || !isDeepStrictEqual(state(after), state(current))) throw Error('Evidence changed while being read');
    return buffer.subarray(0, offset);
  } finally { fs.closeSync(descriptor); }
}

function readJson(filename, maximum = 4 * 1024 * 1024) {
  const bytes = readBytes(filename, maximum);
  return {value: JSON.parse(bytes.toString('utf8')), bytes, sha256: digest(bytes)};
}

function writeExclusiveJson(filename, value, maximum = 8 * 1024 * 1024) {
  const bytes = Buffer.from(JSON.stringify(value) + '\n', 'utf8');
  if (bytes.length > maximum) throw Error('Evaluation evidence exceeds its byte budget');
  const descriptor = fs.openSync(filename, 'wx', 0o600);
  try { fs.writeFileSync(descriptor, bytes); fs.fsyncSync(descriptor); }
  finally { fs.closeSync(descriptor); }
  return {bytes: bytes.length, sha256: digest(bytes)};
}

function progressJournal(filename, header) {
  const descriptor = fs.openSync(filename, 'wx', 0o600);
  let bytesWritten = 0, records = 0, closed = false;
  function append(record) {
    const bytes = Buffer.from(JSON.stringify(record) + '\n', 'utf8');
    if (bytesWritten + bytes.length > 8 * 1024 * 1024) throw Error('Evaluation progress journal exceeds its byte budget');
    let offset = 0;
    while (offset < bytes.length) offset += fs.writeSync(descriptor, bytes, offset, bytes.length - offset);
    fs.fsyncSync(descriptor);
    bytesWritten += bytes.length;
    records++;
  }
  try { append({type: 'started', ...header, startedAt: new Date().toISOString()}); }
  catch (error) { fs.closeSync(descriptor); throw error; }
  return {append, get recordCount() { return records; },
    close() { if (!closed) { fs.closeSync(descriptor); closed = true; } }};
}

function identity() {
  const build = JSON.parse(fs.readFileSync(path.join(ROOT, 'simulator/control-build.json'), 'utf8'));
  const names = [...new Set([...SOURCE_FILES, ...Object.keys(build.sources)])].sort();
  const sources = Object.fromEntries(names.map(filename => {
    const bytes = fs.readFileSync(path.join(ROOT, filename));
    const normalized = filename.endsWith('.wasm') ? bytes : Buffer.from(bytes.toString('utf8').replace(/^\uFEFF/, '').replace(/\r\n/g, '\n'));
    return [filename, digest(normalized)];
  }));
  const engine = loadHeadless().identity;
  return {sources, originalEngineSha256: digest(Buffer.from(JSON.stringify(engine))),
    runtime: {node: process.version, v8: process.versions.v8, platform: process.platform, architecture: process.arch}};
}

function audit(directory) {
  return createLocalFilesystemHoldoutAudit(directory, {readCurrentSourceRuntimeIdentity: identity});
}

function verifySourceManifest(directory, expectedSha256) {
  const manifestFile = readJson(path.join(directory, 'source-manifest.json'));
  if (manifestFile.sha256 !== expectedSha256) throw Error('Training source-manifest hash mismatch');
  const manifest = manifestFile.value;
  if (!manifest || Object.getPrototypeOf(manifest) !== Object.prototype ||
      Object.keys(manifest).sort().join(',') !== 'files,purpose,schemaVersion' ||
      manifest.schemaVersion !== 1 || manifest.purpose !== MANIFEST_PURPOSE ||
      !Array.isArray(manifest.files) || !manifest.files.length || manifest.files.length > 2048) {
    throw Error('Exact versioned training source manifest required');
  }
  const names = new Set();
  let totalBytes = 0;
  for (const row of manifest.files) {
    if (!row || Object.getPrototypeOf(row) !== Object.prototype || Object.keys(row).sort().join(',') !== 'bytes,path,sha256' ||
        typeof row.path !== 'string' || !row.path || row.path.includes('\\') || row.path.startsWith('/') ||
        path.posix.normalize(row.path) !== row.path || row.path.split('/').some(part => !part || part === '.' || part === '..') ||
        !Number.isSafeInteger(row.bytes) || row.bytes < 1 || row.bytes > 8 * 1024 * 1024 ||
        typeof row.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(row.sha256) || names.has(row.path)) {
      throw Error('Malformed or duplicate training source entry');
    }
    names.add(row.path);
    totalBytes += row.bytes;
    if (totalBytes > 160 * 1024 * 1024) throw Error('Training source inventory exceeds its byte budget');
    const target = path.resolve(ROOT, ...row.path.split('/'));
    if (path.relative(ROOT, target).startsWith('..')) throw Error('Training source escaped the project root');
    for (let parent = target; parent !== path.dirname(ROOT); parent = path.dirname(parent)) {
      if (fs.lstatSync(parent).isSymbolicLink()) throw Error('Symlinked training source rejected');
      if (parent === ROOT) break;
    }
    const source = readBytes(target, 8 * 1024 * 1024);
    if (source.length !== row.bytes || digest(source) !== row.sha256) throw Error('Training source differs from frozen manifest: ' + row.path);
  }
  const build = JSON.parse(fs.readFileSync(path.join(ROOT, 'simulator/control-build.json'), 'utf8'));
  const pythonModules = fs.readdirSync(path.join(ROOT, 'roboproof/motion_learning'), {withFileTypes: true})
    .filter(entry => entry.isFile() && entry.name.endsWith('.py'))
    .map(entry => `roboproof/motion_learning/${entry.name}`);
  const required = [...TRAINING_SOURCE_FILES, ...Object.keys(build.sources), ...pythonModules];
  if (fs.existsSync(path.join(ROOT, 'roboproof/motion-exposure-losses.json'))) required.push('roboproof/motion-exposure-losses.json');
  if (required.some(filename => !names.has(filename))) throw Error('Training source manifest omitted required controller, evaluator, or Python source');
  return {manifest, sha256: manifestFile.sha256};
}

function actorWeightsChanged(initial, learned) {
  return !isDeepStrictEqual(initial.data.layers, learned.data.layers);
}

function selectCohort(records) {
  if (!Array.isArray(records) || records.length !== 3) throw Error('Three separately trained GVF runs are required');
  const seeds = records.map(record => record?.config?.seed).sort((left, right) => left - right);
  const config = records[0]?.config;
  const cohort = COHORTS.find(candidate => isDeepStrictEqual(seeds, [...candidate.seeds].sort((left, right) => left - right)) &&
    records.every(record => record?.config?.networkWidth === candidate.width &&
      record?.config?.hiddenLayers === candidate.hiddenLayers));
  if (!cohort || !config || records.some(record => record?.config?.algorithm !== 'td3-tanh-gvf-residual-v1')) {
    throw Error('Only a predeclared primary 512x10 or historical 64x2 three-seed cohort is admissible');
  }
  return cohort;
}

function modelsForRuns(runDirectories) {
  if (!Array.isArray(runDirectories) || runDirectories.length !== REQUIRED_SEEDS.length) throw Error('Three separately trained GVF runs are required');
  const runRecords = runDirectories.map(directory => {
    const absolute = path.resolve(directory), folder = fs.lstatSync(absolute);
    if (!folder.isDirectory() || folder.isSymbolicLink()) throw Error('Regular GVF training run directory required');
    return {absolute, record: readJson(path.join(absolute, 'run.json')).value};
  });
  const cohort = selectCohort(runRecords.map(row => row.record));
  const currentIdentity = loadHeadless().identity;
  const models = runRecords.map(({absolute, record}) => {
    const {value: terminal} = readJson(path.join(absolute, 'terminal.json'));
    if (!isDeepStrictEqual(record, terminal) || !['completed-budget', 'time-budget-stopped'].includes(record.status) ||
        record.unresolvedExposure !== null || record.learnedImprovementVerified !== false ||
        record.independentQualificationPerformed !== false || !Number.isInteger(record.episodes) || record.episodes < 1 ||
        !Number.isInteger(record.decisions) || record.decisions < 1 || !Number.isInteger(record.updates) || record.updates < 1 ||
        record.runId !== path.basename(absolute).replace(/^gvf-td3-/, '') || !record.engineIdentity ||
        !isDeepStrictEqual(record.engineIdentity, currentIdentity) || !record.config ||
        record.config.algorithm !== 'td3-tanh-gvf-residual-v1' || record.config.networkWidth !== cohort.width ||
        record.config.hiddenLayers !== cohort.hiddenLayers || record.config.physics !== 'original-Simulator-iraLIB-only' ||
        record.trainingDevice?.name?.toUpperCase().includes('MI300X') !== true || !record.trainingDevice.hip ||
        record.trainingDevice.cudaBuild !== null || record.trainingDevice.totalMemoryBytes < 180 * 1024 ** 3) {
      throw Error('Completed MI300X GVF training run with matching original-engine identity required');
    }
    const {value: host} = readJson(path.join(absolute, 'host.json'));
    if (host.gpuUsedForOptimizer !== true || !isDeepStrictEqual(host.gpu, record.trainingDevice)) throw Error('Verified MI300X optimizer-use receipt required');
    const manifest = verifySourceManifest(absolute, record.sourceManifestSha256);
    const initialPath = path.join(absolute, 'actors/actor-episode-000000.json');
    const learnedPath = path.join(absolute, 'actors', `actor-episode-${String(record.episodes).padStart(6, '0')}.json`);
    const actorBudget = cohort.width === 512 ? 64 * 1024 * 1024 : 8 * 1024 * 1024;
    const initialSha256 = digest(readBytes(initialPath, actorBudget));
    const learnedSha256 = digest(readBytes(learnedPath, actorBudget));
    const initial = GvfLearner.policy(initialPath, initialSha256);
    const learned = GvfLearner.policy(learnedPath, learnedSha256);
    if (record.config.seed !== initial.data.trainingSeed || initial.data.trainingSeed !== learned.data.trainingSeed ||
        initial.data.steps !== 0 || initial.data.updates !== 0 || initial.data.trained ||
        learned.data.steps !== record.decisions || learned.data.updates !== record.updates || !learned.data.trained ||
        initial.data.width !== cohort.width || learned.data.width !== cohort.width ||
        initial.data.hiddenLayers !== cohort.hiddenLayers || learned.data.hiddenLayers !== cohort.hiddenLayers) {
      throw Error('Run/actor seed, architecture, update count or trained-state mismatch');
    }
    return {seed: record.config.seed, runId: record.runId, runDirectory: absolute,
      sourceManifestSha256: manifest.sha256, config: record.config, trainingDevice: record.trainingDevice,
      initialPath, learnedPath, initialSha256, learnedSha256, initial, learned,
      actorWeightsChanged: actorWeightsChanged(initial, learned)};
  }).sort((left, right) => left.seed - right.seed);
  if (!isDeepStrictEqual(models.map(model => model.seed), [...cohort.seeds].sort((left, right) => left - right)) ||
      new Set(models.map(model => model.runId)).size !== models.length ||
      new Set(models.map(model => model.sourceManifestSha256)).size !== 1 ||
      !models.every(model => isDeepStrictEqual({...model.config, seed: undefined}, {...models[0].config, seed: undefined}) &&
        isDeepStrictEqual(model.trainingDevice, models[0].trainingDevice))) {
    throw Error('Three predeclared independent GVF seeds must share one exact method, source bundle and MI300X runtime');
  }
  return models.map(model => ({...model, cohort: {id: cohort.id, width: cohort.width,
    hiddenLayers: cohort.hiddenLayers, seeds: [...cohort.seeds]}}));
}

function compact(report) { return {reason: report.reason, metrics: report.metrics}; }

function compareCorpus(corpus, models, runners = {}) {
  if (![runners.baseline, runners.zeroResidual, runners.policy, runners.replay].every(runner => typeof runner === 'function')) {
    requireApprovedRemote();
  }
  if (!Array.isArray(corpus?.cases) || !corpus.cases.length || corpus.cases.length > 256 ||
      !Array.isArray(models) || models.length !== 3) throw Error('Bounded corpus and three GVF models required');
  const runBaseline = runners.baseline || baseline;
  const runZero = runners.zeroResidual || runGvfZeroResidual;
  const runPolicy = runners.policy || runGvfPolicy;
  const verifyReplay = runners.replay || replay;
  const onCase = runners.onCase || (() => {});
  const rules = corpus.acceptance || suite(2).acceptance;
  const baselineReports = [], zeroReports = [];
  const perModel = models.map(model => ({initial: [], learned: [], results: []}));
  const onEpisode = runners.onEpisode || (() => {});
  for (const [index, world] of corpus.cases.entries()) {
    const original = runBaseline(world.seed, world.task, world.configuration, {recordTransitions: false});
    const zero = runZero(world.seed, world.task, world.configuration, {recordTransitions: false});
    verifyReplay(original); verifyReplay(zero);
    onEpisode({caseIndex: index, caseId: world.id, seed: world.seed, policyId: 'existing-controller',
      report: compact(original), replayVerified: true});
    onEpisode({caseIndex: index, caseId: world.id, seed: world.seed, policyId: 'gvf-zero-residual',
      report: compact(zero), replayVerified: true});
    baselineReports.push(compact(original)); zeroReports.push(compact(zero));
    const record = {world, baseline: original, zeroResidual: zero, models: []};
    for (const [modelIndex, model] of models.entries()) {
      const initial = runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
        options: {recordTransitions: false, policyIdentity: model.initial.identity},
        act: (_values, observation) => model.initial.act(observation)});
      const learned = runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
        options: {recordTransitions: false, policyIdentity: model.learned.identity},
        act: (_values, observation) => model.learned.act(observation)});
      verifyReplay(initial); verifyReplay(learned);
      const initialCompact = compact(initial), learnedCompact = compact(learned);
      onEpisode({caseIndex: index, caseId: world.id, seed: world.seed, policyId: model.initial.identity.id,
        report: initialCompact, replayVerified: true});
      onEpisode({caseIndex: index, caseId: world.id, seed: world.seed, policyId: model.learned.identity.id,
        report: learnedCompact, replayVerified: true});
      if (!isDeepStrictEqual(initialCompact, compact(zero))) throw Error('Untrained zero-initialized actor differs from the analytic GVF zero-residual policy');
      perModel[modelIndex].initial.push(initialCompact);
      perModel[modelIndex].learned.push(learnedCompact);
      perModel[modelIndex].results.push({id: world.id, seed: world.seed, baseline: compact(original),
        zeroResidual: compact(zero), initial: initialCompact, untrained: initialCompact, learned: learnedCompact});
      record.models.push({seed: model.seed, initial, learned,
        exactReplayVerified: true, initialEqualsZeroResidual: true});
    }
    onCase(record, index);
  }
  const baselineSummary = aggregate(baselineReports), zeroSummary = aggregate(zeroReports);
  const resultModels = models.map((model, index) => {
    const initial = aggregate(perModel[index].initial), learned = aggregate(perModel[index].learned);
    const acceptance = Learner.acceptance(perModel[index].results, baselineSummary, learned, initial, rules);
    const initialSuccessRegressions = perModel[index].results.filter(row => row.initial.reason === 'success' && row.learned.reason !== 'success').map(row => row.id);
    const gainAgainstInitial = 1 - learned.meanEffortProxyVAsAllWorlds / initial.meanEffortProxyVAsAllWorlds;
    return {seed: model.seed, runId: model.runId, policySha256: model.learnedSha256,
      initialSha256: model.initialSha256, actorWeightsChanged: model.actorWeightsChanged,
      baseline: baselineSummary, zeroResidual: zeroSummary, untrained: initial, learned,
      results: perModel[index].results, acceptance, initialSuccessRegressions,
      measuredEfficiencyEvidence: acceptance.passed && model.actorWeightsChanged && initialSuccessRegressions.length === 0 &&
        acceptance.meanEffortImprovementFraction >= 0.05 && gainAgainstInitial >= 0.05,
      meanEffortImprovementAgainstInitial: gainAgainstInitial};
  });
  return {schemaVersion: 1, evaluatedAt: new Date().toISOString(), identity: identity(),
    caseCount: corpus.cases.length, corpusSha256: corpus.corpusSha256,
    specification: corpus.specification, scope: corpus.scope, baseline: baselineSummary, zeroResidual: zeroSummary,
    models: resultModels, learnedImprovementVerified: models.length === 3 && resultModels.every(model =>
      model.acceptance.passed && model.actorWeightsChanged),
    measuredEfficiencyEvidence: resultModels.every(model => model.measuredEfficiencyEvidence),
    exactReplayEpisodes: corpus.cases.length * (2 + models.length * 2),
    interpretation: 'All episodes, failures and independent seeds retained. Original Simulator/iraLIB software evidence only; no physical, full-game or tactical qualification.'};
}

function evaluateDevelopment(runDirectories, directory) {
  requireApprovedRemote();
  const models = modelsForRuns(runDirectories), corpus = suite(2);
  const absolute = path.resolve(directory);
  fs.mkdirSync(absolute, {recursive: false, mode: 0o700});
  const journal = progressJournal(path.join(absolute, 'evaluation-progress.jsonl'),
    {kind: 'gvf-residual-development-progress-v1', caseCount: corpus.cases.length});
  const caseFiles = [];
  try {
    const result = compareCorpus({...corpus, specification: 'gvf-residual-frozen-development-v1',
      scope: 'Previously frozen development worlds; never independent final evidence'}, models, {
      onEpisode: record => journal.append({type: 'episode', ...record}),
      onCase(record, index) {
        const output = writeExclusiveJson(path.join(absolute, `case-${String(index).padStart(3, '0')}.json`), record);
        caseFiles.push({path: `case-${String(index).padStart(3, '0')}.json`, ...output});
        journal.append({type: 'case-completed', caseIndex: index, caseId: record.world.id, ...output});
      }
    });
    const development = {...result, kind: 'gvf-residual-development-evaluation-v1',
      developmentGatesPassed: result.learnedImprovementVerified && result.measuredEfficiencyEvidence,
      candidateCohort: models[0].cohort,
      seeds: models.map(model => ({seed: model.seed, runId: model.runId,
        policySha256: model.learnedSha256, initialSha256: model.initialSha256})), caseFiles};
    writeExclusiveJson(path.join(absolute, 'development-evaluation.json'), development);
    journal.append({type: 'completed', developmentGatesPassed: development.developmentGatesPassed,
      completedAt: new Date().toISOString()});
    return development;
  } catch (error) {
    try { journal.append({type: 'failed', error: String(error), failedAt: new Date().toISOString()}); } catch {}
    throw error;
  } finally { journal.close(); }
}

function prepare(directory, runDirectories, developmentFile) {
  requireApprovedRemote();
  const models = modelsForRuns(runDirectories), development = readJson(developmentFile, 16 * 1024 * 1024);
  const proof = development.value;
  if (proof.kind !== 'gvf-residual-development-evaluation-v1' || proof.developmentGatesPassed !== true ||
      !isDeepStrictEqual(proof.identity, identity()) || proof.models?.length !== 3 ||
      !isDeepStrictEqual(proof.candidateCohort, models[0].cohort) ||
      !isDeepStrictEqual(proof.seeds, models.map(model => ({seed: model.seed, runId: model.runId,
        policySha256: model.learnedSha256, initialSha256: model.initialSha256})))) {
    throw Error('All three exact GVF actors must pass the frozen development gate before drawing an independent final');
  }
  const absolute = path.resolve(directory);
  if (!absolute.startsWith(DEFAULT_HOLDOUT + path.sep) || fs.existsSync(absolute)) throw Error('New unused GVF holdout directory required');
  const saved = snapshotExposureLedger(absolute);
  let cases, exclusion, draws = 0;
  for (; draws < 20; draws++) {
    cases = [...worlds(randomInt(0, 0xffffffff), 8, 'short-reach').cases,
      ...worlds(randomInt(0, 0xffffffff), 16, 'field-reach').cases,
      ...worlds(randomInt(0, 0xffffffff), 8, 'full-contract').cases];
    try { exclusion = assertFreshCases(cases, saved.ledger); break; }
    catch (error) { if (!error.message.includes('Final overlaps known')) throw error; }
  }
  if (!exclusion || cases.length !== 128) throw Error('A fresh complete 128-case GVF final could not be committed');
  const corpus = {schemaVersion: 1, specification: 'gvf-residual-independent-final-v1',
    scope: 'Fresh original Simulator short/field/full reach with four original physical variations per family; not physical or full-game validation',
    split: 'Final only, after all three trained actors and development outcomes are frozen; never use for training or selection',
    familyCounts: {short: 8, field: 16, full: 8}, physicalVariationsPerFamily: 4, cases,
    acceptance: {...suite(2).acceptance},
    candidateCohort: models[0].cohort,
    candidateProtocol: 'Compare unchanged controller, zero residual, each zero-initialized actor and each of the three frozen trained actors; every seed must pass, no best-seed selection',
    developmentEvidenceSha256: development.sha256, candidateSeeds: models.map(model => model.seed),
    knownExposure: exclusion, exposureSnapshotSha256: saved.ledgerSha256, exclusionDraws: draws + 1};
  const commitment = audit(absolute).commitFreshCorpus(corpus);
  writeExclusiveJson(path.join(absolute, 'protocol.json'), {schemaVersion: 1,
    specification: corpus.specification, caseCount: 128, familyCounts: corpus.familyCounts,
    acceptance: corpus.acceptance, candidateSeeds: corpus.candidateSeeds, candidateCohort: corpus.candidateCohort,
    developmentEvidence: {path: path.resolve(developmentFile), sha256: development.sha256},
    knownExposure: exclusion, commitment});
  return commitment;
}

function evaluateFinal(directory, runDirectories, developmentFile) {
  requireApprovedRemote();
  const models = modelsForRuns(runDirectories), absolute = path.resolve(directory), holdout = audit(absolute);
  const protocol = readJson(path.join(absolute, 'protocol.json')).value;
  const development = readJson(developmentFile, 16 * 1024 * 1024);
  if (protocol.specification !== 'gvf-residual-independent-final-v1' || protocol.caseCount !== 128 ||
      protocol.developmentEvidence?.sha256 !== development.sha256 || development.value.developmentGatesPassed !== true ||
      !isDeepStrictEqual(protocol.candidateSeeds, models.map(model => model.seed)) ||
      !isDeepStrictEqual(protocol.candidateCohort, models[0].cohort) ||
      !isDeepStrictEqual(development.value.candidateCohort, models[0].cohort)) throw Error('Frozen GVF final/development protocol mismatch');
  const exposure = assertExposureSnapshotCurrent(absolute);
  if (protocol.knownExposure?.ledgerSha256 !== exposure.ledgerSha256) throw Error('GVF final exposure snapshot changed before reveal');
  const candidates = models.flatMap(model => [
    {id: `initial-${model.seed}`, path: model.initialPath}, {id: `learned-${model.seed}`, path: model.learnedPath}
  ]);
  holdout.freezeCandidateFilesBeforeReveal(candidates);
  const journal = progressJournal(path.join(absolute, 'evaluation-progress.jsonl'),
    {kind: 'gvf-residual-independent-final-progress-v1', caseCount: 128,
      commitmentSha256: protocol.commitment?.commitmentSha256 ?? null});
  let revealed;
  try {
    revealed = holdout.consumeAndRevealOnce();
    journal.append({type: 'holdout-consumed', commitmentSha256: revealed.commitment.commitmentSha256,
      consumedAt: revealed.consumed.consumedAt});
    if (revealed.corpus.knownExposure?.ledgerSha256 !== exposure.ledgerSha256 ||
        !isDeepStrictEqual(assertFreshCases(revealed.corpus.cases, exposure.ledger), revealed.corpus.knownExposure)) {
      throw Error('Fresh GVF corpus failed source-bound exposure admission after one-shot reveal');
    }
    const frozen = models.map(model => {
      const initial = revealed.candidates.find(candidate => candidate.id === `initial-${model.seed}`);
      const learned = revealed.candidates.find(candidate => candidate.id === `learned-${model.seed}`);
      if (!initial || !learned) throw Error('Frozen initial/learned actor pair missing after reveal');
      const initialPolicy = GvfLearner.fromArtifact(initial.json, loadHeadless().identity);
      const learnedPolicy = GvfLearner.fromArtifact(learned.json, loadHeadless().identity);
      initialPolicy.identity = {kind: 'untrained', id: `gvf-td3-seed-${model.seed}-initial`, sha256: initial.sha256};
      learnedPolicy.identity = {kind: 'learned', id: `gvf-td3-seed-${model.seed}-learned`, sha256: learned.sha256};
      return {...model, initial: initialPolicy, learned: learnedPolicy};
    });
    const reportsDirectory = path.join(absolute, 'reports');
    fs.mkdirSync(reportsDirectory, {recursive: false, mode: 0o700});
    const result = compareCorpus(revealed.corpus, frozen, {onCase(record, index) {
      writeExclusiveJson(path.join(reportsDirectory, `case-${String(index).padStart(3, '0')}.json`), record);
      journal.append({type: 'case-completed', caseIndex: index, caseId: record.world.id});
    }, onEpisode: record => journal.append({type: 'episode', ...record})});
    result.independentEvaluationPerformed = true;
    result.independentQualificationPassed = result.learnedImprovementVerified;
    result.holdoutAudit = {commitment: revealed.commitment, consumed: revealed.consumed};
    result.knownExposureExclusion = protocol.knownExposure;
    result.developmentEvidenceSha256 = development.sha256;
    writeExclusiveJson(path.join(absolute, 'evaluation.json'), result, 16 * 1024 * 1024);
    journal.append({type: 'completed', independentQualificationPassed: result.independentQualificationPassed,
      completedAt: new Date().toISOString()});
    return result;
  } catch (error) {
    try { journal.append({type: 'failed', error: String(error), failedAtUtc: new Date().toISOString()}); } catch {}
    if (revealed) writeExclusiveJson(path.join(absolute, 'evaluation-failure.json'), {schemaVersion: 1,
      status: 'consumed-with-incomplete-evaluation', error: String(error),
      retainedJournalRecords: journal.recordCount,
      retainedReportCount: fs.existsSync(path.join(absolute, 'reports')) ?
        fs.readdirSync(path.join(absolute, 'reports')).length : 0,
      commitmentSha256: revealed.commitment.commitmentSha256, failedAtUtc: new Date().toISOString()});
    throw error;
  } finally { journal.close(); }
}

function main(argv) {
  const [operation, ...args] = argv;
  if (operation === 'development' && args.length === 4) {
    const result = evaluateDevelopment(args.slice(0, 3), args[3]);
    process.stdout.write(JSON.stringify({kind: result.kind, caseCount: result.caseCount,
      developmentGatesPassed: result.developmentGatesPassed, models: result.models.map(model => ({
        seed: model.seed, learnedImprovementVerified: model.acceptance.passed,
        measuredEfficiencyEvidence: model.measuredEfficiencyEvidence}))}) + '\n');
    return 0;
  }
  if (operation === 'prepare' && args.length === 5) {
    const commitment = prepare(args[0], args.slice(1, 4), args[4]);
    process.stdout.write(JSON.stringify(commitment) + '\n');
    return 0;
  }
  if (operation === 'final' && args.length === 5) {
    const result = evaluateFinal(args[0], args.slice(1, 4), args[4]);
    process.stdout.write(JSON.stringify({caseCount: result.caseCount,
      independentQualificationPassed: result.independentQualificationPassed,
      learnedImprovementVerified: result.learnedImprovementVerified,
      measuredEfficiencyEvidence: result.measuredEfficiencyEvidence}) + '\n');
    return 0;
  }
  throw Error('Usage: development RUN1 RUN2 RUN3 OUTPUT_DIR | prepare HOLDOUT_DIR RUN1 RUN2 RUN3 DEVELOPMENT_JSON | final HOLDOUT_DIR RUN1 RUN2 RUN3 DEVELOPMENT_JSON');
}

if (require.main === module) {
  try { process.exitCode = main(process.argv.slice(2)); }
  catch (error) { process.stderr.write(`${error.stack || error}\n`); process.exitCode = 1; }
}

module.exports = {REQUIRED_SEEDS, COHORTS, selectCohort, identity, audit, requireApprovedRemote, modelsForRuns,
  compareCorpus, evaluateDevelopment, prepare, evaluateFinal, progressJournal, main};
