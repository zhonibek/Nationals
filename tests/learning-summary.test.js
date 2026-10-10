'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Summary = require('../roboproof/dashboard/learning-summary');
const source = fs.readFileSync(path.join(__dirname, '../roboproof/dashboard/learning-summary.js'), 'utf8');

function diagnostics(overrides = {}) {
  return {meanApproxKL: 0.0123, clipFraction: 0.25, meanEntropy: -1.125, meanActorLoss: -0.0625,
    meanCriticLoss: 0.375, valueExplainedVariance: 0.5, sampledActionMean: [0.75, -0.5, 0.25, -0.125],
    sampledActionStd: [0.125, 0.25, 0.375, 0.5], sampledActionBoundaryFraction: 0.375,
    curriculumStageCounts: {'2': 3, '0': 1, '1': 2}, ...overrides};
}

function model(overrides = {}) {
  return {seed: 42, steps: 64, updates: 2, trained: true, actorWeightsChanged: true,
    policySha256: 'fixture-policy-42', history: [diagnostics()], ...overrides};
}

function evaluatedModel(overrides = {}) {
  return {seed: 42, policySha256: 'fixture-policy-42', baseline: {success: 6, count: 24},
    untrained: {success: 7, count: 24}, learned: {success: 7, count: 24},
    acceptance: {passed: false, regressions: ['failure-one'], contactRegressions: ['contact-one'],
      meanTimeImprovementFraction: -0.125, meanEffortImprovementFraction: 0.0125,
      rule: 'Frozen all-world success, contact and efficiency gates'}, ...overrides};
}

function status(overrides = {}) {
  return {available: true, status: 'completed', motionPolicyTrained: true, learnedImprovementVerified: false,
    models: [model()], scope: 'Canonical reach only; no obstacle or game-policy training', evaluation: null, ...overrides};
}

test('CommonJS exports only the three pure presenter functions', () => {
  assert.deepEqual(Object.keys(Summary), ['describeStatus', 'describeModel', 'describeDiagnostics']);
  assert(Object.values(Summary).every(value => typeof value === 'function'));
});

test('browser loading and presentation do not touch DOM, networking, timers or policy execution', () => {
  const forbidden = () => { throw Error('Presenter attempted an external action'); };
  const realm = vm.createContext({});
  for (const name of ['document', 'fetch', 'location', 'XMLHttpRequest', 'WebSocket', 'setTimeout',
    'setInterval', 'requestAnimationFrame', 'Worker', 'localStorage', 'Policy', 'VexRobotSimulator', 'require']) {
    Object.defineProperty(realm, name, {get: forbidden});
  }
  vm.runInContext(source, realm);
  assert.deepEqual(Object.keys(realm), ['MotionLearningSummary']);
  const presenter = realm.MotionLearningSummary;
  assert.match(presenter.describeStatus(status()).improvement, /not verified/);
  assert.equal(presenter.describeModel(model(), {models: [evaluatedModel()]}).gate, 'Failed');
  assert.match(presenter.describeDiagnostics([diagnostics()]), /entropy -1\.1250/);
});

test('AMD loading exports the same presenter without creating a browser global', () => {
  let presenter;
  const define = (dependencies, factory) => {
    assert.equal(dependencies.length, 0);
    presenter = factory();
  };
  define.amd = {};
  const realm = vm.createContext({define});
  vm.runInContext(source, realm);
  assert.equal(realm.MotionLearningSummary, undefined);
  assert.deepEqual(Object.keys(presenter), Object.keys(Summary));
  assert.equal(presenter.describeModel(model(), evaluatedModel()).learned, '7/24');
});

test('recorded training priors are distinct from executed action constraints or verification', () => {
  const legacy = Summary.describeDiagnostics([diagnostics()]);
  assert.doesNotMatch(legacy, /training-only/);
  const summary = Summary.describeDiagnostics([diagnostics({meanDeadlinePaceLoss: 0.0125, meanResidualPriorKL: 0})]);
  assert.match(summary, /training-only deadline pace loss 0\.0125/);
  assert.match(summary, /training-only residual prior KL 0\.0000/);
  assert.doesNotMatch(summary, /(?:enforced|promoted|verified)/);
  assert.match(Summary.describeDiagnostics([{meanDeadlinePaceLoss: null}]), /training-only deadline pace loss Not recorded/);
  assert.match(Summary.describeStatus(status({models: [model({history: [{meanDeadlinePaceLoss: 0, meanResidualPriorKL: 0}]})]})).improvement,
    /not verified/);
});

test('missing, malformed and older learner records have readable unknown labels', () => {
  for (const value of [undefined, null, {}, [], 'saved', false]) {
    const summary = Summary.describeStatus(value);
    assert.match(summary.availability, /Not recorded/);
    assert.match(summary.run, /Not recorded/);
    assert.match(summary.training, /Not recorded/);
    assert.match(summary.trainedSeeds, /Not recorded/);
    assert.match(summary.independentSeeds, /Not recorded/);
    assert.match(summary.scope, /Scope not recorded/);
    assert.match(summary.improvement, /not verified/);
  }
  const missing = Summary.describeModel();
  assert.match(missing.seed, /Not recorded/);
  assert.match(missing.progress, /Not recorded.*Not recorded/);
  assert.match(missing.training, /Not recorded/);
  assert.match(missing.actorWeights, /Not recorded/);
  assert.equal(missing.baseline, 'Not evaluated');
  assert.equal(missing.gate, 'Not evaluated for this checkpoint');
});

test('imitation phases and retained anchors are training evidence, never a promotion claim', () => {
  const imitation = Summary.describeDiagnostics([{phase: 'measured-controller-imitation', teacherQueries: 3, demonstrationAnchorRows: 16}]);
  assert.match(imitation, /training phase Measured controller imitation/);
  assert.match(imitation, /training teacher queries 3/);
  assert.match(imitation, /training-only anchor rows 16/);
  const ppo = Summary.describeDiagnostics([{phase: 'ppo', demonstrationAnchorRows: 512, meanMeasuredAnchorLoss: 0.0125}]);
  assert.match(ppo, /training phase PPO/);
  assert.match(ppo, /training-only demonstration anchor MSE 0\.0125/);
  assert.doesNotMatch(ppo, /(?:promoted|verified|enforced)/);
  assert.doesNotMatch(Summary.describeDiagnostics([{phase: '__proto__'}]), /training phase/);
});

test('raw pacing loss, weighted loss and coefficient are separate training-only diagnostics', () => {
  const summary = Summary.describeDiagnostics([{meanDeadlinePaceLoss: 0.025, meanWeightedDeadlinePaceLoss: 2.5, deadlinePaceLossWeight: 100}]);
  assert.match(summary, /training-only deadline pace loss 0\.0250/);
  assert.match(summary, /training-only weighted deadline pace loss 2\.5000/);
  assert.match(summary, /training-only deadline pace coefficient 100\.0000/);
  assert.doesNotMatch(summary, /(?:promoted|verified|enforced)/);
  const old = Summary.describeDiagnostics([{meanDeadlinePaceLoss: 0.025}]);
  assert.doesNotMatch(old, /weighted deadline|coefficient/);
});

test('precision anchor units are explicit training-only labels without promotion or action constraints', () => {
  const precision = Summary.describeDiagnostics([{meanMeasuredAnchorLoss: 0.1, demonstrationAnchorLossProfile: 'pose-budget-v2'}]);
  assert.match(precision, /training-only anchor units pose-budget-normalized MSE/);
  assert.doesNotMatch(precision, /(?:promoted|verified|enforced)/);
  const uniform = Summary.describeDiagnostics([{demonstrationAnchorLossProfile: 'uniform-action-v1'}]);
  assert.match(uniform, /training-only anchor units uniform signed action MSE/);
  assert.doesNotMatch(Summary.describeDiagnostics([{demonstrationAnchorLossProfile: '__proto__'}]), /anchor units/);
});

test('sampling concentration is a training-only distribution setting, not deterministic action clipping', () => {
  const summary = Summary.describeDiagnostics([{meanExplorationConcentration: 8}]);
  assert.match(summary, /training-only sampling concentration 8\.0000/);
  assert.doesNotMatch(summary, /(?:promoted|verified|clamped|enforced)/);
  assert.doesNotMatch(Summary.describeDiagnostics([{meanDeadlinePaceLoss: 0}]), /sampling concentration/);
});

test('unavailable or stale saved evidence retains its message and cannot report success', () => {
  const summary = Summary.describeStatus(status({available: false, learnedImprovementVerified: true,
    message: 'Saved learning run is stale'}));
  assert.equal(summary.message, 'Saved learning run is stale');
  assert.equal(summary.availability, 'Saved learner unavailable');
  assert.match(summary.improvement, /not verified/);
  assert.doesNotMatch(summary.improvement, /passed/);
});

test('all existing run statuses distinguish completion, budgets, failure and active training', () => {
  const labels = {running: /running.*not frozen/, completed: /completed/, 'budget-stopped': /budget/,
    'interrupted-or-failed': /interrupted or failed/};
  for (const [savedStatus, label] of Object.entries(labels)) {
    const summary = Summary.describeStatus(status({status: savedStatus}));
    assert.match(summary.run, label);
    assert.match(summary.improvement, /not verified/);
  }
  assert.match(Summary.describeStatus(status({status: 'future-state'})).run, /Unrecognized.*future-state/);
  assert.match(Summary.describeStatus(status({status: 'toString'})).run, /Unrecognized/);
});

test('counts and training flags do not coerce missing values into zero or truthy strings into training', () => {
  const summary = Summary.describeModel(model({seed: 0, steps: 0, updates: null, trained: 'true', actorWeightsChanged: 1}));
  assert.equal(summary.seed, '0');
  assert.match(summary.progress, /^0 \/ Not recorded/);
  assert.match(summary.training, /Not recorded/);
  assert.match(summary.actorWeights, /Not recorded/);
  for (const missing of [null, undefined, '2', -1, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    assert.match(Summary.describeModel(model({steps: missing})).progress, /^Not recorded/);
  }
  assert.equal(Summary.describeStatus(status()).trainedSeeds, '1 trained seeds recorded');
  assert.match(Summary.describeStatus(status({models: [{}]})).trainedSeeds, /Not recorded/);
  assert.match(Summary.describeStatus(status({models: null})).trainedSeeds, /Not recorded/);
  assert.match(Summary.describeModel(model({trained: false, actorWeightsChanged: false})).training, /Untrained/);
  assert.match(Summary.describeModel(model({actorWeightsChanged: false})).actorWeights, /unchanged/);
});

test('only a matching unambiguous checkpoint evaluation supplies model scores and gates', () => {
  const saved = model();
  for (const evaluation of [evaluatedModel(), {models: [evaluatedModel()]}]) {
    const summary = Summary.describeModel(saved, evaluation);
    assert.equal(summary.seed, '42');
    assert.equal(summary.progress, '64 / 2');
    assert.equal(summary.baseline, '6/24');
    assert.equal(summary.untrained, '7/24');
    assert.equal(summary.learned, '7/24');
    assert.equal(summary.gate, 'Failed');
  }
  for (const evaluation of [undefined, null, {}, {models: []}, {models: [null, {}]},
    evaluatedModel({policySha256: 'old-policy'}), evaluatedModel({seed: 43}),
    {models: [evaluatedModel(), evaluatedModel()]}]) {
    const summary = Summary.describeModel(saved, evaluation);
    assert.equal(summary.baseline, 'Not evaluated');
    assert.equal(summary.learned, 'Not evaluated');
    assert.equal(summary.gate, 'Not evaluated for this checkpoint');
  }
  for (const policySha256 of [null, undefined, '', ' ']) {
    assert.equal(Summary.describeModel(model({policySha256}), evaluatedModel({policySha256})).gate,
      'Not evaluated for this checkpoint');
  }
});

test('missing or malformed scores are not fabricated as successes', () => {
  for (const score of [undefined, null, {}, {success: null, count: 24}, {success: 0}]) {
    assert.match(Summary.describeModel(model(), evaluatedModel({learned: score})).learned, /Not recorded/);
  }
  for (const score of [{success: '7', count: 24}, {success: 25, count: 24}, {success: -1, count: 24},
    {success: 7, count: Infinity}, {success: NaN, count: 24}, {success: 0, count: 1.5}]) {
    assert.equal(Summary.describeModel(model(), evaluatedModel({learned: score})).learned, 'Invalid saved score');
  }
  assert.equal(Summary.describeModel(model(), evaluatedModel({learned: {success: 0, count: 24}})).learned, '0/24');
  assert.equal(Summary.describeModel(model(), evaluatedModel({learned: {success: 0, count: 0}})).learned,
    'No evaluated worlds recorded');
});

test('negative gates retain failure, regression counts, rules and signed efficiency fractions', () => {
  const summary = Summary.describeModel(model(), {models: [evaluatedModel()]});
  assert.equal(summary.gate, 'Failed');
  assert.match(summary.gateDetails, /Frozen all-world success, contact and efficiency gates/);
  assert.match(summary.gateDetails, /success regressions 1.*contact regressions 1/);
  assert.match(summary.gateDetails, /time improvement fraction -0\.1250/);
  assert.match(summary.gateDetails, /effort improvement fraction 0\.0125/);
  for (const passed of [undefined, null, 'true', 1]) {
    assert.match(Summary.describeModel(model(), evaluatedModel({acceptance: {passed}})).gate, /Not recorded/);
  }
  assert.match(Summary.describeModel(model(), evaluatedModel({acceptance: null})).gateDetails, /not recorded/);
});

test('missing and legacy histories never backfill diagnostics from older entries or invent zeros', () => {
  for (const history of [undefined, null, {}, [], [null], [{}], [{meanLoss: 0.25}],
    [diagnostics(), {meanLoss: 0.5}], [diagnostics(), null]]) {
    assert.equal(Summary.describeDiagnostics(history), 'Not recorded in this older run');
  }
});

test('diagnostics preserve recorded zero and negative entropy independently of missing KL', () => {
  const summary = Summary.describeDiagnostics([diagnostics({meanApproxKL: null, clipFraction: 0,
    meanActorLoss: 0, meanCriticLoss: 0, valueExplainedVariance: null, sampledActionBoundaryFraction: 0})]);
  assert.match(summary, /KL Not recorded/);
  assert.match(summary, /clip 0\.0000.*entropy -1\.1250.*actor 0\.0000.*critic 0\.0000/);
  assert.match(summary, /explained variance Not recorded/);
  assert.match(summary, /sampled action boundary 0\.0000/);
  assert.doesNotMatch(summary, /NaN|Infinity/);
  for (const missing of [undefined, null, '0', false, NaN, Infinity, -Infinity]) {
    assert.match(Summary.describeDiagnostics([{meanApproxKL: 0, clipFraction: missing}]), /clip Not recorded/);
  }
  assert.match(Summary.describeDiagnostics([{meanEntropy: -1}]), /KL Not recorded.*entropy -1\.0000/);
});

test('sampled actions and numbered curriculum stages retain provenance and explicit missing values', () => {
  const summary = Summary.describeDiagnostics([diagnostics()]);
  assert.match(summary, /sampled action mean 0\.7500\/-0\.5000\/0\.2500\/-0\.1250/);
  assert.match(summary, /sampled action std 0\.1250\/0\.2500\/0\.3750\/0\.5000/);
  assert.match(summary, /sampled action boundary 0\.3750/);
  assert.match(summary, /world stages 0: 1 \/ 1: 2 \/ 2: 3/);
  assert.doesNotMatch(summary, /applied|executed/);
  const partial = Summary.describeDiagnostics([diagnostics({sampledActionStd: [0, null, 0.2, Infinity],
    curriculumStageCounts: {'2': 0, '0': null}})]);
  assert.match(partial, /sampled action std 0\.0000\/Not recorded\/0\.2000\/Not recorded/);
  assert.match(partial, /world stages 0: Not recorded.*1: Not recorded.*2: 0/);
  for (const vector of [null, {}, [], [0, 0, 0], [0, 0, 0, 0, 0]]) {
    assert.match(Summary.describeDiagnostics([diagnostics({sampledActionStd: vector})]), /sampled action std Not recorded/);
  }
});

test('training metrics, a passing seed or an evaluation flag alone never verify or promote a policy', () => {
  const favorable = model({history: [diagnostics({trainingSuccessRate: 1, meanTrainingReward: 10})]});
  for (const verified of [false, undefined, null, 'true', 1]) {
    const summary = Summary.describeStatus(status({learnedImprovementVerified: verified, models: [favorable],
      evaluation: {learnedImprovementVerified: true, models: [evaluatedModel({acceptance: {passed: true}})]}}));
    assert.match(summary.improvement, /not verified/);
    assert.doesNotMatch(summary.improvement, /passed/);
  }
  assert.match(Summary.describeModel(favorable, evaluatedModel({acceptance: {passed: true}})).gate,
    /Pass reported; this seed alone does not verify improvement/);
});

test('explicit verification remains a scoped report and contradictory negative gates remain negative', () => {
  const reported = Summary.describeStatus(status({learnedImprovementVerified: true,
    evaluation: {learnedImprovementVerified: true, independentTrainingSeedsRequirementMet: true}}));
  assert.match(reported.improvement, /Frozen improvement gate reported passed.*no automatic policy promotion/);
  assert.match(reported.independentSeeds, /reported met/);
  assert.match(reported.scope, /no policy promotion.*hardware or AMD validation is inferred/);
  for (const evaluation of [{learnedImprovementVerified: false}, {independentTrainingSeedsRequirementMet: false},
    {models: [evaluatedModel()]}]) {
    const summary = Summary.describeStatus(status({learnedImprovementVerified: true, evaluation}));
    assert.match(summary.improvement, /not verified.*negative gate/);
    assert.doesNotMatch(summary.improvement, /passed/);
  }
  assert.match(Summary.describeStatus(status({status: 'running', learnedImprovementVerified: true})).improvement,
    /not verified while training is running/);
  assert.match(Summary.describeStatus(status({status: null, learnedImprovementVerified: true})).improvement,
    /frozen run status is not recorded/);
});

test('training and evaluation scopes are kept distinct and saved development caveats survive', () => {
  const evaluation = {scope: 'Development-only; never a fresh held-out improvement claim', models: [evaluatedModel()]};
  const summary = Summary.describeStatus(status({evaluation}));
  assert.match(summary.scope, /Canonical reach only; no obstacle or game-policy training/);
  assert.doesNotMatch(summary.scope, /Development-only/);
  assert.match(summary.evaluationScope, /Development-only; never a fresh held-out improvement claim/);
  assert.match(Summary.describeModel(model(), evaluation).scope, /Development-only/);
  assert.match(Summary.describeModel(model(), {models: [evaluatedModel()],
    interpretation: 'Frozen software evaluation only; negative results retained'}).scope, /negative results retained/);
});

test('presentation is deterministic, does not mutate frozen inputs and never retains previous evidence', () => {
  const freeze = value => {
    if (value && typeof value === 'object') {
      for (const child of Object.values(value)) freeze(child);
      Object.freeze(value);
    }
    return value;
  };
  const saved = freeze(status({evaluation: {models: [evaluatedModel()]}}));
  const before = JSON.stringify(saved);
  const first = Summary.describeModel(saved.models[0], saved.evaluation);
  assert.deepEqual(first, Summary.describeModel(saved.models[0], saved.evaluation));
  assert.deepEqual(Summary.describeStatus(saved), Summary.describeStatus(saved));
  first.learned = 'changed by caller';
  assert.equal(Summary.describeModel(saved.models[0], saved.evaluation).learned, '7/24');
  const older = Summary.describeModel(model({history: undefined}));
  assert.equal(older.diagnostics, 'Not recorded in this older run');
  assert.equal(older.learned, 'Not evaluated');
  assert.equal(JSON.stringify(saved), before);
});
