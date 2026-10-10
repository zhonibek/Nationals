(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else if (typeof define === 'function' && define.amd) define([], factory);
  else root.MotionLearningSummary = factory();
})(globalThis, function() {
  'use strict';

  const NOT_RECORDED = 'Not recorded in this older run';
  const SCOPE_LIMIT = 'Software evidence only; no policy promotion, full-game, hardware or AMD validation is inferred';
  const DIAGNOSTICS = [['KL', 'meanApproxKL'], ['clip', 'clipFraction'], ['entropy', 'meanEntropy'],
    ['actor', 'meanActorLoss'], ['critic', 'meanCriticLoss'], ['explained variance', 'valueExplainedVariance']];
  const TRAINING_PRIORS = [['deadline pace loss', 'meanDeadlinePaceLoss'], ['weighted deadline pace loss', 'meanWeightedDeadlinePaceLoss'],
    ['deadline pace coefficient', 'deadlinePaceLossWeight'], ['sampling concentration', 'meanExplorationConcentration'],
    ['residual prior KL', 'meanResidualPriorKL'],
    ['demonstration anchor MSE', 'meanMeasuredAnchorLoss']];
  const TRAINING_PHASES = {ppo: 'PPO', 'measured-controller-imitation': 'Measured controller imitation'};
  const ANCHOR_PROFILES = {'uniform-action-v1': 'uniform signed action MSE', 'pose-budget-v2': 'pose-budget-normalized MSE'};
  const RUN_LABELS = {running: 'Training running; results are not frozen', completed: 'Training completed',
    'budget-stopped': 'Training stopped at its budget', 'interrupted-or-failed': 'Training interrupted or failed'};

  const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const recordedText = value => typeof value === 'string' && value.trim() ? value.trim() : null;
  const validCount = value => Number.isSafeInteger(value) && value >= 0;
  const count = value => validCount(value) ? String(value) : NOT_RECORDED;
  const number = value => Number.isFinite(value) ? value.toFixed(4) : 'Not recorded';
  const flag = (value, positive, negative) => value === true ? positive : value === false ? negative : NOT_RECORDED;

  function describeScope(evidence) {
    const saved = record(evidence) ? recordedText(evidence.scope) || recordedText(evidence.interpretation) : null;
    return `${saved || 'Scope not recorded in this older run'} · ${SCOPE_LIMIT}`;
  }

  function describeDiagnostics(history) {
    const latest = Array.isArray(history) && history.length ? history[history.length - 1] : null;
    const fields = [...DIAGNOSTICS.map(entry => entry[1]), ...TRAINING_PRIORS.map(entry => entry[1]), 'sampledActionMean', 'sampledActionStd',
      'sampledActionBoundaryFraction', 'curriculumStageCounts', 'phase', 'teacherQueries', 'demonstrationAnchorRows', 'demonstrationAnchorLossProfile'];
    if (!record(latest) || !fields.some(field => Object.hasOwn(latest, field))) return NOT_RECORDED;
    const vector = value => Array.isArray(value) && value.length === 4
      ? Array.from(value, number).join('/') : 'Not recorded';
    const stages = record(latest.curriculumStageCounts) ? latest.curriculumStageCounts : {};
    return [...(Object.hasOwn(TRAINING_PHASES, latest.phase) ? [`training phase ${TRAINING_PHASES[latest.phase]}`] : []),
      ...(Object.hasOwn(latest, 'teacherQueries') ? [`training teacher queries ${count(latest.teacherQueries)}`] : []),
      ...(Object.hasOwn(latest, 'demonstrationAnchorRows') ? [`training-only anchor rows ${count(latest.demonstrationAnchorRows)}`] : []),
      ...(Object.hasOwn(ANCHOR_PROFILES, latest.demonstrationAnchorLossProfile) ? [`training-only anchor units ${ANCHOR_PROFILES[latest.demonstrationAnchorLossProfile]}`] : []),
      ...DIAGNOSTICS.map(([label, field]) => `${label} ${number(latest[field])}`),
      ...TRAINING_PRIORS.filter(([, field]) => Object.hasOwn(latest, field)).map(([label, field]) => `training-only ${label} ${number(latest[field])}`),
      `sampled action mean ${vector(latest.sampledActionMean)}`,
      `sampled action std ${vector(latest.sampledActionStd)}`,
      `sampled action boundary ${number(latest.sampledActionBoundaryFraction)}`,
      `world stages ${['0', '1', '2'].map(stage => `${stage}: ${count(stages[stage])}`).join(' / ')}`].join(' · ');
  }

  function matchEvaluation(model, evaluation) {
    if (!recordedText(model.policySha256) || !record(evaluation)) return null;
    const rows = Array.isArray(evaluation.models) ? evaluation.models : [evaluation];
    const matches = rows.filter(row => record(row) && recordedText(row.policySha256) &&
      row.policySha256 === model.policySha256 &&
      (!validCount(row.seed) || !validCount(model.seed) || row.seed === model.seed));
    return matches.length === 1 ? matches[0] : null;
  }

  function describeScore(value) {
    if (!record(value) || value.success == null || value.count == null) return 'Not recorded in this evaluation';
    if (!validCount(value.success) || !validCount(value.count) || value.success > value.count) return 'Invalid saved score';
    return value.count === 0 ? 'No evaluated worlds recorded' : `${value.success}/${value.count}`;
  }

  function describeGateDetails(acceptance) {
    if (!record(acceptance)) return 'Gate details not recorded in this older evaluation';
    const regressions = value => Array.isArray(value) ? String(value.length) : 'Not recorded';
    return [recordedText(acceptance.rule) || 'Gate rule not recorded',
      `success regressions ${regressions(acceptance.regressions)}`,
      `contact regressions ${regressions(acceptance.contactRegressions)}`,
      `mean time improvement fraction ${number(acceptance.meanTimeImprovementFraction)}`,
      `mean effort improvement fraction ${number(acceptance.meanEffortImprovementFraction)}`].join(' · ');
  }

  function describeModel(model, evaluation) {
    const saved = record(model) ? model : {};
    const evaluated = matchEvaluation(saved, evaluation);
    const acceptance = record(evaluated?.acceptance) ? evaluated.acceptance : {};
    return {seed: count(saved.seed), progress: `${count(saved.steps)} / ${count(saved.updates)}`,
      training: flag(saved.trained, 'Trained checkpoint recorded; not proof of improvement', 'Untrained checkpoint recorded'),
      actorWeights: flag(saved.actorWeightsChanged, 'Actor weights changed as recorded', 'Actor weights unchanged as recorded'),
      baseline: evaluated ? describeScore(evaluated.baseline) : 'Not evaluated',
      untrained: evaluated ? describeScore(evaluated.untrained) : 'Not evaluated',
      learned: evaluated ? describeScore(evaluated.learned) : 'Not evaluated',
      gate: evaluated ? flag(acceptance.passed, 'Pass reported; this seed alone does not verify improvement', 'Failed')
        : 'Not evaluated for this checkpoint',
      gateDetails: evaluated ? describeGateDetails(evaluated.acceptance) : 'No matching checkpoint evaluation recorded',
      diagnostics: describeDiagnostics(saved.history), scope: describeScope(evaluated ? evaluation : null)};
  }

  function describeStatus(status) {
    const saved = record(status) ? status : {};
    const evaluation = record(saved.evaluation) ? saved.evaluation : {};
    const models = Array.isArray(saved.models) ? saved.models : null;
    const evaluatedModels = Array.isArray(evaluation.models) ? evaluation.models : [];
    const negativeGate = evaluation.learnedImprovementVerified === false ||
      evaluation.independentTrainingSeedsRequirementMet === false ||
      evaluatedModels.some(model => record(model) && model.acceptance?.passed === false);
    let improvement = 'Improvement not verified; baseline remains the default';
    if (saved.available === true && saved.learnedImprovementVerified === true) {
      if (saved.status === 'running') improvement = 'Improvement not verified while training is running';
      else if (negativeGate) improvement = 'Improvement not verified; saved evaluation reports a negative gate';
      else if (!['completed', 'budget-stopped', 'interrupted-or-failed'].includes(saved.status)) {
        improvement = 'Verification reported, but frozen run status is not recorded';
      } else improvement = 'Frozen improvement gate reported passed; no automatic policy promotion';
    }
    return {availability: flag(saved.available, 'Saved learner available', 'Saved learner unavailable'),
      run: Object.hasOwn(RUN_LABELS, saved.status) ? RUN_LABELS[saved.status]
        : recordedText(saved.status) ? `Unrecognized saved run status: ${saved.status}` : NOT_RECORDED,
      training: flag(saved.motionPolicyTrained, 'Trained policy reported; improvement is a separate gate', 'No trained policy reported'),
      trainedSeeds: models && models.every(model => record(model) && typeof model.trained === 'boolean')
        ? `${models.filter(model => model.trained === true).length} trained seeds recorded` : NOT_RECORDED,
      improvement,
      independentSeeds: flag(evaluation.independentTrainingSeedsRequirementMet,
        'Independent training-seed requirement reported met', 'Independent training-seed requirement not met'),
      scope: describeScope(saved), evaluationScope: describeScope(saved.evaluation),
      message: recordedText(saved.message) || 'No saved learner message recorded'};
  }

  return {describeStatus, describeModel, describeDiagnostics};
});
