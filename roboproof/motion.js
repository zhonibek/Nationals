'use strict';

const assert = require('node:assert/strict');
const {loadHeadless} = require('../simulator/headless');
const {SCHEMA_VERSION, FIXED_DT, DEFAULT_TASK, normalizeEpisodeOptions} = require('../simulator/motion');
const Policy = require('../simulator/policy');
const GvfPolicy = require('../simulator/gvf-policy');

function baseline(seed = 42, task = DEFAULT_TASK, configuration = {}, options = {}) {
  const runner = loadHeadless();
  const episode = runner.createEpisode();
  episode.reset(seed, configuration, task, {...options, controlMode: 'scripted',
    policyIdentity: {kind: 'scripted', id: 'controller-only-baseline'}});
  episode.applyAction({type: 'pose', ...task.goal});
  while (!episode.terminated && !episode.truncated) episode.step();
  return {...episode.report(), identity: runner.identity, agent: 'existing scripted pose -> C++ LTV-LQR -> wheel PI/feedforward'};
}

function runPolicy({seed = 42, task = DEFAULT_TASK, configuration = {}, options = {},
  act = () => [...Policy.DEFAULT_ACTION]} = {}) {
  if (typeof act !== 'function') throw Error('A synchronous policy function is required');
  const runner = loadHeadless();
  const episode = runner.createEpisode();
  episode.reset(seed, configuration, task, {...options, controlMode: 'policy'});
  if (typeof act.reset === 'function') act.reset({referenceDurationSeconds: episode.referenceAdapter.duration});
  while (!episode.terminated && !episode.truncated) {
    const observation = episode.observe();
    episode.policyStep(act(Policy.vector(observation), structuredClone(observation)));
  }
  return {...episode.report(), identity: runner.identity, agent: 'bounded reference policy -> persistent C++ LTV-LQR -> wheel PI/feedforward'};
}

function runGvf(seed = 42, task = DEFAULT_TASK, configuration = {}, options = {}) {
  const requested = Policy.options(options);
  if (requested.policyIdentity.kind === 'learned') throw Error('Existing learned actions are not a GVF policy contract');
  const runner = loadHeadless();
  const episode = runner.createEpisode();
  episode.reset(seed, configuration, task, {...requested, controlMode: 'gvf',
    policyIdentity: {kind: 'scripted', id: 'pedro-gvf-v2-experimental'}});
  episode.applyAction({type: 'pose', ...task.goal});
  while (!episode.terminated && !episode.truncated) episode.step();
  return {...episode.report(), identity: runner.identity,
    agent: 'experimental Pedro GVF v2 -> existing C++ LTV-LQR -> wheel PI/feedforward; not a learned policy'};
}

function runGvfPolicy({seed = 42, task = DEFAULT_TASK, configuration = {}, options = {},
  act = () => [...GvfPolicy.ZERO_ACTION]} = {}) {
  if (typeof act !== 'function') throw Error('A synchronous GVF residual policy function is required');
  const requested = GvfPolicy.options({controlMode: GvfPolicy.CONTRACT_NAME, policyContractName: GvfPolicy.CONTRACT_NAME,
    policyContractVersion: GvfPolicy.CONTRACT_VERSION, ...options});
  const runner = loadHeadless();
  const episode = runner.createEpisode();
  episode.reset(seed, configuration, task, requested);
  if (typeof act.reset === 'function') act.reset({contractName: GvfPolicy.CONTRACT_NAME, contractVersion: GvfPolicy.CONTRACT_VERSION});
  while (!episode.terminated && !episode.truncated) {
    const observation = episode.observe();
    episode.gvfResidualStep(act(GvfPolicy.vector(observation), structuredClone(observation)));
  }
  return {...episode.report(), identity: runner.identity,
    agent: 'bounded GVF residual policy -> persistent native GVF/LTV-LQR -> wheel PI/feedforward; qualification required'};
}

function runGvfZeroResidual(seed = 42, task = DEFAULT_TASK, configuration = {}, options = {}) {
  if (options.policyIdentity?.kind === 'learned') throw Error('Zero residual is not a learned policy');
  return runGvfPolicy({seed, task, configuration, options: {...options,
    policyIdentity: {kind: 'scripted', id: 'persistent-gvf-zero-residual-v2'}}, act: () => [...GvfPolicy.ZERO_ACTION]});
}

function replay(report) {
  if (report?.schemaVersion !== SCHEMA_VERSION || report.taskType !== 'reach-pose-benchmark' || report.fixedDt !== FIXED_DT) throw Error('Unsupported motion replay schema');
  if (!Number.isInteger(report.ticks) || report.ticks < 0 || report.ticks > 6000 || !Array.isArray(report.actions) || report.actions.length > 6001) throw Error('Invalid bounded motion replay');
  const options = normalizeEpisodeOptions(report.options);
  if (!Array.isArray(report.transitions) || report.transitions.length !== (options.recordTransitions ? report.ticks : 0)) throw Error('Invalid transition log length');
  const residual = options.controlMode === GvfPolicy.CONTRACT_NAME;
  if (report.policyContractVersion !== (residual ? GvfPolicy.CONTRACT_VERSION : Policy.CONTRACT_VERSION) ||
    (residual ? report.policyContractName !== GvfPolicy.CONTRACT_NAME : Object.hasOwn(report, 'policyContractName'))) throw Error('Unsupported policy contract');
  let previousTick = -1;
  for (const entry of report.actions) {
    if (!Number.isInteger(entry.tick) || entry.tick < previousTick || entry.tick < 0 || entry.tick > report.ticks) throw Error('Invalid action tick');
    previousTick = entry.tick;
  }
  const runner = loadHeadless();
  assert.deepEqual(report.identity, runner.identity, 'Motion replay engine/controller identity mismatch');
  const episode = runner.createEpisode();
  episode.reset(report.seed, report.configuration, report.task, options);
  let actionIndex = 0;
  for (let tick = 0; tick < report.ticks; tick++) {
    while (actionIndex < report.actions.length && report.actions[actionIndex].tick === tick) episode.applyAction(report.actions[actionIndex++].action);
    episode.step();
  }
  while (actionIndex < report.actions.length && report.actions[actionIndex].tick === report.ticks) episode.applyAction(report.actions[actionIndex++].action);
  const actual = {...episode.report(), identity: runner.identity, agent: report.agent};
  assert.deepEqual(actual, report, 'Recorded motion does not reproduce exactly in this runtime');
  return actual;
}

module.exports = {baseline, runPolicy, runGvf, runGvfPolicy, runGvfZeroResidual, replay};
