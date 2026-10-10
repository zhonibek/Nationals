'use strict';

const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {baseline, runPolicy} = require('./motion');
const {assertTrainingWorld, aggregate} = require('./motion-evaluation');
const {loadHeadless} = require('../simulator/headless');
const Policy = require('../simulator/policy');
const hash = value => createHash('sha256').update(value).digest('hex');
const SPECIFICATION = 'mixed-reach-development-v1';

function worlds(seed = 730139, families = 3, profile = 'short-reach') {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff || !Number.isInteger(families) || families < 1 || families > 32) throw Error('Bounded development seed/families required');
  if (!['short-reach', 'field-reach', 'full-contract'].includes(profile)) throw Error('Unsupported development profile');
  let state = seed >>> 0;
  const uniform = (minimum, maximum) => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return minimum + (maximum - minimum) * state / 4294967296;
  };
  const cases = [];
  for (let family = 0; family < families; family++) {
    const limit = profile === 'full-contract' ? 50 : profile === 'field-reach' ? 40 : 12;
    const start = {xIn: uniform(-limit, limit), yIn: uniform(-limit, limit), headingDeg: uniform(-45, 45)};
    const direction = uniform(-Math.PI, Math.PI), distance = uniform(12, 40);
    const goal = profile === 'full-contract' ? {xIn: uniform(-60, 60), yIn: uniform(-60, 60), headingDeg: uniform(-180, 180)}
      : profile === 'field-reach' ? {xIn: uniform(-45, 45), yIn: uniform(-45, 45), headingDeg: uniform(-175, 175)}
      : {xIn: start.xIn + Math.sin(direction) * distance, yIn: start.yIn + Math.cos(direction) * distance, headingDeg: uniform(-135, 135)};
    const configurations = [{}, {massKg: [10, 13.6], moiKgM2: [0.35, 0.58], muLong: [0.3, 0.55], kSlip: [325, 500]},
      {massKg: [8, 12], muLong: [0.4, 0.8], batteryInternalR: [0.04, 0.12]},
      {massKg: [4, 12], moiKgM2: [0.08, 0.55], muLong: [0.4, 0.95], kSlip: [350, 850], batteryInternalR: [0, 0.12]}];
    if (profile === 'full-contract') {
      configurations[3] = {massKg: [3.4, 13.6], moiKgM2: [0.0725, 0.58], muLong: [0.3, 1.1], muLat: [0.02, 0.15],
        kSlip: [325, 975], batteryInternalR: [0, 0.12]};
      if (family % 2 === 0) {
        const axis = family % 4 === 0 ? 'xIn' : 'yIn';
        goal[axis] = uniform(0, 1) < 0.5 ? -60 : 60;
        goal.headingDeg = [0, 90, 180][Math.floor(uniform(0, 3))];
      }
    }
    for (let variation = 0; variation < configurations.length; variation++) {
      const world = {id: `development-${seed}-${family}-${variation}`, group: `development-${seed}-${family}`,
        seed: Math.floor(uniform(0, 4294967296)), task: {start: {...start}, goal: {...goal}, deadlineSeconds: variation === 2 ? 6 : 10},
        configuration: configurations[variation]};
      cases.push(assertTrainingWorld(world));
    }
  }
  return {schemaVersion: 1, specification: SPECIFICATION, profile, scope: 'Development-only paired reach tasks; never a fresh held-out improvement claim',
    generationSeed: seed, cases, corpusSha256: hash(JSON.stringify(cases))};
}

const paceAction = pace => 2 * (pace - Policy.LIMITS.minimumPace) / (Policy.LIMITS.maximumPace - Policy.LIMITS.minimumPace) - 1;
const STRATEGIES = Object.freeze(['controller', 'nominal-reference', 'pace-0.4', 'pace-0.55', 'pace-0.7', 'pace-0.85',
  'ease-pace', 'turn-pace', 'heading-lead', 'heading-lag', 'lead-0.15', 'lead-0.3', 'lead-0.6']);

function strategy(name) {
  if (!STRATEGIES.includes(name) || name === 'controller') throw Error('Unsupported development reference strategy');
  if (name === 'nominal-reference') return () => [...Policy.DEFAULT_ACTION];
  if (name.startsWith('pace-')) return () => [paceAction(Number(name.slice(5))), 0, 0, 0];
  if (name === 'ease-pace') return observation => [paceAction(observation[25] < 0.7 ? 0.4 : 1), 0, 0, 0];
  if (name === 'turn-pace') return observation => [paceAction(Math.abs(observation[5]) > 0.6 ? 0.55 : 0.9), 0, 0, 0];
  if (name === 'heading-lead' || name === 'heading-lag') {
    let lead = null;
    return observation => {
      lead ??= Math.max(-1, Math.min(1, Math.atan2(observation[12], observation[13]) / 3));
      return [1, 0, 0, name === 'heading-lag' ? -lead : lead];
    };
  }
  const lead = Number(name.slice(5));
  let direction = null;
  return observation => {
    if (!direction) {
      const length = Math.hypot(observation[10], observation[11]);
      const initialHeading = observation[2];
      direction = length > 1e-9 ? [(observation[10] * Math.cos(initialHeading) + observation[11] * Math.sin(initialHeading)) / length,
        (-observation[10] * Math.sin(initialHeading) + observation[11] * Math.cos(initialHeading)) / length] : [0, 0];
    }
    const heading = observation[21];
    return [1, lead * (direction[0] * Math.cos(heading) - direction[1] * Math.sin(heading)),
      lead * (direction[0] * Math.sin(heading) + direction[1] * Math.cos(heading)), 0];
  };
}

function probe(corpus, {names = STRATEGIES, maximumSeconds = 90, maximumRuns = 256} = {}) {
  if (corpus?.specification !== SPECIFICATION || !Array.isArray(corpus.cases) || !corpus.cases.length || corpus.cases.length > 128 ||
      corpus.corpusSha256 !== hash(JSON.stringify(corpus.cases))) throw Error('Invalid development corpus');
  if (!Array.isArray(names) || !names.length || names[0] !== 'controller' || new Set(names).size !== names.length ||
      names.some(name => !STRATEGIES.includes(name)) || !Number.isFinite(maximumSeconds) || maximumSeconds < 1 || maximumSeconds > 300 ||
      !Number.isInteger(maximumRuns) || maximumRuns < 1 || maximumRuns > 1024) throw Error('Invalid bounded development probe');
  for (const world of corpus.cases) assertTrainingWorld(world);
  const started = performance.now(), deadline = started + maximumSeconds * 1000;
  const comparisons = [];
  const reports = new Map(names.map(name => [name, []]));
  let completedRuns = 0;
  for (const world of corpus.cases) {
    if (completedRuns + names.length > maximumRuns || performance.now() >= deadline) break;
    const measured = {};
    for (const name of names) {
      const report = name === 'controller' ? baseline(world.seed, world.task, world.configuration)
        : runPolicy({seed: world.seed, task: world.task, configuration: world.configuration,
          options: {recordTransitions: false, policyIdentity: {kind: 'scripted', id: `development-${name}`}}, act: strategy(name)});
      measured[name] = {reason: report.reason, metrics: report.metrics, configuration: report.configuration};
      reports.get(name).push(report);
      completedRuns++;
    }
    const baselineResult = measured.controller;
    const successful = names.filter(name => measured[name].reason === 'success');
    comparisons.push({id: world.id, group: world.group, seed: world.seed, task: world.task, results: measured,
      alternativeSuccesses: successful.filter(name => name !== 'controller'),
      baselineFailureRecoveredByAnyAlternative: baselineResult.reason !== 'success' && successful.length > 0});
  }
  return {schemaVersion: 1, specification: SPECIFICATION, profile: corpus.profile ?? 'short-reach', evaluatedAt: new Date().toISOString(), corpusSha256: corpus.corpusSha256,
    identity: loadHeadless().identity, policyContractVersion: Policy.CONTRACT_VERSION,
    sourceSha256: hash(fs.readFileSync(__filename, 'utf8').replace(/\r\n/g, '\n')), completedRuns, completedWorlds: comparisons.length,
    requestedWorlds: corpus.cases.length, budgetStopped: comparisons.length !== corpus.cases.length,
    wallSeconds: (performance.now() - started) / 1000,
    strategies: Object.fromEntries([...reports].map(([name, values]) => [name, values.length ? aggregate(values) : null])),
    postHocRecoverableFailures: comparisons.filter(row => row.baselineFailureRecoveredByAnyAlternative).map(row => row.id),
    results: comparisons, learnedImprovementVerified: false,
    interpretation: 'Development-only scripted feasibility probe. Post-hoc best-per-world choices are privileged analysis, not deployable or learned policy, and not final evaluation.'};
}

if (require.main === module) {
  const {writeJson} = require('./core');
  const directory = path.join(__dirname, 'runs/motion/development', `probe-${Date.now()}`);
  const profile = process.argv[2] || 'short-reach';
  const corpus = worlds(730139, profile === 'field-reach' ? 4 : 3, profile);
  writeJson(path.join(directory, 'worlds.json'), corpus);
  const result = probe(corpus);
  writeJson(path.join(directory, 'probe.json'), result);
  console.log(JSON.stringify({directory, completedRuns: result.completedRuns, strategies: result.strategies,
    postHocRecoverableFailures: result.postHocRecoverableFailures, interpretation: result.interpretation}, null, 2));
}

module.exports = {worlds, strategy, probe, STRATEGIES, SPECIFICATION};
