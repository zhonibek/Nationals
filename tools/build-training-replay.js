'use strict';

// Records one full match of an exported tactical policy for the training page (inference only, nothing trains):
// every decision's candidates and scores, the chosen move's inputs and its layer-by-layer neuron trace, the score
// after the move, and the loader-cycle tactic on the same start pose for comparison.
//
//   node tools/build-training-replay.js <policy.json> <seed> <red|blue> [output.json]

//   node tools/build-training-replay.js <v4 policy.json> <seed> <red|blue> <output.json> --task "<typed task>"
// records a multi-task policy carrying out one typed task, against the per-task greedy tactic on the same task.

const fs = require('node:fs');
const path = require('node:path');
const {TacticalEnvironment} = require('../roboproof/tactical_environment');
const {TaskEnvironment} = require('../roboproof/tactical_task_environment');
const {parseTask} = require('../roboproof/tactical_task');
const {chooseForTask} = require('../roboproof/tactical_task_training');
const {loadPolicy} = require('../roboproof/tactical_policy');
const {playMatch, policyChooser} = require('../roboproof/tactical_baselines');

const BASE_FEATURES = ['score Pin', 'score Cup', 'Toggle', 'park', 'alliance side', 'red halves', 'blue halves',
  'yellow halves', 'face up', 'distance to object', 'distance to goal', 'travel time', 'time left', 'score margin',
  'holding Pin', 'holding Cup', 'goal stack height', 'own goal', 'Toggle state', 'goal x', 'goal y', 'endgame',
  'from loader'];
const PREVIEW_FEATURES = ['legal', 'previewed points', 'previewed seconds', 'points per second', 'rank', 'best move',
  'best loader move'];
const TASK_FEATURES = ['task: play', 'task: score', 'task: park', 'objects: any', 'objects: Pin', 'objects: Cup',
  'source: any', 'source: loader', 'source: field', 'goals: any', 'goals: neutral', 'goals: alliance', 'goals: tall',
  'has count target', 'count left', 'has points target', 'points left', 'has time limit', 'task time left',
  'park afterwards', 'parking phase', 'robots parked'];
const TOP_CANDIDATES = 8;
const round = value => Number(value.toPrecision(4));

function kindOf(id) { return id.split(':')[0]; }

async function record(policyFile, seed, alliance) {
  const policy = loadPolicy(policyFile);
  if (policy.featureSet !== 'v3') throw Error('The match replay records a single-goal v3 policy');
  const environment = new TacticalEnvironment({featureSet: 'v3'});
  let observation = environment.reset(seed, alliance);
  const decisions = [];
  const started = Date.now();
  while (!observation.done && observation.actions.length && decisions.length < 100) {
    const scored = observation.actions.map(action => ({action, score: policy.score(action.features)}))
      .sort((first, second) => second.score - first.score);
    const chosen = scored[0];
    const trace = policy.trace(chosen.action.features, 64);
    const clock = observation.clock;
    const result = environment.step(chosen.action.id);
    decisions.push({decision: decisions.length + 1, clock: round(clock), id: chosen.action.id, kind: kindOf(chosen.action.id),
      score: round(chosen.score), candidateCount: scored.length, points: result.reward, seconds: round(result.durationSeconds),
      scoreAfter: result.score, rejected: Boolean(result.rejectedAction),
      candidates: scored.slice(0, TOP_CANDIDATES).map(row => ({id: row.action.id, kind: kindOf(row.action.id), score: round(row.score)})),
      inputs: chosen.action.features.map(round),
      layers: trace.layers.map(layer => ({cells: layer.cells.map(round), contribution: layer.contribution.map(round),
        mean: round(layer.mean), spread: round(layer.spread), peakNeuron: layer.peakNeuron, peakValue: round(layer.peakValue)}))});
    observation = result.observation;
  }
  const baseline = await playMatch({seed, alliance, chooser: policyChooser('loader-cycle', seed)});
  return {schemaVersion: 1, kind: 'tactical-training-replay', seed, alliance, featureNames: [...BASE_FEATURES, ...PREVIEW_FEATURES],
    policy: {width: policy.width, hiddenLayers: policy.hiddenLayers, featureSet: policy.featureSet, algorithm: policy.algorithm,
      provenance: policy.provenance, independentHoldout: policy.independentHoldout},
    finalScore: environment.score(), decisions, inferenceSeconds: (Date.now() - started) / 1000,
    baseline: {policy: 'loader-cycle', finalScore: baseline.finalScore,
      timeline: baseline.log.map(row => ({clock: row.clock, kind: row.kind, points: row.points, scoreAfter: row.score}))},
    scope: 'Abstract Override macro-action model scored by the rules engine; not physical motion.'};
}

function traceRow(policy, chosen, scored) {
  const trace = policy.trace(chosen.action.features, 64);
  return {score: round(chosen.score), candidateCount: scored.length,
    candidates: scored.slice(0, TOP_CANDIDATES).map(row => ({id: row.action.id, kind: kindOf(row.action.id), score: round(row.score)})),
    inputs: chosen.action.features.map(round),
    layers: trace.layers.map(layer => ({cells: layer.cells.map(round), contribution: layer.contribution.map(round),
      mean: round(layer.mean), spread: round(layer.spread), peakNeuron: layer.peakNeuron, peakValue: round(layer.peakValue)}))};
}

// A multi-task policy carrying out one typed task through the app's TaskEnvironment, and task-greedy on the same task.
function recordTask(policyFile, seed, alliance, text) {
  const policy = loadPolicy(policyFile);
  if (policy.featureSet !== 'v4') throw Error('A typed-task replay needs a multi-task v4 policy');
  const spec = parseTask(text);
  const task = new TaskEnvironment({spec, environment: new TacticalEnvironment({featureSet: 'v3'}), featureSet: 'v4'});
  let observation = task.reset(seed, alliance);
  const decisions = [];
  const started = Date.now();
  while (!observation.done) {
    const scored = observation.actions.map(action => ({action, score: policy.score(action.features)}))
      .sort((first, second) => second.score - first.score);
    const chosen = scored[0];
    const clock = observation.clock;
    const {row, observation: next} = task.step(chosen.action.id);
    decisions.push({decision: row.decision, clock: round(clock), id: row.id, kind: row.kind, phase: row.phase, points: row.points,
      ownPoints: row.ownPoints, taskReward: row.taskReward, seconds: row.seconds, scoreAfter: row.score, rejected: row.rejected,
      ...traceRow(policy, chosen, scored)});
    observation = next;
  }
  const result = task.summary();
  const greedy = new TaskEnvironment({spec, environment: new TacticalEnvironment({featureSet: 'v3'}), featureSet: 'v4'});
  let greedyObservation = greedy.reset(seed, alliance);
  while (!greedyObservation.done) {
    const pick = chooseForTask('task-greedy', greedy, greedy.previewAllowed(), null);
    greedyObservation = greedy.step(pick ? pick.id : greedyObservation.actions[0].id).observation;
  }
  const baseline = greedy.summary();
  return {schemaVersion: 1, kind: 'tactical-training-replay', mode: 'task', task: {text, summary: spec.summary},
    seed, alliance, featureNames: [...BASE_FEATURES, ...PREVIEW_FEATURES, ...TASK_FEATURES],
    policy: {width: policy.width, hiddenLayers: policy.hiddenLayers, featureSet: policy.featureSet, algorithm: policy.algorithm,
      provenance: policy.provenance, independentHoldout: policy.independentHoldout},
    finalScore: result.finalScore, result: {achieved: result.achieved, stopReason: result.stopReason, taskReturn: round(result.taskReturn),
      secondsUsed: round(result.clockEnd - result.clockStart), scoringActions: result.scoringActions, parkedRobots: result.parkedRobots},
    decisions, inferenceSeconds: (Date.now() - started) / 1000,
    baseline: {policy: 'task-greedy', finalScore: baseline.finalScore,
      result: {achieved: baseline.achieved, stopReason: baseline.stopReason, taskReturn: round(baseline.taskReturn),
        secondsUsed: round(baseline.clockEnd - baseline.clockStart), scoringActions: baseline.scoringActions},
      timeline: greedy.log.map(row => ({clock: row.clock, kind: row.kind, points: row.points, scoreAfter: row.score}))},
    scope: 'Abstract Override macro-action model scored by the rules engine; not physical motion.'};
}

async function main() {
  const args = process.argv.slice(2);
  const taskIndex = args.indexOf('--task');
  const text = taskIndex >= 0 ? args.splice(taskIndex, 2)[1] : null;
  const [policyFile, seedText, alliance, output = path.join(__dirname, '../roboproof/dashboard/training-replay.json')] = args;
  const seed = Number(seedText);
  if (!policyFile || !Number.isInteger(seed) || !['red', 'blue'].includes(alliance) || (taskIndex >= 0 && !text)) {
    throw Error('Usage: node tools/build-training-replay.js <policy.json> <seed> <red|blue> [output.json] [--task "<typed task>"]');
  }
  const replay = text ? recordTask(policyFile, seed, alliance, text) : await record(policyFile, seed, alliance);
  fs.writeFileSync(output, JSON.stringify(replay) + '\n');
  console.log(JSON.stringify({output, bytes: fs.statSync(output).size, decisions: replay.decisions.length,
    finalMargin: replay.finalScore.margin, loaderCycleMargin: replay.baseline.finalScore.margin, inferenceSeconds: replay.inferenceSeconds}));
}

if (require.main === module) main().catch(error => { console.error(error.stack || error); process.exitCode = 1; });

module.exports = {record, BASE_FEATURES, PREVIEW_FEATURES};
