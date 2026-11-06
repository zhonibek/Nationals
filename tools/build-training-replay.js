'use strict';

// Records one full match of an exported tactical policy for the training page (inference only, nothing trains):
// every decision's candidates and scores, the chosen move's inputs and its layer-by-layer neuron trace, the score
// after the move, and the loader-cycle tactic on the same start pose for comparison.
//
//   node tools/build-training-replay.js <policy.json> <seed> <red|blue> [output.json]

const fs = require('node:fs');
const path = require('node:path');
const {TacticalEnvironment} = require('../roboproof/tactical_environment');
const {loadPolicy} = require('../roboproof/tactical_policy');
const {playMatch, policyChooser} = require('../roboproof/tactical_baselines');

const BASE_FEATURES = ['score Pin', 'score Cup', 'Toggle', 'park', 'alliance side', 'red halves', 'blue halves',
  'yellow halves', 'face up', 'distance to object', 'distance to goal', 'travel time', 'time left', 'score margin',
  'holding Pin', 'holding Cup', 'goal stack height', 'own goal', 'Toggle state', 'goal x', 'goal y', 'endgame',
  'from loader'];
const PREVIEW_FEATURES = ['legal', 'previewed points', 'previewed seconds', 'points per second', 'rank', 'best move',
  'best loader move'];
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

async function main() {
  const [policyFile, seedText, alliance, output = path.join(__dirname, '../roboproof/dashboard/training-replay.json')] =
    process.argv.slice(2);
  const seed = Number(seedText);
  if (!policyFile || !Number.isInteger(seed) || !['red', 'blue'].includes(alliance)) {
    throw Error('Usage: node tools/build-training-replay.js <policy.json> <seed> <red|blue> [output.json]');
  }
  const replay = await record(policyFile, seed, alliance);
  fs.writeFileSync(output, JSON.stringify(replay) + '\n');
  console.log(JSON.stringify({output, bytes: fs.statSync(output).size, decisions: replay.decisions.length,
    finalMargin: replay.finalScore.margin, loaderCycleMargin: replay.baseline.finalScore.margin, inferenceSeconds: replay.inferenceSeconds}));
}

if (require.main === module) main().catch(error => { console.error(error.stack || error); process.exitCode = 1; });

module.exports = {record, BASE_FEATURES, PREVIEW_FEATURES};
