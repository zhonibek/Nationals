#!/usr/bin/env node
'use strict';

// Plays the installed Tactical AI policy and the loader-cycle baseline in the abstract tactical model, then drives
// each plan through the original Simulator (production controller, physical Override rules) from the same start.
// Open loop: the plan is not re-planned when physics differs. A demonstration, not a qualification: a few seeds only,
// and the simulator controller is slower than the team's measured robot timing the abstract model assumes.
//   node tools/tactical-physics-replay.js [--seeds 101,202] [--alliance red] [--policy roboproof/runs/tactical/policy.json]

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const {loadHeadless} = require('../simulator/headless');
const Control = require('../simulator/control-runtime');
const Engine = require('../simulator/engine');
const Game = require('../simulator/override');
const Plan = require('../simulator/override-plan');
const {TacticalEnvironment} = require('../roboproof/tactical_environment');
const {playMatch, policyChooser, MATCH_DECISIONS} = require('../roboproof/tactical_baselines');
const {loadPolicy} = require('../roboproof/tactical_policy');
const {parseTask, runTask} = require('../roboproof/tactical_task');

const MAX_SEEDS = 5;

function options(argv) {
  const value = flag => { const index = argv.indexOf(flag); return index >= 0 ? argv[index + 1] : undefined; };
  const seeds = (value('--seeds') || '101').split(',').map(Number);
  if (!seeds.length || seeds.length > MAX_SEEDS || !seeds.every(seed => Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff)) {
    throw Error(`Give 1 to ${MAX_SEEDS} uint32 seeds`);
  }
  const alliance = value('--alliance') || 'red';
  if (!['red', 'blue'].includes(alliance)) throw Error('Alliance must be red or blue');
  return {seeds, alliance, policyPath: path.resolve(root, value('--policy') || 'roboproof/runs/tactical/policy.json'),
    output: path.resolve(root, value('--output') || 'roboproof/runs/tactical/physics-replay')};
}

async function plans(seed, alliance, policy) {
  const environment = new TacticalEnvironment({maximumDecisions: MATCH_DECISIONS});
  environment.reset(seed, alliance);
  const poses = environment.game.robots.map(robot => ({id: robot.id, x: robot.x, y: robot.y, theta: robot.theta}));
  const baseline = await playMatch({seed, alliance, chooser: policyChooser('loader-cycle', seed)});
  const featureSet = policy.featureSet === 'v4' ? 'v3' : policy.featureSet;
  const trained = runTask({spec: parseTask('play the driver period for the best score'), policy, seed, alliance,
    environment: new TacticalEnvironment({maximumDecisions: MATCH_DECISIONS, featureSet})});
  return {poses, plans: {'loader-cycle': baseline, 'tactical-policy': trained}};
}

function drive(compiled, poses, ids, alliance) {
  const sim = new Engine.VexRobotSimulator(Control.fromModule(compiled), {Fleet: Plan.planFleet(ids)});
  sim.override = new Game({world: 'head-to-head'});
  for (const pose of poses) sim.override.setRobotPose(pose.id, pose);
  const started = sim.startGame('match');
  if (!started.ok) throw Error(started.error);
  while (!sim.override.finalScore && sim.simTime < 130) sim.update(0.05);
  const score = sim.override.score(), other = alliance === 'red' ? 'blue' : 'red';
  const results = sim.fleet.runner.results;
  return {finalScore: {...score, margin: score[alliance] - score[other]}, plannedActions: ids.length,
    attempted: results.length, completed: results.filter(row => row.ok).length, macroActions: results};
}

async function main() {
  const {seeds, alliance, policyPath, output} = options(process.argv.slice(2));
  const identity = loadHeadless().identity;
  const compiled = new WebAssembly.Module(fs.readFileSync(path.join(root, 'simulator/control.wasm')));
  const policy = loadPolicy(policyPath);
  const matches = [];
  for (const seed of seeds) {
    const {poses, plans: played} = await plans(seed, alliance, policy);
    const row = {seed, alliance, startPoses: poses, coordinators: {}};
    for (const [name, match] of Object.entries(played)) {
      const ids = match.log.filter(entry => !entry.rejected).map(entry => entry.id);
      row.coordinators[name] = {abstractFinalScore: match.finalScore, physics: drive(compiled, poses, ids, alliance)};
      const physics = row.coordinators[name].physics;
      console.log(`seed ${seed} ${name}: abstract margin ${match.finalScore.margin}, physics margin ${physics.finalScore.margin} ` +
        `(${physics.completed} of ${physics.attempted} attempted macro-actions completed, ${physics.plannedActions} planned)`);
    }
    matches.push(row);
  }
  const report = {schema: 'roboproof-tactical-physics-replay-v1', createdAt: new Date().toISOString(),
    policy: {path: path.relative(root, policyPath), featureSet: policy.featureSet, algorithm: policy.algorithm, provenance: policy.provenance},
    simulator: identity, executor: 'simulator/override-plan.js (open loop, pose moves on the production controller)',
    scope: 'demonstration on a few seeds; not an independent evaluation; simulator controller slower than measured robot timing',
    learnedImprovementVerified: false, matches};
  fs.mkdirSync(output, {recursive: true});
  const text = `${JSON.stringify(report, null, 2)}\n`;
  const file = path.join(output, `physics-replay-${alliance}-${seeds.join('-')}.json`);
  fs.writeFileSync(file, text);
  fs.writeFileSync(`${file}.sha256`, `${crypto.createHash('sha256').update(text).digest('hex')}  ${path.basename(file)}\n`);
  console.log(`Saved ${path.relative(root, file)}`);
}

main().catch(error => { console.error(error.message); process.exit(1); });
