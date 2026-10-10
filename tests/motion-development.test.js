'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const Development = require('../roboproof/motion-development');
const {assertTrainingWorld, suite} = require('../roboproof/motion-evaluation');
const Policy = require('../simulator/policy');

test('development worlds are deterministic paired families, bounded and distinct from frozen benchmarks', () => {
  const corpus = Development.worlds(789, 2);
  assert.deepEqual(corpus, Development.worlds(789, 2));
  assert.equal(corpus.cases.length, 8);
  assert(corpus.cases.every(world => assertTrainingWorld(world)));
  for (const world of corpus.cases) {
    assert(Math.abs(world.task.goal.xIn) <= 52);
    assert(Math.abs(world.task.goal.yIn) <= 52);
  }
  assert.deepEqual(corpus.cases[0].task.goal, corpus.cases[3].task.goal);
  assert.notEqual(corpus.cases[0].seed, corpus.cases[3].seed);
  const field = Development.worlds(789, 2, 'field-reach');
  assert(field.cases.every(world => Math.abs(world.task.start.xIn) <= 40 && Math.abs(world.task.goal.xIn) <= 45));
  assert.throws(() => Development.worlds(789, 2, 'unknown'), /Unsupported development profile/);
  const full = Development.worlds(789, 2, 'full-contract');
  assert(full.cases.every(world => assertTrainingWorld(world)));
  assert(full.cases.some(world => Math.abs(world.task.goal.xIn) === 60));
  assert(full.cases.some(world => world.configuration.muLat?.[1] === 0.15));
});

test('scripted feasibility actions are bounded and use only policy observation', () => {
  const observation = Array(34).fill(0);
  observation[10] = 0.5; observation[11] = 0.3;
  for (const name of Development.STRATEGIES.filter(name => name !== 'controller')) {
    const act = Development.strategy(name);
    assert.deepEqual(Policy.normalizeAction(act(observation)), act(observation));
  }
  assert.throws(() => Development.strategy('shell'), /Unsupported/);
});

test('bounded probe records original-engine outcomes and never claims learned/final improvement', () => {
  const result = Development.probe(Development.worlds(790, 1), {names: ['controller', 'pace-0.7'], maximumRuns: 2});
  assert.equal(result.completedRuns, 2);
  assert.equal(result.completedWorlds, 1);
  assert.equal(result.budgetStopped, true);
  assert.equal(result.identity.canonicalSimulator, true);
  assert.equal(result.learnedImprovementVerified, false);
  assert.equal(result.strategies.controller.count, result.strategies['pace-0.7'].count);
  assert.match(result.interpretation, /not final evaluation/);
  const bad = Development.worlds(791, 1);
  bad.cases[0] = suite(2).cases[0];
  assert.throws(() => Development.probe(bad), /Invalid development corpus/);
});
