'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const OverrideGame = require('../simulator/override');
const {TacticalEnvironment, FEATURE_COUNT, cloneGame} = require('../roboproof/tactical_environment');

test('tactical episodes start in the driver period with a bounded, finite candidate set', () => {
  const environment = new TacticalEnvironment();
  const observation = environment.reset(17, 'red');
  assert.equal(observation.clock, OverrideGame.constants.AUTO_SECONDS);
  assert.equal(observation.alliance, 'red');
  assert.equal(observation.done, false);
  assert(observation.actions.length > 0);
  assert(observation.actions.length < 300);
  assert(observation.actions.every(action => action.features.length === FEATURE_COUNT && action.features.every(Number.isFinite)));
  assert.match(observation.movementModel, /not a motion\/physics result/);
});

test('previewing actions does not mutate the live simulator state', () => {
  const environment = new TacticalEnvironment();
  environment.reset(21, 'red');
  const before = JSON.stringify(environment.game.getState());
  environment.observe();
  assert.equal(JSON.stringify(environment.game.getState()), before);
});

test('a selected scoring macro uses the original Override score and advances game time', () => {
  const environment = new TacticalEnvironment();
  const observation = environment.reset(31, 'red');
  const action = observation.actions.find(candidate => candidate.id.startsWith('score-pin:red-1:pin-01:g-red-sw'));
  assert(action, 'expected a legal preload-to-alliance-goal candidate');
  const before = environment.score();
  const result = environment.step(action.id);
  assert(result.reward > 0);
  assert(result.durationSeconds > 0);
  assert(result.score.red > before.red);
  assert.equal(environment.game.pins.find(pin => pin.id === 'pin-01').status, 'placed');
});

test('candidate IDs are validated against the current state and game clones preserve engine methods', () => {
  const environment = new TacticalEnvironment();
  const observation = environment.reset(41, 'blue');
  assert.throws(() => environment.step('score-everything'), /current legal candidate set/);
  const copy = cloneGame(environment.game);
  assert(copy instanceof OverrideGame);
  assert.deepEqual(copy.score(), environment.game.score());
  assert.notEqual(copy.pins, environment.game.pins);
  assert.notEqual(copy.goals, environment.game.goals);
  assert(observation.actions.some(action => action.id.includes('blue-1')));
});
