'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {TacticalEnvironment} = require('../roboproof/tactical_environment');
const {DEMONSTRATION_POLICIES, choose, demonstrate} = require('../roboproof/tactical_baselines');

// Only single decisions are previewed here; no full match is played.
const ids = rows => rows.map(row => row.id);

test('observation action IDs are the previewed candidate IDs in the same order', () => {
  const environment = new TacticalEnvironment();
  for (const [seed, alliance] of [[17, 'red'], [23, 'blue']]) {
    let observation = environment.reset(seed, alliance);
    assert.deepEqual(ids(observation.actions), ids(environment.previewCandidates()));
    observation = environment.step(demonstrate(environment).id).observation;
    assert.equal(observation.done, false);
    assert.deepEqual(ids(observation.actions), ids(environment.previewCandidates()));
  }
});

test('a loader-cycle demonstration is the choice the baseline makes from the full preview', () => {
  const environment = new TacticalEnvironment();
  for (const [seed, alliance] of [[17, 'red'], [23, 'blue']]) {
    environment.reset(seed, alliance);
    for (let decision = 0; decision < 2; decision += 1) {
      const pick = demonstrate(environment);
      assert.deepEqual(pick, choose('loader-cycle', environment.previewCandidates()));
      environment.step(pick.id);
    }
  }
});

function scriptedEnvironment(rows) {
  return {done: false, previewed: [],
    previewCandidates(include = () => true) {
      const selected = rows.filter(row => include(row));
      this.previewed.push(selected.length);
      return selected;
    }};
}

const row = (id, points, seconds = 2) => ({id, kind: id.split(':')[0], legal: points !== null, points, seconds,
  pointsPerSecond: points === null ? null : points / seconds});

test('loader-cycle previews the other candidates only when no match load scores', () => {
  const field = row('score-pin:red-1:pin-02:g-red-sw', 6, 3);
  const scoringLoad = row('load-score-cup:red-1:loader-red-1:g-red-sw', 4);
  const idleLoad = row('load-score-cup:red-2:loader-red-1:g-red-se', 0);
  const illegalLoad = row('load-score-pin:red-2:loader-red-2:g-red-se', null);

  const loading = scriptedEnvironment([field, scoringLoad, idleLoad, illegalLoad]);
  assert.equal(demonstrate(loading), scoringLoad);
  assert.deepEqual(loading.previewed, [3], 'only the three load candidates are previewed');

  const fallback = scriptedEnvironment([field, idleLoad, illegalLoad]);
  assert.equal(demonstrate(fallback), field);
  assert.equal(choose('loader-cycle', [field, idleLoad, illegalLoad]), field);
  assert.deepEqual(fallback.previewed, [2, 3]);

  const greedy = scriptedEnvironment([field, scoringLoad]);
  assert.equal(demonstrate(greedy, 'greedy-all'), field);
  assert.deepEqual(greedy.previewed, [2]);
});

test('demonstrations end with the episode and refuse tactics that need a seeded stream', () => {
  const environment = new TacticalEnvironment({maximumDecisions: 1});
  environment.reset(17, 'red');
  assert.equal(environment.step(demonstrate(environment).id).observation.done, true);
  assert.equal(demonstrate(environment), null);
  assert.deepEqual(DEMONSTRATION_POLICIES, ['loader-cycle', 'field-greedy', 'greedy-all']);
  assert.throws(() => demonstrate(environment, 'random'), /Unsupported demonstration policy random/);
  assert.throws(() => demonstrate(environment, 'nemotron'), /Unsupported demonstration policy nemotron/);
});
