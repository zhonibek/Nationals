'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {choose, playMatch, policyChooser, seededRandom} = require('../roboproof/tactical_baselines');
const {shortlist, parseChoice, playNemotronMatch, CANDIDATES_SHOWN} = require('../roboproof/tactical-coordinator');

const metadata = {provider: 'test-fixture-not-real-inference', model: 'robotai-nemotron', baseUrl: 'http://127.0.0.1:8080/v1'};
const call = (args, extra = {}) => ({choices: [{finish_reason: 'tool_calls', message: {role: 'assistant', content: null,
  tool_calls: [{id: 'c1', type: 'function', function: {name: 'choose_action', arguments: JSON.stringify(args)}}]}}],
usage: {total_tokens: 10}, ...extra});

test('baseline policies are deterministic and respect their action families', async () => {
  const rows = [
    {id: 'load-score-pin:a', kind: 'load-score-pin', legal: true, points: 5, pointsPerSecond: 0.8},
    {id: 'toggle:a', kind: 'toggle', legal: true, points: 20, pointsPerSecond: 8},
    {id: 'score-cup:a', kind: 'score-cup', legal: false, points: null, pointsPerSecond: null}];
  assert.equal(choose('loader-cycle', rows, seededRandom(1)).id, 'load-score-pin:a');
  assert.equal(choose('field-greedy', rows, seededRandom(1)).id, 'toggle:a');
  assert.equal(choose('greedy-all', rows, seededRandom(1)).id, 'toggle:a');
  assert.throws(() => choose('oracle', rows, seededRandom(1)), /Unknown tactical policy/);
  const first = await playMatch({seed: 5, chooser: policyChooser('loader-cycle', 5), maximumDecisions: 3});
  const second = await playMatch({seed: 5, chooser: policyChooser('loader-cycle', 5), maximumDecisions: 3});
  assert.deepEqual(first.log, second.log);
  assert.equal(first.decisions, 3);
  assert(first.log.every(row => !row.rejected));
});

test('the Nemotron shortlist is bounded and always offers loader and field options', async () => {
  let offered;
  await playMatch({seed: 9, maximumDecisions: 1, chooser: async ({candidates}) => {
    offered = shortlist(candidates);
    return offered[0];
  }});
  assert(offered.length <= CANDIDATES_SHOWN);
  assert(offered.some(row => row.kind.startsWith('load-score-')));
  assert(offered.some(row => !row.kind.startsWith('load-score-')));
});

test('choose_action responses fail closed unless they pick exactly one offered ID', () => {
  const ids = ['toggle:red-1:toggle-west'];
  assert.equal(parseChoice(call({candidate_id: ids[0], reason: 'fast'}), ids).id, ids[0]);
  assert.throws(() => parseChoice(call({candidate_id: 'score-everything', reason: 'x'}), ids), /not offered/);
  assert.throws(() => parseChoice(call({candidate_id: ids[0], reason: 'x', points: 99}), ids), /Unsupported/);
  assert.throws(() => parseChoice({choices: [{finish_reason: 'stop', message: {role: 'assistant', content: 'toggle west'}}]}, ids), /exactly once/);
  assert.throws(() => parseChoice({choices: [{finish_reason: 'length', message: {}}]}, ids), /bounded/);
});

test('a coordinated match keeps the decisions made before an invalid model answer', async () => {
  let turn = 0;
  const client = {metadata, check: async () => ({available: true}),
    complete: async (messages, signal, tools) => {
      turn += 1;
      const ids = tools[0].function.parameters.properties.candidate_id.enum;
      return turn === 1 ? call({candidate_id: ids[0], reason: 'fixture'}) : call({candidate_id: 'invented', reason: 'fixture'});
    }};
  const match = await playNemotronMatch({seed: 3, client});
  assert.equal(match.status, 'failed-closed');
  assert.match(match.error, /not offered/);
  assert.equal(match.decisions, 1);
  assert.equal(match.log[0].reason, 'fixture');
});
