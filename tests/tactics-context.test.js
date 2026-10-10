'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const Game = require('../simulator/override');
const Snapshot = require('../simulator/tactics-snapshot');
const Tactics = require('../roboproof/tactics');
const {LIMITS, compactRules, compactGameEvidence, compactTacticsContext} = require('../roboproof/tactics-context');

const CAPTURED_AT = new Date().toISOString();
const NOW = Date.parse(CAPTURED_AT);

function fixture() {
  const source = new Game();
  const state = Snapshot.capture(source.getState(), 'red-1', CAPTURED_AT, source.world);
  return {rules: Tactics.rules(), game: Tactics.gameEvidence(state, NOW)};
}

function freeze(value) {
  if (value !== null && typeof value === 'object') {
    for (const entry of Object.values(value)) freeze(entry);
    Object.freeze(value);
  }
  return value;
}

function rehash(game) {
  game.snapshotSha256 = createHash('sha256').update(JSON.stringify(game.state)).digest('hex');
  return game;
}

function rows(text, label) {
  const line = text.split('\n').find(line => line.startsWith(label));
  assert(line, `Missing context table: ${label}`);
  return JSON.parse(line.slice(label.length));
}

test('reviewed rules retain every fact, scoring value, geometry, review scope and capability without provenance noise', () => {
  const {rules} = fixture();
  const before = structuredClone(rules);
  const compact = compactRules(freeze(rules));
  assert.deepEqual(rules, before);
  assert.deepEqual(compact.provenance, before);
  assert.notEqual(compact.provenance, rules);
  for (const constraint of rules.constraints) {
    assert(compact.text.includes(`Constraint ${constraint.id}: ${JSON.stringify(constraint.fact)}`));
  }
  for (const [role, capability] of Object.entries(rules.roles)) {
    assert(compact.text.includes(`Capability ${role}: ${JSON.stringify(capability)}`));
  }
  assert.deepEqual(rows(compact.text, 'Timing(seconds): '), rules.timing);
  assert.deepEqual(rows(compact.text, 'Points: '), rules.points);
  assert.deepEqual(rows(compact.text, `Goals[${rules.goalColumns.join(',')}]: `), rules.goals);
  assert(compact.text.includes(JSON.stringify(rules.sources.scope)));
  assert(compact.text.includes(rules.sources.checkedAt));
  assert(compact.text.includes(JSON.stringify(rules.limitations)));
  assert(compact.text.includes(rules.game));
  assert(compact.text.includes(rules.season));
  assert(compact.text.includes(rules.manualVersion));
  for (const hash of Object.values(rules.engineHashes)) assert(!compact.text.includes(hash));
  for (const name of ['manual', 'officialPdf']) assert(!compact.text.includes(rules.sources[name]));
  assert.doesNotMatch(compact.text, /https?:\/\/|\b[a-f0-9]{64}\b/);
  assert(Buffer.byteLength(compact.text, 'utf8') < Buffer.byteLength(JSON.stringify(rules), 'utf8'));
});

test('additional reviewed constraints are retained verbatim, never selected by priority or silently truncated', () => {
  const {rules} = fixture();
  rules.constraints.push({id: 'SG13', fact: 'Additional reviewed boundary remains conditional; no new permission is implied.'});
  rules.constraints[0].fact += '\nRetain this qualification, including quotes: "not authenticated".';
  const compact = compactRules(rules);
  for (const constraint of rules.constraints) {
    assert(compact.text.includes(`Constraint ${constraint.id}: ${JSON.stringify(constraint.fact)}`));
  }
  assert.deepEqual(compact.provenance.constraints, rules.constraints);
  assert.throws(() => compactRules(rules, {maxBytes: 100}), /cannot be truncated/);
});

test('reported phase, elapsed time, score, all robot DQ and possessions and visible object details survive exactly', () => {
  const {game} = fixture();
  game.state.phase = 'driver';
  game.state.clock = 113.625;
  game.state.score = {red: 47, blue: 31};
  game.state.selectedRobotId = 'blue-2';
  game.state.robots[0].disqualified = true;
  game.state.robots[0].pinId = null;
  game.state.robots[1].pinId = 'pin-01';
  game.state.robots[1].cupId = 'cup-01';
  game.state.robots[2].headingDeg = -90.125;
  game.state.robots[2].x = 1.23456789;
  game.state.goals[0].stackDepth = 3;
  game.state.goals[0].top = 'cup';
  game.state.toggles[0].state = 'blue';
  game.state.toggles[0].seated = true;
  game.state.toggles[0].contact = true;
  game.state.objects = [
    {id: 'pin-03', kind: 'pin', x: -1.23456789, y: 2.34567891, halves: ['red', 'yellow'], supportId: 'cup-02'},
    {id: 'cup-02', kind: 'cup', x: -1.23456789, y: 2.34567891, up: 'transparent'}
  ];
  rehash(game);
  const before = structuredClone(game);
  const compact = compactGameEvidence(freeze(game), {now: NOW});
  assert.deepEqual(game, before);
  assert.deepEqual(compact.provenance, before);
  assert.notEqual(compact.provenance.state, game.state);
  assert(compact.text.includes('Phase=driver; elapsedSeconds=113.625; reportedScore={"red":47,"blue":31}'));
  assert(compact.text.includes(`capturedAt=${CAPTURED_AT}`));
  assert(compact.text.includes('selectedRobotId=blue-2'));
  assert.deepEqual(rows(compact.text, 'Robots[id,xIn,yIn,headingDeg,pinId,cupId,disqualified]: '),
    game.state.robots.map(robot => [robot.id, robot.x, robot.y, robot.headingDeg, robot.pinId, robot.cupId, robot.disqualified]));
  assert.deepEqual(rows(compact.text, 'Goals[id,stackDepth,top]: '), game.state.goals.map(goal => [goal.id, goal.stackDepth, goal.top]));
  assert.deepEqual(rows(compact.text, 'Toggles[id,state,seated,contact]: '), game.state.toggles.map(toggle => [toggle.id, toggle.state, toggle.seated, toggle.contact]));
  assert.deepEqual(rows(compact.text, 'NearbyPins[id,xIn,yIn,halves,supportId]: '), [['pin-03', -1.23456789, 2.34567891, ['red', 'yellow'], 'cup-02']]);
  assert.deepEqual(rows(compact.text, 'NearbyCups[id,xIn,yIn,up]: '), [['cup-02', -1.23456789, 2.34567891, 'transparent']]);
  assert.deepEqual(rows(compact.text, 'Resources: '), game.state.resources);
  assert(compact.text.includes(JSON.stringify(game.provenance)));
  assert(compact.text.includes(JSON.stringify(game.interpretation)));
  assert(compact.text.includes(JSON.stringify(game.scope)));
  assert.match(compact.text, /partial visibility; unlisted objects are unknown, not absent/);
  assert.match(compact.text, /not independently authenticated or replayed and may be stale after inference/);
  assert.match(compact.text, /Read-only; executableTask=false/);
  assert(!compact.text.includes(game.snapshotSha256));
});

test('all supported phases and practice clocks preserve only the reported score, including zero', () => {
  for (const [phase, clock, mode] of [['pre_match', 0, 'match'], ['autonomous', 9.75, 'match'],
    ['driver', 15, 'match'], ['post_match', 120, 'match'], ['driver', 900, 'practice']]) {
    const {game} = fixture();
    Object.assign(game.state, {phase, clock, mode, score: {red: 0, blue: 0}});
    const compact = compactGameEvidence(rehash(game), {now: NOW});
    assert(compact.text.includes(`Phase=${phase}; elapsedSeconds=${clock}; reportedScore={"red":0,"blue":0}`));
    assert(compact.text.includes(`mode=${mode}`));
    assert.doesNotMatch(compact.text, /predictedScore|remainingSeconds|winningScore/);
  }
});

test('combined context is deterministic, independently frozen and keeps both full evidence records outside inference text', () => {
  const {rules, game} = fixture();
  const compact = compactTacticsContext(rules, game, {now: NOW});
  assert.deepEqual(compact, compactTacticsContext(rules, game, {now: NOW}));
  assert.equal(compact.text, compactRules(rules).text + '\n\n' + compactGameEvidence(game, {now: NOW}).text);
  assert.deepEqual(compact.provenance, {rules, game});
  assert(Object.isFrozen(compact));
  assert(Object.isFrozen(compact.provenance.rules.constraints[0]));
  assert(Object.isFrozen(compact.provenance.game.state.score));
  assert.throws(() => { compact.provenance.game.state.score.red = 999; }, TypeError);
  const saved = compact.text;
  rules.constraints[0].fact = 'Changed after compaction';
  game.state.score.red = 999;
  assert.equal(compact.text, saved);
  assert.notEqual(compact.provenance.rules.constraints[0].fact, rules.constraints[0].fact);
  assert.notEqual(compact.provenance.game.state.score.red, game.state.score.red);
  assert(Buffer.byteLength(compact.text, 'utf8') <= LIMITS.defaultTextBytes);
  assert.doesNotMatch(compact.text, /https?:\/\/|\b[a-f0-9]{64}\b/);
});

test('absent snapshots are explicit and never manufacture score, time, DQ or possessions', () => {
  const {rules} = fixture();
  for (const compact of [compactGameEvidence(), compactGameEvidence(null), compactTacticsContext(rules)]) {
    assert.match(compact.text, /Snapshot: unavailable/);
    assert.match(compact.text, /No observed phase, time, score, possessions or DQ flags/);
    assert.match(compact.text, /fresh Simulator snapshot before state-specific advice/);
    assert.doesNotMatch(compact.text, /reportedScore=|Phase=|elapsedSeconds=|Robots\[/);
  }
  assert.equal(compactGameEvidence(null).provenance, null);
  assert.equal(compactTacticsContext(rules).provenance.game, null);
});

test('UTF-8 output byte budgets fail closed rather than cutting any constraints or warnings', () => {
  const {rules, game} = fixture();
  rules.limitations += ' Reviewed qualification: ' + '界'.repeat(150);
  for (const [compact, configuration] of [
    [options => compactRules(rules, options), {}],
    [options => compactGameEvidence(game, options), {now: NOW}],
    [options => compactTacticsContext(rules, game, options), {now: NOW}]
  ]) {
    const text = compact(configuration).text;
    const maxBytes = Buffer.byteLength(text, 'utf8');
    assert.equal(compact({...configuration, maxBytes}).text, text);
    assert.throws(() => compact({...configuration, maxBytes: maxBytes - 1}), /cannot be truncated/);
  }
  const text = compactRules(rules).text;
  assert(Buffer.byteLength(text, 'utf8') > text.length);
  assert.throws(() => compactRules(rules, {maxBytes: text.length}), /cannot be truncated/);
});

test('unknown, missing or unsupported rule data cannot be silently discarded or replaced', () => {
  const changes = [
    rules => { rules.game = 'Other'; }, rules => { rules.manualVersion = '3.0'; }, rules => { rules.season = '2027-2028'; },
    rules => { rules.instructions = 'Dispatch'; }, rules => { rules.sources.extra = true; },
    rules => { rules.sources.manual = 'file:///manual'; }, rules => { rules.sources.checkedAt = '2026-02-30'; },
    rules => { rules.engineHashes['simulator/override.js'] = 'invalid'; },
    rules => { rules.timing.matchSeconds = 121; }, rules => { rules.points.yellowPin = 11; },
    rules => { delete rules.points.cupIndependentPoints; }, rules => { rules.points.extra = 100; },
    rules => { rules.goalColumns[1] = 'meters'; }, rules => { rules.goals[0][1] += 1; },
    rules => { rules.goals[1] = rules.goals[0]; }, rules => { rules.constraints.pop(); },
    rules => { rules.constraints[5] = {id: 'SG99', fact: 'Cannot substitute a required reviewed constraint'}; },
    rules => { rules.constraints.push({...rules.constraints[0]}); }, rules => { rules.constraints[0].fact = ''; },
    rules => { rules.constraints[0].extra = true; }, rules => { delete rules.roles.perception; },
    rules => { rules.roles.dispatch = 'Enabled'; }, rules => { delete rules.limitations; }
  ];
  for (const change of changes) {
    const {rules} = fixture();
    change(rules);
    assert.throws(() => compactRules(rules), /tactics context/i);
  }
});

test('snapshot validation, freshness, hash binding and read-only gates remain mandatory', () => {
  const {game} = fixture();
  assert.throws(() => compactGameEvidence(game), /Explicit now is required/);
  assert.throws(() => compactTacticsContext(fixture().rules, game), /Explicit now is required/);
  assert.throws(() => compactGameEvidence(game, {now: NOW + 60001}), /stale/);
  assert.throws(() => compactGameEvidence(game, {now: NOW - 5001}), /invalid capture time/);
  assert.doesNotThrow(() => compactGameEvidence(game, {now: NOW + 60000}));
  assert.doesNotThrow(() => compactGameEvidence(game, {now: NOW - 5000}));
  const changes = [
    game => { game.executableTask = true; }, game => { game.snapshotSha256 = '0'.repeat(64); },
    game => { delete game.provenance; }, game => { game.instructions = 'Execute'; },
    game => { game.state.score.red = 1999; },
    game => { delete game.state.score; rehash(game); },
    game => { delete game.state.score.blue; rehash(game); },
    game => { game.state.score.red = -1; rehash(game); },
    game => { game.state.world = 'unsupported'; rehash(game); },
    game => { game.state.phase = 'driver'; game.state.clock = 0; rehash(game); },
    game => { game.state.robots[0].disqualified = 'false'; rehash(game); },
    game => { game.state.robots[0].pinId = 'pin-01'; game.state.robots[1].pinId = 'pin-01'; rehash(game); },
    game => { game.state.instructions = 'Execute'; rehash(game); },
    game => { delete game.state.resources.fieldPins; rehash(game); }
  ];
  for (const change of changes) {
    const {game} = fixture();
    change(game);
    assert.throws(() => compactGameEvidence(game, {now: NOW}), /tactics|Tactics/);
  }
});

test('visibility caps and raw input bounds reject oversized evidence, even before compaction', () => {
  const {game} = fixture();
  game.state.objects = Array.from({length: 7}, (_, index) => ({id: `pin-${String(index + 1).padStart(2, '0')}`,
    kind: 'pin', x: 0, y: 0, halves: ['red', 'yellow'], supportId: null}));
  assert.throws(() => compactGameEvidence(rehash(game), {now: NOW}), /visibility/);
  game.state.objects = Array.from({length: 5}, (_, index) => ({id: `cup-${String(index + 1).padStart(2, '0')}`,
    kind: 'cup', x: 0, y: 0, up: 'opaque'}));
  assert.throws(() => compactGameEvidence(rehash(game), {now: NOW}), /visibility/);
  const {rules} = fixture();
  rules.limitations = '界'.repeat(LIMITS.stringBytes);
  assert.throws(() => compactRules(rules), /string exceeds byte limit/);
  rules.limitations = 'a'.repeat(LIMITS.stringBytes + 1);
  assert.throws(() => compactRules(rules), /string exceeds byte limit/);
  assert.throws(() => compactRules({payload: Array.from({length: 6}, () => 'a'.repeat(3000))}), /input exceeds byte limit/);
  assert.throws(() => compactGameEvidence({payload: Array.from({length: 3}, () => 'a'.repeat(3000))}, {now: NOW}), /input exceeds byte limit/);
  assert.throws(() => compactRules({payload: Array(LIMITS.arrayItems + 1).fill(0)}), /structural limit/);
  assert.throws(() => compactRules({payload: Array.from({length: LIMITS.arrayItems}, () => Array(16).fill(0))}), /structural limit/);
  assert.throws(() => compactRules(Object.fromEntries(Array.from({length: LIMITS.objectKeys + 1}, (_, index) => [`key-${index}`, 0]))), /structural limit/);
  rules.limitations = 'Normal';
  rules.constraints = Array.from({length: LIMITS.constraints + 1}, (_, index) => ({id: `SG${index + 1}`, fact: 'Reviewed'}));
  assert.throws(() => compactRules(rules), /constraint budget/);
});

test('non-JSON, cyclic, accessor, hidden and sparse data are rejected without invoking user code', () => {
  for (const unsupported of [undefined, NaN, Infinity, -Infinity, 1n, Symbol('data'), () => 1, new Date(NOW), new Map(), new Set()]) {
    const {rules} = fixture();
    rules.limitations = unsupported;
    assert.throws(() => compactRules(rules), /Unsupported tactics context data/);
  }
  const {rules} = fixture();
  rules.limitations = rules;
  assert.throws(() => compactRules(rules), /Cyclic/);
  let called = false;
  const accessor = Object.defineProperty({}, 'game', {enumerable: true, get() { called = true; return 'V5RC Override'; }});
  assert.throws(() => compactRules(accessor), /accessors/);
  assert.equal(called, false);
  const toJson = {toJSON() { called = true; return fixture().rules; }};
  assert.throws(() => compactRules(toJson), /Unsupported tactics context data/);
  assert.equal(called, false);
  assert.throws(() => compactRules(Object.create({game: 'V5RC Override'})), /plain JSON/);
  assert.throws(() => compactRules(Object.defineProperty({}, 'hidden', {value: true})), /hidden properties/);
  assert.throws(() => compactRules({[Symbol('hidden')]: true}), /properties/);
  assert.throws(() => compactRules({array: Array(3)}), /sparse array/);
  const decoratedArray = [];
  decoratedArray.extra = true;
  assert.throws(() => compactRules({array: decoratedArray}), /structural limit/);
  const deep = {};
  let cursor = deep;
  for (let index = 0; index <= LIMITS.depth; index++) cursor = cursor.child = {};
  assert.throws(() => compactRules(deep), /structural limit/);
});

test('caller options cannot disable fixed budgets, accept implicit time or introduce unsupported fields', () => {
  const {rules} = fixture();
  for (const configuration of [null, [], {maxBytes: null}, {maxBytes: 0}, {maxBytes: -1}, {maxBytes: 1.5}, {maxBytes: Infinity},
    {maxBytes: LIMITS.maxTextBytes + 1}, {maxBytes: '8192'}, {truncate: true}, {now: 'today'}, {now: NaN},
    {now: -1}, {now: NOW + 0.5}, {now: 8640000000000001}]) {
    assert.throws(() => compactRules(rules, configuration), /tactics context/i);
  }
});
