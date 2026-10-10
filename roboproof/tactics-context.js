'use strict';

const {createHash} = require('node:crypto');
const Game = require('../simulator/override');
const Snapshot = require('../simulator/tactics-snapshot');

const LIMITS = Object.freeze({rulesBytes: 16384, gameBytes: 8192, defaultTextBytes: 8192,
  maxTextBytes: 16384, stringBytes: 4096, depth: 12, nodes: 2048, arrayItems: 128, objectKeys: 64, constraints: 32});
const RULE_KEYS = ['game', 'season', 'manualVersion', 'sources', 'engineHashes', 'timing', 'points',
  'goalColumns', 'goals', 'constraints', 'roles', 'limitations'];
const SOURCE_KEYS = ['manual', 'officialPdf', 'checkedAt', 'scope'];
const ENGINE_KEYS = ['simulator/override.js', 'simulator/override-geometry.js', 'simulator/override-dynamics.js'];
const TIMING = {autonomousSeconds: Game.constants.AUTO_SECONDS, matchSeconds: Game.constants.MATCH_SECONDS,
  endgameSeconds: Game.constants.ENDGAME_SECONDS};
const POINTS = {...Game.constants.POINTS, cupIndependentPoints: 0, tiedAutonomousBonus: Game.constants.POINTS.autonomousBonus / 2};
const GOAL_COLUMNS = ['id', 'xIn', 'yIn', 'alliance', 'heightIn'];
const REQUIRED_CONSTRAINTS = ['SC2/SC3', 'SC4/SC5', 'SC6/SC7', 'SG6', 'SG7/SG8/SG9/SG10', 'SG12'];
const ROLE_KEYS = ['tactics', 'motion', 'manipulation', 'perception', 'referee'];
const GAME_KEYS = ['snapshotSha256', 'state', 'provenance', 'interpretation', 'scope', 'executableTask'];
const validHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const plainObject = value => value !== null && typeof value === 'object' &&
  [Object.prototype, null].includes(Object.getPrototypeOf(value));

function boundedCopy(value, maximumBytes) {
  let bytes = 0, nodes = 0;
  const ancestors = new Set();
  function account(chunk) {
    bytes += Buffer.byteLength(chunk, 'utf8');
    if (bytes > maximumBytes) throw Error('Tactics context input exceeds byte limit');
  }
  function string(value) {
    if (value.length > LIMITS.stringBytes || Buffer.byteLength(value, 'utf8') > LIMITS.stringBytes) {
      throw Error('Tactics context string exceeds byte limit');
    }
    account(JSON.stringify(value));
  }
  function visit(entry, depth) {
    if (++nodes > LIMITS.nodes || depth > LIMITS.depth) throw Error('Tactics context exceeds structural limit');
    if (entry === null || typeof entry === 'boolean' || (typeof entry === 'number' && Number.isFinite(entry))) {
      account(JSON.stringify(entry));
      return entry;
    }
    if (typeof entry === 'string') {
      string(entry);
      return entry;
    }
    const array = Array.isArray(entry);
    if ((!array && !plainObject(entry)) || (array && Object.getPrototypeOf(entry) !== Array.prototype)) {
      throw Error('Unsupported tactics context data; plain JSON required');
    }
    if (ancestors.has(entry)) throw Error('Cyclic tactics context data');
    const ownKeys = Reflect.ownKeys(entry);
    if (array ? entry.length > LIMITS.arrayItems || ownKeys.length !== entry.length + 1 : ownKeys.length > LIMITS.objectKeys) {
      throw Error('Tactics context exceeds structural limit or contains a sparse array');
    }
    if (ownKeys.some(key => typeof key !== 'string' || (!array && ['__proto__', 'constructor', 'prototype'].includes(key)))) {
      throw Error('Unsupported tactics context properties');
    }
    ancestors.add(entry);
    const copy = array ? [] : Object.create(Object.getPrototypeOf(entry));
    const ordered = array ? Array.from({length: entry.length}, (_, index) => String(index)) : ownKeys;
    account(array ? '[' : '{');
    for (const [index, key] of ordered.entries()) {
      const descriptor = Object.getOwnPropertyDescriptor(entry, key);
      if (!descriptor || !descriptor.enumerable || !Object.hasOwn(descriptor, 'value')) {
        throw Error('Unsupported tactics context accessors or hidden properties');
      }
      if (index) account(',');
      if (!array) {
        string(key);
        account(':');
      }
      copy[key] = visit(descriptor.value, depth + 1);
    }
    account(array ? ']' : '}');
    ancestors.delete(entry);
    return copy;
  }
  return visit(value, 0);
}

function keys(value, required, label, optional = []) {
  if (!plainObject(value) || required.some(key => !Object.hasOwn(value, key)) ||
      Object.keys(value).some(key => !required.includes(key) && !optional.includes(key))) {
    throw Error(`Unsupported tactics context ${label} fields`);
  }
}

function text(value, maximumBytes, label) {
  if (typeof value !== 'string' || !value.trim() || Buffer.byteLength(value, 'utf8') > maximumBytes) {
    throw Error(`Invalid tactics context ${label}`);
  }
}

function options(value) {
  const copy = boundedCopy(value, 512);
  keys(copy, [], 'options', ['maxBytes', 'now']);
  const maxBytes = Object.hasOwn(copy, 'maxBytes') ? copy.maxBytes : LIMITS.defaultTextBytes;
  if (!Number.isInteger(maxBytes) || maxBytes < 1 || maxBytes > LIMITS.maxTextBytes) {
    throw Error('Tactics context maxBytes must be within the fixed text limit');
  }
  if (copy.now !== undefined && (!Number.isSafeInteger(copy.now) || copy.now < 0 || copy.now > 8640000000000000)) {
    throw Error('Invalid tactics context explicit now timestamp');
  }
  return {maxBytes, now: copy.now};
}

function reviewedRules(value) {
  const copy = boundedCopy(value, LIMITS.rulesBytes);
  keys(copy, RULE_KEYS, 'rules');
  if (copy.game !== 'V5RC Override' || copy.season !== '2026-2027' || copy.manualVersion !== '2.0') {
    throw Error('Unsupported tactics context game, season or manual version');
  }
  keys(copy.sources, SOURCE_KEYS, 'rule sources');
  for (const name of ['manual', 'officialPdf']) {
    text(copy.sources[name], 1024, 'source URL');
    let url;
    try { url = new URL(copy.sources[name]); } catch { throw Error('Invalid tactics context source URL'); }
    if (url.protocol !== 'https:' || url.username || url.password) throw Error('Unsupported tactics context source URL');
  }
  if (typeof copy.sources.checkedAt !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(copy.sources.checkedAt) ||
      !Number.isFinite(Date.parse(copy.sources.checkedAt)) ||
      new Date(copy.sources.checkedAt).toISOString().slice(0, 10) !== copy.sources.checkedAt) {
    throw Error('Invalid tactics context rule review date');
  }
  text(copy.sources.scope, 1024, 'review scope');
  keys(copy.engineHashes, ENGINE_KEYS, 'engine hashes');
  if (!Object.values(copy.engineHashes).every(validHash)) throw Error('Invalid tactics context engine hash');
  for (const [label, expected] of [['timing', TIMING], ['points', POINTS]]) {
    keys(copy[label], Object.keys(expected), label);
    if (Object.entries(expected).some(([key, expectedValue]) => copy[label][key] !== expectedValue)) {
      throw Error(`Unsupported tactics context ${label} for the reviewed engine`);
    }
  }
  if (!Array.isArray(copy.goalColumns) || copy.goalColumns.length !== GOAL_COLUMNS.length ||
      copy.goalColumns.some((column, index) => column !== GOAL_COLUMNS[index]) ||
      !Array.isArray(copy.goals) || copy.goals.length !== Game.layouts.goals.length) {
    throw Error('Unsupported tactics context goal schema');
  }
  const goalIds = new Set();
  for (const row of copy.goals) {
    const goal = Array.isArray(row) && Game.layouts.goals.find(goal => goal.id === row[0]);
    if (!goal || row.length !== GOAL_COLUMNS.length || goalIds.has(row[0]) ||
        row[1] !== goal.x - Game.constants.FIELD_HALF || row[2] !== goal.y - Game.constants.FIELD_HALF ||
        row[3] !== goal.alliance || row[4] !== goal.height) throw Error('Unsupported tactics context goal geometry');
    goalIds.add(row[0]);
  }
  if (!Array.isArray(copy.constraints) || copy.constraints.length < REQUIRED_CONSTRAINTS.length ||
      copy.constraints.length > LIMITS.constraints) throw Error('Invalid tactics context reviewed constraint budget');
  const constraintIds = new Set();
  for (const constraint of copy.constraints) {
    keys(constraint, ['id', 'fact'], 'constraint');
    text(constraint.id, 64, 'constraint ID');
    if (!/^[A-Z]{1,4}[0-9]+(?:\/[A-Z]{1,4}[0-9]+)*$/.test(constraint.id) || constraintIds.has(constraint.id)) {
      throw Error('Invalid tactics context constraint ID');
    }
    text(constraint.fact, 1024, 'reviewed constraint fact');
    constraintIds.add(constraint.id);
  }
  if (REQUIRED_CONSTRAINTS.some(id => !constraintIds.has(id))) throw Error('Missing tactics context reviewed constraint');
  keys(copy.roles, ROLE_KEYS, 'capabilities');
  for (const role of ROLE_KEYS) text(copy.roles[role], 1024, 'capability');
  text(copy.limitations, 2048, 'rule limitations');
  return copy;
}

function observedGame(value, now) {
  if (value === null) return null;
  if (now === undefined) throw Error('Explicit now is required for tactics context freshness validation');
  const copy = boundedCopy(value, LIMITS.gameBytes);
  keys(copy, GAME_KEYS, 'game evidence');
  if (!validHash(copy.snapshotSha256) || copy.executableTask !== false) throw Error('Invalid tactics context read-only game evidence');
  for (const name of ['provenance', 'interpretation', 'scope']) text(copy[name], 1024, name);
  Snapshot.validate(copy.state, now);
  if (Buffer.byteLength(JSON.stringify(copy.state), 'utf8') > Snapshot.MAX_BYTES ||
      copy.state.objects.filter(object => object.kind === 'pin').length > 6 ||
      copy.state.objects.filter(object => object.kind === 'cup').length > 4) throw Error('Tactics context snapshot exceeds visibility or byte budget');
  const digest = createHash('sha256').update(JSON.stringify(copy.state)).digest('hex');
  if (copy.snapshotSha256 !== digest) throw Error('Tactics context snapshot hash does not match supplied state');
  return copy;
}

function rulesText(value) {
  return [
    `Game: ${value.game}; season=${value.season}; manual=${value.manualVersion}`,
    `Review: checkedAt=${value.sources.checkedAt}; scope=${JSON.stringify(value.sources.scope)}`,
    `Timing(seconds): ${JSON.stringify(value.timing)}`,
    `Points: ${JSON.stringify(value.points)}`,
    `Goals[${value.goalColumns.join(',')}]: ${JSON.stringify(value.goals)}`,
    ...value.constraints.map(constraint => `Constraint ${constraint.id}: ${JSON.stringify(constraint.fact)}`),
    ...ROLE_KEYS.map(role => `Capability ${role}: ${JSON.stringify(value.roles[role])}`),
    `Limitations: ${JSON.stringify(value.limitations)}`
  ].join('\n');
}

function gameText(value) {
  if (value === null) return 'Snapshot: unavailable. No observed phase, time, score, possessions or DQ flags. ' +
    'Request a fresh Simulator snapshot before state-specific advice. Read-only; executableTask=false.';
  const state = value.state;
  const robots = state.robots.map(robot => [robot.id, robot.x, robot.y, robot.headingDeg, robot.pinId, robot.cupId, robot.disqualified]);
  const goals = state.goals.map(goal => [goal.id, goal.stackDepth, goal.top]);
  const toggles = state.toggles.map(toggle => [toggle.id, toggle.state, toggle.seated, toggle.contact]);
  const pins = state.objects.filter(object => object.kind === 'pin').map(object => [object.id, object.x, object.y, object.halves, object.supportId]);
  const cups = state.objects.filter(object => object.kind === 'cup').map(object => [object.id, object.x, object.y, object.up]);
  return [
    `Snapshot: schema=${state.schemaVersion}; capturedAt=${state.capturedAt}; season=${state.season}; manual=${state.manualVersion}; ` +
      `world=${state.world}; mode=${state.mode}; selectedRobotId=${state.selectedRobotId}`,
    `Phase=${state.phase}; elapsedSeconds=${state.clock}; reportedScore=${JSON.stringify(state.score)}`,
    `Robots[id,xIn,yIn,headingDeg,pinId,cupId,disqualified]: ${JSON.stringify(robots)}`,
    `Goals[id,stackDepth,top]: ${JSON.stringify(goals)}`,
    `Toggles[id,state,seated,contact]: ${JSON.stringify(toggles)}`,
    `NearbyPins[id,xIn,yIn,halves,supportId]: ${JSON.stringify(pins)}`,
    `NearbyCups[id,xIn,yIn,up]: ${JSON.stringify(cups)}`,
    `Resources: ${JSON.stringify(state.resources)}`,
    `Provenance: ${JSON.stringify(value.provenance)}`,
    `Interpretation: ${JSON.stringify(value.interpretation)}`,
    `Visibility: ${JSON.stringify(value.scope)}`,
    'Warning: partial visibility; unlisted objects are unknown, not absent. User-supplied state is not independently ' +
      'authenticated or replayed and may be stale after inference.',
    'Read-only; executableTask=false.'
  ].join('\n');
}

function freeze(value) {
  if (value !== null && typeof value === 'object') {
    for (const entry of Object.values(value)) freeze(entry);
    Object.freeze(value);
  }
  return value;
}

function result(text, provenance, maxBytes) {
  if (Buffer.byteLength(text, 'utf8') > maxBytes) throw Error('Tactics context text exceeds byte budget; evidence cannot be truncated');
  return freeze({text, provenance});
}

function compactRules(value, configuration = {}) {
  const {maxBytes} = options(configuration);
  const provenance = reviewedRules(value);
  return result(rulesText(provenance), provenance, maxBytes);
}

function compactGameEvidence(value = null, configuration = {}) {
  const {maxBytes, now} = options(configuration);
  const provenance = observedGame(value, now);
  return result(gameText(provenance), provenance, maxBytes);
}

function compactTacticsContext(ruleEvidence, gameEvidence = null, configuration = {}) {
  const {maxBytes, now} = options(configuration);
  const rules = reviewedRules(ruleEvidence), game = observedGame(gameEvidence, now);
  return result(`${rulesText(rules)}\n\n${gameText(game)}`, {rules, game}, maxBytes);
}

module.exports = {LIMITS, compactRules, compactGameEvidence, compactTacticsContext};
