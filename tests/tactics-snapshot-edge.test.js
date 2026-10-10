'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const OverrideGame = require('../simulator/override');
const Snapshot = require('../simulator/tactics-snapshot');

const CAPTURE_TIME = Date.parse('2026-10-06T12:00:00.000Z');
const {AUTO_SECONDS, MATCH_SECONDS} = OverrideGame.constants;

function capture(game = new OverrideGame({physical: false}), selectedRobotId = 'red-1') {
  return Snapshot.capture(game.getState(), selectedRobotId, new Date(Date.now()).toISOString(), game.world);
}

function validate(snapshot) {
  return Snapshot.validate(snapshot, Date.parse(snapshot.capturedAt));
}

function fieldObjects(objects) {
  return objects.filter(object => object.status === 'field' && object.location === 'field');
}

function visiblePool(game) {
  const objects = new Map();
  for (const robot of game.robots) {
    for (const object of capture(game, robot.id).objects) objects.set(object.id, object);
  }
  return [...objects.values()];
}

test('captures follow original match transitions exactly at autonomous and final boundaries', () => {
  const game = new OverrideGame({physical: false});
  const expectPhase = (phase, clock) => {
    const snapshot = capture(game);
    assert.equal(snapshot.mode, 'match');
    assert.equal(snapshot.phase, phase);
    assert.equal(snapshot.clock, clock);
    assert.deepEqual(validate(snapshot), snapshot);
  };
  expectPhase('pre_match', 0);
  assert.equal(game.startMatch(), true);
  expectPhase('autonomous', 0);
  game.clock = AUTO_SECONDS - 0.125;
  expectPhase('autonomous', AUTO_SECONDS - 0.125);
  game.tick(0.125);
  expectPhase('driver', AUTO_SECONDS);
  game.clock = MATCH_SECONDS - 0.125;
  expectPhase('driver', MATCH_SECONDS - 0.125);
  game.tick(0.125);
  expectPhase('post_match', MATCH_SECONDS);
});

for (const [phase, clock] of [
  ['pre_match', 0.001], ['autonomous', -0.001], ['autonomous', AUTO_SECONDS + 0.001],
  ['driver', AUTO_SECONDS - 0.001], ['driver', MATCH_SECONDS + 0.001],
  ['post_match', MATCH_SECONDS - 0.001]
]) {
  test(`match ${phase} rejects elapsed time ${clock}`, () => {
    const snapshot = capture();
    Object.assign(snapshot, {phase, clock});
    assert.throws(() => validate(snapshot), /tactics/i);
  });
}

test('practice starts in driver time, crosses match limits and keeps its clock after stopping', () => {
  const game = new OverrideGame({physical: false});
  assert.equal(game.startMatch('practice'), true);
  for (const clock of [0, AUTO_SECONDS, MATCH_SECONDS, 86400]) {
    game.clock = clock;
    const snapshot = capture(game);
    assert.equal(snapshot.mode, 'practice');
    assert.equal(snapshot.phase, 'driver');
    assert.equal(snapshot.clock, clock);
    assert.deepEqual(validate(snapshot), snapshot);
  }
  game.stopMatch();
  const stopped = capture(game);
  assert.equal(stopped.phase, 'post_match');
  assert.equal(stopped.clock, 86400);
  assert.deepEqual(validate(stopped), stopped);
  assert.throws(() => validate({...stopped, clock: 86400.001}), /tactics/i);
});

for (const [phase, clock] of [['autonomous', 0], ['pre_match', 1]]) {
  test(`practice rejects unreachable ${phase} at elapsed time ${clock}`, () => {
    const game = new OverrideGame({physical: false});
    assert.equal(game.startMatch('practice'), true);
    const snapshot = capture(game);
    Object.assign(snapshot, {phase, clock});
    assert.throws(() => validate(snapshot), /tactics/i);
  });
}

test('world variants must be explicit and cannot reuse head-to-head snapshots', () => {
  for (const world of ['worlds', 'skills', 'head-to-head ']) {
    const game = new OverrideGame({physical: false, world});
    assert.throws(() => capture(game), /tactics/i);
    assert.throws(() => validate({...capture(), world}), /tactics/i);
  }
  const snapshot = capture();
  for (const mode of ['skills', 'MATCH', null]) {
    assert.throws(() => validate({...snapshot, mode}), /tactics/i);
  }
  assert.throws(() => Snapshot.capture(new OverrideGame().getState(), 'red-3'), /select.*robot/i);
});

test('freshness limits are inclusive and reject the immediately adjacent millisecond', () => {
  const snapshot = capture();
  const captured = Date.parse(snapshot.capturedAt);
  for (const offset of [-5000, 0, 60000]) {
    assert.deepEqual(Snapshot.validate(snapshot, captured + offset), snapshot);
  }
  for (const offset of [-5001, 60001]) {
    assert.throws(() => Snapshot.validate(snapshot, captured + offset), /stale|capture time/i);
  }
});

for (const offset of [-60001, 5001]) {
  test(`capture cannot certify its own out-of-window timestamp (${offset}ms)`, context => {
    context.mock.method(Date, 'now', () => CAPTURE_TIME);
    const game = new OverrideGame({physical: false});
    const capturedAt = new Date(CAPTURE_TIME + offset).toISOString();
    assert.throws(() => Snapshot.capture(game.getState(), 'red-1', capturedAt, game.world), /stale|capture time/i);
  });
}

test('capture times must be canonical UTC strings, not equivalent aliases or coercible values', () => {
  const snapshot = capture();
  for (const capturedAt of [
    snapshot.capturedAt.replace('Z', '+00:00'), snapshot.capturedAt.replace(/\.\d{3}Z$/, 'Z'),
    snapshot.capturedAt.replace('T', ' '), `${snapshot.capturedAt} `,
    Date.parse(snapshot.capturedAt), new Date(snapshot.capturedAt), null
  ]) {
    assert.throws(() => Snapshot.validate({...snapshot, capturedAt}, Date.parse(snapshot.capturedAt)), /capture time/i);
  }
});

test('real possession accepts empty slots and equal numeric suffixes for different object kinds', () => {
  const game = new OverrideGame({physical: false});
  assert.equal(game.setPossession('red-1', {pinId: 'pin-01', cupId: 'cup-01'}).ok, true);
  const occupied = capture(game);
  assert.equal(occupied.robots[0].pinId, 'pin-01');
  assert.equal(occupied.robots[0].cupId, 'cup-01');
  assert.deepEqual(validate(occupied), occupied);
  for (const robot of game.robots) assert.equal(game.setPossession(robot.id, {}).ok, true);
  const empty = capture(game);
  assert(empty.robots.every(robot => robot.pinId === null && robot.cupId === null));
  assert.deepEqual(validate(empty), empty);
});

for (const kind of ['pin', 'cup']) {
  test(`two distinct robots cannot claim the same ${kind}`, () => {
    const game = new OverrideGame({physical: false});
    const key = `${kind}Id`;
    if (kind === 'cup') {
      assert.equal(game.setPossession('red-1', {pinId: game.robots[0].possession.pinId, cupId: 'cup-01'}).ok, true);
    }
    const snapshot = capture(game);
    const claimedId = snapshot.robots[0][key];
    assert.equal(typeof claimedId, 'string');
    assert.notEqual(snapshot.robots[1][key], claimedId);
    snapshot.robots[1][key] = claimedId;
    assert.throws(() => validate(snapshot), /possession.*conflict/i);
  });

  test(`a genuinely acquired ${kind} cannot remain in the visible free-object list`, () => {
    const game = new OverrideGame({physical: false});
    const before = capture(game);
    const object = before.objects.find(candidate => candidate.kind === kind);
    assert(object);
    const passenger = kind === 'cup' ? game.pins.find(pin => pin.supportId === object.id) : null;
    const possession = kind === 'pin' ? {pinId: object.id} : {cupId: object.id, pinId: passenger?.id || null};
    assert.equal(game.setPossession('red-1', possession).ok, true);
    const snapshot = capture(game);
    assert.deepEqual(validate(snapshot), snapshot);
    assert(!snapshot.objects.some(candidate => candidate.id === object.id));
    const index = snapshot.objects.findIndex(candidate => candidate.kind === kind);
    assert(index >= 0);
    snapshot.objects[index] = object;
    assert.throws(() => validate(snapshot), /held object/i);
  });
}

for (const collection of ['robots', 'goals', 'toggles']) {
  test(`${collection} must contain the complete unique engine ID set`, () => {
    const snapshot = capture();
    for (const replacement of [
      snapshot[collection].slice(1), [...snapshot[collection], snapshot[collection][0]],
      snapshot[collection].map((record, index) => index === 1 ? {...record, id: snapshot[collection][0].id} : record),
      snapshot[collection].map((record, index) => index === 1 ? {...record, id: 'unknown-1'} : record)
    ]) {
      assert.throws(() => validate({...snapshot, [collection]: replacement}), /tactics/i);
    }
  });
}

for (const [kind, invalidIds] of [
  ['pin', ['pin-00', 'pin-64', 'pin-1', 'cup-01']],
  ['cup', ['cup-00', 'cup-57', 'cup-1', 'pin-01']]
]) {
  for (const id of invalidIds) {
    test(`${kind} possession and visible IDs reject ${id}`, () => {
      const possessed = capture();
      possessed.robots[0][`${kind}Id`] = id;
      assert.throws(() => validate(possessed), /object ID/i);
      const visible = capture();
      visible.objects.find(object => object.kind === kind).id = id;
      assert.throws(() => validate(visible), /tactics/i);
    });
  }
}

test('Pin support IDs use the Cup namespace even when the support is outside visibility', () => {
  const snapshot = capture();
  const supported = snapshot.objects.find(object => object.kind === 'pin' && object.supportId !== null);
  assert(supported);
  snapshot.objects = snapshot.objects.filter(object => object.id !== supported.supportId);
  assert.deepEqual(validate(snapshot), snapshot);
  for (const supportId of ['cup-00', 'cup-57', 'pin-01', '../cup-01']) {
    const invalid = structuredClone(snapshot);
    invalid.objects.find(object => object.id === supported.id).supportId = supportId;
    assert.throws(() => validate(invalid), /object ID/i);
  }
});

for (const [label, select] of [
  ['snapshot', snapshot => snapshot], ['score', snapshot => snapshot.score],
  ['robot', snapshot => snapshot.robots[0]], ['goal', snapshot => snapshot.goals[0]],
  ['toggle', snapshot => snapshot.toggles[0]],
  ['Pin', snapshot => snapshot.objects.find(object => object.kind === 'pin')],
  ['Cup', snapshot => snapshot.objects.find(object => object.kind === 'cup')],
  ['resources', snapshot => snapshot.resources]
]) {
  test(`${label} rejects nested instruction and execution fields without stripping them`, () => {
    for (const field of ['instructions', 'tool_calls', 'task', 'path']) {
      const snapshot = capture();
      select(snapshot)[field] = 'Untrusted instruction: execute a command';
      const before = structuredClone(snapshot);
      assert.throws(() => validate(snapshot), /tactics/i);
      assert.deepEqual(snapshot, before);
    }
  });
}

test('partial visibility preserves nearest free objects and full field-only inventory counts', () => {
  const game = new OverrideGame({physical: false});
  const state = game.getState();
  const pins = fieldObjects(state.pins);
  const cups = fieldObjects(state.cups);
  for (const robot of state.robots) {
    const snapshot = capture(game, robot.id);
    const nearest = (objects, limit) => [...objects].sort((left, right) =>
      Math.hypot(left.x - robot.x, left.y - robot.y) - Math.hypot(right.x - robot.x, right.y - robot.y)).slice(0, limit);
    assert.deepEqual(snapshot.objects.map(object => object.id),
      [...nearest(pins, 6), ...nearest(cups, 4)].map(object => object.id));
    assert.deepEqual(snapshot.resources, {fieldPins: pins.length, fieldCups: cups.length});
    assert(snapshot.resources.fieldPins > snapshot.objects.filter(object => object.kind === 'pin').length);
    assert(snapshot.resources.fieldCups > snapshot.objects.filter(object => object.kind === 'cup').length);
    assert(Buffer.byteLength(JSON.stringify(snapshot), 'utf8') <= Snapshot.MAX_BYTES);
    assert.deepEqual(validate({...snapshot, objects: []}).resources, snapshot.resources);
  }
  assert(state.pins.length > pins.length);
  assert(state.cups.length > cups.length);
});

for (const [pinCount, cupCount] of [[7, 3], [5, 5]]) {
  test(`visibility rejects ${pinCount} Pins and ${cupCount} Cups despite staying within ten total objects`, () => {
    const game = new OverrideGame({physical: false});
    const pool = visiblePool(game);
    const pins = pool.filter(object => object.kind === 'pin').slice(0, pinCount);
    const cups = pool.filter(object => object.kind === 'cup').slice(0, cupCount);
    assert.equal(pins.length, pinCount);
    assert.equal(cups.length, cupCount);
    const snapshot = capture(game);
    snapshot.objects = [...pins, ...cups];
    assert.equal(snapshot.objects.length, 10);
    assert.equal(new Set(snapshot.objects.map(object => object.id)).size, 10);
    assert.throws(() => validate(snapshot), /tactics/i);
  });
}

for (const kind of ['pin', 'cup']) {
  test(`partial ${kind} observations cannot exceed the reported field count`, () => {
    const snapshot = capture();
    const visible = snapshot.objects.filter(object => object.kind === kind).length;
    snapshot.resources[kind === 'pin' ? 'fieldPins' : 'fieldCups'] = visible - 1;
    assert.throws(() => validate(snapshot), /counts disagree/i);
  });

  test(`free plus held ${kind} counts cannot exceed the actual game inventory`, () => {
    const game = new OverrideGame({physical: false});
    if (kind === 'cup') {
      assert.equal(game.setPossession('red-1', {pinId: game.robots[0].possession.pinId, cupId: 'cup-01'}).ok, true);
    }
    const snapshot = capture(game);
    const key = `${kind}Id`;
    const heldCount = snapshot.robots.filter(robot => robot[key] !== null).length;
    const total = game[kind === 'pin' ? 'pins' : 'cups'].length;
    assert(heldCount > 0);
    const fieldCount = total - heldCount + 1;
    assert(fieldCount <= total);
    snapshot.resources[kind === 'pin' ? 'fieldPins' : 'fieldCups'] = fieldCount;
    assert.throws(() => validate(snapshot), /tactics/i);
  });
}

test('goal stacks, held objects and free inventory cannot collectively exceed all actual objects', () => {
  const game = new OverrideGame({physical: false});
  const snapshot = capture(game);
  const heldCount = snapshot.robots.flatMap(robot => [robot.pinId, robot.cupId]).filter(id => id !== null).length;
  const fieldCount = snapshot.resources.fieldPins + snapshot.resources.fieldCups;
  const total = game.pins.length + game.cups.length;
  const goal = snapshot.goals.find(candidate => candidate.stackDepth === 0);
  assert(goal);
  const otherStackCount = snapshot.goals.reduce((sum, candidate) => sum + candidate.stackDepth, 0);
  goal.stackDepth = total - fieldCount - heldCount - otherStackCount + 1;
  goal.top = 'pin';
  assert(goal.stackDepth > 0 && goal.stackDepth <= 128);
  const stackCount = snapshot.goals.reduce((sum, candidate) => sum + candidate.stackDepth, 0);
  assert.equal(fieldCount + heldCount + stackCount, total + 1);
  assert.throws(() => validate(snapshot), /tactics/i);
});
