(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./override'));
  else root.NationalsTacticsSnapshot = factory(root.OverrideGame);
})(globalThis, function(OverrideGame) {
  'use strict';

  const ROBOTS = ['red-1', 'red-2', 'blue-1', 'blue-2'];
  const COLORS = ['red', 'blue', 'yellow'];
  const GOALS = OverrideGame.layouts.goals;
  const TOGGLES = OverrideGame.layouts.toggles;
  const HALF = OverrideGame.constants.FIELD_HALF;
  const MAX_BYTES = 4096;
  function keys(value, allowed, label) {
    if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !allowed.includes(key))) {
      throw Error(`Invalid tactics ${label}`);
    }
  }
  function number(value, minimum, maximum, label) {
    if (!Number.isFinite(value) || value < minimum || value > maximum) throw Error(`Invalid tactics ${label}`);
    return value;
  }
  function count(value, maximum, label) {
    if (!Number.isInteger(value)) throw Error(`Invalid tactics ${label}`);
    return number(value, 0, maximum, label);
  }
  function objectId(value, kind, nullable = false) {
    if (nullable && value === null) return value;
    if (typeof value !== 'string' || !new RegExp(`^${kind}-[0-9]{2}$`).test(value) ||
        Number(value.slice(-2)) < 1 || Number(value.slice(-2)) > (kind === 'pin' ? 63 : 56)) throw Error('Invalid tactics object ID');
    return value;
  }
  function listed(values, expected, label) {
    if (!Array.isArray(values) || values.length !== expected.length ||
        new Set(values.map(value => value?.id)).size !== expected.length || values.some(value => !expected.includes(value?.id))) {
      throw Error(`Invalid tactics ${label}`);
    }
    return values;
  }
  function validate(value, now = Date.now()) {
    keys(value, ['schemaVersion', 'capturedAt', 'selectedRobotId', 'manualVersion', 'season', 'world', 'phase', 'mode',
      'clock', 'score', 'robots', 'goals', 'toggles', 'objects', 'resources'], 'snapshot');
    if (value.schemaVersion !== 1 || value.manualVersion !== '2.0' || value.season !== '2026-2027' || value.world !== 'head-to-head' ||
        !ROBOTS.includes(value.selectedRobotId) || !['pre_match', 'autonomous', 'driver', 'post_match'].includes(value.phase) ||
        !['match', 'practice'].includes(value.mode)) throw Error('Unsupported tactics game or phase');
    const captured = Date.parse(value.capturedAt);
    if (!Number.isFinite(captured) || new Date(captured).toISOString() !== value.capturedAt || now - captured > 60000 || captured - now > 5000) {
      throw Error('Tactics snapshot is stale or has an invalid capture time; capture the Simulator again');
    }
    number(value.clock, 0, value.mode === 'match' ? 120 : 86400, 'clock');
    if (value.mode === 'match' && ((value.phase === 'autonomous' && value.clock >= 15) ||
        (value.phase === 'driver' && (value.clock < 15 || value.clock >= 120)) ||
        (value.phase === 'pre_match' && value.clock !== 0) || (value.phase === 'post_match' && value.clock !== 120))) {
      throw Error('Tactics clock and match phase disagree');
    }
    if (value.mode === 'practice' && !['driver', 'post_match'].includes(value.phase)) throw Error('Unsupported tactics practice phase');
    keys(value.score, ['red', 'blue'], 'score');
    for (const alliance of ['red', 'blue']) count(value.score[alliance], 2000, 'reported score');
    for (const robot of listed(value.robots, ROBOTS, 'robots')) {
      keys(robot, ['id', 'x', 'y', 'headingDeg', 'pinId', 'cupId', 'disqualified'], 'robot');
      number(robot.x, -HALF, HALF, 'robot X'); number(robot.y, -HALF, HALF, 'robot Y');
      number(robot.headingDeg, -1000000, 1000000, 'robot heading');
      objectId(robot.pinId, 'pin', true); objectId(robot.cupId, 'cup', true);
      if (typeof robot.disqualified !== 'boolean') throw Error('Invalid tactics disqualification flag');
    }
    const held = value.robots.flatMap(robot => [robot.pinId, robot.cupId]).filter(Boolean);
    if (new Set(held).size !== held.length) throw Error('Tactics possession IDs conflict');
    for (const goal of listed(value.goals, GOALS.map(goal => goal.id), 'goals')) {
      keys(goal, ['id', 'stackDepth', 'top'], 'goal');
      count(goal.stackDepth, 128, 'stack depth');
      if (!['empty', 'pin', 'cup'].includes(goal.top) || (goal.stackDepth === 0) !== (goal.top === 'empty')) throw Error('Invalid tactics goal top');
    }
    for (const toggle of listed(value.toggles, TOGGLES.map(toggle => toggle.id), 'toggles')) {
      keys(toggle, ['id', 'state', 'seated', 'contact'], 'toggle');
      if (!COLORS.includes(toggle.state) || typeof toggle.seated !== 'boolean' || typeof toggle.contact !== 'boolean') throw Error('Invalid tactics toggle');
    }
    if (!Array.isArray(value.objects) || value.objects.length > 10 || new Set(value.objects.map(object => object?.id)).size !== value.objects.length) {
      throw Error('Invalid tactics nearby-object budget');
    }
    if (value.objects.filter(object => object.kind === 'pin').length > 6 || value.objects.filter(object => object.kind === 'cup').length > 4) throw Error('Invalid tactics visibility limits');
    for (const object of value.objects) {
      keys(object, ['id', 'kind', 'x', 'y', 'halves', 'up', 'supportId'], 'nearby object');
      if (!['pin', 'cup'].includes(object.kind)) throw Error('Invalid tactics object kind');
      objectId(object.id, object.kind);
      if (held.includes(object.id)) throw Error('A held object cannot also be a nearby free object');
      number(object.x, -HALF, HALF, 'object X'); number(object.y, -HALF, HALF, 'object Y');
      if (object.kind === 'pin') {
        if (!Array.isArray(object.halves) || object.halves.length !== 2 || object.halves.some(color => !COLORS.includes(color)) || object.up !== undefined) throw Error('Invalid tactics Pin halves');
        objectId(object.supportId, 'cup', true);
      } else if (!['opaque', 'transparent'].includes(object.up) || object.halves !== undefined || object.supportId !== undefined) throw Error('Invalid tactics Cup orientation');
    }
    keys(value.resources, ['fieldPins', 'fieldCups'], 'resource counts');
    count(value.resources.fieldPins, 63, 'Pin count'); count(value.resources.fieldCups, 56, 'Cup count');
    const heldPins = value.robots.filter(robot => robot.pinId !== null).length;
    const heldCups = value.robots.filter(robot => robot.cupId !== null).length;
    if (value.resources.fieldPins + heldPins > 63 || value.resources.fieldCups + heldCups > 56 ||
        value.resources.fieldPins + value.resources.fieldCups + heldPins + heldCups + value.goals.reduce((total, goal) => total + goal.stackDepth, 0) > 119) throw Error('Invalid tactics combined inventory counts');
    if (value.objects.filter(object => object.kind === 'pin').length > value.resources.fieldPins ||
        value.objects.filter(object => object.kind === 'cup').length > value.resources.fieldCups) throw Error('Tactics object counts disagree');
    if (JSON.stringify(value).length > MAX_BYTES) throw Error('Tactics snapshot exceeds context budget');
    return JSON.parse(JSON.stringify(value));
  }
  function capture(state, selectedRobotId, capturedAt = new Date().toISOString(), world = state.world) {
    const selected = state.robots.find(robot => robot.id === selectedRobotId);
    if (!selected) throw Error('Select a Simulator robot before requesting tactics');
    const pins = state.pins.filter(pin => pin.status === 'field' && pin.location === 'field');
    const cups = state.cups.filter(cup => cup.status === 'field' && cup.location === 'field');
    const nearest = (objects, maximum) => [...objects].sort((left, right) =>
      Math.hypot(left.x - selected.x, left.y - selected.y) - Math.hypot(right.x - selected.x, right.y - selected.y)).slice(0, maximum);
    const value = {schemaVersion: 1, capturedAt, selectedRobotId, manualVersion: state.rules.manualVersion,
      season: state.rules.season, world, phase: state.phase, mode: state.mode, clock: state.clock, score: {...state.score},
      robots: state.robots.map(robot => ({id: robot.id, x: robot.x, y: robot.y, headingDeg: robot.theta,
        pinId: robot.possession.pinId, cupId: robot.possession.cupId, disqualified: state.disqualifiedRobots.includes(robot.id)})),
      goals: state.goals.map(goal => ({id: goal.id, stackDepth: goal.stack.length, top: goal.stack.at(-1)?.type || 'empty'})),
      toggles: state.toggles.map(toggle => ({id: toggle.id, state: toggle.state, seated: toggle.seated, contact: toggle.robotContact})),
      objects: [...nearest(pins, 6).map(pin => ({id: pin.id, kind: 'pin', x: pin.x, y: pin.y, halves: [...pin.halves], supportId: pin.supportId})),
        ...nearest(cups, 4).map(cup => ({id: cup.id, kind: 'cup', x: cup.x, y: cup.y, up: cup.up}))],
      resources: {fieldPins: pins.length, fieldCups: cups.length}};
    return validate(value);
  }
  return {capture, validate, MAX_BYTES};
});
