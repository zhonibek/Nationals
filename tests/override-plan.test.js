'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Plan = require('../simulator/override-plan');
const Game = require('../simulator/override');
const Engine = require('../simulator/engine');
const Control = require('../simulator/control-runtime');
const {loadHeadless} = require('../simulator/headless');

test('plan action IDs from the tactical model are parsed, anything else is refused', () => {
  assert.deepEqual(Plan.parseAction('load-score-cup:red-2:loader-red-n:g-neutral-tall'),
    {id: 'load-score-cup:red-2:loader-red-n:g-neutral-tall', kind: 'load-score-cup', robotId: 'red-2', loaderId: 'loader-red-n', goalId: 'g-neutral-tall'});
  assert.deepEqual(Plan.parseAction('toggle:red-1:toggle-west'), {id: 'toggle:red-1:toggle-west', kind: 'toggle', robotId: 'red-1', targetId: 'toggle-west'});
  for (const bad of ['teleport:red-1', 'score-pin:red-1', 42, 'x'.repeat(201)]) assert.throws(() => Plan.parseAction(bad));
});

test('routes go around goals, and approaches keep the 18 in body inside the walls', () => {
  const game = new Game({world: 'head-to-head', physical: false});
  const obstacles = game.goals.map(goal => ({x: goal.x, y: goal.y, pass: Plan.constants.GOAL_PASS, turn: Plan.constants.GOAL_TURN}));
  const points = Plan.route({x: 30, y: 0}, {x: -30, y: 0}, obstacles);
  assert.ok(points.length > 1, 'the tall goal is in the way');
  let from = {x: 30, y: 0};
  for (const point of points) {
    for (let k = 0; k <= 40; k++) {
      const x = from.x + (point.x - from.x) * k / 40, y = from.y + (point.y - from.y) * k / 40;
      assert.ok(Math.hypot(x, y) >= Plan.constants.GOAL_PASS - 1e-9, `(${x}, ${y}) passes the tall goal`);
    }
    from = point;
  }
  assert.deepEqual(points.at(-1), {x: -30, y: 0});
  // A wall Toggle cannot be reached facing it diagonally; the robot squares up to the wall instead.
  const toggle = game.toggles.find(candidate => candidate.id === 'toggle-west');
  const plan = Plan.approach(toggle, {x: -40, y: -43}, 13, game.goals, []);
  assert.equal(plan.fits, true);
  assert.equal(Math.round(plan.heading), -90);
  assert.ok(Plan.bodyFits(plan.centre, plan.heading, game.goals));
  assert.ok(Plan.squaresOverlap({x: 0, y: 0}, 0, {x: 20, y: 0}, 45, 0));
  assert.ok(!Plan.squaresOverlap({x: 0, y: 0}, 0, {x: 19, y: 0}, 0, 0.5));
});

test('the original Simulator carries out a planned placement through the physical rules', () => {
  loadHeadless();
  const compiled = new WebAssembly.Module(fs.readFileSync(path.join(__dirname, '../simulator/control.wasm')));
  const sim = new Engine.VexRobotSimulator(Control.fromModule(compiled), {Fleet: Plan.planFleet(['score-pin:red-1:pin-01:g-red-sw'])});
  sim.override = new Game({world: 'head-to-head'});
  assert.equal(sim.startGame('match').ok, true);
  const runner = sim.fleet.runner;
  while (!runner.done && sim.simTime < 45) sim.update(0.05);
  assert.equal(runner.done, true);
  const [result] = runner.results;
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.ok(result.motions >= 1 && result.steps.some(step => step.type === 'game' && step.action === 'place' && step.ok));
  assert.ok(sim.override.goal('g-red-sw').stack.some(item => item.id === 'pin-01'));
  assert.ok(result.marginGain > 0);
  assert.equal(sim.override.robots.find(robot => robot.id === 'red-1').possession.pinId, null);
  assert.throws(() => Plan.planFleet(['park:blue-1', 'park:red-1']) && new Engine.VexRobotSimulator(Control.fromModule(compiled),
    {Fleet: Plan.planFleet(['park:blue-1', 'park:red-1'])}).startGame('match'), /outside the planning alliance/);
});
