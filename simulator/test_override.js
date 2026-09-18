"use strict";
const assert = require("assert");
const OverrideGame = require("./override.js");

const game = new OverrideGame();
let state = game.getState();
assert.strictEqual(state.field.width, 140.4);
assert.strictEqual(state.rules.manualVersion, "2.0");
assert.strictEqual(state.goals.length, 9);
assert.strictEqual(state.toggles.length, 4);
assert.strictEqual(state.loaders.length, 4);
assert.strictEqual(state.cups.length, 56);
assert.strictEqual(state.pins.length, 63);

assert(game.startMatch());
for (let i = 0; i < 15; i++) game.tick(1);
assert.strictEqual(game.getState().phase, "driver");
for (let i = 0; i < 95; i++) game.tick(1);
assert.strictEqual(game.getState().endgame, true);

assert.strictEqual(game.placePin("pin-01", "g-red-sw", { visibleHalf: "red" }).ok, true);
assert.strictEqual(game.score().red, 5);
assert.strictEqual(game.setToggle("toggle-west", "red"), true);
assert.strictEqual(game.placePin("pin-02", "g-red-sw", { visibleHalf: "red" }).ok, true);
assert.strictEqual(game.score().red, 10);

assert.strictEqual(game.setRobotPose("red-1", { x: 0, y: 0, perimeterContact: false }).ok, true);
assert.strictEqual(game.score().red, 18);
assert.strictEqual(game.setRobotSize("red-1", { width: 25 }).ok, false);
assert.strictEqual(game.setRobotSize("red-1", { width: 24, length: 24, height: 50 }).ok, true);

assert.strictEqual(game.setPossession("red-1", { pinId: "pin-10", cupId: "cup-01" }).ok, true);
assert.strictEqual(game.setPossession("blue-1", { pinId: "pin-10" }).ok, false);

const yellow = game.pins.find(pin => pin.color === "yellow");
assert(yellow);
assert.strictEqual(game.placePin(yellow.id, "g-red-sw").ok, true);
assert.strictEqual(game.score().red, 28);

while (!game.matchEnded) game.tick(1);
assert.strictEqual(game.getState().phase, "post_match");
assert.strictEqual(game.getState().autonomousBonus.red, 12);
console.log("Override rules tests passed");
