'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {declare, evaluate} = require('../tools/evaluate-gvf-turns');
const root = path.resolve(__dirname, '..');

test('GVF repair declares every development and historical case before simulation', () => {
  const protocol = declare();
  assert.equal(protocol.cases.length, 40);
  assert.equal(new Set(protocol.cases.map(world => world.id)).size, 40);
  const turns = protocol.cases.slice(0, 16);
  assert.deepEqual([...new Set(turns.map(world => world.task.goal.headingDeg))], [-180, -135, -90, -45, 45, 90, 135, 180]);
  assert.deepEqual(turns.map(world => world.seed), Array.from({length: 16}, (_, index) => 2026101601 + index));
  assert(turns.every(world => world.task.deadlineSeconds === 8 && /not an independent final/.test(world.scope)));
  assert(protocol.cases.slice(16).every(world => /historical development/.test(world.scope)));
  assert.deepEqual(protocol.modes, ['scripted', 'gvf']);
  assert.equal(protocol.maximumEpisodes, 160);
  assert.equal(protocol.exactReplayEpisodes, 160);
  assert.equal(protocol.maximumPhysicsTicksIncludingReplay,
    protocol.cases.reduce((sum, world) => sum + Math.ceil(world.task.deadlineSeconds / 0.01) * 8, 0));
  for (const flag of ['parameterSearch', 'training', 'independentQualification', 'promotion']) assert.equal(protocol[flag], false);
});

test('GVF repair protocol binds the real built WASM and complete evaluator bytes', () => {
  const protocol = declare();
  const source = fs.readFileSync(path.join(root, 'tools/evaluate-gvf-turns.js'));
  const wasm = fs.readFileSync(path.join(root, 'simulator/control.wasm'));
  assert.equal(protocol.scriptSha256, createHash('sha256').update(source).digest('hex'));
  assert.equal(protocol.currentWasmSha256, createHash('sha256').update(wasm).digest('hex'));
  assert.notEqual(protocol.currentWasmSha256, protocol.preservedWasmSha256);
});

test('GVF campaign cannot run without explicit remote authorization', () => {
  const previous = process.env.ROBOPROOF_APPROVED_REMOTE;
  delete process.env.ROBOPROOF_APPROVED_REMOTE;
  try {
    assert.throws(() => evaluate(), /Approved remote Linux/);
  } finally {
    if (previous !== undefined) process.env.ROBOPROOF_APPROVED_REMOTE = previous;
  }
});
