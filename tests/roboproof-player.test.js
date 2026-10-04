'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {prepare, sampleIndex, fieldTransform, headingVector, create} = require('../roboproof/dashboard/player');
const {sampleScenarios, clone} = require('../roboproof/core');
const {runScenario} = require('../roboproof/sim');
const result = runScenario(sampleScenarios(122, 42)[121], {telemetry: true});

test('successful exported scenario plays actual telemetry without inventing frames or a final sample', () => {
  assert.equal(result.scenario.scenario_id, 'mc-42-121');
  assert.equal(result.scenario.random_seed, 4004312486);
  assert.equal(result.passed, true);
  const prepared = prepare(result);
  assert.strictEqual(prepared.rows, result.telemetry);
  assert.equal(prepared.rows.length, 1000);
  assert.equal(prepared.start, 0);
  assert.equal(prepared.end, 9.99);
  assert.notDeepEqual(prepared.rows[0].truth, result.scenario.task.start);
  assert.notDeepEqual(prepared.rows.at(-1).truth, result.final_truth);
});

test('timeline selects recorded samples, holds between samples, and clamps at the ends', () => {
  const rows = result.telemetry;
  assert.equal(sampleIndex(rows, -10), 0);
  assert.equal(sampleIndex(rows, 0.105), 10);
  assert.equal(sampleIndex(rows, rows[350].time), 350);
  assert.equal(sampleIndex(rows, 100), 999);
  assert.throws(() => sampleIndex([], 0), /samples/);
  assert.throws(() => sampleIndex(rows, NaN), /finite/);
});

test('field projection preserves metric aspect ratio and puts +Y upwards', () => {
  const prepared = prepare(result), geometry = fieldTransform(prepared.bounds);
  const origin = geometry.point([0, 0]), right = geometry.point([1, 0]), up = geometry.point([0, 1]);
  assert(Math.abs(right[0] - origin[0] - (origin[1] - up[1])) < 1e-10);
  assert(up[1] < origin[1]);
  for (const row of prepared.rows) for (const name of ['truth', 'estimate', 'reference']) {
    const point = geometry.point(row[name]);
    assert(point[0] >= 40 && point[0] <= 760);
    assert(point[1] >= 40 && point[1] <= 460);
  }
});

test('heading arrows use clockwise-from-+Y convention, not a Cartesian angle', () => {
  assert.deepEqual(headingVector(0), [0, -1]);
  assert(Math.abs(headingVector(Math.PI / 2)[0] - 1) < 1e-12);
  assert(Math.abs(headingVector(Math.PI / 2)[1]) < 1e-12);
  assert(Math.abs(headingVector(-Math.PI / 2)[0] + 1) < 1e-12);
});

test('missing, nonfinite, unordered and unrenderable telemetry is rejected explicitly', () => {
  assert.throws(() => prepare({...result, telemetry: []}), /No recorded/);
  for (const change of [
    copy => { copy.telemetry[1].truth[0] = NaN; },
    copy => { delete copy.telemetry[0].estimate; },
    copy => { copy.telemetry[1].time = copy.telemetry[0].time; },
    copy => { copy.telemetry[1].time = -1; },
    copy => { copy.telemetry[1].time = Infinity; },
    copy => { copy.scenario.environment.radius = 0; },
    copy => { copy.telemetry[0].truth[0] = -1e308; copy.telemetry[1].truth[0] = 1e308; }
  ]) {
    const copy = clone(result);
    change(copy);
    assert.throws(() => prepare(copy));
  }
});

function fakePlayer() {
  const elements = new Map(), callbacks = new Map(), documentListeners = new Map();
  let nextId = 0;
  class Element {
    constructor() { this.attributes = {}; this.listeners = new Map(); this.children = []; this.value = ''; this.hidden = false; }
    setAttribute(name, value) { this.attributes[name] = String(value); if (name === 'id') elements.set(value, this); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
  }
  for (const name of ['play', 'restart', 'end', 'timeline', 'speed', 'time', 'position', 'heading', 'motion', 'motors', 'sample', 'close', 'field', 'scenario', 'outcome', 'summary']) elements.set(`sim-${name}`, new Element());
  elements.get('sim-speed').value = '1';
  const doc = {
    hidden: false,
    defaultView: {
      requestAnimationFrame: callback => { const frameId = ++nextId; callbacks.set(frameId, callback); return frameId; },
      cancelAnimationFrame: frameId => callbacks.delete(frameId)
    },
    createElementNS: () => new Element(),
    addEventListener: (name, listener) => documentListeners.set(name, listener)
  };
  const host = {ownerDocument: doc, hidden: true, querySelector: selector => elements.get(selector.slice(1))};
  const player = create(host);
  const advance = timestamp => { const pending = [...callbacks.values()]; callbacks.clear(); pending.forEach(callback => callback(timestamp)); };
  const click = id => elements.get(id).listeners.get('click')();
  const pose = () => JSON.parse(elements.get('sim-truth-robot').attributes['data-pose']);
  return {player, host, doc, elements, callbacks, advance, click, pose, documentListeners};
}

test('animation uses elapsed time and playback speed; pause, seek, restart and end use exact poses', () => {
  const harness = fakePlayer();
  harness.player.load(result);
  assert.equal(harness.host.hidden, false);
  assert.deepEqual(harness.pose(), result.telemetry[0].truth);
  harness.player.play();
  assert.equal(harness.callbacks.size, 1);
  harness.advance(1000);
  harness.advance(2250);
  assert.deepEqual(harness.pose(), result.telemetry[125].truth);
  harness.elements.get('sim-speed').value = '2';
  harness.advance(2750);
  assert.deepEqual(harness.pose(), result.telemetry[225].truth);
  harness.player.pause();
  assert.equal(harness.callbacks.size, 0);
  assert.equal(harness.elements.get('sim-play').textContent, 'Play');
  harness.player.seek(450);
  assert.deepEqual(harness.pose(), result.telemetry[450].truth);
  harness.click('sim-end');
  assert.deepEqual(harness.pose(), result.telemetry.at(-1).truth);
  harness.click('sim-play');
  assert.deepEqual(harness.pose(), result.telemetry[0].truth);
  harness.advance(3000);
  harness.advance(13000);
  assert.equal(harness.callbacks.size, 0);
  assert.deepEqual(harness.pose(), result.telemetry.at(-1).truth);
  harness.click('sim-restart');
  assert.deepEqual(harness.pose(), result.telemetry[0].truth);
  assert.equal(harness.callbacks.size, 0);
});

test('closing, changing scenarios or hiding the document cancels playback', () => {
  const harness = fakePlayer();
  harness.player.load(result);
  harness.player.play();
  harness.doc.hidden = true;
  harness.documentListeners.get('visibilitychange')();
  assert.equal(harness.callbacks.size, 0);
  harness.player.play();
  harness.player.load(result);
  assert.equal(harness.callbacks.size, 0);
  harness.player.play();
  harness.click('sim-close');
  assert.equal(harness.host.hidden, true);
  assert.equal(harness.callbacks.size, 0);
  harness.player.play();
  assert.equal(harness.callbacks.size, 0);
});

test('one recorded sample is displayable without scheduling an endless animation', () => {
  const harness = fakePlayer();
  harness.player.load({...result, telemetry: [result.telemetry[0]]});
  harness.player.play();
  assert.equal(harness.callbacks.size, 0);
  assert.deepEqual(harness.pose(), result.telemetry[0].truth);
});
