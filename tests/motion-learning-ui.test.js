'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('learning history and promotion UI preserve failures and never train or infer on load', async () => {
  const elements = new Map();
  const makeElement = () => ({children: [], listeners: {}, textContent: '', hidden: false, value: '',
    addEventListener(name, callback) { this.listeners[name] = callback; },
    replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); }});
  const element = id => {
    if (!elements.has(id)) elements.set(id, makeElement());
    return elements.get(id);
  };
  let saved = {available: true, status: 'completed', motionPolicyTrained: true,
    learnedImprovementVerified: false, models: [{seed: 42, steps: 33, updates: 1, trained: true, policySha256: 'one'}],
    evaluation: {models: [{seed: 42, policySha256: 'one', baseline: {success: 6, count: 24},
      untrained: {success: 7, count: 24}, learned: {success: 7, count: 24}, acceptance: {passed: false}}]}};
  const requests = [];
  const realm = vm.createContext({document: {getElementById: element, createElement: makeElement},
    location: {protocol: 'http:', hostname: '127.0.0.1'}, AbortController, console,
    MotionLearningSummary: require('../roboproof/dashboard/learning-summary'),
    fetch: async (url, options) => { requests.push({url, options});
      return {ok: true, json: async () => url.endsWith('/learning') ? saved : {readyForBoundedCpuExperiment: false}}; }});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../roboproof/dashboard/motion.js'), 'utf8'), realm);
  const settle = async () => { await new Promise(resolve => setImmediate(resolve)); };
  await settle();
  assert.deepEqual(requests.map(row => row.url), ['/api/motion/status']);
  element('motion-learning-refresh').listeners.click();
  await settle();
  assert.equal(requests.at(-1).url, '/api/motion/learning');
  assert.equal(requests.at(-1).options.method, undefined);
  assert.equal(element('motion-compare-learned').disabled, false);
  assert.equal(element('motion-use-verified').disabled, true);
  assert.match(element('motion-learning-status').textContent, /IMPROVEMENT NOT VERIFIED/);
  const row = element('motion-learning-table').children[0].children[1];
  assert.deepEqual(row.children.map(cell => cell.textContent), ['42', '33 / 1', '6/24', '7/24', '7/24', 'Failed', 'Not recorded in this older run']);
  saved = {...saved, independentResearch: {available: true, sourceRuntimeMatches: false, gateReportedPassed: false,
    models: [{seed: 42, policySha256: 'one', baseline: {success: 87, count: 128},
      untrained: {success: 80, count: 128}, learned: {success: 75, count: 128},
      gateReportedPassed: false, successRegressions: 12}]}};
  element('motion-learning-refresh').listeners.click();
  await settle();
  assert.match(element('motion-learning-status').textContent, /Historical source\/runtime.*75\/128.*12 success regressions.*INDEPENDENT GATE FAILED/);
  assert.equal(element('motion-learning-table').children[0].children[1].children[5].textContent, 'Failed · independent research');
  assert.equal(element('motion-use-verified').disabled, true);
  saved = {...saved, independentResearch: {...saved.independentResearch, gateReportedPassed: true}};
  element('motion-learning-refresh').listeners.click();
  await settle();
  assert.equal(element('motion-use-verified').disabled, true);
  assert.match(element('motion-learning-status').textContent, /research only, no promotion/);
  element('motion-use-verified').listeners.click();
  assert.equal(element('motion-mode').value, '');
  saved = {...saved, independentResearch: undefined, learnedImprovementVerified: true, models: saved.models.map(model => ({...model,
    history: [{meanApproxKL: 0.01, clipFraction: 0.2, meanEntropy: -1.1, meanActorLoss: 0.1,
      meanCriticLoss: 0.3, curriculumStageCounts: {'0': 1, '1': 2, '2': 3}}]}))};
  element('motion-learning-refresh').listeners.click();
  await settle();
  assert.equal(element('motion-use-verified').disabled, false);
  assert.match(element('motion-learning-table').children[0].children[1].children[6].textContent, /KL 0.0100.*world stages 0: 1 \/ 1: 2 \/ 2: 3/);
  const count = requests.length;
  element('motion-use-verified').listeners.click();
  assert.equal(element('motion-mode').value, 'learned-experiment');
  assert.equal(requests.length, count);
  assert(requests.every(row => row.options.method === undefined));
});
