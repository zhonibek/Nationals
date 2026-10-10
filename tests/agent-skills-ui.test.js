'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('skill UI loads metadata without inference, explicitly selects evidence and disables analysis execution', async () => {
  const elements = new Map();
  const makeElement = () => ({children: [], listeners: {}, textContent: '', hidden: false, value: '',
    addEventListener(name, callback) { this.listeners[name] = callback; },
    replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); }});
  const element = id => {
    if (!elements.has(id)) elements.set(id, makeElement());
    return elements.get(id);
  };
  const requests = [];
  const realm = vm.createContext({document: {getElementById: element, createElement: makeElement},
    location: {protocol: 'http:', hostname: '127.0.0.1'}, AbortController, console,
    fetch: async (url, options = {}) => {
      requests.push({url, options});
      const value = url.endsWith('/chat/latest') ? {error: 'No saved chat yet'}
        : url.endsWith('/skills') ? {skills: [{name: 'diagnose-motion', description: 'Fixture skill'}]}
        : url.endsWith('/latest') ? {id: 'saved-motion-id'}
        : url.endsWith('/plan') ? {id: 'saved-analysis-id', status: 'analyzed', message: '<script>not executable</script>',
          skills: {loaded: [{name: 'diagnose-motion'}]}, analysisEvidence: {motion: {reason: 'time_limit'}}}
        : {model: 'fixture', baseUrl: 'http://127.0.0.1:8080/v1'};
      return {ok: !url.endsWith('/chat/latest'), json: async () => value};
    }});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../roboproof/dashboard/nemotron.js'), 'utf8'), realm);
  const settle = async () => { await new Promise(resolve => setImmediate(resolve)); };
  await settle();
  assert.deepEqual(requests.map(row => row.url), ['/api/nemotron/chat/latest', '/api/nemotron/status', '/api/nemotron/skills']);
  assert(requests.every(row => row.options.method === undefined));
  assert.equal(element('nemotron-skills').children.length, 1);
  element('nemotron-diagnose-latest').listeners.click();
  await settle();
  assert.equal(requests.at(-2).url, '/api/motion/latest');
  assert.equal(requests.at(-1).url, '/api/nemotron/plan');
  assert.deepEqual(JSON.parse(requests.at(-1).options.body).evidence, {source: 'motion', id: 'saved-motion-id'});
  assert.equal(element('nemotron-analysis').hidden, false);
  assert.equal(element('nemotron-proposal').hidden, true);
  assert.equal(element('nemotron-run').disabled, true);
  assert.equal(element('nemotron-explain-run').disabled, true);
  assert.match(element('nemotron-message').textContent, /<script>/);
  assert.match(element('nemotron-analysis-facts').textContent, /time_limit/);
  assert.match(element('nemotron-loaded-skills').textContent, /diagnose-motion/);
  assert.equal(element('nemotron-download').disabled, false);
  assert.equal(requests.some(row => row.url.endsWith('/run')), false);
});
