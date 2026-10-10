'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const settle = () => new Promise(resolve => setImmediate(resolve));
function dashboard(handler, saved = null) {
  const elements = new Map(), calls = [];
  const makeElement = () => ({children: [], listeners: {}, textContent: '', hidden: false, value: '',
    addEventListener(name, callback) { this.listeners[name] = callback; },
    replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); },
    requestSubmit() { this.listeners.submit({preventDefault() {}}); }});
  const element = id => {
    if (!elements.has(id)) elements.set(id, makeElement());
    return elements.get(id);
  };
  const realm = vm.createContext({document: {getElementById: element, createElement: makeElement},
    location: {protocol: 'http:', hostname: '127.0.0.1'}, AbortController, console,
    fetch: async (url, options = {}) => {
      calls.push({url, options});
      if (url.endsWith('/chat/latest')) return {ok: !!saved, json: async () => saved || {error: 'No saved chat yet'}};
      if (url.endsWith('/skills')) return {ok: true, json: async () => ({skills: []})};
      if (url.endsWith('/status')) return {ok: true, json: async () => ({model: 'fixture', baseUrl: 'http://127.0.0.1:8080/v1'})};
      return handler(url, options);
    }});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../roboproof/dashboard/nemotron.js'), 'utf8'), realm);
  return {element: id => element(`nemotron-${id}`), calls};
}
const chat = messages => ({id: 'fixture-chat-id', kind: 'chat', messages});

test('ordinary messages use chat, show readable safe bubbles and continue with saved conversation ID', async () => {
  const messages = [];
  const ui = dashboard(async (url, options) => {
    const body = JSON.parse(options.body);
    messages.push({role: 'user', content: body.message}, {role: 'assistant', content: '<img src=x onerror=alert(1)>\nReadable fixture reply.'});
    return {ok: true, json: async () => chat([...messages])};
  });
  await settle();
  assert(ui.calls.every(call => call.options.method === undefined));
  ui.element('prompt').value = 'Hello';
  ui.element('form').requestSubmit();
  assert.equal(ui.element('thinking').hidden, false);
  assert.equal(ui.element('plan').disabled, true);
  await settle();
  assert.equal(ui.calls.at(-1).url, '/api/nemotron/chat');
  assert.deepEqual(JSON.parse(ui.calls.at(-1).options.body), {message: 'Hello'});
  assert.equal(ui.element('messages').children.length, 2);
  const reply = ui.element('messages').children[1];
  assert.equal(reply.children[0].textContent, 'Nemotron');
  assert.match(reply.children[1].textContent, /<img/);
  assert.equal(reply.children[1].innerHTML, undefined);
  assert.equal(ui.element('prompt').value, '');
  assert.equal(ui.element('run').disabled, true);
  assert.equal(ui.element('thinking').hidden, true);
  ui.element('prompt').value = 'Explain that';
  ui.element('form').requestSubmit();
  await settle();
  assert.deepEqual(JSON.parse(ui.calls.at(-1).options.body), {message: 'Explain that', id: 'fixture-chat-id'});
  assert.equal(ui.element('messages').children.length, 4);
  assert.equal(ui.calls.some(call => /\/(plan|run)$/.test(call.url)), false);
});

test('saved chat restores without inference; New chat removes context but does not delete stored history', async () => {
  const saved = chat([{role: 'user', content: 'Previous question'}, {role: 'assistant', content: 'Previous reply'}]);
  const ui = dashboard(async (url, options) => ({ok: true, json: async () => chat([
    {role: 'user', content: JSON.parse(options.body).message}, {role: 'assistant', content: 'New reply'}])}), saved);
  await settle();
  assert.equal(ui.element('messages').children.length, 2);
  assert(ui.calls.every(call => call.options.method === undefined));
  const count = ui.calls.length;
  ui.element('new-chat').listeners.click();
  assert.equal(ui.calls.length, count);
  assert.equal(ui.element('messages').children.length, 0);
  assert.equal(ui.element('chat-empty').hidden, false);
  ui.element('prompt').value = 'New conversation';
  ui.element('form').requestSubmit();
  await settle();
  assert.equal(JSON.parse(ui.calls.at(-1).options.body).id, undefined);
});

test('Enter submits, Shift+Enter and IME composition do not send messages', async () => {
  const ui = dashboard(async () => ({ok: true, json: async () => chat([
    {role: 'user', content: 'Hello'}, {role: 'assistant', content: 'Reply'}])}));
  await settle();
  ui.element('prompt').value = 'Hello';
  const count = ui.calls.length;
  const prevented = [];
  for (const flags of [{shiftKey: true}, {isComposing: true}, {ctrlKey: true}]) {
    ui.element('prompt').listeners.keydown({key: 'Enter', ...flags, preventDefault() { prevented.push(flags); }});
  }
  assert.equal(ui.calls.length, count);
  assert.equal(prevented.length, 0);
  ui.element('prompt').listeners.keydown({key: 'Enter', preventDefault() { prevented.push('send'); }});
  await settle();
  assert.equal(ui.calls.at(-1).url, '/api/nemotron/chat');
  assert.deepEqual(prevented, ['send']);
});

test('Stop cancels the request, retains the draft and shows an honest status instead of a fake reply', async () => {
  const ui = dashboard(async (url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(Object.assign(Error('Stopped'), {name: 'AbortError'})), {once: true});
  }));
  await settle();
  ui.element('prompt').value = 'Keep this draft';
  ui.element('form').requestSubmit();
  assert.equal(ui.element('cancel').disabled, false);
  ui.element('cancel').listeners.click();
  await settle();
  assert.equal(ui.element('prompt').value, 'Keep this draft');
  assert.equal(ui.element('thinking').hidden, true);
  assert.match(ui.element('status').textContent, /Stopped/);
  assert.equal(ui.element('messages').children.filter(message => message.className.endsWith('assistant')).length, 0);
  assert.equal(ui.element('messages').children.at(-1).children[0].textContent, 'Chat status');
});

test('movement preparation remains an explicit action with human-readable coordinates and separate approval', async () => {
  const task = {start: {xIn: 0, yIn: 0, headingDeg: 0}, goal: {xIn: 0, yIn: 24, headingDeg: 0}, deadlineSeconds: 10};
  const ui = dashboard(async url => ({ok: true, json: async () => {
    assert.equal(url, '/api/nemotron/plan');
    return {id: 'fixture-task-id', status: 'prepared', task, message: 'Move forward 24 inches. This is a proposal, not a result.'};
  }}));
  await settle();
  ui.element('prompt').value = 'From (0,0), go to (0,24) inches';
  await ui.element('prepare').listeners.click();
  assert.equal(ui.calls.at(-1).url, '/api/nemotron/plan');
  assert.match(ui.element('task-summary').textContent, /\(0, 24\).*inches.*10 seconds/);
  assert.equal(ui.element('message').textContent, 'Move forward 24 inches. This is a proposal, not a result.');
  assert.equal(ui.element('run').disabled, false);
  assert.equal(ui.calls.some(call => call.url.endsWith('/run')), false);
  const html = fs.readFileSync(path.join(__dirname, '../roboproof/dashboard/index.html'), 'utf8');
  assert.match(html, /<summary>Robot tools<\/summary>/);
  assert.match(html, /<details><summary>Exact task coordinates<\/summary>/);
  assert.match(html, /<summary>Connection, privacy &amp; limitations<\/summary>/);
});
