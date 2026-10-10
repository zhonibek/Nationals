'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const {Worker} = require('node:worker_threads');
const {validate} = require('./core');
const {createClient, planTask} = require('./nemotron');
const {createStore} = require('./nemotron-store');
const {normalizeTask} = require('../simulator/motion');
const {readiness: perceptionReadiness} = require('./perception');
const {readiness: motionReadiness} = require('./motion-readiness');
const {createMotionStore} = require('./motion-store');
const MotionLearner = require('./motion-learner');
const {readIndependentEvaluation} = require('./independent-learning');
const {createRegistry} = require('./agent-skills');
const {motionEvidence} = require('./skill-evidence');
const {chatReply, validateConversation} = require('./nemotron-chat');
const Tactics = require('./tactics');

function createServer({reportPath = path.join(__dirname, 'runs/latest/report.json'), timeout = 120000,
  nemotronClient = createClient(), nemotronDirectory, nemotronChatDirectory, motionDirectory, learningDirectory, agentTimeout = 300000} = {}) {
  let report = fs.existsSync(reportPath) ? JSON.parse(fs.readFileSync(reportPath, 'utf8')) : null;
  let activeWorker = null;
  let activeAgent = null;
  const store = createStore(nemotronDirectory);
  const chatStore = createStore(nemotronChatDirectory || path.join(__dirname, 'runs/nemotron-chat'));
  const motionStore = createMotionStore(motionDirectory);
  const skills = createRegistry();
  const assets = new Map([['/', ['index.html', 'text/html']], ['/index.html', ['index.html', 'text/html']], ['/app.js', ['app.js', 'text/javascript']], ['/nemotron.js', ['nemotron.js', 'text/javascript']], ['/motion.js', ['motion.js', 'text/javascript']], ['/learning-summary.js', ['learning-summary.js', 'text/javascript']], ['/perception.js', ['perception.js', 'text/javascript']], ['/player.js', ['player.js', 'text/javascript']], ['/style.css', ['style.css', 'text/css']]]);
  const simulatorFiles = new Set(['index.html', 'simulator.css', 'simulator.js', 'engine.js', 'motion.js', 'policy.js', 'motion-replay.js',
    'control-runtime.js', 'control.wasm', 'robot-config.js', 'override.js', 'override-view.js', 'override-ui.js', 'override-geometry.js',
    'override-dynamics.js', 'override-autonomy.js', 'nemotron-task.js', 'cad-runtime.js', 'tactics-snapshot.js', 'tactics-ui.js', 'models/robot.stl', 'models/robot.preview.glb', 'models/robot.preview.json']);
  function launchWorker(filename, workerData, response) {
    return new Promise((resolve, reject) => {
      const worker = new Worker(path.join(__dirname, filename), {workerData});
      activeWorker = worker;
      let finished = false;
      const finish = (error, result) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        response.removeListener('close', cancelled);
        if (activeWorker === worker) activeWorker = null;
        if (error) { void worker.terminate(); reject(error); }
        else resolve(result);
      };
      const cancelled = () => { if (!response.writableEnded) finish(Error('Simulation cancelled before completion')); };
      const timer = setTimeout(() => finish(Error('Simulation exceeded server time budget')), timeout);
      worker.once('message', result => finish(result.error ? Error(result.error) : null, result));
      worker.once('error', error => finish(error));
      worker.once('exit', code => { if (!finished) finish(Error(`Worker exited before responding (${code})`)); });
      response.once('close', cancelled);
    });
  }
  const server = http.createServer(async (request, response) => {
    const reply = (status, data) => {
      if (response.destroyed || response.writableEnded) return;
      response.writeHead(status, {'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'});
      response.end(JSON.stringify(data));
    };
    const authority = `127.0.0.1:${server.address().port}`;
    if (request.headers.host !== authority && request.headers.host !== `localhost:${server.address().port}`) { reply(403, {error: 'Loopback Host required'}); return; }
    if (request.headers.origin && ![`http://${authority}`, `http://localhost:${server.address().port}`].includes(request.headers.origin)) { reply(403, {error: 'Same-origin requests only'}); return; }
    const pathname = request.url.split('?')[0];
    if (request.method === 'GET' && pathname === '/api/report') { reply(report ? 200 : 404, report || {error: 'No report yet; run scenarios or load report.json'}); return; }
    if (request.method === 'GET' && pathname === '/api/perception/status') { reply(200, perceptionReadiness()); return; }
    if (request.method === 'GET' && pathname === '/api/motion/status') { reply(200, motionReadiness()); return; }
    if (request.method === 'GET' && pathname === '/api/motion/learning') {
      const status = MotionLearner.status(learningDirectory);
      const independentResearch = readIndependentEvaluation(status, learningDirectory);
      reply(200, {...status, independentResearch,
        learnedImprovementVerified: status.learnedImprovementVerified === true &&
          !(independentResearch.available && !independentResearch.gateReportedPassed)});
      return;
    }
    if (request.method === 'GET' && ['/api/motion/latest', '/api/motion/session'].includes(pathname)) {
      try {
        const record = pathname.endsWith('/latest') ? motionStore.latest()
          : motionStore.read(new URL(request.url, `http://${authority}`).searchParams.get('id'));
        reply(record ? 200 : 404, record || {error: 'No motion experiment saved yet'});
      } catch (error) { reply(400, {error: error.message}); }
      return;
    }
    if (request.method === 'GET' && pathname === '/api/nemotron/status') { reply(200, {...nemotronClient.metadata, available: 'not checked', liveInferenceVerified: false}); return; }
    if (request.method === 'GET' && ['/api/nemotron/chat/latest', '/api/nemotron/chat/session'].includes(pathname)) {
      try {
        const record = pathname.endsWith('/latest') ? chatStore.latest()
          : chatStore.read(new URL(request.url, `http://${authority}`).searchParams.get('id'));
        reply(record ? 200 : 404, record ? validateConversation(record) : {error: 'No saved chat yet'});
      } catch (error) { reply(400, {error: error.message}); }
      return;
    }
    if (request.method === 'GET' && pathname === '/api/nemotron/skills') {
      try { reply(200, {schemaVersion: 1, skills: skills.catalog(), inferencePerformed: false,
        trainingPerformed: false, arbitraryScriptsEnabled: false}); }
      catch (error) { reply(400, {error: error.message}); }
      return;
    }
    if (request.method === 'GET' && ['/api/nemotron/latest', '/api/nemotron/session'].includes(pathname)) {
      try {
        const record = pathname.endsWith('/latest') ? store.latest() : store.read(new URL(request.url, `http://${authority}`).searchParams.get('id'));
        reply(record ? 200 : 404, record || {error: 'No Nemotron session saved yet'});
      } catch (error) { reply(400, {error: error.message}); }
      return;
    }
    if (request.method === 'GET' && pathname.startsWith('/simulator/')) {
      const filename = pathname.slice('/simulator/'.length) || 'index.html';
      if (!simulatorFiles.has(filename)) { reply(404, {error: 'Simulator asset not found'}); return; }
      const type = {'.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.wasm': 'application/wasm', '.stl': 'application/octet-stream', '.glb': 'model/gltf-binary', '.json': 'application/json'}[path.extname(filename)];
      response.writeHead(200, {'Content-Type': type, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"});
      fs.createReadStream(path.join(__dirname, '../simulator', filename)).pipe(response);
      return;
    }
    if (request.method === 'GET' && assets.has(pathname)) {
      const [filename, type] = assets.get(pathname);
      const file = path.join(__dirname, 'dashboard', filename);
      if (!fs.existsSync(file)) { reply(404, {error: 'Dashboard asset missing'}); return; }
      response.writeHead(200, {'Content-Type': `${type}; charset=utf-8`, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"});
      fs.createReadStream(file).pipe(response);
      return;
    }
    if (request.method !== 'POST' || !['/api/run', '/api/replay', '/api/motion/check', '/api/motion/run', '/api/motion/replay', '/api/motion/evaluate', '/api/motion/compare-learned', '/api/nemotron/check', '/api/nemotron/plan', '/api/nemotron/run', '/api/nemotron/chat'].includes(pathname)) { reply(404, {error: 'Not found'}); return; }
    if (!request.headers['content-type']?.startsWith('application/json')) { reply(415, {error: 'application/json required'}); return; }
    if (activeWorker || activeAgent) { reply(409, {error: 'A simulation or Nemotron request is already running'}); return; }
    try {
      const chunks = [];
      let length = 0;
      for await (const chunk of request) {
        length += chunk.length;
        if (length > 65536) { reply(413, {error: 'Request body exceeds 64 KiB'}); request.resume(); return; }
        chunks.push(chunk);
      }
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      if (activeWorker || activeAgent) { reply(409, {error: 'A simulation or Nemotron request is already running'}); return; }
      if (pathname.startsWith('/api/motion/')) {
        const operation = pathname.slice('/api/motion/'.length);
        const allowed = operation === 'run' ? ['mode', 'seed', 'task', 'configuration'] : operation === 'replay' ? ['id'] : [];
        if (!body || typeof body !== 'object' || Array.isArray(body) || Object.keys(body).some(key => !allowed.includes(key))) throw Error('Unsupported motion request fields');
        let workerData = {operation, learningDirectory};
        if (operation === 'run') {
          if (!['scripted', 'policy-fixture', 'random-fixture', 'learned-experiment'].includes(body.mode) ||
            !Number.isInteger(body.seed) || body.seed < 0 || body.seed > 0xffffffff) throw Error('Supported fixture mode and uint32 seed required');
          workerData = {...workerData, mode: body.mode, seed: body.seed, task: normalizeTask(body.task), configuration: body.configuration ?? {}};
        } else if (operation === 'replay') workerData.report = motionStore.read(body.id).report;
        const result = await launchWorker('motion-worker.js', workerData, response);
        reply(200, operation === 'run' ? motionStore.create(result) : result);
        return;
      }
      if (pathname.startsWith('/api/nemotron/')) {
        const chatting = pathname === '/api/nemotron/chat';
        const allowed = chatting ? ['message', 'id'] : pathname.endsWith('/plan') ? ['prompt', 'evidence', 'gameSnapshot'] : pathname.endsWith('/run') ? ['id'] : [];
        if (!body || typeof body !== 'object' || Array.isArray(body) || Object.keys(body).some(key => !allowed.includes(key))) throw Error('Unsupported Nemotron request fields');
        const conversation = chatting && body.id !== undefined ? validateConversation(chatStore.read(body.id)) : null;
        if (body.gameSnapshot !== undefined) {
          if (body.evidence !== undefined) throw Error('Select game or motion evidence, not both');
          Tactics.gameEvidence(body.gameSnapshot);
        }
        let selectedEvidence = null;
        if (body.evidence !== undefined) {
          const selection = body.evidence;
          if (!selection || typeof selection !== 'object' || Array.isArray(selection) ||
              Object.keys(selection).some(key => !['source', 'id'].includes(key)) ||
              !['motion', 'nemotron'].includes(selection.source)) throw Error('Select motion or Nemotron evidence by saved session ID only');
          const record = selection.source === 'motion' ? motionStore.read(selection.id) : store.read(selection.id);
          selectedEvidence = motionEvidence(record, selection.source);
        }
        if (pathname.endsWith('/run')) {
          const record = store.read(body.id);
          if (record.status !== 'prepared') throw Error('A validated Nemotron task must be prepared before approval');
          if (record.run) { reply(200, record); return; }
          const task = normalizeTask(record.task);
          record.approvedAt = new Date().toISOString();
          record.execution = 'approved; simulation running';
          store.save(record);
          try {
            record.run = await launchWorker('nemotron-worker.js', {task}, response);
            record.execution = 'completed in original Simulator; no physical robot was accessed';
          } catch (error) {
            record.execution = 'interrupted or failed; no complete result';
            record.runError = error.message;
            store.save(record);
            throw error;
          }
          delete record.runError;
          store.save(record);
          reply(200, record);
          return;
        }
        const controller = new AbortController();
        activeAgent = controller;
        const timer = setTimeout(() => controller.abort(), pathname.endsWith('/check') ? 10000 : agentTimeout);
        const cancelled = () => { if (!response.writableEnded) controller.abort(); };
        response.once('close', cancelled);
        try {
          if (pathname.endsWith('/check')) reply(200, await nemotronClient.check(controller.signal));
          else if (chatting) {
            const record = await chatReply(body.message, conversation, nemotronClient, {signal: controller.signal});
            controller.signal.throwIfAborted();
            reply(200, conversation ? chatStore.save({...record, id: conversation.id, createdAt: conversation.createdAt})
              : chatStore.create(body.message, record));
          }
          else {
            const plan = await planTask(body.prompt, nemotronClient, {signal: controller.signal,
              registry: skills, evidence: selectedEvidence, learningDirectory, gameSnapshot: body.gameSnapshot});
            controller.signal.throwIfAborted();
            reply(200, store.create(body.prompt, plan));
          }
        } finally {
          clearTimeout(timer);
          response.removeListener('close', cancelled);
          if (activeAgent === controller) activeAgent = null;
        }
        return;
      }
      let workerData;
      if (pathname === '/api/replay') { validate(body.scenario); workerData = {action: 'replay', scenario: body.scenario}; }
      else {
        if (!Number.isInteger(body.count) || body.count < 1 || body.count > 1000 || !Number.isInteger(body.seed) || body.seed < 0 || body.seed > 0xffffffff) throw Error('count must be 1..1000; seed must be a uint32');
        workerData = {action: 'run', count: body.count, seed: body.seed};
      }
      const result = await launchWorker('worker.js', workerData, response);
      if (workerData.action === 'run') report = result;
      reply(200, result);
    } catch (error) { reply(error.message.includes('time budget') ? 504 : 400, {error: error.message}); }
  });
  server.on('close', () => { activeAgent?.abort(); if (activeWorker) void activeWorker.terminate(); });
  server.requestTimeout = timeout + 5000;
  return server;
}

if (require.main === module) {
  const port = Number(process.argv[2] || 8766);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw Error('Port must be 1024..65535');
  const server = createServer({reportPath: process.argv[3] ? path.resolve(process.argv[3]) : undefined});
  server.on('error', error => { console.error(error.message); process.exitCode = 1; });
  server.listen(port, '127.0.0.1', () => console.log(`RoboProof: http://127.0.0.1:${port}`));
}

module.exports = {createServer};
