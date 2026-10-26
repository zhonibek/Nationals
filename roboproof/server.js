'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const {Worker} = require('node:worker_threads');
const {validate} = require('./core');
const {configuration, createClient, planTask, NEBIUS_BASE_URL, NEBIUS_MODEL} = require('./nemotron');
const {createStore} = require('./nemotron-store');
const {normalizeTask} = require('../simulator/motion');
const {readiness: perceptionReadiness} = require('./perception');
const {readiness: motionReadiness} = require('./motion-readiness');
const {createMotionStore} = require('./motion-store');
const MotionLearner = require('./motion-learner');
const {readIndependentEvaluation, currentImprovementVerified} = require('./independent-learning');
const {createRegistry} = require('./agent-skills');
const {motionEvidence} = require('./skill-evidence');
const {chatReply, validateConversation} = require('./nemotron-chat');
const Tactics = require('./tactics');
const {readResults: readCosmosResults} = require('./cosmos-results');
const {POLICIES: TACTICAL_POLICIES, playMatch: playTacticalMatch, policyChooser} = require('./tactical_baselines');
const {playNemotronMatch} = require('./tactical-coordinator');
const {loadPolicy: loadTacticalPolicy} = require('./tactical_policy');
const {parseTask: parseTacticalTask, runTask: runTacticalTask} = require('./tactical_task');

function createServer({reportPath = path.join(__dirname, 'runs/latest/report.json'), timeout = 120000,
  nemotronClient = createClient(), nemotronDirectory, nemotronChatDirectory, motionDirectory, learningDirectory, cosmosDirectory, agentTimeout = 300000,
  tacticalPolicyPath = process.env.ROBOPROOF_TACTICAL_POLICY || path.join(__dirname, 'runs/tactical/policy.json')} = {}) {
  let report = fs.existsSync(reportPath) ? JSON.parse(fs.readFileSync(reportPath, 'utf8')) : null;
  let activeWorker = null;
  let activeAgent = null;
  const store = createStore(nemotronDirectory);
  const chatStore = createStore(nemotronChatDirectory || path.join(__dirname, 'runs/nemotron-chat'));
  const motionStore = createMotionStore(motionDirectory);
  const skills = createRegistry();
  let tacticalPolicy = null;
  // The exported trained tactical actor, reloaded when the file changes. It is Git-ignored, so it may be missing.
  function currentTacticalPolicy() {
    if (!fs.existsSync(tacticalPolicyPath)) {
      throw Error('No trained tactical policy is exported yet. Run: python -m roboproof.tactical_learning.export_policy --checkpoint <preserved run>/run/latest.pt');
    }
    const {mtimeMs, size} = fs.statSync(tacticalPolicyPath);
    if (!tacticalPolicy || tacticalPolicy.mtimeMs !== mtimeMs || tacticalPolicy.size !== size) {
      tacticalPolicy = {mtimeMs, size, policy: loadTacticalPolicy(tacticalPolicyPath)};
    }
    return tacticalPolicy.policy;
  }
  const assets = new Map([['/', ['index.html', 'text/html']], ['/index.html', ['index.html', 'text/html']], ['/app.js', ['app.js', 'text/javascript']], ['/nemotron.js', ['nemotron.js', 'text/javascript']], ['/motion.js', ['motion.js', 'text/javascript']], ['/learning-summary.js', ['learning-summary.js', 'text/javascript']], ['/perception.js', ['perception.js', 'text/javascript']], ['/player.js', ['player.js', 'text/javascript']], ['/style.css', ['style.css', 'text/css']]]);
  assets.set('/cosmos.js', ['cosmos.js', 'text/javascript']);
  const simulatorFiles = new Set(['index.html', 'portal.html', 'simulator.css', 'simulator.js', 'engine.js', 'motion.js', 'policy.js', 'gvf-policy.js', 'motion-replay.js',
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
      const allowedOrigins = [`http://${authority}`, `http://localhost:${server.address().port}`, 'http://127.0.0.1:8765', 'http://localhost:8765'];
      const originHeader = request.headers.origin && allowedOrigins.includes(request.headers.origin) ? request.headers.origin : `http://${authority}`;
      response.writeHead(status, {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'no-store',
        'X-Content-Type-Options': 'nosniff',
        'Access-Control-Allow-Origin': originHeader,
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
      });
      response.end(JSON.stringify(data));
    };
    const authority = `127.0.0.1:${server.address().port}`;
    if (request.headers.host !== authority && request.headers.host !== `localhost:${server.address().port}`) { reply(403, {error: 'Loopback Host required'}); return; }
    const allowedOrigins = [`http://${authority}`, `http://localhost:${server.address().port}`, 'http://127.0.0.1:8765', 'http://localhost:8765'];
    if (request.headers.origin && !allowedOrigins.includes(request.headers.origin)) { reply(403, {error: 'Same-origin requests only'}); return; }
    if (request.method === 'OPTIONS') {
      const originHeader = request.headers.origin && allowedOrigins.includes(request.headers.origin) ? request.headers.origin : `http://${authority}`;
      response.writeHead(204, {
        'Access-Control-Allow-Origin': originHeader,
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
      });
      response.end();
      return;
    }
    const pathname = request.url.split('?')[0];
    if (request.method === 'GET' && pathname === '/api/report') { reply(report ? 200 : 404, report || {error: 'No report yet; run scenarios or load report.json'}); return; }
    if (request.method === 'GET' && pathname === '/api/perception/status') { reply(200, perceptionReadiness()); return; }
    if (request.method === 'GET' && pathname === '/api/cosmos/results') { reply(200, readCosmosResults(cosmosDirectory)); return; }
    if (request.method === 'GET' && pathname === '/api/motion/status') { reply(200, motionReadiness()); return; }
    if (request.method === 'GET' && pathname === '/api/motion/learning') {
      const status = MotionLearner.status(learningDirectory);
      const independentResearch = readIndependentEvaluation(status, learningDirectory);
      reply(200, {...status, independentResearch,
        learnedImprovementVerified: currentImprovementVerified(status, independentResearch)});
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
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'wasm-unsafe-eval' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self' http://127.0.0.1:8765 http://localhost:8765 http://127.0.0.1:8766 http://localhost:8766"});
      fs.createReadStream(path.join(__dirname, '../simulator', filename)).pipe(response);
      return;
    }
    if (request.method === 'GET' && assets.has(pathname)) {
      const [filename, type] = assets.get(pathname);
      const file = path.join(__dirname, 'dashboard', filename);
      if (!fs.existsSync(file)) { reply(404, {error: 'Dashboard asset missing'}); return; }
      response.writeHead(200, {'Content-Type': `${type}; charset=utf-8`, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self' http://127.0.0.1:8765 http://localhost:8765 http://127.0.0.1:8766 http://localhost:8766"});
      fs.createReadStream(file).pipe(response);
      return;
    }
    if (request.method !== 'POST' || !['/api/run', '/api/replay', '/api/motion/check', '/api/motion/run', '/api/motion/replay', '/api/motion/evaluate', '/api/motion/compare-learned', '/api/tactics/match', '/api/tactics/task', '/api/nemotron/check', '/api/nemotron/plan', '/api/nemotron/run', '/api/nemotron/chat'].includes(pathname)) { reply(404, {error: 'Not found'}); return; }
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
          if (!['scripted', 'gvf', 'policy-fixture', 'random-fixture', 'learned-experiment'].includes(body.mode) ||
            !Number.isInteger(body.seed) || body.seed < 0 || body.seed > 0xffffffff) throw Error('Supported fixture mode and uint32 seed required');
          workerData = {...workerData, mode: body.mode, seed: body.seed, task: normalizeTask(body.task), configuration: body.configuration ?? {}};
        } else if (operation === 'replay') workerData.report = motionStore.read(body.id).report;
        const result = await launchWorker('motion-worker.js', workerData, response);
        reply(200, operation === 'run' ? motionStore.create(result) : result);
        return;
      }
      if (pathname === '/api/tactics/match') {
        // One abstract driver-period match: Nemotron or a non-learning tactic chooses macro-actions; the rules engine scores.
        if (!body || typeof body !== 'object' || Array.isArray(body) || Object.keys(body).some(key => !['coordinator', 'seed', 'alliance'].includes(key))) {
          throw Error('Unsupported tactical match fields');
        }
        if (body.coordinator !== 'nemotron' && !Object.hasOwn(TACTICAL_POLICIES, body.coordinator)) throw Error('Unknown tactical coordinator');
        if (!Number.isInteger(body.seed) || body.seed < 0 || body.seed > 0xffffffff) throw Error('Tactical match seed must be a uint32');
        const alliance = body.alliance ?? 'red';
        if (!['red', 'blue'].includes(alliance)) throw Error('Tactical alliance must be red or blue');
        const controller = new AbortController();
        activeAgent = controller;
        const timer = setTimeout(() => controller.abort(), agentTimeout);
        const cancelled = () => { if (!response.writableEnded) controller.abort(); };
        response.once('close', cancelled);
        try {
          const result = body.coordinator === 'nemotron'
            ? await playNemotronMatch({seed: body.seed, alliance, client: nemotronClient, signal: controller.signal})
            : await playTacticalMatch({seed: body.seed, alliance, signal: controller.signal, chooser: policyChooser(body.coordinator, body.seed)});
          controller.signal.throwIfAborted();
          reply(200, {schemaVersion: 1, coordinator: body.coordinator, ...result,
            scope: 'abstract macro-action model scored by the Override rules engine; no robot, motion or physics simulation was driven'});
        } finally {
          clearTimeout(timer);
          response.removeListener('close', cancelled);
          if (activeAgent === controller) activeAgent = null;
        }
        return;
      }
      if (pathname === '/api/tactics/task') {
        // A typed task carried out by the trained tactical policy in the abstract macro-action model; inference only.
        if (!body || typeof body !== 'object' || Array.isArray(body) || Object.keys(body).some(key => !['task', 'seed', 'alliance'].includes(key))) {
          throw Error('Unsupported tactical task fields');
        }
        if (!Number.isInteger(body.seed) || body.seed < 0 || body.seed > 0xffffffff) throw Error('Tactical task seed must be a uint32');
        const alliance = body.alliance ?? 'red';
        if (!['red', 'blue'].includes(alliance)) throw Error('Tactical alliance must be red or blue');
        const spec = parseTacticalTask(body.task);
        const policy = currentTacticalPolicy();
        const result = runTacticalTask({spec, policy, seed: body.seed, alliance});
        reply(200, {schemaVersion: 1, summary: spec.summary, ...result,
          policy: {algorithm: policy.algorithm, width: policy.width, provenance: policy.provenance, evaluation: policy.evaluation},
          learnedImprovementVerified: false, independentQualificationPerformed: false,
          scope: 'abstract macro-action model scored by the Override rules engine; no robot, motion or physics simulation was driven'});
        return;
      }
      if (pathname.startsWith('/api/nemotron/')) {
        const chatting = pathname === '/api/nemotron/chat';
        const allowed = chatting ? ['message', 'id'] : pathname.endsWith('/plan') ? ['prompt', 'evidence', 'gameSnapshot'] : pathname.endsWith('/run') ? ['id'] : pathname.endsWith('/check') ? ['apiKey', 'baseUrl', 'model'] : [];
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
          if (pathname.endsWith('/check')) {
            if (['apiKey', 'baseUrl', 'model'].some(key => body[key] !== undefined && (typeof body[key] !== 'string' || body[key].length > 4096))) {
              throw Error('Nemotron connection fields must be strings');
            }
            // Switching endpoints: verify the candidate first, so a bad key never replaces a working client or reaches disk.
            let candidate = nemotronClient;
            const apiKey = body.apiKey?.trim();
            if (apiKey || body.baseUrl || body.model) {
              const baseUrl = body.baseUrl || NEBIUS_BASE_URL;
              const {isNebius} = configuration({ROBOTAI_NEMOTRON_BASE_URL: baseUrl, ROBOTAI_NEMOTRON_MODEL: 'nemotron'});
              candidate = createClient(configuration({
                ROBOTAI_NEMOTRON_BASE_URL: baseUrl,
                ROBOTAI_NEMOTRON_MODEL: body.model || (isNebius ? NEBIUS_MODEL : 'robotai-nemotron'),
                ...(apiKey ? {ROBOTAI_NEMOTRON_API_KEY: apiKey} : {})
              }));
            }
            let checked;
            try { checked = await candidate.check(controller.signal); }
            catch (e) {
              reply(503, {...candidate.metadata, available: false, liveInferenceVerified: false, error: e.message});
              return;
            }
            if (candidate !== nemotronClient) {
              nemotronClient = candidate;
              if (apiKey && candidate.metadata.provider === 'nebius-token-factory') {
                const nebiusKeyFile = path.join(__dirname, '../.cache/robotai-nemotron/nebius-api-key.txt');
                fs.mkdirSync(path.dirname(nebiusKeyFile), {recursive: true});
                fs.writeFileSync(nebiusKeyFile, apiKey, {encoding: 'utf8', mode: 0o600});
              }
            }
            reply(200, checked);
          }
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
    } catch (error) {
      const status = /time budget|timed out|cancelled/i.test(error.message) ? 504
        : /Cannot connect to the approved Nemotron|Local Nemotron returned HTTP/.test(error.message) ? 503 : 400;
      reply(status, {error: error.message});
    }
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
