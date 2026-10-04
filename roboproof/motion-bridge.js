'use strict';

const {loadHeadless} = require('../simulator/headless');
const Policy = require('../simulator/policy');
const {FIXED_DT, SCHEMA_VERSION} = require('../simulator/motion');
const PROTOCOL_VERSION = 1;
const MAX_REQUEST_BYTES = 16384;

function createBridge() {
  const runner = loadHeadless();
  const episode = runner.createEpisode();
  let count = 0;
  function handle(request) {
    if (!request || typeof request !== 'object' || Array.isArray(request) ||
      request.protocolVersion !== PROTOCOL_VERSION || !Number.isInteger(request.id) || request.id < 0) throw Error('Invalid protocol identity');
    if (++count > 1000000) throw Error('Bridge operation budget exhausted');
    const allowed = {contract: [], reset: ['seed', 'configuration', 'task', 'options'],
      step: ['action'], observe: [], report: [], stop: [], close: []};
    if (!Object.hasOwn(allowed, request.op) || Object.keys(request).some(key =>
      !['id', 'protocolVersion', 'op', ...allowed[request.op]].includes(key))) throw Error('Unsupported bridge operation/fields');
    let result;
    if (request.op === 'contract') result = {protocolVersion: PROTOCOL_VERSION, episodeSchemaVersion: SCHEMA_VERSION,
      actionSize: Policy.ACTION_SIZE, observationSize: Policy.OBSERVATION_SIZE, actionBounds: [-1, 1],
      fixedDt: FIXED_DT, policyLimits: Policy.LIMITS, identity: runner.identity,
      physics: 'Original Simulator only; Python performs no physics', motionTrainingImplemented: false};
    else if (request.op === 'reset') {
      const observation = episode.reset(request.seed, request.configuration, request.task,
        {...request.options, controlMode: 'policy'});
      result = {observation, vector: Policy.vector(observation), options: episode.options, identity: runner.identity};
    } else if (request.op === 'step') result = episode.policyStep(request.action);
    else if (request.op === 'observe') result = {observation: episode.observe(), vector: Policy.vector(episode.observe())};
    else if (request.op === 'stop') {
      episode.applyAction({type: 'stop'});
      result = episode.step();
    } else if (request.op === 'report') result = {...episode.report(), identity: runner.identity,
      agent: 'local Python action stream -> original Simulator -> persistent iraLIB controller'};
    else result = {closed: true};
    return {protocolVersion: PROTOCOL_VERSION, id: request.id, ok: true, result};
  }
  return {handle};
}

function serve() {
  const bridge = createBridge();
  let pending = '';
  let closed = false;
  function respond(line) {
    let request;
    let reply;
    try { request = JSON.parse(line); reply = bridge.handle(request); }
    catch (error) { reply = {protocolVersion: PROTOCOL_VERSION, id: request?.id ?? null, ok: false, error: error.message}; }
    process.stdout.write(JSON.stringify(reply) + '\n');
    if (reply.ok && request.op === 'close') { closed = true; process.stdin.destroy(); }
  }
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => {
    if (closed) return;
    pending += chunk;
    let boundary;
    while (!closed && (boundary = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, boundary);
      pending = pending.slice(boundary + 1);
      if (Buffer.byteLength(line) > MAX_REQUEST_BYTES) {
        process.stderr.write('Bridge request exceeds 16 KiB\n');
        process.exitCode = 1;
        closed = true;
        process.stdin.destroy();
      } else respond(line);
    }
    if (!closed && Buffer.byteLength(pending) > MAX_REQUEST_BYTES) {
      process.stderr.write('Bridge request exceeds 16 KiB\n');
      process.exitCode = 1;
      closed = true;
      process.stdin.destroy();
    }
  });
  process.stdin.on('end', () => {
    if (pending.trim() && !closed) { process.stderr.write('Truncated bridge request\n'); process.exitCode = 1; }
  });
}

if (require.main === module) {
  try { serve(); } catch (error) { process.stderr.write(error.message + '\n'); process.exitCode = 1; }
}

module.exports = {createBridge, PROTOCOL_VERSION, MAX_REQUEST_BYTES};
