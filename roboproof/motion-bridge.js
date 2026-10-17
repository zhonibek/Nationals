'use strict';

const {loadHeadless} = require('../simulator/headless');
const Policy = require('../simulator/policy');
const Gvf = require('../simulator/gvf-policy');
const {FIXED_DT, SCHEMA_VERSION} = require('../simulator/motion');
const PROTOCOL_VERSION = 1;
const MAX_REQUEST_BYTES = 16384;

function createBridge({controlMode = 'policy', ...unknown} = {}) {
  if (!['policy', 'gvf-residual'].includes(controlMode) || Object.keys(unknown).length) throw Error('Unknown bridge controlMode/selector');
  const gvf = controlMode === 'gvf-residual';
  const contractIdentity = {controlMode, policyContractVersion: Gvf.CONTRACT_VERSION, policyContractName: Gvf.CONTRACT_NAME};
  const tags = {contractName: Gvf.CONTRACT_NAME, contractVersion: Gvf.CONTRACT_VERSION, observationSchema: Gvf.OBSERVATION_SCHEMA};
  const runner = loadHeadless();
  const episode = runner.createEpisode();
  let needsReset = true;
  let count = 0;
  function stop() {
    needsReset = true;
    if (gvf && (!episode.sim || episode.terminated || episode.truncated)) {
      if (episode.sim) episode.neutral();
      return {stopped: true, requiresReset: true};
    }
    episode.applyAction({type: 'stop'});
    return episode.step();
  }
  function handle(request) {
    if (!request || typeof request !== 'object' || Array.isArray(request) ||
      request.protocolVersion !== PROTOCOL_VERSION || !Number.isInteger(request.id) || request.id < 0) throw Error('Invalid protocol identity');
    if (++count > 1000000) throw Error('Bridge operation budget exhausted');
    const allowed = {contract: [], reset: ['seed', 'configuration', 'task', 'options'],
      step: ['action'], observe: [], report: [], stop: [], close: []};
    if (!Object.hasOwn(allowed, request.op) || Object.keys(request).some(key =>
      !['id', 'protocolVersion', 'op', ...allowed[request.op]].includes(key))) throw Error('Unsupported bridge operation/fields');
    let result;
    if (request.op === 'contract') result = gvf ? {...Gvf.CONTRACT, ...contractIdentity,
      protocolVersion: PROTOCOL_VERSION, episodeSchemaVersion: SCHEMA_VERSION,
      actionSize: Gvf.ACTION_SIZE, observationSize: Gvf.OBSERVATION_SIZE, observationFields: Gvf.OBSERVATION_FIELDS,
      actionBounds: [-1, 1], zeroAction: [...Gvf.ZERO_ACTION], fixedDt: FIXED_DT, policyLimits: Gvf.LIMITS,
      policyIntervalTicks: 5, actionTimeoutTicks: 10, identity: runner.identity,
      physics: 'Original Simulator only; Python performs no physics', motionTrainingImplemented: false}
      : {protocolVersion: PROTOCOL_VERSION, episodeSchemaVersion: SCHEMA_VERSION,
      actionSize: Policy.ACTION_SIZE, observationSize: Policy.OBSERVATION_SIZE, actionBounds: [-1, 1],
      fixedDt: FIXED_DT, policyLimits: Policy.LIMITS, identity: runner.identity,
      physics: 'Original Simulator only; Python performs no physics', motionTrainingImplemented: true};
    else if (request.op === 'reset') {
      needsReset = true;
      try {
        let options = {...request.options, controlMode: 'policy'};
        if (gvf) {
          if (request.options !== undefined && (!request.options || typeof request.options !== 'object' || Array.isArray(request.options))) {
            throw Error('Unsupported episode options');
          }
          for (const [key, expected] of Object.entries(contractIdentity)) {
            if (Object.hasOwn(request.options ?? {}, key) && request.options[key] !== expected) throw Error(`Process-fixed ${key} mismatch`);
          }
          options = {...request.options, ...contractIdentity};
        } else if (request.options?.controlMode === 'gvf-residual') throw Error('Process-fixed controlMode mismatch');
        const observation = episode.reset(request.seed, request.configuration, request.task, options);
        result = gvf ? {...tags, observation, vector: Gvf.vector(observation), options: episode.options, identity: runner.identity}
          : {observation, vector: Policy.vector(observation), options: episode.options, identity: runner.identity,
            referenceDurationSeconds: episode.referenceAdapter.duration};
        needsReset = false;
      } catch (error) {
        if (gvf && episode.sim) episode.neutral();
        throw error;
      }
    } else if (request.op === 'step') {
      if (gvf) {
        if (needsReset) throw Error('Reset before stepping, including after an episode ends');
        try {
          result = episode.gvfResidualStep(Gvf.normalizeAction(request.action));
          result = {...result, ...tags, vector: Gvf.vector(result.observation)};
          needsReset = result.terminated || result.truncated;
        } catch (error) {
          stop();
          throw error;
        }
      } else result = episode.policyStep(request.action);
      result.info = {...result.info, rewardComponents: {...result.rewardComponents},
        cumulativeEffortProxyVAs: episode.effortProxyVAs, cumulativeContactSeconds: episode.sim.contactSeconds};
    }
    else if (request.op === 'observe') {
      const observation = episode.observe();
      result = gvf ? {...tags, observation, vector: Gvf.vector(observation)} : {observation, vector: Policy.vector(observation)};
    }
    else if (request.op === 'stop') {
      result = stop();
    } else if (request.op === 'report') result = {...episode.report(), identity: runner.identity,
      agent: 'local Python action stream -> original Simulator -> persistent iraLIB controller'};
    else result = {closed: true};
    return {protocolVersion: PROTOCOL_VERSION, id: request.id, ok: true, result};
  }
  return {handle};
}

function serve() {
  const args = process.argv.slice(2);
  if (args.length > 1 || (args.length && !['--contract=policy', '--contract=gvf-residual'].includes(args[0]))) {
    throw Error('Unknown bridge argument/contract selector');
  }
  const bridge = createBridge({controlMode: args.length ? args[0].slice('--contract='.length) : 'policy'});
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
