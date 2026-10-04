'use strict';

const {normalizeTask, DEFAULT_TASK, FIXED_DT} = require('../simulator/motion');
const fs = require('node:fs');
const path = require('node:path');

const MAX_TURNS = 4;
const MAX_TOKENS = 1024;
const CONTRACT = Object.freeze({
  environment: 'original Nationals Simulator + source-checked iraLIB C++ WASM',
  task: 'single reach-pose, nominal physical configuration, no obstacle planner',
  coordinates: 'field center is (0,0); +X right; +Y forward; inches; heading degrees clockwise from +Y',
  bounds: {xIn: [-60, 60], yIn: [-60, 60], headingDeg: [-180, 180], deadlineSeconds: [0.01, 60]},
  defaultStart: DEFAULT_TASK.start, defaultDeadlineSeconds: 10, fixedDt: FIXED_DT,
  success: 'truth position <0.02032m, heading <0.035rad, speed <0.0254m/s, yaw rate <0.0873rad/s for 15 consecutive ticks',
  limitations: ['No camera input or pixel-to-field calibration', 'No scoring, training, gain tuning or physical robot access',
    'Simulator approximations are not physical validation', 'A proposed task is not a measured success; user approval is required to run']
});
const poseSchema = {type: 'object', additionalProperties: false, required: ['xIn', 'yIn', 'headingDeg'], properties: {
  xIn: {type: 'number', minimum: -60, maximum: 60}, yIn: {type: 'number', minimum: -60, maximum: 60},
  headingDeg: {type: 'number', minimum: -180, maximum: 180}
}};
const TOOLS = [
  {type: 'function', function: {name: 'get_motion_contract', description: 'Read the real supported simulator task, units, bounds and limitations before proposing a task.',
    parameters: {type: 'object', properties: {}, additionalProperties: false}}},
  {type: 'function', function: {name: 'prepare_reach_pose', description: 'Validate and propose one simulated task for user review. Does not move a robot or run a simulation.',
    parameters: {type: 'object', additionalProperties: false, required: ['task', 'summary'], properties: {
      task: {type: 'object', additionalProperties: false, required: ['start', 'goal', 'deadlineSeconds'], properties: {
        start: poseSchema, goal: poseSchema, deadlineSeconds: {type: 'number', minimum: 0.01, maximum: 60}
      }}, summary: {type: 'string', minLength: 1, maxLength: 1500}
    }}}}
];
const INSTRUCTIONS = `You are Nemotron, the task coordinator inside RobotAI/RoboProof.
First call get_motion_contract. Then call prepare_reach_pose if the user's request defines a supported single target.
Use a default start (0,0,0), default heading 0 degrees, and deadline 10 seconds unless specified; state these defaults in the summary.
Convert explicitly stated meters or centimeters to inches. Coordinates are absolute field coordinates; +X right, +Y forward.
For relative movement, compute the goal from the stated start and heading. Never interpret pixels as field positions.
If a target/units are ambiguous, ask a brief clarification instead of guessing. Unsupported camera, obstacles, scoring,
training, shell execution, files, firmware or hardware requests must be declined or clarified.
Only the listed tools exist. Never claim movement, success, optimized routes or learned improvement before measured evidence.
Produce one tool call at a time. Be concise. Your task summary is a proposal, not evidence.`;

function objectKeys(value, allowed, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error(`${label} must be an object`);
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw Error(`Unsupported ${label}.${key}`);
}

function configuration(env = process.env) {
  let base;
  try { base = new URL(env.ROBOTAI_NEMOTRON_BASE_URL || 'http://127.0.0.1:8080/v1'); }
  catch { throw Error('Invalid Nemotron base URL'); }
  if (!['http:', 'https:'].includes(base.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname) ||
      base.username || base.password || base.search || base.hash || base.pathname.replace(/\/$/, '') !== '/v1') {
    throw Error('Nemotron requires a loopback /v1 endpoint; cloud access is disabled');
  }
  const model = env.ROBOTAI_NEMOTRON_MODEL || 'robotai-nemotron';
  if (typeof model !== 'string' || model.length > 200 || !/nemotron/i.test(model) || /[\r\n]/.test(model)) throw Error('Configure a Nemotron model ID or alias');
  const timeoutMs = Number(env.ROBOTAI_NEMOTRON_TIMEOUT_MS || 120000);
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 300000) throw Error('Nemotron timeout must be 1000..300000 ms');
  const keyFile = path.join(__dirname, '../.cache/robotai-nemotron/local-api-key.txt');
  const portableDefault = base.href.replace(/\/$/, '') === 'http://127.0.0.1:8080/v1';
  const apiKey = env.ROBOTAI_NEMOTRON_API_KEY ?? (portableDefault && fs.existsSync(keyFile) ? fs.readFileSync(keyFile, 'utf8').trim() : '');
  return {baseUrl: base.href.replace(/\/$/, ''), model, timeoutMs, apiKey};
}

async function readJson(response, maximum = 256 * 1024) {
  if (!response.ok) {
    await response.body?.cancel();
    throw Error(`Local Nemotron returned HTTP ${response.status}; check the runtime logs`);
  }
  const chunks = [];
  let length = 0;
  for await (const chunk of response.body) {
    length += chunk.length;
    if (length > maximum) throw Error('Nemotron response exceeds 256 KiB');
    chunks.push(Buffer.from(chunk));
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); }
  catch { throw Error('Nemotron returned invalid JSON'); }
}

function createClient(config = configuration(), fetchImpl = fetch) {
  configuration({ROBOTAI_NEMOTRON_BASE_URL: config.baseUrl, ROBOTAI_NEMOTRON_MODEL: config.model,
    ROBOTAI_NEMOTRON_TIMEOUT_MS: String(config.timeoutMs)});
  const metadata = {provider: 'local-openai-compatible', model: config.model, baseUrl: config.baseUrl,
    cloudEnabled: false, weightsVerifiedBy: 'tools/setup-nemotron.ps1; an arbitrary endpoint alias alone does not prove model identity'};
  async function request(route, body, signal) {
    const deadline = AbortSignal.timeout(config.timeoutMs);
    const combined = signal ? AbortSignal.any([signal, deadline]) : deadline;
    try {
      return await readJson(await fetchImpl(`${config.baseUrl}/${route}`, {
        method: body ? 'POST' : 'GET', redirect: 'error', signal: combined,
        headers: {'Content-Type': 'application/json', ...(config.apiKey ? {Authorization: `Bearer ${config.apiKey}`} : {})},
        ...(body ? {body: JSON.stringify(body)} : {})
      }));
    } catch (error) {
      if (combined.aborted) throw Error('Nemotron request cancelled or timed out; no task was executed');
      if (error.message.startsWith('Nemotron') || error.message.startsWith('Local Nemotron')) throw error;
      throw Error('Cannot connect to local Nemotron. Start tools/start-nemotron.ps1 and check the runtime logs');
    }
  }
  return {
    metadata,
    async check(signal) {
      const result = await request('models', undefined, signal);
      if (!Array.isArray(result.data) || !result.data.some(row => row.id === config.model)) throw Error('Configured Nemotron model is not served by this endpoint');
      return {...metadata, available: true, liveInferenceVerified: false};
    },
    complete: (messages, signal) => request('chat/completions', {model: config.model, messages, tools: TOOLS,
      tool_choice: 'auto', parallel_tool_calls: false, stream: false, max_tokens: MAX_TOKENS,
      temperature: 0.6, top_p: 0.95, chat_template_kwargs: {enable_thinking: false}}, signal)
  };
}

async function planTask(prompt, client = createClient(), {signal} = {}) {
  if (typeof prompt !== 'string' || !prompt.trim() || prompt.length > 4000) throw Error('Task prompt must contain 1..4000 characters');
  const started = performance.now();
  await client.check(signal);
  const messages = [{role: 'system', content: INSTRUCTIONS}, {role: 'user', content: prompt.trim()}];
  const log = [], usage = [];
  const ids = new Set();
  for (let turn = 0; turn < MAX_TURNS; turn++) {
    const response = await client.complete(messages, signal);
    const choice = response.choices?.[0];
    if (!choice || ['length', 'content_filter'].includes(choice.finish_reason)) throw Error('Nemotron did not complete a bounded response; no task was executed');
    const message = choice.message;
    if (message?.role !== 'assistant') throw Error('Invalid Nemotron assistant response');
    const counts = {};
    for (const key of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
      if (Number.isInteger(response.usage?.[key]) && response.usage[key] >= 0) counts[key] = response.usage[key];
    }
    usage.push(counts);
    const common = {schemaVersion: 1, provider: {...client.metadata}, inferencePerformed: true, toolLog: log, usage,
      inferenceWallSeconds: (performance.now() - started) / 1000,
      execution: 'not run; a separate user approval is required', motionLearning: 'not implemented'};
    if (!message.tool_calls?.length) {
      if (typeof message.content !== 'string' || !message.content.trim() || message.content.length > 4000) throw Error('Nemotron returned no supported task or clarification');
      return {...common, status: 'clarification', message: message.content.trim()};
    }
    if (!Array.isArray(message.tool_calls) || message.tool_calls.length !== 1) throw Error('Only one Nemotron tool call per turn is allowed');
    const call = message.tool_calls[0];
    if (call.type !== 'function' || typeof call.id !== 'string' || !call.id || call.id.length > 200 || ids.has(call.id)) throw Error('Invalid or duplicate Nemotron tool call');
    ids.add(call.id);
    let argumentsValue;
    try { argumentsValue = JSON.parse(call.function.arguments); }
    catch { throw Error('Invalid Nemotron tool arguments JSON'); }
    let result;
    if (call.function.name === 'get_motion_contract') {
      objectKeys(argumentsValue, [], 'arguments');
      result = CONTRACT;
    } else if (call.function.name === 'prepare_reach_pose') {
      if (!log.some(row => row.name === 'get_motion_contract')) throw Error('Nemotron must inspect the motion contract before proposing a task');
      objectKeys(argumentsValue, ['task', 'summary'], 'arguments');
      const task = normalizeTask(argumentsValue.task);
      if (typeof argumentsValue.summary !== 'string' || !argumentsValue.summary.trim() || argumentsValue.summary.length > 1500) throw Error('Task summary must contain 1..1500 characters');
      result = {task, execution: 'awaiting user approval'};
      log.push({name: call.function.name, arguments: {task, summary: argumentsValue.summary}, result});
      return {...common, status: 'prepared', task, message: argumentsValue.summary.trim()};
    } else throw Error('Unsupported Nemotron tool; no command or file access is permitted');
    log.push({name: call.function.name, arguments: argumentsValue, result});
    messages.push({role: 'assistant', content: null, tool_calls: [{id: call.id, type: 'function', function: call.function}]});
    messages.push({role: 'tool', tool_call_id: call.id, content: JSON.stringify(result)});
  }
  throw Error('Nemotron reached its 4-turn budget; no task was executed');
}

module.exports = {configuration, createClient, planTask, CONTRACT, TOOLS, MAX_TURNS, MAX_TOKENS};
