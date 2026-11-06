'use strict';

// Any wording -> a task spec, through one Nemotron tool call. The model may only fill the allowlisted fields of
// set_task; the result goes through the same validateSpec as the pattern parser and the trainer, and anything else
// fails closed. Nemotron only describes the task: the Tactical AI plays it and the rules engine scores it.

const {validateSpec, SPEC_VALUES} = require('./tactical_task_environment');
const {describe, MAX_TASK_LENGTH} = require('./tactical_task');

const NOTE_LIMIT = 200;
const INSTRUCTIONS = `You turn a team member's request for a VEX V5RC Override robot alliance into one task for its tactical AI.
Call set_task exactly once. Fill only what the request says; leave the rest at its default.
- mode: "score" to place Pins or Cups, "park" to only park, "play" to play the driver period for the best score.
- objects: "pin", "cup" or "any". source: "loader" (match loads from the loading zone), "field" (objects on the field) or "any".
- goals: "neutral", "alliance" (our own goals), "tall" (the neutral tall centre goal) or "any".
- count: how many objects to place, or null. points: how many points to gain, or null. Give both only if the request gives both.
- seconds: a time limit in seconds (at most 105), or null. park: true to park after scoring (always true for mode "park").
If the request is not a task this robot can do in the match (for example a question, or something about another game),
set understood to false and explain why in note. Never invent numbers that the request does not give.`;

function taskTool() {
  const nullableInteger = (minimum, maximum) => ({type: ['integer', 'null'], minimum, maximum});
  return [{type: 'function', function: {name: 'set_task',
    description: 'Set the one task the tactical AI will carry out in the match.',
    parameters: {type: 'object', additionalProperties: false,
      required: ['understood', 'mode', 'objects', 'source', 'goals', 'count', 'points', 'seconds', 'park', 'note'],
      properties: {
        understood: {type: 'boolean'},
        mode: {type: 'string', enum: SPEC_VALUES.mode}, objects: {type: 'string', enum: SPEC_VALUES.objects},
        source: {type: 'string', enum: SPEC_VALUES.source}, goals: {type: 'string', enum: SPEC_VALUES.goals},
        count: nullableInteger(1, 30), points: nullableInteger(1, 500), seconds: nullableInteger(1, 105),
        park: {type: 'boolean'}, note: {type: 'string', maxLength: NOTE_LIMIT}}}}}];
}

const KEYS = ['understood', 'mode', 'objects', 'source', 'goals', 'count', 'points', 'seconds', 'park', 'note'];

// Validates one response; throws (fail closed) on anything but a single valid set_task call describing a real task.
function parseTaskCall(response) {
  const choice = response?.choices?.[0];
  if (!choice || ['length', 'content_filter'].includes(choice.finish_reason)) throw Error('Nemotron did not complete a bounded response');
  const calls = choice.message?.tool_calls;
  if (!Array.isArray(calls) || calls.length !== 1) throw Error('Nemotron must call set_task exactly once');
  const call = calls[0];
  if (call.type !== 'function' || call.function?.name !== 'set_task' || typeof call.function.arguments !== 'string' ||
      call.function.arguments.length > 4096) throw Error('Invalid Nemotron tool call');
  let args;
  try { args = JSON.parse(call.function.arguments); } catch { throw Error('Nemotron tool arguments are not valid JSON'); }
  if (!args || typeof args !== 'object' || Array.isArray(args) || Object.keys(args).some(key => !KEYS.includes(key))) {
    throw Error('Unsupported set_task arguments');
  }
  const note = typeof args.note === 'string' ? args.note.replace(/[\r\n]+/g, ' ').trim().slice(0, NOTE_LIMIT) : '';
  if (args.understood !== true) throw Error(`Nemotron did not understand this as a task${note ? `: ${note}` : ''}`);
  const spec = {mode: args.mode, objects: args.objects ?? 'any', source: args.source ?? 'any', goals: args.goals ?? 'any',
    count: args.count ?? null, points: args.points ?? null, seconds: args.seconds ?? null,
    park: args.mode === 'park' ? true : Boolean(args.park)};
  if (spec.mode !== 'score') Object.assign(spec, {objects: 'any', source: 'any', goals: 'any', count: null, points: null});
  if (spec.mode === 'play') spec.park = false;
  validateSpec(spec);
  const usage = {};
  for (const key of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
    if (Number.isInteger(response.usage?.[key]) && response.usage[key] >= 0) usage[key] = response.usage[key];
  }
  return {spec: {...spec, summary: describe(spec)}, note, usage};
}

async function parseTaskWithNemotron(text, client, {signal} = {}) {
  if (typeof text !== 'string' || !text.trim() || text.length > MAX_TASK_LENGTH) {
    throw Error(`Describe the task in 1 to ${MAX_TASK_LENGTH} characters`);
  }
  const messages = [{role: 'system', content: INSTRUCTIONS}, {role: 'user', content: text.trim()}];
  const response = await client.complete(messages, signal, taskTool());
  return {...parseTaskCall(response), parser: 'nemotron',
    provider: {provider: client.metadata?.provider, model: client.metadata?.model}};
}

module.exports = {INSTRUCTIONS, taskTool, parseTaskCall, parseTaskWithNemotron};
