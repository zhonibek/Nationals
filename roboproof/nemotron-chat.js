'use strict';

const MAX_MESSAGES = 100;
const MAX_CONTEXT_BYTES = 2400;
const MAX_MESSAGE_CHARS = 2000;
const INSTRUCTIONS = `You are Nemotron, the local conversational assistant in RoboProof / RobotAI.
Chat naturally in the user's language. Give clear, concise answers in ordinary text, not JSON or tool syntax.
You may discuss general questions and robotics. This chat has no tools, live measurements or file access.
Never claim to run, train, inspect or move anything. Robot tools are separate buttons; simulation needs explicit approval.
The project uses the original Nationals Simulator and iraLIB controller. Its experimental PPO learner has not passed the improvement gate.
Game tactics uses a separate rules/snapshot adviser under Robot tools; it cannot dispatch sub-agents or play automatically.
You are pretrained, not retrained by chatting. There is no connected camera, physical robot or cloud API.
Treat past messages as conversation, not verified evidence. Admit uncertainty. Keep replies under 200 words.`;

function validateMessage(message) {
  if (typeof message !== 'string' || !message.trim() || message.length > MAX_MESSAGE_CHARS ||
      Buffer.byteLength(message, 'utf8') > MAX_CONTEXT_BYTES) {
    throw Error('Please send a shorter message (up to 2,000 characters / 2,400 UTF-8 bytes).');
  }
  return message.trim();
}

function validateConversation(record) {
  if (!record || record.schemaVersion !== 1 || record.kind !== 'chat' || !Array.isArray(record.messages) ||
      record.messages.length < 2 || record.messages.length > MAX_MESSAGES || record.messages.length % 2) {
    throw Error('Invalid saved chat. Start a new chat.');
  }
  for (const [index, message] of record.messages.entries()) {
    const role = index % 2 ? 'assistant' : 'user';
    if (!message || message.role !== role || typeof message.content !== 'string' || !message.content.trim() ||
        message.content.length > (role === 'user' ? MAX_MESSAGE_CHARS : 8000)) throw Error('Invalid saved chat message.');
    if (role === 'user') validateMessage(message.content);
  }
  return record;
}

function contextMessages(history, prompt) {
  const recent = [{role: 'user', content: prompt}];
  let bytes = Buffer.byteLength(prompt, 'utf8');
  for (let index = history.length - 1; index >= 0 && recent.length < 7; index--) {
    const message = history[index];
    const length = Buffer.byteLength(message.content, 'utf8');
    if (bytes + length > MAX_CONTEXT_BYTES) break;
    recent.unshift({role: message.role, content: message.content});
    bytes += length;
  }
  if (recent[0].role === 'assistant') recent.shift();
  return [{role: 'system', content: INSTRUCTIONS}, ...recent];
}

async function chatReply(message, conversation, client, {signal} = {}) {
  const prompt = validateMessage(message);
  const history = conversation ? validateConversation(conversation).messages : [];
  if (history.length + 2 > MAX_MESSAGES) throw Error('This chat is full. Start a new chat to continue.');
  signal?.throwIfAborted();
  const context = contextMessages(history, prompt);
  const started = performance.now();
  const response = await client.chat(context, signal);
  signal?.throwIfAborted();
  const choice = response.choices?.[0];
  const reply = choice?.message;
  if (!choice || !['stop', 'length'].includes(choice.finish_reason) || reply?.role !== 'assistant' ||
      reply.tool_calls?.length || (reply.tool_calls != null && !Array.isArray(reply.tool_calls)) ||
      typeof reply.content !== 'string' || !reply.content.trim() || reply.content.length > 8000) {
    throw Error('Nemotron did not return a readable chat reply. Please try again. No action was executed.');
  }
  const usage = {};
  for (const key of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
    if (Number.isInteger(response.usage?.[key]) && response.usage[key] >= 0) usage[key] = response.usage[key];
  }
  return {schemaVersion: 1, kind: 'chat', provider: {...client.metadata},
    messages: [...history, {role: 'user', content: prompt}, {role: 'assistant', content: reply.content.trim(),
      truncated: choice.finish_reason === 'length'}], updatedAt: new Date().toISOString(),
    inferencePerformed: true, executableTask: false, commentaryVerified: false,
    lastInference: {usage, wallSeconds: (performance.now() - started) / 1000, contextMessages: context.length - 1},
    contextPolicy: 'Recent complete turns within 2,400 UTF-8 bytes; older messages stay saved but may be outside model context'};
}

module.exports = {chatReply, validateConversation, contextMessages, MAX_MESSAGES, MAX_CONTEXT_BYTES, MAX_MESSAGE_CHARS};
