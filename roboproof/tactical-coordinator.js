'use strict';

const {performance} = require('node:perf_hooks');
const {createClient} = require('./nemotron');
const {playMatch, MATCH_DECISIONS} = require('./tactical_baselines');

// Nemotron as the tactical coordinator of one alliance. Each decision is one stateless tool call:
// the model sees a compact match state and engine-previewed candidates and must pick one candidate ID.
// The rules engine executes and scores the choice; the model never supplies points or motion.

const CANDIDATES_SHOWN = 12;
const REASON_LIMIT = 300;

const INSTRUCTIONS = `You are the tactical coordinator for one VEX V5RC Override alliance in a simulated driver period.
Each turn, call choose_action exactly once with one candidate_id from the list. Never invent an ID.
"points" and "seconds" are previews computed by the official rules engine for that single action; the engine, not you, scores the result.
Goal: maximise the final score margin. Rank options by pointsPerSecond first. Choose a move with zero or negative points only when it clearly sets up a better next move (for example a Cup that lets a Pin score on top), and say which move it enables.
Kinds: score-* takes a field object to a goal; load-score-* drives to the alliance loader, takes a match load and scores it; toggle flips a Toggle (affects yellow ownership); park moves a robot into midfield.
Match loads are limited (see matchLoadsRemaining); on the real robot the team finds loading-zone cycles fast and repeatable, so when a load cycle and a field action score similarly per second, prefer the load cycle. The neutral tall goal is closed in the last 10 seconds.
Give a short reason (at most 25 words).`;

function compactCandidate(row) {
  const parts = row.id.split(':');
  return {id: row.id, kind: row.kind, robot: row.robotId,
    ...(row.loaderId ? {loader: row.loaderId} : {}), ...(row.targetId ? {object: row.targetId} : {}),
    ...(row.goalId ? {goal: row.goalId} : parts[0] === 'toggle' ? {toggle: parts[2]} : {}),
    points: row.points, seconds: row.seconds, pointsPerSecond: row.pointsPerSecond};
}

// Bounded shortlist: best loader and non-loader options are always included so the model can compare both.
function shortlist(candidates, limit = CANDIDATES_SHOWN) {
  const legal = candidates.filter(row => row.legal);
  const ranked = rows => [...rows].sort((first, second) => second.pointsPerSecond - first.pointsPerSecond ||
    second.points - first.points || first.id.localeCompare(second.id));
  const loads = ranked(legal.filter(row => row.kind.startsWith('load-score-')));
  const others = ranked(legal.filter(row => !row.kind.startsWith('load-score-')));
  const chosen = new Map();
  for (const row of [...loads.slice(0, 3), ...others.slice(0, 3), ...ranked(legal)]) {
    if (chosen.size >= limit) break;
    chosen.set(row.id, row);
  }
  return [...chosen.values()];
}

function matchState(environment, observation) {
  const alliance = environment.alliance;
  const score = environment.score();
  return {alliance, clock: Number(observation.clock.toFixed(2)),
    secondsLeft: Number((120 - observation.clock).toFixed(2)),
    score: {own: score[alliance], opponent: score[alliance === 'red' ? 'blue' : 'red'], margin: score.margin},
    robots: environment.game.robots.filter(robot => robot.alliance === alliance).map(robot => ({id: robot.id,
      x: Number(robot.x.toFixed(1)), y: Number(robot.y.toFixed(1)), holding: robot.possession, midfield: Boolean(robot.midfield)})),
    matchLoadsRemaining: environment.matchLoadsRemaining(),
    toggles: Object.fromEntries(environment.game.toggles.map(toggle => [toggle.id, toggle.state]))};
}

function chooseTool(ids) {
  return [{type: 'function', function: {name: 'choose_action',
    description: 'Choose the next macro-action for the alliance from the listed candidates.',
    parameters: {type: 'object', additionalProperties: false, required: ['candidate_id', 'reason'],
      properties: {candidate_id: {type: 'string', enum: ids}, reason: {type: 'string', maxLength: REASON_LIMIT}}}}}];
}

// Validates one response; throws (fail closed) on anything but a single valid choose_action call.
function parseChoice(response, ids) {
  const choice = response?.choices?.[0];
  if (!choice || ['length', 'content_filter'].includes(choice.finish_reason)) throw Error('Nemotron did not complete a bounded response');
  const calls = choice.message?.tool_calls;
  if (!Array.isArray(calls) || calls.length !== 1) throw Error('Nemotron must call choose_action exactly once');
  const call = calls[0];
  if (call.type !== 'function' || call.function?.name !== 'choose_action' || typeof call.function.arguments !== 'string' ||
      call.function.arguments.length > 4096) throw Error('Invalid Nemotron tool call');
  let args;
  try { args = JSON.parse(call.function.arguments); } catch { throw Error('Nemotron tool arguments are not valid JSON'); }
  if (!args || typeof args !== 'object' || Array.isArray(args) || Object.keys(args).some(key => !['candidate_id', 'reason'].includes(key))) {
    throw Error('Unsupported choose_action arguments');
  }
  if (!ids.includes(args.candidate_id)) throw Error('Nemotron chose an ID that was not offered');
  const reason = typeof args.reason === 'string' ? args.reason.replace(/[\r\n]+/g, ' ').trim().slice(0, REASON_LIMIT) : '';
  const usage = {};
  for (const key of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
    if (Number.isInteger(response.usage?.[key]) && response.usage[key] >= 0) usage[key] = response.usage[key];
  }
  return {id: args.candidate_id, reason, usage};
}

function nemotronChooser(client, {signal, onDecision} = {}) {
  return async ({environment, observation, candidates}) => {
    const offered = shortlist(candidates);
    if (!offered.length) return null;
    const ids = offered.map(row => row.id);
    const messages = [{role: 'system', content: INSTRUCTIONS},
      {role: 'user', content: JSON.stringify({state: matchState(environment, observation), candidates: offered.map(compactCandidate)})}];
    const started = performance.now();
    const response = await client.complete(messages, signal, chooseTool(ids));
    const latencyMs = Math.round(performance.now() - started);
    const parsed = parseChoice(response, ids);
    const row = offered.find(candidate => candidate.id === parsed.id);
    const best = offered.reduce((top, candidate) => candidate.pointsPerSecond > top.pointsPerSecond ? candidate : top, offered[0]);
    const decision = {...row, reason: parsed.reason, meta: {latencyMs, usage: parsed.usage, offered: ids.length,
      greedyChoice: best.id, matchedGreedy: best.id === row.id}};
    onDecision?.(decision);
    return decision;
  };
}

async function playNemotronMatch({seed, alliance = 'red', client = createClient(), signal, onDecision,
  maximumDecisions = MATCH_DECISIONS, environmentOptions = {}} = {}) {
  await client.check(signal);
  const started = performance.now();
  const chooser = nemotronChooser(client, {signal, onDecision});
  const result = await playMatch({seed, alliance, maximumDecisions, environmentOptions, signal, chooser});
  const wallSeconds = Number(((performance.now() - started) / 1000).toFixed(2));
  const latencies = result.log.map(row => row.meta?.latencyMs).filter(Number.isFinite);
  return {coordinator: 'nemotron', status: result.stopReason === 'match-complete' ? 'completed' : result.stopReason,
    provider: {provider: client.metadata.provider, model: client.metadata.model, baseUrl: client.metadata.baseUrl},
    wallSeconds, meanDecisionLatencyMs: latencies.length ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : null,
    tokens: result.log.reduce((sum, row) => sum + (row.meta?.usage?.total_tokens || 0), 0),
    greedyAgreement: result.log.length ? result.log.filter(row => row.meta?.matchedGreedy).length / result.log.length : null,
    ...result};
}

module.exports = {INSTRUCTIONS, CANDIDATES_SHOWN, shortlist, compactCandidate, matchState, chooseTool, parseChoice,
  nemotronChooser, playNemotronMatch};
