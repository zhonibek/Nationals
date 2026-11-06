'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {playMatch} = require('./tactical_baselines');

// A recorded Nemotron tactical match lets anyone without a Nebius key see what the model chose.
// Replay re-executes the recorded candidate IDs in the rules engine and checks every score against the recording;
// no inference happens during replay.

const ROOT = path.resolve(__dirname, '..');
const RECORDINGS = path.join(__dirname, 'tactical_recordings');
const ENGINE_FILES = ['roboproof/tactical_environment.js', 'simulator/override.js', 'simulator/override-geometry.js', 'simulator/override-dynamics.js'];

function engineHashes() {
  return Object.fromEntries(ENGINE_FILES.map(file => [file,
    crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT, file))).digest('hex')]));
}

function recordingFromMatch(match, {recordedAt = new Date().toISOString()} = {}) {
  if (match.coordinator !== 'nemotron' || match.stopReason !== 'match-complete') throw Error('Only completed Nemotron matches are recorded');
  return {schemaVersion: 1, kind: 'nemotron-tactical-match', recordedAt, provider: match.provider,
    seed: match.seed, alliance: match.alliance, timing: match.timing, actionSeconds: match.actionSeconds, engineSha256: engineHashes(),
    finalScore: match.finalScore, decisions: match.log.map(row => ({decision: row.decision, id: row.id, points: row.points,
      margin: row.score.margin, reason: row.reason ?? '', latencyMs: row.meta?.latencyMs ?? null})),
    meanDecisionLatencyMs: match.meanDecisionLatencyMs, tokens: match.tokens};
}

const NAME = /^[a-z0-9-]{1,80}$/;

function readRecording(name, directory = RECORDINGS) {
  if (typeof name !== 'string' || !NAME.test(name)) throw Error('Invalid recording name');
  const record = JSON.parse(fs.readFileSync(path.join(directory, `${name}.json`), 'utf8'));
  if (record.schemaVersion !== 1 || record.kind !== 'nemotron-tactical-match' || !Array.isArray(record.decisions)) throw Error('Unsupported recording');
  return record;
}

function listRecordings(directory = RECORDINGS) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory).filter(file => file.endsWith('.json') && NAME.test(file.slice(0, -5)))
    .map(file => file.slice(0, -5)).sort();
}

// Saves a completed Nemotron match as `nemotron-s<seed>-<alliance>` (the latest match for that start replaces it).
function saveRecording(match, directory = RECORDINGS) {
  const record = recordingFromMatch(match);
  const name = `nemotron-s${record.seed}-${record.alliance}`;
  fs.mkdirSync(directory, {recursive: true});
  const file = path.join(directory, `${name}.json`);
  const temporary = `${file}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(record, null, 2)}\n`);
  fs.renameSync(temporary, file);
  return name;
}

// What the app shows in its recording picker; a recording that cannot be read is listed as unreadable, not hidden.
function describeRecordings(directory = RECORDINGS) {
  return listRecordings(directory).map(name => {
    try {
      const record = readRecording(name, directory);
      return {name, seed: record.seed, alliance: record.alliance, recordedAt: record.recordedAt,
        model: record.provider?.model ?? null, finalMargin: record.finalScore?.margin ?? null, decisions: record.decisions.length};
    } catch (error) {
      return {name, error: error.message};
    }
  });
}

// Replays the recorded choices. Fails closed if an ID is no longer offered or any score differs.
async function replayRecording(record, {signal} = {}) {
  let index = 0;
  const chooser = async ({candidates}) => {
    const step = record.decisions[index];
    if (!step) return null;
    const row = candidates.find(candidate => candidate.id === step.id && candidate.legal);
    if (!row) throw Error(`Recorded decision ${step.decision} (${step.id}) is not a legal candidate in this engine`);
    index += 1;
    return {...row, reason: step.reason, meta: {latencyMs: step.latencyMs, recorded: true}};
  };
  const match = await playMatch({seed: record.seed, alliance: record.alliance, signal, chooser,
    environmentOptions: {timing: record.timing}});
  const mismatches = match.log.filter((row, i) => row.points !== record.decisions[i]?.points || row.score.margin !== record.decisions[i]?.margin)
    .map(row => row.decision);
  const sameEngine = Object.entries(engineHashes()).every(([file, hash]) => record.engineSha256?.[file] === hash);
  const reproduced = match.stopReason !== 'failed-closed' && index === record.decisions.length && mismatches.length === 0 &&
    match.finalScore.margin === record.finalScore.margin;
  return {...match, coordinator: 'nemotron-recording', provider: record.provider, recordedAt: record.recordedAt,
    meanDecisionLatencyMs: record.meanDecisionLatencyMs, inferencePerformed: false,
    replay: {reproduced, decisionsReplayed: index, decisionsRecorded: record.decisions.length, mismatches, sameEngine,
      recordedFinalScore: record.finalScore}};
}

module.exports = {RECORDINGS, engineHashes, recordingFromMatch, readRecording, listRecordings, saveRecording, describeRecordings,
  replayRecording};
