'use strict';

const {TacticalEnvironment} = require('./tactical_environment');

// Non-learning tactical policies. Each one chooses among the environment's legal candidates using the
// rules-engine preview (points = change in score margin, seconds = game time used); none invents scores.
const POLICIES = Object.freeze({
  'loader-cycle': 'Team tactic: score match loads from the loading zone; other actions only when no scoring loader cycle is legal',
  'field-greedy': 'Best previewed points per second among field objects, Toggles and parking; never uses match loads',
  'greedy-all': 'Best previewed points per second among all legal candidates, including match loads',
  random: 'Seeded uniformly random legal candidate'
});

const MATCH_DECISIONS = 60;

function seededRandom(seed) {
  let state = (seed ^ 0x9E3779B9) >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

const isLoad = row => row.kind.startsWith('load-score-');

function rank(rows) {
  return [...rows].sort((first, second) => second.pointsPerSecond - first.pointsPerSecond ||
    second.points - first.points || first.id.localeCompare(second.id));
}

// Prefer candidates that gain points; fall back to the best remaining one so a policy never stalls.
function best(rows) {
  const scoring = rows.filter(row => row.points > 0);
  return rank(scoring.length ? scoring : rows)[0] || null;
}

function choose(policy, rows, random) {
  const legal = rows.filter(row => row.legal);
  if (!legal.length) return null;
  if (policy === 'random') return legal[Math.floor(random() * legal.length)];
  if (policy === 'greedy-all') return best(legal);
  if (policy === 'field-greedy') return best(legal.filter(row => !isLoad(row)));
  if (policy === 'loader-cycle') {
    const loads = legal.filter(row => isLoad(row) && row.points > 0);
    return loads.length ? rank(loads)[0] : best(legal.filter(row => !isLoad(row)));
  }
  throw Error(`Unknown tactical policy ${policy}`);
}

function summarize(log) {
  const counts = {};
  for (const row of log) counts[row.kind] = (counts[row.kind] || 0) + 1;
  const loaderDecisions = log.filter(isLoad);
  return {actionCounts: counts, matchLoadsScored: loaderDecisions.length,
    pointsFromMatchLoads: loaderDecisions.reduce((sum, row) => sum + row.points, 0)};
}

// Plays one driver period. `chooser({environment, observation, candidates})` returns a candidate row
// (optionally with `reason` / `meta`) or null to stop. Every chosen ID is re-checked by environment.step().
async function playMatch({seed, alliance = 'red', chooser, maximumDecisions = MATCH_DECISIONS, environmentOptions = {}, signal} = {}) {
  const environment = new TacticalEnvironment({...environmentOptions, maximumDecisions});
  let observation = environment.reset(seed, alliance);
  const log = [];
  let stopReason = 'match-complete';
  let error = null;
  while (!observation.done) {
    if (signal?.aborted) { stopReason = 'cancelled'; break; }
    const candidates = environment.previewCandidates();
    let pick;
    try { pick = await chooser({environment, observation, candidates}); }
    catch (failure) {
      // Fail closed: an invalid or failed choice ends the match; the decisions so far are kept.
      stopReason = signal?.aborted ? 'cancelled' : 'failed-closed';
      error = failure.message;
      break;
    }
    if (!pick) { stopReason = 'no-choice'; break; }
    const clock = observation.clock;
    const result = environment.step(pick.id);
    log.push({decision: log.length + 1, clock: Number(clock.toFixed(2)), id: pick.id, kind: pick.kind, robotId: pick.robotId,
      points: result.reward, seconds: Number(result.durationSeconds.toFixed(3)), rejected: Boolean(result.rejectedAction),
      score: result.score, ...(pick.reason ? {reason: pick.reason} : {}), ...(pick.meta ? {meta: pick.meta} : {})});
    observation = result.observation;
  }
  return {seed, alliance, stopReason, ...(error ? {error} : {}), timing: environment.timing, actionSeconds: environment.actionSeconds, finalScore: environment.score(), decisions: log.length, ...summarize(log), log,
    movementModel: observation.movementModel};
}

// Deterministic tactics a learner can imitate one decision at a time ('random' needs a per-match seeded stream).
const DEMONSTRATION_POLICIES = Object.freeze(['loader-cycle', 'field-greedy', 'greedy-all']);

// The candidate `policy` picks in the environment's current state, exactly as playMatch would pick it, or null
// once the episode is over or nothing is legal. When a match load scores, loader-cycle chooses among the loads
// alone, so it previews those first and previews the other candidates only when no load scores.
function demonstrate(environment, policy = 'loader-cycle') {
  if (!DEMONSTRATION_POLICIES.includes(policy)) throw Error(`Unsupported demonstration policy ${policy}`);
  if (environment.done) return null;
  if (policy === 'loader-cycle') {
    const loads = environment.previewCandidates(isLoad);
    if (loads.some(row => row.legal && row.points > 0)) return choose(policy, loads);
  }
  return choose(policy, environment.previewCandidates());
}

function policyChooser(policy, seed) {
  if (!POLICIES[policy]) throw Error(`Unknown tactical policy ${policy}`);
  const random = seededRandom(seed);
  return async ({candidates}) => choose(policy, candidates, random);
}

async function evaluatePolicies({policies = Object.keys(POLICIES), seeds, alliance = 'red', environmentOptions = {}} = {}) {
  if (!Array.isArray(seeds) || !seeds.length) throw Error('Provide evaluation seeds');
  const results = {};
  for (const policy of policies) {
    const matches = [];
    for (const seed of seeds) matches.push(await playMatch({seed, alliance, environmentOptions, chooser: policyChooser(policy, seed)}));
    const margins = matches.map(match => match.finalScore.margin);
    results[policy] = {description: POLICIES[policy], seeds, alliance,
      meanMargin: margins.reduce((sum, value) => sum + value, 0) / margins.length,
      minMargin: Math.min(...margins), maxMargin: Math.max(...margins),
      meanOwnScore: matches.reduce((sum, match) => sum + match.finalScore[alliance], 0) / matches.length,
      meanMatchLoadsScored: matches.reduce((sum, match) => sum + match.matchLoadsScored, 0) / matches.length,
      matches: matches.map(({log, ...rest}) => rest)};
  }
  return results;
}

module.exports = {POLICIES, MATCH_DECISIONS, DEMONSTRATION_POLICIES, choose, demonstrate, playMatch, policyChooser,
  evaluatePolicies, seededRandom};
