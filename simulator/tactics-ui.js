(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./tactics-snapshot'));
  else root.NationalsTacticsUI = factory(root.NationalsTacticsSnapshot);
})(globalThis, function(Snapshot) {
  'use strict';

  function attach(sim, {replay = false, document = globalThis.document, fetch = globalThis.fetch,
    location = globalThis.location} = {}) {
    const button = document.getElementById('askTactics');
    const stop = document.getElementById('stopTactics');
    const status = document.getElementById('tacticsStatus');
    const result = document.getElementById('tacticsReply');
    const message = document.getElementById('tacticsMessage');
    if (!button || !stop || !status || !result || !message) return {};
    let controller = null;
    const local = ['http:', 'https:'].includes(location.protocol) && ['127.0.0.1', 'localhost'].includes(location.hostname) &&
      (location.pathname.startsWith('/simulator/') || location.port === '8765' || location.pathname.endsWith('/index.html'));
    button.disabled = replay || !local;
    if (!local) status.textContent = 'For AI, open the original Simulator through RoboProof: http://127.0.0.1:8766/simulator/index.html';
    stop.disabled = true;
    stop.addEventListener('click', () => controller?.abort());

    button.addEventListener('click', async () => {
      if (controller || replay || !local) return;
      if (!sim.fleet) {
        const btnStart = document.getElementById('btnOverrideStart');
        if (btnStart) btnStart.click();
      }
      if (!sim.fleet && !sim.override) { status.textContent = 'Start the game with the Override button first, then ask for tactics.'; return; }
      controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 310000);
      button.disabled = true; stop.disabled = false; result.hidden = true;
      try {
        const snapshot = Snapshot.capture(sim.override.getState(), sim.activeRobotId || 'red-1', new Date().toISOString(), sim.override.world || 'head-to-head');
        status.textContent = 'Nemotron is reading the rules and the game snapshot. This is advice, not an action; computing…';
        const apiBase = location.port === '8765' ? 'http://127.0.0.1:8766' : '';
        const response = await fetch(`${apiBase}/api/nemotron/plan`, {method: 'POST', signal: controller.signal,
          headers: {'Content-Type': 'application/json'}, body: JSON.stringify({gameSnapshot: snapshot,
            prompt: 'Suggest three conditional tactical priorities for the selected robot alliance based on the Override rules and the selected snapshot. Read the rules and the snapshot through the plan-game-tactics skill. Give the rule basis, the motion/manipulation/perception roles and the limitations. Do not execute anything; do not promise points or a win.'})});
        const record = await response.json();
        if (!response.ok) throw Error(record.error || `HTTP ${response.status}`);
        if (!['analyzed', 'clarification'].includes(record.status) || typeof record.message !== 'string') throw Error('Expected tactical advice only, not an executable task');
        message.textContent = record.message;
        result.hidden = false;
        status.textContent = `Advice saved. Snapshot: ${snapshot.capturedAt}, ${snapshot.phase}, ${snapshot.clock.toFixed(1)} s. The game may have changed; no actions were run.`;
      } catch (error) { status.textContent = error.name === 'AbortError' ? 'Request stopped. No tactic was executed.' : error.message; }
      finally { clearTimeout(timer); controller = null; button.disabled = replay || !local; stop.disabled = true; }
    });
    attachTacticalMatch({document, fetch, location, local});
    const tacticalTask = attachTacticalTask({document, fetch, location, local});
    // Nemotron advises; the Tactical PPO optimizer (the trained tactical policy) executes in the tactical model.
    const toOptimizer = document.getElementById('btnTacticsToPpo');
    if (toOptimizer && tacticalTask) toOptimizer.addEventListener('click', () => tacticalTask.runFromAdvice());
    return {cancel: () => controller?.abort()};
  }

  function policyLabel(policy) {
    const run = policy?.provenance?.runId ? `run ${String(policy.provenance.runId).slice(0, 8)}` : 'unknown run';
    const evaluation = policy?.evaluation;
    const heldOut = evaluation && Number.isFinite(evaluation.meanScoreMargin)
      ? `, held-out margin ${evaluation.meanScoreMargin} vs loader-cycle ${evaluation.baselineMeanScoreMargin?.['loader-cycle'] ?? '?'}` : '';
    return `${policy?.algorithm || 'trained policy'} (${run}, seed ${policy?.provenance?.trainingSeed ?? '?'}${heldOut})`;
  }

  const DEFAULT_TASK = 'play the driver period for the best score';

  // Sends one typed task to the trained tactical policy and lists every decision. Server text is shown as plain text only.
  function attachTacticalTask({document, fetch, location, local}) {
    const run = document.getElementById('runTacticalTask');
    const text = document.getElementById('tacticalTaskText');
    const status = document.getElementById('tacticalTaskStatus');
    const log = document.getElementById('tacticalTaskLog');
    const seedInput = document.getElementById('tacticalTaskSeed');
    const allianceInput = document.getElementById('tacticalTaskAlliance');
    if (!run || !text || !status || !log || !seedInput || !allianceInput) return null;
    const info = document.getElementById('tacticalPolicyInfo');
    run.disabled = !local;
    if (!local) status.textContent = 'Open the Simulator through RoboProof to give Tactical AI a task.';
    const apiBase = () => location.port === '8765' ? 'http://127.0.0.1:8766' : '';
    let infoLoaded = false;
    // Which trained policy RoboProof has loaded; read only when the task box is first used, never on page load.
    const loadInfo = async () => {
      if (infoLoaded || !local || !info) return;
      infoLoaded = true;
      try {
        const response = await fetch(`${apiBase()}/api/tactics/policy`);
        const policy = await response.json();
        info.textContent = policy.available ? `Loaded policy: ${policyLabel(policy)}.` : `No trained policy is loaded: ${policy.error}`;
      } catch (error) { infoLoaded = false; info.textContent = `Could not read the loaded policy: ${error.message}`; }
    };
    text.addEventListener('focus', loadInfo);
    let busy = false;
    const execute = async () => {
      if (busy || !local) return;
      const task = String(text.value || '').trim();
      const seed = Number(seedInput.value);
      if (!task) { status.textContent = 'Type a task first, for example: score 3 match loads into the neutral goals, then park.'; return; }
      if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) { status.textContent = 'Seed must be a whole number from 0 to 4294967295.'; return; }
      busy = true;
      run.disabled = true;
      log.replaceChildren();
      status.textContent = 'Tactical AI is carrying out the task in the abstract tactical model…';
      try {
        const response = await fetch(`${apiBase()}/api/tactics/task`, {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({task, seed, alliance: allianceInput.value})});
        const result = await response.json();
        if (!response.ok) throw Error(result.error || `HTTP ${response.status}`);
        for (const row of result.log || []) {
          const tr = document.createElement('tr');
          for (const value of [row.decision, `${row.clock.toFixed(1)} s`, row.rejected ? `${row.id} (rejected)` : row.id,
            row.ownPoints, row.seconds.toFixed(2), row.score.margin, row.policyScore]) {
            const td = document.createElement('td');
            td.textContent = String(value);
            tr.appendChild(td);
          }
          log.appendChild(tr);
        }
        const parked = result.parkedRobots?.length ? `, parked ${result.parkedRobots.join(' and ')}` : '';
        if (info) info.textContent = `Loaded policy: ${policyLabel(result.policy)}.`;
        status.textContent = `Understood: ${result.summary}. ${result.achieved ? 'Done' : 'Not completed'} (${result.stopReason}): ` +
          `${result.scoringActions} placements${parked}, ${result.ownPointsGained >= 0 ? '+' : ''}${result.ownPointsGained} own points, ` +
          `final margin ${result.finalScore.margin}, game clock ${result.clockStart}–${result.clockEnd} s. ` +
          `Policy: ${policyLabel(result.policy)}. Abstract tactical model; no robot was driven.`;
      } catch (error) {
        status.textContent = `Tactical task failed: ${error.message}`;
      } finally {
        busy = false;
        run.disabled = !local;
      }
    };
    run.addEventListener('click', execute);
    text.addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault?.(); execute(); } });
    // Used by "Run with the Tactical PPO optimizer" next to Nemotron's advice: run the typed task, or the whole period.
    const runFromAdvice = () => {
      const panel = document.getElementById('tacticalTaskPanel');
      if (panel) panel.open = true;
      if (!String(text.value || '').trim()) text.value = DEFAULT_TASK;
      text.scrollIntoView?.({block: 'center'});
      return execute();
    };
    return {execute, runFromAdvice};
  }

  // Runs one abstract tactical match through RoboProof and lists every decision. Model text is shown as plain text only.
  function attachTacticalMatch({document, fetch, location, local}) {
    const run = document.getElementById('runTacticalMatch');
    const status = document.getElementById('tacticalMatchStatus');
    const log = document.getElementById('tacticalMatchLog');
    const coordinator = document.getElementById('tacticalCoordinator');
    const seedInput = document.getElementById('tacticalSeed');
    const allianceInput = document.getElementById('tacticalAlliance');
    if (!run || !status || !log || !coordinator || !seedInput || !allianceInput) return;
    run.disabled = !local;
    if (!local) status.textContent = 'Open the Simulator through RoboProof to run tactical matches.';
    run.addEventListener('click', async () => {
      const seed = Number(seedInput.value);
      if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) { status.textContent = 'Seed must be a whole number from 0 to 4294967295.'; return; }
      run.disabled = true;
      log.replaceChildren();
      const nemotron = coordinator.value === 'nemotron';
      status.textContent = nemotron ? 'Nemotron is choosing each macro-action on Nebius Token Factory…'
        : coordinator.value === 'tactical-policy' ? 'The trained tactical policy is choosing each macro-action…' : 'Running the baseline tactic…';
      try {
        const apiBase = location.port === '8765' ? 'http://127.0.0.1:8766' : '';
        const response = await fetch(`${apiBase}/api/tactics/match`, {method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({coordinator: coordinator.value, seed, alliance: allianceInput.value})});
        const result = await response.json();
        if (!response.ok) throw Error(result.error || `HTTP ${response.status}`);
        for (const row of result.log || []) {
          const tr = document.createElement('tr');
          for (const value of [row.decision, `${row.clock.toFixed(1)} s`, row.id, row.points, row.seconds.toFixed(2), row.score.margin, row.reason || '']) {
            const td = document.createElement('td');
            td.textContent = String(value);
            tr.appendChild(td);
          }
          log.appendChild(tr);
        }
        const own = result.finalScore[result.alliance];
        status.textContent = `${result.policy ? `${result.coordinator}, ${policyLabel(result.policy)}` : result.coordinator}: final margin ${result.finalScore.margin} (own score ${own}), ` +
          `${result.decisions} decisions, ${result.matchLoadsScored} match loads scored` +
          (nemotron && result.meanDecisionLatencyMs ? `, ${result.meanDecisionLatencyMs} ms per Nemotron decision (${result.provider?.model})` : '') +
          (result.stopReason !== 'match-complete' ? ` · stopped: ${result.stopReason}${result.error ? ` (${result.error})` : ''}` : '') +
          '. Abstract tactical model; no robot was driven.';
      } catch (error) {
        status.textContent = `Tactical match failed: ${error.message}`;
      } finally {
        run.disabled = !local;
      }
    });
  }
  return {attach};
});
