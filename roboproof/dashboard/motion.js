'use strict';

(() => {
  const element = id => document.getElementById(`motion-${id}`);
  const local = ['http:', 'https:'].includes(location.protocol) && ['127.0.0.1', 'localhost'].includes(location.hostname);
  const text = (id, value) => { element(id).textContent = value; };
  const json = value => JSON.stringify(value, null, 2);
  let controller = null;
  let session = null;
  let learning = null;

  function controls() {
    for (const id of ['check', 'restore', 'evaluate', 'run']) element(id).disabled = !local || !!controller;
    element('cancel').disabled = !controller;
    element('replay').disabled = !local || !!controller || !session;
    element('download').disabled = !!controller || !session;
    element('learning-refresh').disabled = !local || !!controller;
    element('compare-learned').disabled = !local || !!controller || !learning?.motionPolicyTrained || learning.status === 'running';
    element('use-verified').disabled = !local || !!controller || !learning?.learnedImprovementVerified;
  }

  async function request(operation, body) {
    const response = await fetch(`/api/motion/${operation}`, {cache: 'no-store', signal: controller?.signal,
      ...(body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: json(body)})});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || `HTTP ${response.status}`);
    return result;
  }

  async function operation(message, callback) {
    if (!local || controller) return;
    controller = new AbortController();
    controls();
    text('status', message);
    try { await callback(); }
    catch (error) { text('status', error.name === 'AbortError' ? 'Cancelled; no completed experiment is claimed. Restore saved evidence to inspect completed runs.' : error.message); }
    finally { controller = null; controls(); }
  }

  function render(record) {
    session = record;
    const report = record.report;
    element('result').hidden = false;
    text('outcome', `${report.reason.toUpperCase()} · ${report.metrics.elapsedSeconds.toFixed(2)} simulated seconds · ${report.identity.engine} · ${report.options.policyIdentity.id}`);
    text('metrics', json({metrics: report.metrics, exactReplayVerified: record.exactReplayVerified,
      motionPolicyTrained: record.motionPolicyTrained, transitions: report.transitions.length, savedAt: record.savedAt,
      controllerWasmSha256: report.identity.controllerWasmSha256, learningStatus: report.learningStatus}));
    element('simulator').href = `/simulator/index.html?motion=${encodeURIComponent(record.id)}`;
    controls();
  }

  element('form').addEventListener('submit', event => {
    event.preventDefault();
    void operation('Running and replay-checking the original Simulator. Saving measured transitions locally…', async () => {
      const number = id => Number(element(id).value);
      const task = {start: {xIn: number('start-x'), yIn: number('start-y'), headingDeg: number('start-heading')},
        goal: {xIn: number('goal-x'), yIn: number('goal-y'), headingDeg: number('goal-heading')}, deadlineSeconds: number('deadline')};
      const record = await request('run', {mode: element('mode').value, seed: number('seed'), task});
      render(record);
      text('status', record.motionPolicyTrained
        ? 'Learned-policy experiment saved and exact-replay verified. One run does not prove improvement. Open the original Simulator replay.'
        : 'Fixture experiment saved. Open the original Simulator replay. This run is not training or learned improvement.');
    });
  });
  element('check').addEventListener('click', () => operation('Checking baseline, policy actions, transition replay and safe stop…', async () => {
    await request('check', {});
    const status = await request('status');
    text('readiness', status.readyForBoundedCpuExperiment ? 'BOUNDED CPU ENVIRONMENT READY' : 'READINESS NOT VERIFIED');
    text('gates', json(status));
    text('status', status.readyForBoundedCpuExperiment ? 'Environment smoke passed. Motion training and physical/game validation remain separate.' : status.issue);
  }));
  element('restore').addEventListener('click', () => operation('Loading a disk-backed motion experiment…', async () => {
    render(await request('latest'));
    text('status', 'Restored saved evidence. Use exact replay verification to check it against the current engine.');
  }));
  element('evaluate').addEventListener('click', () => operation('Comparing the baseline and nominal reference adapter on 24 frozen worlds. No training…', async () => {
    const result = await request('evaluate', {});
    element('evaluation').hidden = false;
    text('evaluation-result', json(result));
    text('status', `Frozen evaluation complete: baseline ${result.baseline.success}/${result.baseline.count}, adapter ${result.adapter.success}/${result.adapter.count}. No learned policy was evaluated.`);
  }));
  function renderLearning(result) {
    learning = result;
    element('learning-results').hidden = !result.available;
    text('learning-status', !result.available ? result.message :
      `${result.status.toUpperCase()} · ${result.models.filter(row => row.trained).length} trained seeds · ` +
      (result.learnedImprovementVerified ? 'FROZEN IMPROVEMENT GATE PASSED' : 'IMPROVEMENT NOT VERIFIED · baseline remains the default'));
    text('learning-details', json(result));
    const container = element('learning-table');
    container.replaceChildren();
    if (result.available) {
      const table = document.createElement('table');
      const heading = document.createElement('tr');
      for (const label of ['Seed', 'Steps / updates', 'Baseline', 'Untrained', 'Learned', 'Gate']) {
        const cell = document.createElement('th');
        cell.textContent = label;
        heading.appendChild(cell);
      }
      table.appendChild(heading);
      for (const model of result.models) {
        const evaluated = result.evaluation?.models.find(row => row.policySha256 === model.policySha256);
        const score = value => value ? `${value.success}/${value.count}` : 'Not evaluated';
        const row = document.createElement('tr');
        for (const value of [model.seed, `${model.steps} / ${model.updates}`, score(evaluated?.baseline),
          score(evaluated?.untrained), score(evaluated?.learned), evaluated ? (evaluated.acceptance.passed ? 'Pass' : 'Failed') : 'Pending']) {
          const cell = document.createElement('td');
          cell.textContent = String(value);
          row.appendChild(cell);
        }
        table.appendChild(row);
      }
      container.appendChild(table);
    }
    controls();
  }
  element('learning-refresh').addEventListener('click', () => operation('Reading disk-backed motion checkpoints. No training…', async () => {
    renderLearning(await request('learning'));
    text('status', 'Training history refreshed. Use the terminal command for a new bounded run.');
  }));
  element('compare-learned').addEventListener('click', () => operation('Frozen evaluation: controller, untrained and learned policies. No tuning…', async () => {
    await request('compare-learned', {});
    renderLearning(await request('learning'));
    text('status', learning.learnedImprovementVerified ? 'Independent frozen improvement gate passed.' : 'Comparison saved. Improvement gate did not pass; inspect failed cases. No automatic policy promotion.');
  }));
  element('use-verified').addEventListener('click', () => {
    if (!learning?.learnedImprovementVerified || controller) return;
    element('mode').value = 'learned-experiment';
    text('status', 'Verified checkpoint selected for the next explicit experiment. No robot command was sent.');
  });
  element('replay').addEventListener('click', () => operation('Recomputing saved actions with the source-checked original Simulator…', async () => {
    const result = await request('replay', {id: session.id});
    text('status', `Exact same-runtime replay verified: ${result.ticks} physics ticks. No policy/model was called.`);
  }));
  element('cancel').addEventListener('click', () => controller?.abort());
  element('download').addEventListener('click', () => {
    if (!session) return;
    const url = URL.createObjectURL(new Blob([json(session) + '\n'], {type: 'application/json'}));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'robotai-motion-evidence.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  controls();
  if (local) void request('status').then(status => {
    text('readiness', status.readyForBoundedCpuExperiment ? 'BOUNDED CPU ENVIRONMENT READY' : 'READINESS NOT VERIFIED');
    text('gates', json(status));
  }).catch(() => text('status', 'Start the updated local RoboProof server for the motion lab.'));
  else text('status', 'Motion experiments require the local RoboProof server. Loading this page does not run experiments.');
})();
