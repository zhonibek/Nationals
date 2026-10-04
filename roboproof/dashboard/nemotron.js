'use strict';

(() => {
  const element = id => document.getElementById(`nemotron-${id}`);
  const local = ['http:', 'https:'].includes(location.protocol) && ['127.0.0.1', 'localhost'].includes(location.hostname);
  let session = null, controller = null;
  const text = (id, value) => { element(id).textContent = value; };
  const json = value => JSON.stringify(value, null, 2);
  function controls() {
    for (const id of ['check', 'restore', 'plan', 'prompt']) element(id).disabled = !local || !!controller;
    element('cancel').disabled = !controller;
    element('run').disabled = !local || !!controller || session?.status !== 'prepared' || !!session?.run;
    element('download').disabled = !!controller || !session;
  }
  async function request(route, body) {
    const response = await fetch(`/api/nemotron/${route}`, {cache: 'no-store', signal: controller?.signal,
      ...(body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: json(body)} : {})});
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
    catch (error) { text('status', error.name === 'AbortError' ? 'Cancelled. No completed result is claimed. Restore the session to check saved state.' : error.message); }
    finally { controller = null; controls(); }
  }
  function render(record) {
    session = record;
    element('session').hidden = false;
    element('proposal').hidden = record.status !== 'prepared';
    element('result').hidden = !record.run;
    text('message', `${record.status === 'prepared' ? 'Proposal' : 'Model response / no executable task'}: ${record.message}`);
    text('task', json(record.task || null));
    text('task-title', record.run ? 'Validated task · measured result recorded' : 'Validated proposal · not yet measured');
    if (record.inferencePerformed) text('connection', 'SAVED LOCAL INFERENCE');
    text('log', json({id: record.id, createdAt: record.createdAt, provider: record.provider,
      inferencePerformed: record.inferencePerformed, toolLog: record.toolLog, usage: record.usage,
      execution: record.execution, approvedAt: record.approvedAt, runError: record.runError,
      identity: record.run?.report.identity}));
    element('simulator').href = `/simulator/index.html?nemotron=${encodeURIComponent(record.id)}`;
    if (record.run) {
      const report = record.run.report;
      text('outcome', `${report.reason === 'success' ? 'SUCCESS' : report.reason.toUpperCase()} · ${report.metrics.elapsedSeconds.toFixed(2)} simulated seconds · exact same-runtime replay ${record.run.exactReplayVerified ? 'verified' : 'not verified'}`);
      text('metrics', json(report.metrics));
    }
    controls();
  }
  element('check').addEventListener('click', () => operation('Checking local model availability…', async () => {
    const result = await request('check', {});
    text('connection', 'LOCAL MODEL AVAILABLE');
    text('provider', `${result.model} · ${result.baseUrl} · cloud disabled`);
    text('status', 'Model endpoint is available. Availability is not a verified inference run.');
  }));
  element('restore').addEventListener('click', () => operation('Loading saved session…', async () => {
    const record = await request('latest');
    render(record);
    element('prompt').value = record.prompt;
    text('status', `Restored saved session: ${record.execution}`);
  }));
  element('form').addEventListener('submit', event => {
    event.preventDefault();
    void operation('Running local Nemotron. CPU inference may take time; no simulation runs until you approve.', async () => {
      session = null;
      element('session').hidden = true;
      const record = await request('plan', {prompt: element('prompt').value});
      render(record);
      text('connection', 'LOCAL INFERENCE COMPLETED');
      text('status', record.status === 'prepared' ? 'Task validated and saved. Review coordinates before approving the simulation.' : 'No executable task prepared. Clarify your request and ask again.');
    });
  });
  element('run').addEventListener('click', () => operation('Running and replay-checking the original Simulator with the actual iraLIB controller…', async () => {
    render(await request('run', {id: session.id}));
    text('status', 'Measured simulation and replay saved locally. This is not motion training or physical validation.');
  }));
  element('cancel').addEventListener('click', () => controller?.abort());
  element('download').addEventListener('click', () => {
    if (!session) return;
    const url = URL.createObjectURL(new Blob([json(session) + '\n'], {type: 'application/json'}));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'robotai-nemotron-session.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  controls();
  if (local) void request('status').then(result => text('provider', `${result.model} · ${result.baseUrl} · cloud disabled`))
    .catch(() => text('status', 'Start the updated RoboProof server to use Nemotron.'));
  else text('status', 'Nemotron requires the local RoboProof server. File-based legacy reports remain available.');
})();
