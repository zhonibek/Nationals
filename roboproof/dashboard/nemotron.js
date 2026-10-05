'use strict';

(() => {
  const element = id => document.getElementById(`nemotron-${id}`);
  const local = ['http:', 'https:'].includes(location.protocol) && ['127.0.0.1', 'localhost'].includes(location.hostname);
  let session = null, controller = null;
  const text = (id, value) => { element(id).textContent = value; };
  const json = value => JSON.stringify(value, null, 2);
  function controls() {
    for (const id of ['check', 'restore', 'plan', 'prompt', 'inspect', 'learning', 'diagnose-latest']) element(id).disabled = !local || !!controller;
    element('cancel').disabled = !controller;
    element('run').disabled = !local || !!controller || session?.status !== 'prepared' || !!session?.run;
    element('download').disabled = !!controller || !session;
    element('explain-run').disabled = !local || !!controller || !session?.run;
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
    element('analysis').hidden = record.status !== 'analyzed';
    text('message', `${record.status === 'prepared' ? 'Proposal' : record.status === 'analyzed' ? 'Model commentary / read-only analysis' : 'Model response / no executable task'}: ${record.message}`);
    text('analysis-facts', json(record.analysisEvidence || {}));
    text('loaded-skills', `Activated skills: ${(record.skills?.loaded ?? []).map(skill => skill.name).join(', ') || 'None recorded (possibly a legacy session)'}`);
    text('task', json(record.task || null));
    text('task-title', record.run ? 'Validated task · measured result recorded' : 'Validated proposal · not yet measured');
    if (record.inferencePerformed) text('connection', 'SAVED LOCAL INFERENCE');
    text('log', json({id: record.id, createdAt: record.createdAt, provider: record.provider,
      inferencePerformed: record.inferencePerformed, toolLog: record.toolLog, usage: record.usage,
      execution: record.execution, approvedAt: record.approvedAt, runError: record.runError,
      skills: record.skills, commentaryVerified: record.commentaryVerified,
      identity: record.run?.report.identity}));
    element('simulator').href = `/simulator/index.html?nemotron=${encodeURIComponent(record.id)}`;
    if (record.run) {
      const report = record.run.report;
      text('outcome', `${report.reason === 'success' ? 'SUCCESS' : report.reason.toUpperCase()} · ${report.metrics.elapsedSeconds.toFixed(2)} simulated seconds · exact same-runtime replay ${record.run.exactReplayVerified ? 'verified' : 'not verified'}`);
      text('metrics', json(report.metrics));
    }
    controls();
  }
  async function ask(prompt, evidence) {
    session = null;
    element('session').hidden = true;
    const record = await request('plan', {prompt, ...(evidence ? {evidence} : {})});
    render(record);
    text('connection', 'LOCAL INFERENCE COMPLETED');
    text('status', record.status === 'prepared' ? 'Task validated and saved. Review coordinates before approving the simulation.'
      : record.status === 'analyzed' ? 'Analysis and read-only evidence saved. No simulation, training or policy promotion was performed.'
      : 'No executable task prepared. Clarify your request and ask again.');
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
      await ask(element('prompt').value);
    });
  });
  element('inspect').addEventListener('click', () => operation('Local inference: inspecting nominal configuration and the controller path…',
    () => ask('Explain our robot configuration, drive geometry and active simulated controller path. Read the actual robot profile and state the limitations.')));
  element('learning').addEventListener('click', () => operation('Local inference: reviewing saved PPO checkpoints and frozen evaluation only…',
    () => ask('Read our saved motion learning summary. Have weights changed, and has learned movement passed the improvement gate? Include every seed, comparators and failures. Do not train or rerun evaluation.')));
  element('diagnose-latest').addEventListener('click', () => operation('Selecting the latest saved Motion lab experiment, then running local diagnosis…', async () => {
    const response = await fetch('/api/motion/latest', {cache: 'no-store', signal: controller.signal});
    const record = await response.json();
    if (!response.ok) throw Error(record.error || 'No saved Motion lab experiment');
    await ask('Diagnose the selected saved movement experiment. Read measured evidence, report the outcome and settling checks, and separate facts from hypotheses. Do not rerun or train.',
      {source: 'motion', id: record.id});
  }));
  element('explain-run').addEventListener('click', () => {
    if (!session?.run) return;
    const id = session.id;
    void operation('Local inference: analyzing this saved Simulator result without rerunning it…',
      () => ask('Diagnose this selected saved Nemotron movement experiment. Use measured evidence, settling checks and telemetry; distinguish facts from hypotheses. Do not rerun or train.',
        {source: 'nemotron', id}));
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
  if (local) {
    void request('status').then(result => text('provider', `${result.model} · ${result.baseUrl} · cloud disabled`))
      .catch(() => text('status', 'Start the updated RoboProof server to use Nemotron.'));
    void request('skills').then(result => {
      element('skills').replaceChildren();
      for (const skill of result.skills) {
        const card = document.createElement('div');
        const title = document.createElement('h3');
        const description = document.createElement('p');
        title.textContent = skill.name;
        description.textContent = skill.description;
        card.appendChild(title); card.appendChild(description); element('skills').appendChild(card);
      }
      text('skills-status', `${result.skills.length} reviewed skills available. Instructions load on demand; availability is not inference evidence.`);
    }).catch(error => text('skills-status', `Skill catalog unavailable: ${error.message}`));
  }
  else text('status', 'Nemotron requires the local RoboProof server. File-based legacy reports remain available.');
})();
