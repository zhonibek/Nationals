'use strict';

(() => {
  const element = id => document.getElementById(`nemotron-${id}`);
  const local = ['http:', 'https:'].includes(location.protocol) && ['127.0.0.1', 'localhost'].includes(location.hostname);
  let session = null, conversation = null, controller = null, lastPrompt = '', chatRevision = 0;
  const text = (id, value) => { element(id).textContent = value; };
  const json = value => JSON.stringify(value, null, 2);
  function controls() {
    for (const id of ['check', 'restore', 'plan', 'prompt', 'inspect', 'learning', 'diagnose-latest', 'prepare', 'new-chat', 'restore-chat', 'tactics']) element(id).disabled = !local || !!controller;
    element('cancel').disabled = !controller;
    element('thinking').hidden = !controller;
    element('chat-download').disabled = !!controller || !conversation;
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
    text('thinking', message);
    controls();
    text('status', message);
    try { await callback(); }
    catch (error) {
      const message = error.name === 'AbortError' ? 'Stopped. Your last completed replies remain saved. Restore chat or the experiment to check saved state.' : error.message;
      text('status', message);
      appendMessage('notice', message);
    }
    finally { controller = null; controls(); }
  }
  function appendMessage(role, content) {
    element('chat-empty').hidden = true;
    const bubble = document.createElement('article');
    bubble.className = `nemotron-chat-message ${role}`;
    const author = document.createElement('strong');
    author.textContent = role === 'user' ? 'You' : role === 'assistant' ? 'Nemotron' : 'Chat status';
    const body = document.createElement('div');
    body.className = 'nemotron-chat-content';
    body.textContent = content;
    bubble.appendChild(author); bubble.appendChild(body); element('messages').appendChild(bubble);
    const viewport = element('messages').parentElement;
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }
  function renderConversation(record) {
    conversation = record;
    element('messages').replaceChildren();
    element('chat-empty').hidden = !!record;
    for (const message of record?.messages ?? []) {
      appendMessage(message.role, message.content);
      if (message.truncated) appendMessage('notice', 'The reply reached the model output limit. Ask it to continue.');
    }
    lastPrompt = record?.messages?.filter(message => message.role === 'user').at(-1)?.content || '';
    controls();
  }
  function render(record) {
    session = record;
    element('session').hidden = false;
    element('proposal').hidden = record.status !== 'prepared';
    element('result').hidden = !record.run;
    element('analysis').hidden = record.status !== 'analyzed';
    text('message', record.message);
    text('analysis-facts', json(record.analysisEvidence || {}));
    text('loaded-skills', `Activated skills: ${(record.skills?.loaded ?? []).map(skill => skill.name).join(', ') || 'None recorded (possibly a legacy session)'}`);
    text('task', json(record.task || null));
    text('task-title', record.run ? 'Movement task · measured result recorded' : 'Proposed movement · waiting for your approval');
    if (record.task) {
      const {start, goal, deadlineSeconds} = record.task;
      text('task-summary', `From (${start.xIn}, ${start.yIn}) at ${start.headingDeg}° to (${goal.xIn}, ${goal.yIn}) at ${goal.headingDeg}°. Coordinates in inches; deadline ${deadlineSeconds} seconds.`);
    }
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
    const prompt = element('prompt').value.trim();
    if (!prompt) return;
    void operation('Nemotron is replying… Local CPU inference can take a minute or more. Use Stop to cancel.', async () => {
      chatRevision++;
      renderConversation(conversation);
      lastPrompt = prompt;
      appendMessage('user', prompt);
      element('prompt').value = '';
      try {
        renderConversation(await request('chat', {message: prompt, ...(conversation ? {id: conversation.id} : {})}));
        text('connection', 'LOCAL REPLY RECEIVED');
        text('status', 'Reply saved. Send another message to keep talking.');
      } catch (error) { element('prompt').value = prompt; throw error; }
    });
  });
  element('prompt').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey && !event.isComposing) {
      event.preventDefault();
      if (!controller && element('prompt').value.trim()) element('form').requestSubmit();
    }
  });
  element('new-chat').addEventListener('click', () => {
    if (controller) return;
    chatRevision++;
    renderConversation(null);
    session = null;
    element('session').hidden = true;
    element('prompt').value = '';
    text('status', 'New chat. Older conversations remain saved locally.');
    controls();
  });
  element('restore-chat').addEventListener('click', () => operation('Restoring the latest saved chat…', async () => {
    chatRevision++;
    renderConversation(await request('chat/latest'));
    text('status', 'Conversation restored. No new model request was made.');
  }));
  element('prepare').addEventListener('click', () => operation('Preparing a movement proposal. Nothing runs until you approve.',
    () => ask(element('prompt').value.trim() || lastPrompt)));
  element('inspect').addEventListener('click', () => operation('Local inference: inspecting nominal configuration and the controller path…',
    () => ask('Explain our robot configuration, drive geometry and active simulated controller path. Read the actual robot profile and state the limitations.')));
  element('learning').addEventListener('click', () => operation('Local inference: reviewing saved PPO checkpoints and frozen evaluation only…',
    () => ask('Read our saved motion learning summary. Have weights changed, and has learned movement passed the improvement gate? Include every seed, comparators and failures. Do not train or rerun evaluation.')));
  element('tactics').addEventListener('click', () => operation('Reading reviewed Override rules and planning roles; no game state or execution…',
    () => ask('Use plan-game-tactics to read the reviewed Override rules and snapshot availability. Explain general tactical priorities and how tactics could coordinate motion, manipulation and future perception. Without a snapshot do not claim current game state, points, an executable plan or a winning policy.')));
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
  element('chat-download').addEventListener('click', () => {
    if (!conversation) return;
    const url = URL.createObjectURL(new Blob([json(conversation) + '\n'], {type: 'application/json'}));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'robotai-nemotron-chat.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  controls();
  if (local) {
    void request('chat/latest').then(record => {
      if (!controller && !conversation && chatRevision === 0) {
        renderConversation(record);
        text('status', 'Your last conversation is restored. Send a message to continue.');
      }
    }).catch(error => {
      if (error.message !== 'No saved chat yet' && chatRevision === 0) text('status', `Could not restore chat: ${error.message}`);
    });
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
