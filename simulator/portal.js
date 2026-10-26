'use strict';

// Portal for the simulator and RoboProof. It works when served by the RoboProof server
// (http://127.0.0.1:8766/simulator/portal.html, one Node process) and by the standalone
// simulator server on port 8765.
(function () {
  const standalone = location.port === '8765';
  const roboproofUrl = standalone ? 'http://127.0.0.1:8766/' : '/';
  const apiBase = standalone ? 'http://127.0.0.1:8766' : '';
  const simulatorUrl = 'index.html';

  const frameContainer = document.getElementById('frameContainer');
  const tabs = {app1: document.getElementById('tabApp1'), app2: document.getElementById('tabApp2'), split: document.getElementById('tabSplit')};
  const toast = document.getElementById('toast');
  let currentMode = 'app1';

  for (const link of document.querySelectorAll('[data-roboproof-link]')) link.href = roboproofUrl;

  function frame(src, title) {
    const element = document.createElement('iframe');
    element.src = src;
    element.title = title;
    return element;
  }

  function switchView(mode) {
    currentMode = mode;
    for (const [name, tab] of Object.entries(tabs)) tab?.classList.toggle('active', name === mode);
    frameContainer.replaceChildren();
    if (mode === 'split') {
      const split = document.createElement('div');
      split.className = 'split-view';
      split.append(frame(simulatorUrl, 'App 1 Simulator'), frame(roboproofUrl, 'App 2 RoboProof'));
      frameContainer.append(split);
    } else {
      const element = mode === 'app2' ? frame(roboproofUrl, 'App 2 RoboProof') : frame(simulatorUrl, 'App 1 Simulator');
      element.id = 'mainFrame';
      frameContainer.append(element);
    }
  }

  for (const button of document.querySelectorAll('[data-view]')) {
    button.addEventListener('click', () => switchView(button.dataset.view));
  }

  document.getElementById('btnOpenCurrentTab')?.addEventListener('click', () => {
    if (currentMode !== 'app2') window.open(simulatorUrl, '_blank', 'noopener');
    if (currentMode !== 'app1') window.open(roboproofUrl, '_blank', 'noopener');
  });

  document.getElementById('btnMasterLaunch')?.addEventListener('click', () => {
    document.getElementById('workspaceSection').scrollIntoView({behavior: 'smooth'});
    toast.style.display = 'flex';
    setTimeout(() => { toast.style.display = 'none'; }, 4000);
    switchView('split');
  });

  const dot = document.getElementById('dotNebius');
  const badgeText = document.getElementById('textNebius');
  function badge(color, text) {
    dot.style.background = color;
    dot.style.boxShadow = color === '#64748b' ? 'none' : `0 0 8px ${color}`;
    badgeText.textContent = text;
  }

  // The server runs one operation at a time; a check that collides with another request gets 409, so retry briefly.
  async function check(body = {}, attempts = 4) {
    for (let attempt = 1; ; attempt++) {
      const response = await fetch(`${apiBase}/api/nemotron/check`, {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      if (response.status === 409 && attempt < attempts) { await new Promise(resolve => setTimeout(resolve, 1500)); continue; }
      return {ok: response.ok, data: await response.json()};
    }
  }

  async function refreshNebius() {
    try {
      const {ok, data} = await check();
      if (ok && data.available && data.provider === 'nebius-token-factory') {
        badge('#10b981', 'Nebius Token Factory: online');
        badgeButton.title = `NVIDIA ${data.model} on Nebius Token Factory`;
      }
      else badge('#eab308', 'Nebius: click to connect');
    } catch {
      badge('#64748b', 'Nebius: RoboProof server not reachable');
    }
  }

  // Inline key form instead of prompt()/alert(); the key is sent only to the local RoboProof server.
  const badgeButton = document.getElementById('badgeNebius');
  const form = document.getElementById('nebiusKeyForm');
  const input = document.getElementById('nebiusKeyInput');
  const formStatus = document.getElementById('nebiusKeyStatus');
  function showForm(visible) {
    form.hidden = !visible;
    badgeButton.setAttribute('aria-expanded', String(visible));
    if (visible) input.focus();
  }
  badgeButton?.addEventListener('click', () => showForm(form.hidden));
  document.getElementById('nebiusKeyCancel')?.addEventListener('click', () => { input.value = ''; showForm(false); });
  form?.addEventListener('submit', async event => {
    event.preventDefault();
    const key = input.value.trim();
    if (!key) return;
    formStatus.textContent = 'Checking the key with Nebius Token Factory…';
    try {
      const {ok, data} = await check({apiKey: key});
      if (ok && data.available) {
        input.value = '';
        formStatus.textContent = `Connected: ${data.model}. The key is saved in the local .cache/ folder.`;
        await refreshNebius();
        setTimeout(() => showForm(false), 1500);
      } else {
        formStatus.textContent = `Not connected: ${data.error || 'Nebius rejected the key'}`;
      }
    } catch (error) {
      formStatus.textContent = `Could not reach the RoboProof server: ${error.message}`;
    }
  });

  refreshNebius();
})();
