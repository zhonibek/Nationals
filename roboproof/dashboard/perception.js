'use strict';

(async function() {
  const status = document.getElementById('perception-status');
  if (location.protocol === 'file:') {
    status.textContent = 'Offline documentation view. Start the local server to inspect preparation status. No camera or inference is active.';
    return;
  }
  try {
    const response = await fetch('/api/perception/status', {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw Error(`Status request failed (${response.status})`);
    const readiness = await response.json();
    document.getElementById('perception-stage').textContent = readiness.stage;
    document.getElementById('perception-model').textContent = readiness.model;
    status.textContent = 'Preparation only. Live inference, camera connection and automatic movement are disabled.';
    const gates = document.getElementById('perception-gates');
    for (const gate of readiness.nextGates) {
      const item = document.createElement('li');
      item.textContent = gate;
      gates.append(item);
    }
  } catch (error) { status.textContent = `Preparation status unavailable: ${error.message}. Camera-driven movement remains disabled.`; }
})();
