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
      location.pathname.startsWith('/simulator/');
    button.disabled = replay || !local;
    if (!local) status.textContent = 'Для AI откройте исходный Simulator через RoboProof: http://127.0.0.1:8766/simulator/index.html';
    stop.disabled = true;
    stop.addEventListener('click', () => controller?.abort());
    button.addEventListener('click', async () => {
      if (controller || replay || !local) return;
      if (!sim.fleet) { status.textContent = 'Сначала запустите игру кнопкой Override, затем запросите тактику.'; return; }
      controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 310000);
      button.disabled = true; stop.disabled = false; result.hidden = true;
      try {
        const snapshot = Snapshot.capture(sim.override.getState(), sim.activeRobotId, new Date().toISOString(), sim.override.world);
        status.textContent = 'Nemotron читает правила и снимок игры. Это совет, не запуск действий; CPU может отвечать несколько минут.';
        const response = await fetch('/api/nemotron/plan', {method: 'POST', signal: controller.signal,
          headers: {'Content-Type': 'application/json'}, body: JSON.stringify({gameSnapshot: snapshot,
            prompt: 'Предложи три условных тактических приоритета для альянса выбранного робота по правилам Override и выбранному снимку. Прочитай правила и снимок через skill plan-game-tactics. Укажи основания правил, роли motion/manipulation/perception и ограничения. Ничего не выполняй; не обещай очки или победу.'})});
        const record = await response.json();
        if (!response.ok) throw Error(record.error || `HTTP ${response.status}`);
        if (!['analyzed', 'clarification'].includes(record.status) || typeof record.message !== 'string') throw Error('Ожидался только тактический совет, не исполняемая задача');
        message.textContent = record.message;
        result.hidden = false;
        status.textContent = `Совет сохранён. Снимок: ${snapshot.capturedAt}, ${snapshot.phase}, ${snapshot.clock.toFixed(1)} с. Игра могла измениться; никаких действий не запущено.`;
      } catch (error) { status.textContent = error.name === 'AbortError' ? 'Запрос остановлен. Тактика не выполнена.' : error.message; }
      finally { clearTimeout(timer); controller = null; button.disabled = replay || !local; stop.disabled = true; }
    });
    return {cancel: () => controller?.abort()};
  }
  return {attach};
});
