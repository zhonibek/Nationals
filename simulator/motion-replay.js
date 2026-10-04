(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./motion'));
  else root.RobotAIMotionReplay = factory(root.NationalsMotion);
})(globalThis, function(Motion) {
  'use strict';

  function matching(actual, recorded, location = 'trace') {
    if (typeof actual === 'number' && typeof recorded === 'number') {
      if (!Number.isFinite(actual) || !Number.isFinite(recorded) || Math.abs(actual - recorded) > 1e-9) throw Error(`Replay numerical mismatch at ${location}`);
    } else if (actual && recorded && typeof actual === 'object' && typeof recorded === 'object') {
      const actualKeys = Object.keys(actual);
      const recordedKeys = Object.keys(recorded);
      if (JSON.stringify(actualKeys.sort()) !== JSON.stringify(recordedKeys.sort())) throw Error(`Replay shape mismatch at ${location}`);
      for (const key of actualKeys) matching(actual[key], recorded[key], `${location}.${key}`);
    } else if (actual !== recorded) throw Error(`Replay value mismatch at ${location}`);
  }

  class Playback {
    constructor(report, controlFactory) {
      if (report?.schemaVersion !== Motion.SCHEMA_VERSION || report.fixedDt !== Motion.FIXED_DT ||
        report.taskType !== 'reach-pose-benchmark' || !Number.isInteger(report.ticks) || report.ticks < 1 || report.ticks > 6000 ||
        !Array.isArray(report.actions) || report.actions.length > 6001 ||
        !Array.isArray(report.transitions) || report.transitions.length !== report.ticks || !report.options?.recordTransitions ||
        !report.identity?.canonicalSimulator) throw Error('A current canonical report with complete transitions is required');
      let previous = -1;
      for (const entry of report.actions) {
        if (!Number.isInteger(entry.tick) || entry.tick < previous || entry.tick < 0 || entry.tick >= report.ticks) throw Error('Invalid replay action tick');
        previous = entry.tick;
      }
      this.record = JSON.parse(JSON.stringify(report));
      this.episode = new Motion.MotionEpisode(controlFactory);
      this.restart();
    }

    restart() {
      const speed = this.sim?.timeScale;
      this.episode.reset(this.record.seed, this.record.configuration, this.record.task, this.record.options);
      if (this.sim) {
        Object.assign(this.sim, this.episode.sim);
        this.episode.sim = this.sim;
      } else this.sim = this.episode.sim;
      if (speed !== undefined) this.sim.timeScale = speed;
      this.sim.isPaused = true;
      this.actionIndex = 0;
      this.finished = false;
      this.verified = false;
      this.error = null;
    }

    step(dt = Motion.FIXED_DT) {
      if (this.sim.isPaused || this.finished) return;
      try {
        const tick = this.episode.ticks;
        while (this.actionIndex < this.record.actions.length && this.record.actions[this.actionIndex].tick === tick) {
          this.episode.applyAction(this.record.actions[this.actionIndex++].action);
        }
        this.episode.step(dt);
        matching(this.episode.transitions[tick], this.record.transitions[tick], `tick[${tick}]`);
        if (this.episode.ticks === this.record.ticks) {
          const actual = this.episode.report();
          for (const key of ['reason', 'terminated', 'truncated', 'totalReward', 'metrics']) matching(actual[key], this.record[key], key);
          this.finished = this.verified = true;
          this.sim.isPaused = true;
        }
      } catch (error) {
        this.episode.neutral();
        this.sim.isPaused = true;
        this.finished = true;
        this.error = error.message;
        throw error;
      }
    }
  }

  async function attach(control) {
    const id = new URLSearchParams(location.search).get('motion');
    if (!id) return null;
    const panel = document.getElementById('robotAITask');
    const status = document.getElementById('robotAITaskStatus');
    const button = document.getElementById('robotAILoadTask');
    panel.hidden = false;
    button.disabled = true;
    button.textContent = 'Restart recorded experiment';
    try {
      if (!control) throw Error('Production C++ controller is unavailable');
      const response = await fetch(`/api/motion/session?id=${encodeURIComponent(id)}`, {cache: 'no-store'});
      if (!response.ok) throw Error('Saved motion evidence is unavailable');
      const record = await response.json();
      const sources = ['engine.js', 'motion.js', 'policy.js', 'control-runtime.js', 'robot-config.js',
        'override.js', 'override-geometry.js', 'override-dynamics.js', 'override-autonomy.js'];
      async function digest(bytes) {
        return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
          .map(value => value.toString(16).padStart(2, '0')).join('');
      }
      for (const filename of sources) {
        const asset = await fetch(filename, {cache: 'no-store'});
        if (!asset.ok || await digest(new TextEncoder().encode((await asset.text()).replace(/\r\n/g, '\n'))) !== record.report.identity.sources[filename]) throw Error(`Recorded browser engine is stale: ${filename}`);
      }
      const wasm = await fetch('control.wasm', {cache: 'no-store'});
      if (!wasm.ok || await digest(await wasm.arrayBuffer()) !== record.report.identity.controllerWasmSha256) throw Error('Recorded controller is stale');
      const playback = new Playback(record.report, () => control.fork());
      status.textContent = 'Recorded actions loaded in the original Simulator. Press Play. Every transition is checked at 1e-9 absolute numerical tolerance; this browser check is separate from exact server replay. No model or policy inference runs.';
      button.disabled = false;
      button.addEventListener('click', () => {
        playback.restart();
        status.textContent = 'Replay reset to its recorded world and seed. Press Play.';
      });
      const tick = playback.step.bind(playback);
      playback.step = dt => {
        try {
          tick(dt);
          if (playback.verified) status.textContent = `Original-engine replay verified at 1e-9 tolerance: ${playback.record.reason}, ${playback.record.ticks} physics ticks. This is recorded movement, not learning or physical validation.`;
        } catch (error) { status.textContent = error.message + '. Playback stopped; no reproduction claim.'; }
      };
      const permitted = new Set(['robotAILoadTask', 'btnPlay', 'btnPause', 'btnReset', 'btnStep',
        'tab2D', 'tab3D', 'camIso', 'camTop', 'camFollow', 'toggleTarget', 'toggleTrail', 'toggleEKF', 'toggleCoord']);
      for (const input of document.querySelectorAll('button, input, select')) {
        if (!permitted.has(input.id) && !input.dataset.speed) input.disabled = true;
      }
      return playback;
    } catch (error) { status.textContent = `Replay unavailable: ${error.message}. No recorded result is being reproduced.`; return null; }
  }

  return {Playback, matching, attach};
});
