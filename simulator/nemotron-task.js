'use strict';

window.RobotAINemotron = {
  async attach(sim) {
    const id = new URLSearchParams(location.search).get('nemotron');
    if (!id) return;
    const panel = document.getElementById('robotAITask');
    const status = document.getElementById('robotAITaskStatus');
    const button = document.getElementById('robotAILoadTask');
    panel.hidden = false;
    try {
      const response = await fetch(`/api/nemotron/session?id=${encodeURIComponent(id)}`, {cache: 'no-store'});
      if (!response.ok) throw Error('Saved task is unavailable; open this Simulator through RoboProof');
      const record = await response.json();
      if (record.status !== 'prepared') throw Error('No executable task in this session');
      const task = NationalsMotion.normalizeTask(record.task);
      status.textContent = `Nemotron proposal: (${task.start.xIn}, ${task.start.yIn}, ${task.start.headingDeg}°) → (${task.goal.xIn}, ${task.goal.yIn}, ${task.goal.headingDeg}°). Positions in inches. Load, then press Play. Visual execution is separate; the RoboProof server report contains measured deadline/success/replay.`;
      button.disabled = !sim.productionControl;
      button.addEventListener('click', () => {
        sim.resetSimulation();
        sim.setPose(task.start.xIn, task.start.yIn, task.start.headingDeg);
        sim.queueAction({type: 'pose', targetX: task.goal.xIn, targetY: task.goal.yIn, targetTheta: task.goal.headingDeg});
        sim.isRunning = true;
        sim.isPaused = true;
        status.textContent = 'Nemotron task loaded. Use normal Play/Pause controls to watch it. Recorded success and deadline are in RoboProof; this visual execution is not motion-policy training.';
      });
    } catch (error) { button.disabled = true; status.textContent = error.message; }
  }
};
