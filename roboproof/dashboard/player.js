'use strict';

const RoboProofPlayer = (() => {
  const poseValid = pose => Array.isArray(pose) && pose.length >= 3 && pose.slice(0, 3).every(Number.isFinite);

  function prepare(result) {
    const rows = result?.telemetry;
    const task = result?.scenario?.task;
    const radius = result?.scenario?.environment?.radius;
    if (!Array.isArray(rows) || !rows.length) throw Error('No recorded telemetry. Recalculate this scenario before opening the simulator.');
    if (!poseValid(task?.start) || !poseValid(task?.goal) || !Number.isFinite(radius) || radius <= 0 || !Number.isFinite(task?.thresholds?.endpoint) || task.thresholds.endpoint <= 0) throw Error('Invalid scenario geometry for playback.');
    let previous = -Infinity;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    const include = pose => { xmin = Math.min(xmin, pose[0]); xmax = Math.max(xmax, pose[0]); ymin = Math.min(ymin, pose[1]); ymax = Math.max(ymax, pose[1]); };
    for (const row of rows) {
      if (!Number.isFinite(row.time) || row.time < 0 || row.time <= previous) throw Error('Playback requires finite, strictly increasing sample times.');
      previous = row.time;
      for (const name of ['truth', 'estimate', 'reference']) {
        if (!poseValid(row[name])) throw Error(`Invalid recorded ${name} pose; playback cannot invent missing positions.`);
        include(row[name]);
      }
    }
    include(task.start);
    include(task.goal);
    const padding = Math.max(radius, task.thresholds.endpoint) + 0.12;
    const bounds = {xmin: xmin - padding, xmax: xmax + padding, ymin: ymin - padding, ymax: ymax + padding};
    if (![bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin].every(size => Number.isFinite(size) && size > 0)) throw Error('Recorded geometry is outside the finite playback range.');
    return {rows, start: rows[0].time, end: rows.at(-1).time, bounds};
  }

  function sampleIndex(rows, time) {
    if (!rows.length || !Number.isFinite(time)) throw Error('A finite playback time and recorded samples are required.');
    let low = 0, high = rows.length - 1;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (rows[middle].time <= time) low = middle;
      else high = middle - 1;
    }
    return low;
  }

  function fieldTransform(bounds, width = 800, height = 500, margin = 40) {
    const scale = Math.min((width - 2 * margin) / (bounds.xmax - bounds.xmin), (height - 2 * margin) / (bounds.ymax - bounds.ymin));
    const middleX = (bounds.xmin + bounds.xmax) / 2, middleY = (bounds.ymin + bounds.ymax) / 2;
    return {scale, point: pose => [width / 2 + (pose[0] - middleX) * scale, height / 2 - (pose[1] - middleY) * scale]};
  }

  function headingVector(heading) { return [Math.sin(heading), -Math.cos(heading)]; }

  function create(host) {
    const doc = host.ownerDocument, view = doc.defaultView;
    const element = id => host.querySelector(`#${id}`);
    const format = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : 'not recorded';
    const svgNode = (tag, attrs, text) => {
      const result = doc.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [name, value] of Object.entries(attrs)) result.setAttribute(name, String(value));
      if (text !== undefined) result.textContent = text;
      return result;
    };
    let prepared = null, geometry = null, frame = null, previousTime = null, time = 0, index = 0;
    let truthRobot, estimateRobot, referenceMarker, truthTrail, estimateTrail, truthPoints, estimatePoints;

    function pause() {
      if (frame !== null) view.cancelAnimationFrame(frame);
      frame = null;
      previousTime = null;
      element('sim-play').textContent = 'Play';
      element('sim-play').setAttribute('aria-pressed', 'false');
    }

    function draw() {
      if (!prepared) return;
      const row = prepared.rows[index];
      const place = (glyph, pose) => {
        const point = geometry.point(pose);
        glyph.setAttribute('transform', `translate(${point[0]} ${point[1]}) rotate(${pose[2] * 180 / Math.PI})`);
        glyph.setAttribute('data-pose', JSON.stringify(pose));
      };
      place(truthRobot, row.truth);
      place(estimateRobot, row.estimate);
      const reference = geometry.point(row.reference);
      referenceMarker.setAttribute('cx', reference[0]);
      referenceMarker.setAttribute('cy', reference[1]);
      truthTrail.setAttribute('points', truthPoints.slice(0, index + 1).join(' '));
      estimateTrail.setAttribute('points', estimatePoints.slice(0, index + 1).join(' '));
      element('sim-timeline').value = String(index);
      element('sim-timeline').setAttribute('aria-valuetext', `${row.time.toFixed(2)} seconds, sample ${index + 1} of ${prepared.rows.length}`);
      element('sim-time').textContent = `${row.time.toFixed(2)} / ${prepared.end.toFixed(2)} s`;
      element('sim-position').textContent = `X ${format(row.truth[0])} m · Y ${format(row.truth[1])} m`;
      element('sim-heading').textContent = `Heading ${format(row.truth[2] * 180 / Math.PI, 1)}° clockwise from +Y`;
      element('sim-motion').textContent = `Speed ${format(row.velocity ? Math.hypot(row.velocity[0], row.velocity[1]) : NaN)} m/s`;
      element('sim-motors').textContent = `Motor commands: ${['FL', 'BL', 'FR', 'BR'].map((name, wheel) => `${name} ${format(row.command?.[wheel], 2)} V`).join(' · ')}`;
      element('sim-sample').textContent = `Recorded sample ${index + 1} / ${prepared.rows.length}${index === prepared.rows.length - 1 ? ' · End of recorded playback' : ''}`;
    }

    function tick(timestamp) {
      if (previousTime !== null) time = Math.min(prepared.end, time + (timestamp - previousTime) / 1000 * Number(element('sim-speed').value));
      previousTime = timestamp;
      index = sampleIndex(prepared.rows, time);
      draw();
      if (time >= prepared.end) pause();
      else frame = view.requestAnimationFrame(tick);
    }

    function play() {
      if (!prepared || frame !== null) return;
      if (index === prepared.rows.length - 1) { index = 0; time = prepared.start; draw(); }
      if (prepared.rows.length === 1) return;
      element('sim-play').textContent = 'Pause';
      element('sim-play').setAttribute('aria-pressed', 'true');
      previousTime = null;
      frame = view.requestAnimationFrame(tick);
    }

    function seek(sample) {
      if (!prepared || !Number.isFinite(sample)) return;
      pause();
      index = Math.max(0, Math.min(prepared.rows.length - 1, Math.round(sample)));
      time = prepared.rows[index].time;
      draw();
    }

    function clear() {
      pause();
      prepared = null;
      host.hidden = true;
      element('sim-field').replaceChildren();
    }

    function load(result) {
      const next = prepare(result);
      pause();
      prepared = next;
      geometry = fieldTransform(prepared.bounds);
      const scenario = result.scenario, task = scenario.task;
      const svg = svgNode('svg', {viewBox: '0 0 800 500', role: 'img', 'aria-label': `Top-down recorded robot simulation for ${scenario.scenario_id}. Green is physical truth; orange is the sensor estimate; blue is the reference.`});
      const bounds = prepared.bounds;
      const gridStep = 10 ** Math.floor(Math.log10(Math.max(bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin))) / 2;
      const firstX = Math.ceil(bounds.xmin / gridStep) * gridStep, firstY = Math.ceil(bounds.ymin / gridStep) * gridStep;
      for (let tickIndex = 0; tickIndex <= 20; tickIndex++) {
        const gridX = firstX + tickIndex * gridStep;
        if (gridX > bounds.xmax) break;
        const start = geometry.point([gridX, bounds.ymin]), end = geometry.point([gridX, bounds.ymax]);
        svg.append(svgNode('line', {x1: start[0], y1: start[1], x2: end[0], y2: end[1], class: 'chart-grid'}));
        svg.append(svgNode('text', {x: start[0], y: start[1] + 18, 'text-anchor': 'middle', class: 'chart-axis'}, format(gridX, 2)));
      }
      for (let tickIndex = 0; tickIndex <= 20; tickIndex++) {
        const gridY = firstY + tickIndex * gridStep;
        if (gridY > bounds.ymax) break;
        const start = geometry.point([bounds.xmin, gridY]), end = geometry.point([bounds.xmax, gridY]);
        svg.append(svgNode('line', {x1: start[0], y1: start[1], x2: end[0], y2: end[1], class: 'chart-grid'}));
        svg.append(svgNode('text', {x: start[0] - 8, y: start[1] + 4, 'text-anchor': 'end', class: 'chart-axis'}, format(gridY, 2)));
      }
      svg.append(svgNode('text', {x: 400, y: 492, 'text-anchor': 'middle', class: 'chart-axis'}, 'Field X / m · equal X/Y scale · +Y up'));
      const goal = geometry.point(task.goal), start = geometry.point(task.start);
      svg.append(svgNode('circle', {cx: goal[0], cy: goal[1], r: task.thresholds.endpoint * geometry.scale, class: 'sim-goal'}));
      const direction = headingVector(task.goal[2]);
      svg.append(svgNode('line', {x1: goal[0], y1: goal[1], x2: goal[0] + direction[0] * 24, y2: goal[1] + direction[1] * 24, class: 'sim-goal-heading'}));
      svg.append(svgNode('text', {x: goal[0], y: goal[1] - task.thresholds.endpoint * geometry.scale - 12, 'text-anchor': 'middle', class: 'chart-axis'}, 'GOAL'));
      svg.append(svgNode('circle', {cx: start[0], cy: start[1], r: 4, class: 'sim-start'}));
      svg.append(svgNode('text', {x: start[0], y: start[1] + scenario.environment.radius * geometry.scale + 22, 'text-anchor': 'middle', class: 'chart-axis'}, 'START'));
      const points = name => prepared.rows.map(row => geometry.point(row[name]).join(','));
      truthPoints = points('truth');
      estimatePoints = points('estimate');
      svg.append(svgNode('polyline', {points: points('reference').join(' '), class: 'plot-line sim-reference', 'stroke-dasharray': '8 5'}));
      truthTrail = svgNode('polyline', {class: 'plot-line sim-truth'});
      estimateTrail = svgNode('polyline', {class: 'plot-line sim-estimate', 'stroke-dasharray': '2 4'});
      referenceMarker = svgNode('circle', {r: 5, class: 'sim-reference-marker'});
      svg.append(estimateTrail, truthTrail, referenceMarker);
      const half = scenario.environment.radius / Math.sqrt(2) * geometry.scale;
      const robot = (id, estimated) => {
        const group = svgNode('g', {id, class: estimated ? 'sim-estimated-robot' : 'sim-true-robot'});
        group.append(svgNode('rect', {x: -half, y: -half, width: 2 * half, height: 2 * half, rx: 5}));
        if (!estimated) for (const offsetX of [-half, half]) for (const offsetY of [-half, half]) {
          group.append(svgNode('rect', {x: offsetX - half * 0.17, y: offsetY - half * 0.25, width: half * 0.34, height: half * 0.5, rx: 2, class: 'sim-wheel', transform: `rotate(${offsetX * offsetY > 0 ? -45 : 45} ${offsetX} ${offsetY})`}));
        }
        group.append(svgNode('path', {d: `M0,${half * 0.4}V${-half * 0.65}m${-half * 0.25},${half * 0.25}L0,${-half * 0.65}l${half * 0.25},${half * 0.25}`, class: 'sim-front'}));
        return group;
      };
      estimateRobot = robot('sim-estimate-robot', true);
      truthRobot = robot('sim-truth-robot', false);
      svg.append(estimateRobot, truthRobot);
      element('sim-field').replaceChildren(svg);
      element('sim-scenario').textContent = `${scenario.scenario_id} · seed ${scenario.random_seed} · ${format(scenario.environment.battery_voltage, 2)} V battery`;
      element('sim-outcome').textContent = `Full-run result: ${result.passed === true ? 'PASS' : result.passed === false ? 'FAIL' : 'UNKNOWN'}`;
      element('sim-outcome').className = `badge ${result.passed === true ? 'pass' : result.passed === false ? 'fail' : ''}`;
      element('sim-summary').textContent = `Final endpoint error ${format(result.metrics?.endpoint_error)} m / ${format(task.thresholds.endpoint)} m allowed · final heading error ${format(result.metrics?.final_heading_error * 180 / Math.PI, 1)}° / ${format(task.thresholds.heading * 180 / Math.PI, 1)}° allowed.`;
      element('sim-timeline').max = String(prepared.rows.length - 1);
      host.hidden = false;
      seek(0);
    }

    element('sim-play').addEventListener('click', () => frame === null ? play() : pause());
    element('sim-restart').addEventListener('click', () => seek(0));
    element('sim-end').addEventListener('click', () => { if (prepared) seek(prepared.rows.length - 1); });
    element('sim-timeline').addEventListener('input', () => seek(Number(element('sim-timeline').value)));
    element('sim-close').addEventListener('click', clear);
    doc.addEventListener('visibilitychange', () => { if (doc.hidden) pause(); });
    return {load, clear, play, pause, seek};
  }

  return {prepare, sampleIndex, fieldTransform, headingVector, create};
})();

if (typeof module !== 'undefined' && module.exports) module.exports = RoboProofPlayer;
