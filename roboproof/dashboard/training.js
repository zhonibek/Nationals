'use strict';

// Tactical AI training page: verified best scores, every run's learning curve replayed game by game, the best model's
// full match decision by decision, and its neuron layers for each decision. Reads only saved JSON; trains nothing.
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const $ = id => document.getElementById(id);
  const fmt = (value, digits = 1) => (Number.isFinite(value) ? value.toFixed(digits) : '–');
  const signed = (value, digits = 1) => (Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${value.toFixed(digits)}` : '–');
  const COLORS = {best: '#6be4cc', run: '#93b9ff', ref: '#ffc17a', baseline: '#ffc17a', ai: '#6be4cc', grid: '#243440',
    muted: '#acbdc9', critic: '#dab0ff', entropy: '#93b9ff', chosen: '#6be4cc', other: '#2c4050'};

  function el(tag, attributes = {}, parent = null, text = null) {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  function html(tag, attributes = {}, parent = null, text = null) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  const clear = node => { while (node.firstChild) node.removeChild(node.firstChild); };
  const scale = (domain, range) => value => range[0] + (value - domain[0]) / ((domain[1] - domain[0]) || 1) * (range[1] - range[0]);
  const pathOf = (points, x, y) => points.map(([px, py], index) => `${index ? 'L' : 'M'}${x(px).toFixed(1)} ${y(py).toFixed(1)}`).join(' ');

  function axes(svg, box, xDomain, yDomain, {xLabel, yLabel, yTicks = 5}) {
    const x = scale(xDomain, [box.left, box.right]);
    const y = scale(yDomain, [box.bottom, box.top]);
    for (let tick = 0; tick <= yTicks; tick += 1) {
      const value = yDomain[0] + (yDomain[1] - yDomain[0]) * tick / yTicks;
      el('line', {x1: box.left, x2: box.right, y1: y(value), y2: y(value), class: 'chart-grid'}, svg);
      el('text', {x: box.left - 8, y: y(value) + 4, 'text-anchor': 'end', class: 'chart-axis'}, svg, fmt(value, Math.abs(yDomain[1] - yDomain[0]) < 5 ? 2 : 0));
    }
    for (let tick = 0; tick <= 5; tick += 1) {
      const value = xDomain[0] + (xDomain[1] - xDomain[0]) * tick / 5;
      el('text', {x: x(value), y: box.bottom + 18, 'text-anchor': 'middle', class: 'chart-axis'}, svg, fmt(value, 0));
    }
    if (xLabel) el('text', {x: (box.left + box.right) / 2, y: box.bottom + 36, 'text-anchor': 'middle', class: 'chart-axis'}, svg, xLabel);
    if (yLabel) el('text', {x: box.left, y: box.top - 10, class: 'chart-axis'}, svg, yLabel);
    return {x, y};
  }

  // ---------- best-score cards ----------
  function renderCards(data, replay) {
    const cards = $('best-cards');
    clear(cards);
    const card = (title, value, detail, note) => {
      const article = html('article', {class: 'training-card'}, cards);
      html('p', {class: 'eyebrow'}, article, title);
      html('p', {class: 'training-big'}, article, value);
      html('p', {}, article, detail);
      if (note) html('p', {class: 'hint'}, article, note);
    };
    const single = data?.singleGoal?.independentCheck;
    if (single) {
      const primary = single.primary;
      const versus = primary.versusLoaderCycle;
      card('SINGLE-GOAL AI · MAXIMIZE SCORE', `${fmt(primary.independentMeanMargin)} vs ${fmt(primary.loaderCycleMeanMargin)}`,
        `Score margin against the loading-zone team tactic on ${single.episodes} start poses it never saw: ${signed(versus.meanDifference)} points (95% interval ${signed(versus.bootstrap95[0])} to ${signed(versus.bootstrap95[1])}), won ${versus.wins} of ${versus.episodes}.`,
        `Run ${String(primary.index).padStart(2, '0')}, seed ${primary.seed}, chosen before the check.`);
    }
    const multi = data?.multiTask?.independentCheck;
    if (multi) {
      const primary = multi.primary;
      const versus = primary.versusTaskGreedy;
      card('MULTI-TASK AI · ANY TYPED TASK', `${fmt(primary.independentMeanTaskReturn)} vs ${fmt(primary.taskGreedyMeanTaskReturn)}`,
        `Task return against the best simple task tactic on ${multi.episodes} tasks it never saw: ${signed(versus.meanDifference)} (95% interval ${signed(versus.bootstrap95[0])} to ${signed(versus.bootstrap95[1])}). Tasks completed: ${fmt(primary.successRate * 100, 0)}% vs ${fmt(primary.taskGreedySuccessRate * 100, 0)}%.`,
        `Run ${String(primary.index).padStart(2, '0')}, seed ${primary.seed}. Wins ${versus.wins}, ties ${versus.ties}, losses ${versus.losses}.`);
    } else {
      card('MULTI-TASK AI · ANY TYPED TASK', 'pending', 'Its training logs are still being copied from the AMD droplet; rebuild training-data.json when they arrive.');
    }
    if (replay) {
      card('THE MATCH BELOW', `${signed(replay.finalScore.margin, 0)} vs ${signed(replay.baseline.finalScore.margin, 0)}`,
        `Final score margin of the best model as ${replay.alliance} on start pose ${replay.seed}, against loader-cycle on the same pose: ${replay.decisions.length} decisions.`,
        `${replay.policy.width} × ${replay.policy.hiddenLayers} network, ${replay.policy.featureSet} features.`);
    }
  }

  // ---------- training replay ----------
  const training = {campaign: 'singleGoal', cursor: 1, max: 1, playing: false, frame: null};

  function campaignData(data) { return data?.[training.campaign]; }

  function drawTraining(data) {
    const section = campaignData(data);
    const svg = $('training-chart');
    clear(svg);
    el('title', {}, svg, 'Learning curves of every run');
    const legend = $('training-legend');
    clear(legend);
    if (!section) {
      el('text', {x: 480, y: 180, 'text-anchor': 'middle', class: 'chart-axis'}, svg, 'This campaign is not in the data file yet.');
      $('training-readout').textContent = '';
      return;
    }
    const multi = section.metric === 'taskReturn';
    const runs = section.runs.filter(run => run.curve.length);
    const values = runs.flatMap(run => run.curve.map(point => point[1]));
    const reference = multi ? section.best.heldOutTaskGreedy : section.best.heldOutLoaderCycle;
    const low = Math.min(...values, reference ?? Infinity);
    const high = Math.max(...values, section.best.heldOut ?? -Infinity, reference ?? -Infinity);
    const pad = (high - low) * 0.08;
    const box = {left: 64, right: 940, top: 28, bottom: 310};
    const {x, y} = axes(svg, box, [1, training.max], [low - pad, high + pad],
      {xLabel: 'training game', yLabel: multi ? 'task return (moving average of 256 games)' : 'score margin (moving average of 64 games)'});
    if (Number.isFinite(reference)) {
      el('line', {x1: box.left, x2: box.right, y1: y(reference), y2: y(reference), stroke: COLORS.ref, 'stroke-dasharray': '6 5', 'stroke-width': 1.5}, svg);
      el('text', {x: box.right - 4, y: y(reference) - 6, 'text-anchor': 'end', class: 'chart-axis', fill: COLORS.ref}, svg,
        multi ? `task-greedy on held-out tasks ${fmt(reference)}` : `loader-cycle team tactic ${fmt(reference)}`);
    }
    if (Number.isFinite(section.best.heldOut)) {
      el('line', {x1: box.left, x2: box.right, y1: y(section.best.heldOut), y2: y(section.best.heldOut), stroke: COLORS.best, 'stroke-dasharray': '2 5', 'stroke-width': 1.2}, svg);
      el('text', {x: box.right - 4, y: y(section.best.heldOut) - 6, 'text-anchor': 'end', class: 'chart-axis', fill: COLORS.best}, svg,
        `best run, deterministic held-out ${fmt(section.best.heldOut)}`);
    }
    const visible = run => {
      const points = run.curve.filter(point => point[0] <= training.cursor);
      return points.length ? points : run.curve.slice(0, 1);
    };
    for (const run of runs) {
      if (run.index === section.best.index) continue;
      el('path', {d: pathOf(visible(run), x, y), fill: 'none', stroke: COLORS.run, 'stroke-opacity': 0.28, 'stroke-width': 1}, svg);
    }
    const bestRun = runs.find(run => run.index === section.best.index);
    if (bestRun) el('path', {d: pathOf(visible(bestRun), x, y), fill: 'none', stroke: COLORS.best, 'stroke-width': 2.6}, svg);
    el('line', {x1: x(training.cursor), x2: x(training.cursor), y1: box.top, y2: box.bottom, stroke: COLORS.muted, 'stroke-opacity': 0.6}, svg);
    html('li', {class: 'training-key-best'}, legend, `Best run ${String(section.best.index).padStart(2, '0')} (seed ${section.best.seed})`);
    html('li', {class: 'training-key-run'}, legend, `Other ${runs.length - 1} runs, trained in parallel`);
    html('li', {class: 'training-key-ref'}, legend, multi ? 'Best simple task tactic' : 'Team loading-zone tactic');
    drawPpo(section);
    readTraining(section, runs);
  }

  function valueAt(curve, episode) {
    let value = curve[0]?.[1];
    for (const point of curve) { if (point[0] > episode) break; value = point[1]; }
    return value;
  }

  function readTraining(section, runs) {
    const multi = section.metric === 'taskReturn';
    const ppo = section.best.ppo.filter(row => row.episodes <= training.cursor);
    const last = ppo.at(-1);
    const reference = multi ? null : section.best.heldOutLoaderCycle;
    const above = reference === null ? null : runs.filter(run => valueAt(run.curve, training.cursor) > reference).length;
    const bestRun = section.runs.find(run => run.index === section.best.index);
    const bestNow = valueAt(bestRun.curve, training.cursor);
    const number = value => value.toLocaleString('en-US');
    const bestGame = Math.min(training.cursor, bestRun.episodes);
    const parts = [`Game ${number(training.cursor)} of ${number(training.max)}`,
      `best run ${multi ? 'task return' : 'margin'} ${fmt(bestNow)}${training.cursor > bestRun.episodes ? ` (it finished at game ${number(bestRun.episodes)})` : ''}`,
      `PPO update ${number(Math.min(section.best.ppoUpdates, Math.floor(bestGame / 8)))} of ${number(section.best.ppoUpdates)}`];
    if (last) parts.push(`critic loss ${fmt(last.criticLoss, 2)}`, `entropy ${fmt(last.entropy, 2)}`);
    if (above !== null) parts.push(`${above} of ${runs.length} runs' training average above the team tactic's ${fmt(reference)}`);
    $('training-readout').textContent = parts.join(' · ');
  }

  function drawPpo(section) {
    const svg = $('ppo-chart');
    clear(svg);
    const rows = section.best.ppo.filter(row => Number.isFinite(row.criticLoss) && Number.isFinite(row.entropy));
    if (!rows.length) return;
    const box = {left: 34, right: 446, top: 16, bottom: 168};
    const x = scale([rows[0].episodes, rows.at(-1).episodes], [box.left, box.right]);
    for (const [key, color] of [['criticLoss', COLORS.critic], ['entropy', COLORS.entropy]]) {
      const values = rows.map(row => row[key]);
      const y = scale([Math.min(...values), Math.max(...values)], [box.bottom, box.top]);
      const shown = rows.filter(row => row.episodes <= training.cursor);
      el('path', {d: pathOf(rows.map(row => [row.episodes, row[key]]), x, y), fill: 'none', stroke: color, 'stroke-opacity': 0.2, 'stroke-width': 1}, svg);
      if (shown.length) el('path', {d: pathOf(shown.map(row => [row.episodes, row[key]]), x, y), fill: 'none', stroke: color, 'stroke-width': 2}, svg);
    }
    el('line', {x1: box.left, x2: box.right, y1: box.bottom, y2: box.bottom, class: 'chart-grid'}, svg);
    el('text', {x: box.left, y: box.bottom + 20, class: 'chart-axis'}, svg, `game ${rows[0].episodes}`);
    el('text', {x: box.right, y: box.bottom + 20, 'text-anchor': 'end', class: 'chart-axis'}, svg, `game ${rows.at(-1).episodes}`);
    el('line', {x1: x(Math.min(training.cursor, rows.at(-1).episodes)), x2: x(Math.min(training.cursor, rows.at(-1).episodes)), y1: box.top, y2: box.bottom, stroke: COLORS.muted, 'stroke-opacity': 0.6}, svg);
  }

  function drawWarm(section) {
    const svg = $('warm-chart');
    clear(svg);
    const warm = section?.best?.warmStart;
    if (!warm?.lossCurve?.length) { $('warm-readout').textContent = 'No warm start was recorded for this run.'; return; }
    const box = {left: 40, right: 446, top: 16, bottom: 168};
    const values = warm.lossCurve.map(point => point[1]);
    const x = scale([warm.lossCurve[0][0], warm.lossCurve.at(-1)[0]], [box.left, box.right]);
    const y = scale([0, Math.max(...values)], [box.bottom, box.top]);
    el('path', {d: pathOf(warm.lossCurve, x, y), fill: 'none', stroke: COLORS.ref, 'stroke-width': 2}, svg);
    el('line', {x1: box.left, x2: box.right, y1: box.bottom, y2: box.bottom, class: 'chart-grid'}, svg);
    el('text', {x: box.left, y: box.bottom + 20, class: 'chart-axis'}, svg, 'epoch 1');
    el('text', {x: box.right, y: box.bottom + 20, 'text-anchor': 'end', class: 'chart-axis'}, svg, `epoch ${warm.lossCurve.at(-1)[0]}`);
    el('text', {x: box.left, y: box.top + 4, class: 'chart-axis'}, svg, 'imitation loss');
    $('warm-readout').textContent = `Before PPO, the network copied the ${warm.policy} tactic from ${warm.demonstrations} demonstration games: it picked the same move as the tactic ${fmt((warm.accuracyBefore ?? 0) * 100, 1)}% of the time at first and ${fmt((warm.accuracyAfter ?? 0) * 100, 1)}% after ${warm.epochs} epochs. PPO then improved on it.`;
  }

  function selectCampaign(data, name) {
    training.campaign = name;
    for (const [id, key] of [['campaign-single', 'singleGoal'], ['campaign-multi', 'multiTask']]) {
      $(id).setAttribute('aria-pressed', String(key === name));
      $(id).classList.toggle('secondary', key !== name);
    }
    const section = campaignData(data);
    training.max = section ? Math.max(...section.runs.map(run => run.episodes || 1)) : 1;
    training.cursor = training.max;
    $('training-cursor').max = String(training.max);
    $('training-cursor').value = String(training.cursor);
    const configuration = section?.configuration;
    $('training-explainer').textContent = !section ? '' : section.metric === 'taskReturn'
      ? `${section.runs.length} runs trained in parallel, ${configuration.parallelEnvironments ?? 1} rules engines each, one random typed task per game. Each run first imitated a per-task greedy tactic, then trained with PPO for up to ${configuration.episodes.toLocaleString('en-US')} games.`
      : `${section.runs.length} runs trained in parallel on one AMD MI300X for 2 hours each. Each first imitated the loading-zone tactic for ${configuration.warmStartDemonstrations} games, then trained with PPO, updating the network every 8 games.`;
    drawTraining(data);
    drawWarm(section);
  }

  function playTraining(data) {
    if (training.playing) { stopTraining(); return; }
    if (training.cursor >= training.max) training.cursor = 1;
    training.playing = true;
    $('training-play').textContent = 'Pause';
    const started = performance.now() - training.cursor / training.max * 14000;
    const step = now => {
      training.cursor = Math.min(training.max, Math.max(1, Math.round((now - started) / 14000 * training.max)));
      $('training-cursor').value = String(training.cursor);
      drawTraining(data);
      if (training.cursor < training.max && training.playing) training.frame = requestAnimationFrame(step);
      else stopTraining();
    };
    training.frame = requestAnimationFrame(step);
  }
  function stopTraining() {
    training.playing = false;
    if (training.frame) cancelAnimationFrame(training.frame);
    $('training-play').textContent = 'Play training';
  }

  // ---------- best match ----------
  const match = {index: 0, timer: null, replay: null, replays: {}};

  function humanMove(id) {
    const [kind, robot, target, goal] = id.split(':');
    const what = {'score-pin': 'Pin from the field', 'score-cup': 'Cup from the field', 'load-score-pin': 'Pin match load',
      'load-score-cup': 'Cup match load', toggle: 'Toggle', park: 'Park'}[kind] || kind;
    if (kind === 'park') return `${robot}: park in midfield`;
    if (kind === 'toggle') return `${robot}: flip ${target}`;
    return `${robot}: ${what}${kind.startsWith('load') ? ` via ${target}` : ` ${target}`} → ${goal}`;
  }

  function drawMatch(replay) {
    const svg = $('match-chart');
    clear(svg);
    const ai = [[replay.decisions[0].clock, 0]];
    for (const row of replay.decisions) ai.push([row.clock + row.seconds, row.scoreAfter.margin]);
    const start = replay.decisions[0].clock;
    const base = [[start, 0]];
    const timeline = replay.baseline.timeline;
    // A typed task can end long before the match does, so its chart ends shortly after both sides finished.
    const baselineEnd = replay.mode === 'task' ? start + (replay.baseline.result?.secondsUsed ?? 0) : 120;
    timeline.forEach((row, index) => base.push([timeline[index + 1]?.clock ?? baselineEnd, row.scoreAfter.margin]));
    const end = replay.mode === 'task' ? Math.max(ai.at(-1)[0], baselineEnd) + 1 : 120;
    const startMargin = 0;
    const offsetAi = ai.map(point => point);
    const values = [...offsetAi, ...base].map(point => point[1]);
    const box = {left: 64, right: 940, top: 24, bottom: 250};
    const {x, y} = axes(svg, box, [start, end], [Math.min(startMargin, ...values) - 10, Math.max(...values) + 20],
      {xLabel: 'match clock · seconds', yLabel: 'score margin'});
    const steps = points => points.map(([px, py], index) => index ? `L${x(px).toFixed(1)} ${y(points[index - 1][1]).toFixed(1)} L${x(px).toFixed(1)} ${y(py).toFixed(1)}` : `M${x(px).toFixed(1)} ${y(py).toFixed(1)}`).join(' ');
    el('path', {d: steps(base), fill: 'none', stroke: COLORS.baseline, 'stroke-width': 2, 'stroke-dasharray': '5 4'}, svg);
    el('path', {d: steps(ai), fill: 'none', stroke: COLORS.ai, 'stroke-opacity': 0.25, 'stroke-width': 2}, svg);
    el('path', {d: steps(ai.slice(0, match.index + 2)), fill: 'none', stroke: COLORS.ai, 'stroke-width': 2.6}, svg);
    const current = replay.decisions[match.index];
    el('circle', {cx: x(current.clock + current.seconds), cy: y(current.scoreAfter.margin), r: 5, fill: COLORS.ai}, svg);
  }

  function readMatch(replay) {
    const row = replay.decisions[match.index];
    const readout = $('match-readout');
    clear(readout);
    const task = replay.mode === 'task' ? ` · ${row.phase === 'park' ? 'parking' : 'scoring'} phase · task reward ${signed(row.taskReward, 2)}` : '';
    html('p', {}, readout, `Decision ${row.decision} of ${replay.decisions.length} · clock ${fmt(row.clock)} s · ${humanMove(row.id)} · ${signed(row.points, 0)} points in ${fmt(row.seconds, 2)} s · margin now ${signed(row.scoreAfter.margin, 0)}${task}`);
    html('p', {class: 'hint'}, readout, `The network scored ${row.candidateCount} ${replay.mode === 'task' ? 'allowed' : 'legal'} candidate moves and played the highest (score ${fmt(row.score, 2)}).`);
    if (replay.mode === 'task' && match.index === replay.decisions.length - 1) {
      const ai = replay.result;
      const greedy = replay.baseline.result;
      html('p', {class: 'hint'}, readout, `Task ${ai.achieved ? 'completed' : 'not completed'} in ${fmt(ai.secondsUsed)} s, task return ${fmt(ai.taskReturn, 2)}; task-greedy ${greedy.achieved ? 'completed it' : 'did not complete it'} in ${fmt(greedy.secondsUsed)} s, task return ${fmt(greedy.taskReturn, 2)}.`);
    }
  }

  function setDecision(replay, index, animate = true) {
    match.index = Math.max(0, Math.min(replay.decisions.length - 1, index));
    $('match-cursor').value = String(match.index + 1);
    drawMatch(replay);
    readMatch(replay);
    drawNetwork(replay, animate);
  }

  function playMatch(replay) {
    if (match.timer) { clearInterval(match.timer); match.timer = null; $('match-play').textContent = 'Play match'; return; }
    if (match.index >= replay.decisions.length - 1) setDecision(replay, 0);
    $('match-play').textContent = 'Pause';
    match.timer = setInterval(() => {
      if (match.index >= replay.decisions.length - 1) { clearInterval(match.timer); match.timer = null; $('match-play').textContent = 'Play match'; return; }
      setDecision(replay, match.index + 1);
    }, 1600);
  }

  // ---------- neuron layers ----------
  let networkTimers = [];

  function cellColor(value, extent) {
    const t = Math.max(-1, Math.min(1, value / (extent || 1)));
    const strength = Math.abs(t);
    const [r, g, b] = t >= 0 ? [255, 193, 122] : [147, 185, 255];
    return `rgba(${r},${g},${b},${(0.08 + 0.92 * strength).toFixed(3)})`;
  }

  function drawNetwork(replay, animate) {
    for (const timer of networkTimers) clearTimeout(timer);
    networkTimers = [];
    const svg = $('network-chart');
    clear(svg);
    const row = replay.decisions[match.index];
    const top = 36;
    const cellHeight = 6;
    const names = replay.featureNames;
    const inputHeight = Math.max(11, row.layers[0].cells.length * cellHeight / row.inputs.length);
    const contentBottom = top + Math.max(row.layers[0].cells.length * cellHeight + 24, row.inputs.length * inputHeight);
    svg.setAttribute('viewBox', `0 0 960 ${Math.ceil(contentBottom + 20)}`);
    el('text', {x: 8, y: 18, class: 'chart-axis'}, svg, 'Inputs for this move');
    row.inputs.forEach((value, index) => {
      const yTop = top + index * inputHeight;
      el('text', {x: 138, y: yTop + inputHeight / 2 + 4, 'text-anchor': 'end', class: 'chart-axis training-input-name'}, svg, names[index] ?? `input ${index + 1}`);
      el('rect', {x: 146, y: yTop + 1, width: 72, height: Math.max(2, inputHeight - 2), fill: COLORS.other, 'fill-opacity': 0.35}, svg);
      const width = Math.min(1, Math.abs(value)) * 36;
      el('rect', {x: value >= 0 ? 182 : 182 - width, y: yTop + 1, width, height: Math.max(2, inputHeight - 2), fill: value >= 0 ? COLORS.ref : COLORS.entropy}, svg);
    });
    const columnWidth = 40;
    const gap = 9;
    const firstColumn = 240;
    row.layers.forEach((layer, layerIndex) => {
      const group = el('g', {class: 'training-layer', opacity: animate ? 0.12 : 1}, svg);
      const xLeft = firstColumn + layerIndex * (columnWidth + gap);
      el('text', {x: xLeft + columnWidth / 2, y: 18, 'text-anchor': 'middle', class: 'chart-axis'}, group, `L${layerIndex + 1}`);
      const deviations = layer.cells.map(value => value - layer.mean);
      const extent = Math.max(...deviations.map(Math.abs));
      deviations.forEach((value, cell) => {
        const rect = el('rect', {x: xLeft, y: top + cell * cellHeight, width: columnWidth, height: cellHeight - 1, fill: cellColor(value, extent)}, group);
        el('title', {}, rect, `Layer ${layerIndex + 1}, neurons ${cell * 16}–${cell * 16 + 15}: average ${fmt(layer.cells[cell], 3)} (layer average ${fmt(layer.mean, 3)})`);
      });
      el('text', {x: xLeft + columnWidth / 2, y: top + layer.cells.length * cellHeight + 16, 'text-anchor': 'middle', class: 'chart-axis'}, group, `σ ${fmt(layer.spread, 2)}`);
      if (animate) networkTimers.push(setTimeout(() => group.setAttribute('opacity', '1'), 90 * (layerIndex + 1)));
    });
    const last = row.layers.at(-1);
    const pushLeft = firstColumn + row.layers.length * (columnWidth + gap) - gap + 6;
    const pushGroup = el('g', {opacity: animate ? 0.12 : 1}, svg);
    el('text', {x: pushLeft + 6, y: 18, 'text-anchor': 'middle', class: 'chart-axis'}, pushGroup, 'push');
    const pushExtent = Math.max(...last.contribution.map(Math.abs));
    last.contribution.forEach((value, cell) => {
      const rect = el('rect', {x: pushLeft, y: top + cell * cellHeight, width: 12, height: cellHeight - 1, fill: cellColor(value, pushExtent)}, pushGroup);
      el('title', {}, rect, `Neurons ${cell * 16}–${cell * 16 + 15} push the move score by ${signed(value, 3)}`);
    });
    if (animate) networkTimers.push(setTimeout(() => pushGroup.setAttribute('opacity', '1'), 90 * (row.layers.length + 1)));
    const outLeft = pushLeft + 30;
    const outGroup = el('g', {opacity: animate ? 0.12 : 1}, svg);
    el('text', {x: outLeft, y: 18, class: 'chart-axis'}, outGroup, 'Move scores');
    const scores = row.candidates.map(candidate => candidate.score);
    const lowest = Math.min(...scores);
    const span = (Math.max(...scores) - lowest) || 1;
    row.candidates.forEach((candidate, index) => {
      const yTop = top + index * 46;
      const chosen = index === 0;
      el('rect', {x: outLeft, y: yTop, width: 18 + 92 * (candidate.score - lowest) / span, height: 16, fill: chosen ? COLORS.chosen : COLORS.run, 'fill-opacity': chosen ? 1 : 0.45}, outGroup);
      el('text', {x: outLeft, y: yTop + 30, class: 'chart-axis'}, outGroup, `${chosen ? '▶ ' : ''}${fmt(candidate.score, 2)} ${candidate.kind}`);
      el('title', {}, outGroup.lastChild, candidate.id);
    });
    if (animate) networkTimers.push(setTimeout(() => outGroup.setAttribute('opacity', '1'), 90 * (row.layers.length + 2)));
    const readout = $('network-readout');
    clear(readout);
    const strongest = row.layers.reduce((best, layer, index) => (layer.spread > best.spread ? {spread: layer.spread, index} : best), {spread: -1, index: 0});
    html('p', {class: 'hint'}, readout, `Chosen move: ${humanMove(row.id)}. Layer spread grows from σ ${fmt(row.layers[0].spread, 2)} to σ ${fmt(last.spread, 2)} as the residual blocks add up; layer ${strongest.index + 1} varies most. Most active neuron in the last layer: #${last.peakNeuron} (${fmt(last.peakValue, 2)}).`);
  }

  // ---------- start ----------
  async function load(name) {
    try {
      const response = await fetch(name, {cache: 'no-store'});
      return response.ok ? await response.json() : null;
    } catch { return null; }
  }

  async function start() {
    const [data, replay, taskReplay] = await Promise.all([load('training-data.json'), load('training-replay.json'),
      load('training-replay-task.json')]);
    $('data-status').textContent = data ? 'Saved training data loaded' : 'training-data.json is missing';
    $('data-source').textContent = data
      ? `Single goal: ${data.singleGoal ? data.singleGoal.campaign : 'missing'} · Multi-task: ${data.multiTask ? data.multiTask.campaign : 'not copied yet'}`
      : 'Run: py -3 tools/build_training_dashboard.py';
    renderCards(data, replay);
    if (data) {
      selectCampaign(data, data.singleGoal ? 'singleGoal' : 'multiTask');
      $('campaign-single').addEventListener('click', () => { stopTraining(); selectCampaign(data, 'singleGoal'); });
      $('campaign-multi').addEventListener('click', () => { stopTraining(); selectCampaign(data, 'multiTask'); });
      $('training-play').addEventListener('click', () => playTraining(data));
      $('training-end').addEventListener('click', () => { stopTraining(); training.cursor = training.max; $('training-cursor').value = String(training.max); drawTraining(data); });
      $('training-cursor').addEventListener('input', event => { stopTraining(); training.cursor = Number(event.target.value); drawTraining(data); });
    }
    match.replays = {match: replay, task: taskReplay};
    if (replay || taskReplay) {
      const choose = key => {
        if (!match.replays[key]) return;
        if (match.timer) { clearInterval(match.timer); match.timer = null; $('match-play').textContent = 'Play match'; }
        match.replay = match.replays[key];
        for (const [id, name] of [['replay-match', 'match'], ['replay-task', 'task']]) {
          $(id).setAttribute('aria-pressed', String(name === key));
          $(id).classList.toggle('secondary', name !== key);
        }
        const current = match.replay;
        $('match-badge').textContent = `${current.alliance.toUpperCase()} · START POSE ${current.seed}`;
        $('match-baseline-key').textContent = current.mode === 'task' ? 'Task-greedy tactic, same task and start pose' : 'Loader-cycle team tactic, same start pose';
        $('match-explainer').textContent = current.mode === 'task'
          ? `The multi-task model (${current.policy.width} × ${current.policy.hiddenLayers}, task features) carries out the typed task "${current.task.text}" (${current.task.summary}). Only moves the task allows are offered; it plays the highest-scoring one.`
          : `The single-goal model (${current.policy.width} × ${current.policy.hiddenLayers}) plays a full driver period for the best score margin.`;
        $('match-cursor').max = String(current.decisions.length);
        setDecision(current, 0, false);
      };
      $('replay-match').disabled = !replay;
      $('replay-task').disabled = !taskReplay;
      $('replay-match').addEventListener('click', () => choose('match'));
      $('replay-task').addEventListener('click', () => choose('task'));
      $('match-play').addEventListener('click', () => playMatch(match.replay));
      $('match-prev').addEventListener('click', () => setDecision(match.replay, match.index - 1));
      $('match-next').addEventListener('click', () => setDecision(match.replay, match.index + 1));
      $('match-cursor').addEventListener('input', event => setDecision(match.replay, Number(event.target.value) - 1));
      $('network-replay').addEventListener('click', () => drawNetwork(match.replay, true));
      choose(replay ? 'match' : 'task');
    } else {
      $('match-readout').textContent = 'training-replay.json is missing. Run: node tools/build-training-replay.js <policy.json> <seed> <red|blue>';
    }
  }

  document.addEventListener('DOMContentLoaded', start);
})();
