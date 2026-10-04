'use strict';

(() => {
  const element = id => document.getElementById(id);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const format = (value, digits = 3) => finite(value) ? value.toLocaleString('en', {maximumFractionDigits: digits}) : 'Not recorded';
  const json = value => JSON.stringify(value ?? null, null, 2);
  const local = ['http:', 'https:'].includes(location.protocol) && ['127.0.0.1', 'localhost'].includes(location.hostname);
  const colors = ['#93b9ff', '#6be4cc', '#ffc17a', '#dab0ff'];
  const state = {report: null, entries: [], groups: [], selected: null, busy: false, replay: new Map()};
  const player = RoboProofPlayer.create(element('simulator'));
  const text = (id, value) => { element(id).textContent = value; };
  const node = (tag, value, className) => {
    const result = document.createElement(tag);
    if (value !== undefined) result.textContent = value;
    if (className) result.className = className;
    return result;
  };
  const svgNode = (tag, attrs, value) => {
    const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, item] of Object.entries(attrs)) result.setAttribute(key, String(item));
    if (value !== undefined) result.textContent = value;
    return result;
  };

  function status(message, kind = '') {
    text('status', message);
    element('status').className = `status full-width ${kind}`;
  }

  function controls(busy) {
    state.busy = busy;
    for (const id of ['run', 'load-report']) element(id).disabled = busy || !local;
    element('replay').disabled = busy || !local || !state.selected;
    element('watch').disabled = busy || !state.selected || (!local && !telemetryRows().length);
    element('download').disabled = busy || !state.selected;
    element('report').disabled = busy;
    element('scenario').disabled = busy;
    element('count').disabled = busy || !local;
    element('seed').disabled = busy || !local;
  }

  function table(id, headers, rows, caption) {
    const host = element(id);
    host.replaceChildren();
    if (!rows.length) { host.append(node('p', 'No data recorded.', 'hint')); return; }
    const result = node('table');
    if (caption) result.append(node('caption', caption));
    const head = node('thead'), heading = node('tr');
    headers.forEach(label => { const cell = node('th', label); cell.scope = 'col'; heading.append(cell); });
    head.append(heading);
    const body = node('tbody');
    for (const row of rows) {
      const line = node('tr');
      for (const value of row) {
        const cell = node('td');
        if (value instanceof Node) cell.append(value);
        else cell.textContent = value == null ? 'Not recorded' : String(value);
        line.append(cell);
      }
      body.append(line);
    }
    result.append(head, body);
    host.append(result);
  }

  function chart(id, series, {equal = false, xlabel = 'Time / s', ylabel = '', points = false} = {}) {
    const host = element(id);
    host.replaceChildren();
    const values = series.flatMap(line => line.values).filter(point => finite(point[0]) && finite(point[1]));
    if (!values.length) { host.append(node('p', 'No finite measurements recorded.', 'chart-empty')); return; }
    const width = 640, height = 340, margin = {left: 65, right: 20, top: 25, bottom: 50};
    const plotWidth = width - margin.left - margin.right, plotHeight = height - margin.top - margin.bottom;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const point of values) { xmin = Math.min(xmin, point[0]); xmax = Math.max(xmax, point[0]); ymin = Math.min(ymin, point[1]); ymax = Math.max(ymax, point[1]); }
    const xPadding = Math.max(0.02, (xmax - xmin) * 0.06), yPadding = Math.max(0.02, (ymax - ymin) * 0.08);
    xmin -= xPadding; xmax += xPadding; ymin -= yPadding; ymax += yPadding;
    if (equal) {
      const scale = Math.max((xmax - xmin) / plotWidth, (ymax - ymin) / plotHeight);
      const middleX = (xmin + xmax) / 2, middleY = (ymin + ymax) / 2;
      xmin = middleX - scale * plotWidth / 2; xmax = middleX + scale * plotWidth / 2;
      ymin = middleY - scale * plotHeight / 2; ymax = middleY + scale * plotHeight / 2;
    }
    const screenX = value => margin.left + (value - xmin) / (xmax - xmin) * plotWidth;
    const screenY = value => margin.top + plotHeight - (value - ymin) / (ymax - ymin) * plotHeight;
    const svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `${ylabel} against ${xlabel}. Recorded values are also available in the table.`});
    for (let index = 0; index <= 4; index++) {
      const xvalue = xmin + (xmax - xmin) * index / 4, yvalue = ymin + (ymax - ymin) * index / 4;
      svg.append(svgNode('line', {x1: screenX(xvalue), x2: screenX(xvalue), y1: margin.top, y2: margin.top + plotHeight, class: 'chart-grid'}));
      svg.append(svgNode('line', {x1: margin.left, x2: width - margin.right, y1: screenY(yvalue), y2: screenY(yvalue), class: 'chart-grid'}));
      svg.append(svgNode('text', {x: screenX(xvalue), y: height - 28, 'text-anchor': 'middle', class: 'chart-axis'}, format(xvalue, 2)));
      svg.append(svgNode('text', {x: margin.left - 9, y: screenY(yvalue) + 4, 'text-anchor': 'end', class: 'chart-axis'}, format(yvalue, 2)));
    }
    svg.append(svgNode('text', {x: width / 2, y: height - 5, 'text-anchor': 'middle', class: 'chart-axis'}, xlabel));
    svg.append(svgNode('text', {x: margin.left, y: 14, class: 'chart-axis'}, ylabel));
    series.forEach((line, lineIndex) => {
      const valid = line.values.filter(point => finite(point[0]) && finite(point[1]));
      const color = line.color || colors[lineIndex % colors.length];
      if (!points) {
        const stride = Math.max(1, Math.ceil(valid.length / 800));
        const samples = valid.filter((_, index) => index % stride === 0 || index === valid.length - 1);
        const polyline = svgNode('polyline', {points: samples.map(point => `${screenX(point[0])},${screenY(point[1])}`).join(' '), stroke: color, class: 'plot-line', 'stroke-dasharray': line.dash || ''});
        polyline.append(svgNode('title', {}, line.name));
        svg.append(polyline);
      } else {
        for (const point of valid) {
          const marker = point[2].result.passed === true ? svgNode('circle', {cx: screenX(point[0]), cy: screenY(point[1]), r: 4}) : point[2].result.passed === false ? svgNode('path', {d: `M${screenX(point[0]) - 4},${screenY(point[1]) - 4}l8,8m-8,0l8,-8`}) : svgNode('rect', {x: screenX(point[0]) - 3, y: screenY(point[1]) - 3, width: 6, height: 6});
          marker.setAttribute('fill', 'none');
          marker.setAttribute('stroke', point[2].result.passed === true ? '#6be4cc' : '#ff9b9e');
          marker.setAttribute('stroke-width', '1.7');
          marker.setAttribute('class', 'parameter-point');
          marker.append(svgNode('title', {}, `${point[2].result.scenario.scenario_id}: friction ${format(point[0])}, battery ${format(point[1])} V`));
          marker.addEventListener('click', () => { select(point[2].key); element('explorer').scrollIntoView({behavior: 'smooth'}); });
          svg.append(marker);
        }
      }
    });
    host.append(svg);
  }

  function groups(report) {
    const candidates = [['baseline', report.kind === 'ml-confirmed-experiments' ? 'ML-confirmed experiment' : 'Baseline', report.results], ['search', 'Adversarial search', report.search?.results], ['holdout', 'Holdout baseline', report.verification?.holdout_before], ['after', 'Candidate training', report.verification?.after], ['holdout-after', 'Candidate holdout', report.verification?.holdout_after], ['search-after', 'Candidate adversarial', report.verification?.adversarial_after]];
    return candidates.filter(([, , rows]) => Array.isArray(rows) && rows.length);
  }

  function comparison() {
    const report = state.report, verification = report.verification;
    const accepted = verification?.accepted === true, rejected = verification?.accepted === false;
    text('verdict', accepted ? 'MODEL GATES PASS' : rejected ? 'REJECTED' : 'NOT VERIFIED');
    element('verdict').className = `verdict ${accepted ? 'accepted' : rejected ? 'rejected' : ''}`;
    text('acceptance-rule', verification?.acceptance_rule || 'Positive net improvement on training and unseen holdout; zero new regressions, including adversarial cases. No firmware patch is applied.');
    const comparisons = ['training', 'holdout', 'adversarial'].filter(name => verification?.[name]);
    table('comparison-table', ['Cohort', 'Before pass', 'After pass', 'Fixed', 'Regressions', 'Net change'], comparisons.map(name => {
      const item = verification[name];
      return [name, `${item.before?.passed ?? '?'} / ${item.before?.count ?? '?'}`, `${item.after?.passed ?? '?'} / ${item.after?.count ?? '?'}`, item.failures_fixed, node('strong', format(item.new_regressions, 0), item.new_regressions ? 'fail' : 'pass'), item.net_improvement];
    }));
    text('candidate', verification ? json(verification.candidate) : 'No candidate evaluated in this report.');
    text('proposal-diff', report.proposal?.diff || '');
    text('hypothesis', report.proposal?.hypothesis || 'No hypothesis recorded.');
    text('proposal-risk', report.proposal?.risk || '');
    text('required-tests', report.proposal?.required_tests ? `Required tests: ${report.proposal.required_tests.join(', ')}` : '');
    table('changes-table', ['Cohort', 'Scenario', 'Change'], comparisons.flatMap(name => (verification[name].changes || []).map(change => [name, change.id, node('strong', change.change, change.change === 'REGRESSION' ? 'fail' : 'pass')])));
  }

  function evidence() {
    const report = state.report, diagnosis = report.diagnosis;
    const unsupportedHistoricalClaim = diagnosis?.mode === 'local-expert-ai-agent';
    text('diagnosis-mode', diagnosis?.mode || 'No diagnosis recorded');
    text('diagnosis-scope', diagnosis ? `Report counterexample: ${report.worst_scenario_id}. LLM: ${diagnosis.llm_status || 'See recorded provider metadata'}. This diagnosis does not change when you select another scenario.` : 'Run the CLI demo or diagnosis tool to record an investigation. A new scalar batch alone has no diagnosis.');
    if (unsupportedHistoricalClaim) text('diagnosis-scope', 'Historical deterministic report with unsupported neural/causal labels. Its text is retained as unverified history, not evidence of PyTorch, LLM inference or proven causation. Rerun diagnosis with the current investigator.');
    element('observations').replaceChildren(node('pre', diagnosis ? json(diagnosis.observations) : 'No observations recorded.'));
    text('inference', diagnosis?.inference || 'No inference recorded.');
    text('confidence', unsupportedHistoricalClaim ? 'UNCALIBRATED; historical HIGH confidence was not supported.' : diagnosis ? `Confidence: ${diagnosis.confidence ?? 'UNCALIBRATED'}` : '');
    table('sensitivity-table', ['Parameter', 'Test value', 'Failure score', 'Pass'], (diagnosis?.sensitivities || []).flatMap(sensitivity => (sensitivity.results || []).map(result => [sensitivity.parameter, format(result.value), format(result.failure_score), result.passed === true ? 'PASS' : 'FAIL'])));
    text('tool-log', diagnosis ? json(diagnosis.tool_log) : 'No tool transcript recorded.');
    text('search-algorithm', report.search?.algorithm || 'No adversarial search recorded.');
    table('search-table', ['Generation', 'Worst failure score', 'Failures'], (report.search?.history || []).map(row => [row.generation, format(row.worst_score), row.failures ?? 'Not recorded']));
    table('category-table', ['Category', 'Count'], Object.entries(report.summary?.categories || {}));
    const metadata = {};
    for (const key of ['schema_version', 'software_version', 'simulation_version', 'controller_wasm_sha256', 'runtime', 'platform', 'source_hashes', 'distribution', 'reproducibility', 'benchmark', 'model_sha256', 'selection_sha256', 'package_sha256', 'canonical_ready']) metadata[key] = report[key];
    text('provenance', json(metadata));
  }

  function load(report, source) {
    if (!report || typeof report !== 'object' || !Array.isArray(report.results) || !report.results.length || !report.summary) throw Error('Expected a RoboProof report containing results and summary, not a standalone scenario.');
    const found = groups(report);
    for (const [, , rows] of found) for (const result of rows) if (!result?.scenario?.scenario_id || !result.scenario.environment || !result.scenario.task || !result.metrics) throw Error('Result is missing scenario or metrics.');
    state.report = report;
    state.groups = found;
    state.replay.clear();
    state.entries = found.flatMap(([id, label, rows]) => rows.map((result, index) => ({key: `${id}:${index}`, group: id, label, result})));
    if (report.counterexample?.telemetry) {
      const matching = state.entries.find(entry => entry.result.scenario_sha256 === report.counterexample.scenario_sha256 && entry.result.scenario.scenario_id === report.counterexample.scenario.scenario_id);
      if (matching) state.replay.set(matching.key, report.counterexample);
      else state.entries.push({key: 'counterexample', group: 'counterexample', label: 'Counterexample', result: report.counterexample});
    }
    element('scenario').replaceChildren(...state.entries.map(entry => {
      const option = node('option', `${entry.label} · ${entry.result.scenario.scenario_id} · ${entry.result.passed ? 'PASS' : 'FAIL'}`);
      option.value = entry.key;
      return option;
    }));
    element('map-group').replaceChildren(...found.map(([id, label]) => { const option = node('option', label); option.value = id; return option; }));
    text('source', source);
    const learned = report.kind === 'ml-confirmed-experiments';
    text('pass-rate-label', learned ? 'Observed selected-cohort pass rate' : 'Observed baseline pass rate');
    text('failure-count-label', learned ? 'Selected-cohort failures' : 'Baseline failures');
    text('pass-rate', `${format(report.summary.robustness_percent, 2)}%`);
    text('pass-count', `${format(report.summary.passed, 0)} / ${format(report.summary.count, 0)} scenarios`);
    text('failure-count', format(report.summary.failed, 0));
    text('nominal-status', `Nominal gate: ${report.nominal?.passed === true ? 'PASS' : report.nominal?.passed === false ? 'FAIL' : 'Not recorded'}`);
    text('throughput', finite(report.benchmark?.simulations_per_second) ? `${format(report.benchmark.simulations_per_second, 2)} / s` : 'NOT MEASURED');
    text('benchmark-time', finite(report.benchmark?.seconds) ? `${format(report.benchmark.seconds)} seconds, recorded batch` : 'No wall-clock measurement');
    text('backend', report.backend || 'Not recorded');
    text('gpu', `GPU: ${report.benchmark?.gpu || 'NOT MEASURED'}`);
    text('interpretation', `${report.summary.interpretation || 'Pass fraction within the recorded scenario set only.'} Distribution: ${report.distribution || 'not recorded'}.`);
    element('empty-state').hidden = true;
    element('report-content').hidden = false;
    comparison();
    evidence();
    select((state.entries.find(entry => entry.result.scenario.scenario_id === report.worst_scenario_id) || state.entries[0]).key);
    renderMap();
    status(`Loaded ${format(state.entries.length, 0)} results from ${source}.`);
  }

  function learnedEvidence() {
    const report = state.report;
    const learned = report.kind === 'ml-confirmed-experiments';
    element('ml-evidence').hidden = !learned;
    if (!learned) return;
    const training = report.model_training;
    text('ml-training', training ? `Reported gradient training: ${training.actual_gradient_training === true ? 'performed' : 'not confirmed'} · device: ${training.runtime?.device || 'not recorded'} · epochs: ${training.epochs_completed ?? 'not recorded'} · model SHA-256: ${report.model_sha256 || 'not recorded'}` : 'Training metadata not recorded.');
    const prediction = report.predictions?.find(row => row.scenario_id === state.selected.result.scenario.scenario_id);
    text('ml-selection', `${state.selected.result.scenario.scenario_id} · selection: ${prediction?.reason || 'not recorded'} · entropy heuristic: ${format(prediction?.predictive_entropy)}`);
    table('ml-prediction-table', ['Category', 'Predicted probability (not measured)', 'Observed in simulation'], Object.entries(prediction?.category_probabilities || {}).map(([name, probability]) => [name, finite(probability) && probability >= 0 && probability <= 1 ? `${format(probability * 100, 1)}%` : 'Not recorded', state.selected.result.categories?.includes(name) ? 'YES' : 'NO']));
  }

  function telemetryRows() {
    const selected = state.selected && (state.replay.get(state.selected.key) || state.selected.result);
    return Array.isArray(selected?.telemetry) ? selected.telemetry : [];
  }

  function select(key) {
    player.clear();
    state.selected = state.entries.find(entry => entry.key === key);
    if (!state.selected) return;
    element('scenario').value = key;
    const result = state.replay.get(key) || state.selected.result, scenario = result.scenario;
    text('selected-outcome', result.passed === true ? 'PASS' : result.passed === false ? 'FAIL' : 'UNKNOWN');
    element('selected-outcome').className = `badge ${result.passed ? 'pass' : 'fail'}`;
    text('selected-meta', `${state.selected.label} · ${scenario.scenario_id} · seed ${scenario.random_seed} · localization: ${scenario.task.localization} · ${scenario.task.localization === 'encoders' ? 'IMU inactive / not fitted to the current configuration' : 'IMU or explicitly configured truth baseline'}`);
    learnedEvidence();
    text('selected-categories', (result.categories || []).join(' · '));
    const metricValues = [
      ['Endpoint error', result.metrics.endpoint_error, 'm'], ['Max path deviation', result.metrics.max_path_deviation, 'm'],
      ['Heading error', result.metrics.final_heading_error, 'rad'], ['Max localization error', result.metrics.max_localization_error, 'm'],
      ['Battery', scenario.environment.battery_voltage, 'V'], ['Friction', scenario.environment.friction, ''],
      ['Encoder latency', scenario.environment.encoder_latency * 1000, 'ms'], ['Saturation fraction', result.metrics.saturation_fraction, ''],
      ['Completion time', result.metrics.completion_time, 's'], ['Failure score', result.failure_score, ''],
      ['First path/localization divergence', result.metrics.first_divergence, 's'], ['Motor efficiencies L / R', `${format(scenario.environment.left_motor_efficiency)} / ${format(scenario.environment.right_motor_efficiency)}`, '']
    ];
    element('selected-metrics').replaceChildren(...metricValues.map(([label, value, unit]) => {
      const card = node('div', undefined, 'metric');
      card.append(node('span', label), node('strong', typeof value === 'string' ? value : value === null ? 'Not reached' : `${format(value)} ${unit}`));
      return card;
    }));
    text('scenario-json', json({...result, telemetry: undefined}));
    const rows = telemetryRows();
    element('telemetry-plots').hidden = rows.length === 0;
    element('telemetry-details').open = false;
    element('telemetry-table').replaceChildren();
    text('telemetry-status', rows.length ? `${format(rows.length, 0)} recorded samples. Polylines use at most ~800 samples per trace; the table retains every sample.` : 'No trajectory was stored for this result. Replay this exact scenario through the local API to obtain real telemetry; no trajectory is inferred from final metrics.');
    if (rows.length) {
      const names = ['reference', 'truth', 'estimate'];
      const dashes = ['8 5', '', '2 4'];
      chart('path-chart', names.map((name, index) => ({name, dash: dashes[index], values: rows.map(row => [row[name]?.[0], row[name]?.[1]])})), {equal: true, xlabel: 'Field X / m', ylabel: 'Field Y / m'});
      chart('heading-chart', names.map((name, index) => ({name, dash: dashes[index], values: rows.map(row => [row.time, row[name]?.[2]])})), {ylabel: 'Heading / rad'});
      const wheelColors = ['#6be4cc', '#93b9ff', '#ffc17a', '#dab0ff'];
      chart('motor-chart', ['command', 'applied'].flatMap(signal => ['FL', 'BL', 'FR', 'BR'].map((name, index) => ({name: `${name} ${signal}`, color: wheelColors[index], dash: signal === 'applied' ? '6 4' : '', values: rows.map(row => [row.time, row[signal]?.[index]])}))), {ylabel: 'Voltage / V'});
      chart('error-chart', ['path_error', 'tracking_error', 'localization_error'].map(name => ({name, values: rows.map(row => [row.time, row[name]])})), {ylabel: 'Position error / m'});
      chart('velocity-chart', [{name: 'Linear speed', color: colors[1], values: rows.map(row => [row.time, row.velocity ? Math.hypot(row.velocity[0], row.velocity[1]) : null])}, {name: 'Yaw rate', color: colors[2], values: rows.map(row => [row.time, row.velocity?.[2]])}], {ylabel: 'm/s, rad/s'});
    }
    controls(state.busy);
  }

  function renderMap() {
    const entries = state.entries.filter(entry => entry.group === element('map-group').value);
    const stride = Math.max(1, Math.ceil(entries.length / 4000));
    const displayed = entries.filter((_, index) => index % stride === 0);
    chart('parameter-chart', [{name: 'Recorded results', values: displayed.map(entry => [entry.result.scenario.environment.friction, entry.result.scenario.environment.battery_voltage, entry])}], {xlabel: 'Friction coefficient', ylabel: 'Battery / V', points: true});
    text('map-status', `${displayed.length} plotted of ${entries.length} results in this cohort. ${stride > 1 ? 'Regular-index subsampling for display only; all results remain selectable.' : 'No interpolation or synthetic surface.'}`);
    renderParameterTable(entries, 0);
  }

  function renderParameterTable(entries, page) {
    const pageSize = 200;
    table('parameter-table', ['Select scenario', 'Battery / V', 'Friction', 'Encoder latency / ms', 'Status'], entries.slice(page * pageSize, (page + 1) * pageSize).map(entry => {
      const button = node('button', entry.result.scenario.scenario_id, 'table-button');
      button.type = 'button';
      button.addEventListener('click', () => select(entry.key));
      const environment = entry.result.scenario.environment;
      return [button, format(environment.battery_voltage), format(environment.friction), format(environment.encoder_latency * 1000), entry.result.passed ? 'PASS' : 'FAIL'];
    }), `Results ${page * pageSize + 1}–${Math.min(entries.length, (page + 1) * pageSize)} of ${entries.length}`);
    if (page > 0) {
      const previous = node('button', 'Previous 200', 'secondary');
      previous.type = 'button'; previous.addEventListener('click', () => renderParameterTable(entries, page - 1)); element('parameter-table').append(previous);
    }
    if ((page + 1) * pageSize < entries.length) {
      const next = node('button', 'Next 200', 'secondary');
      next.type = 'button'; next.addEventListener('click', () => renderParameterTable(entries, page + 1)); element('parameter-table').append(next);
    }
  }

  async function request(endpoint, body) {
    if (!local) throw Error('Live tools require the loopback RoboProof server. Uploaded reports still work offline.');
    const response = await fetch(endpoint, body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {cache: 'no-store'});
    const result = await response.json();
    if (!response.ok) throw Error(result.error || `HTTP ${response.status}`);
    return result;
  }

  async function action(operation) {
    if (state.busy) return;
    controls(true);
    try { await operation(); } catch (error) { status(error.message, 'error'); } finally { controls(false); }
  }

  async function replaySelected() {
    const selected = state.selected;
    if (!selected) throw Error('Select a scenario first.');
    status(`Replaying ${selected.result.scenario.scenario_id}…`, 'busy');
    const result = await request('/api/replay', {scenario: selected.result.scenario});
    state.replay.set(selected.key, result);
    if (state.selected?.key !== selected.key) throw Error('Selection changed during replay. Open the simulator for the selected result again.');
    select(selected.key);
    const matches = json(result.metrics) === json(selected.result.metrics);
    status(matches ? 'Replay metrics exactly match the recorded result.' : 'Replay differs from the recorded result. Compare source/runtime provenance before drawing conclusions.', matches ? '' : 'error');
    return result;
  }

  function showSimulation(result, autoplay = false) {
    player.load(result);
    element('simulator').scrollIntoView({behavior: 'instant', block: 'start'});
    element('sim-play').focus({preventScroll: true});
    if (autoplay) player.play();
  }

  function loadSingleResult(result, source) {
    load({...result, results: [result], summary: {
      count: 1, passed: result.passed ? 1 : 0, failed: result.passed ? 0 : 1,
      robustness_percent: result.passed ? 100 : 0,
      interpretation: 'One imported scenario only, not a population robustness estimate.',
      categories: Object.fromEntries((result.categories || []).map(category => [category, 1]))
    }, distribution: 'Single supplied scenario; no random population sampled.'}, source);
  }

  element('report').addEventListener('change', () => action(async () => {
    const file = element('report').files[0];
    if (!file) return;
    if (file.size > 64 * 1024 * 1024) throw Error('Report exceeds the 64 MiB browser limit; use a smaller batch.');
    const data = JSON.parse((await file.text()).replace(/^\uFEFF/, ''));
    if (data?.scenario_id && data?.task && data?.environment) {
      if (!local) throw Error('A scenario file contains no motion recording. Open the local RoboProof server to simulate it; full reports with telemetry work offline.');
      status(`Simulating imported scenario ${data.scenario_id}…`, 'busy');
      const result = await request('/api/replay', {scenario: data});
      loadSingleResult(result, file.name);
      showSimulation(result);
      status(`${data.scenario_id}: ${result.passed ? 'PASS' : 'FAIL'} in the current model. Press Play in the simulator. The imported file contains no previous result to compare.`);
    } else if (data?.scenario && data?.metrics && Array.isArray(data.telemetry)) {
      loadSingleResult(data, file.name);
      showSimulation(data);
    } else load(data, file.name);
  }));
  element('load-report').addEventListener('click', () => action(async () => { status('Loading recorded server report…', 'busy'); load(await request('/api/report'), 'local server report'); }));
  element('run-form').addEventListener('submit', event => {
    event.preventDefault();
    void action(async () => {
      const count = Number(element('count').value), seed = Number(element('seed').value);
      if (!Number.isInteger(count) || count < 1 || count > 1000 || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw Error('Use 1–1,000 scenarios and an integer seed between 0 and 4294967295.');
      status(`Running ${count} actual C++ controller scenarios. Waiting for measured results…`, 'busy');
      load(await request('/api/run', {count, seed}), `live CPU batch · seed ${seed}`);
    });
  });
  element('scenario').addEventListener('change', () => select(element('scenario').value));
  element('map-group').addEventListener('change', renderMap);
  element('replay').addEventListener('click', () => action(replaySelected));
  element('watch').addEventListener('click', () => action(async () => {
    if (!state.selected) throw Error('Select a scenario first.');
    const result = telemetryRows().length ? (state.replay.get(state.selected.key) || state.selected.result) : await replaySelected();
    showSimulation(result, true);
  }));
  element('download').addEventListener('click', () => {
    if (!state.selected) return;
    const url = URL.createObjectURL(new Blob([json(state.selected.result.scenario) + '\n'], {type: 'application/json'}));
    const anchor = node('a'); anchor.href = url; anchor.download = 'roboproof-scenario.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  element('telemetry-details').addEventListener('toggle', () => {
    if (!element('telemetry-details').open) return;
    table('telemetry-table', ['Time / s', 'True X / m', 'True Y / m', 'True heading / rad', 'Estimated X / m', 'Estimated Y / m', 'Estimated heading / rad', 'Path error / m', 'Heading error / rad', 'Command FL / V', 'Command BL / V', 'Command FR / V', 'Command BR / V'], telemetryRows().map(row => [row.time, ...row.truth, ...row.estimate, row.path_error, row.heading_error, ...row.command].map(value => format(value, 6))));
  });
  text('connection-mode', local ? 'Loopback API + file explorer' : 'Offline file explorer');
  controls(false);
  if (local) void action(async () => {
    try { load(await request('/api/report'), 'local server report'); }
    catch { status('No server report available. Upload report.json or run a local experiment.'); }
  });
})();
