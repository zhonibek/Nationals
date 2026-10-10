'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const dashboardSource = fs.readFileSync(path.join(__dirname, '../roboproof/dashboard/motion.js'), 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));

function ppoDiagnostics(overrides = {}) {
  return {update: 2, meanApproxKL: 0.0123, clipFraction: 0.25, meanEntropy: -1.125,
    meanActorLoss: -0.0625, meanCriticLoss: 0.375, valueExplainedVariance: 0.5,
    sampledActionMean: [0.75, -0.5, 0.25, -0.125], sampledActionStd: [0.125, 0.25, 0.375, 0.5],
    sampledActionBoundaryFraction: 0.375, curriculumStageCounts: {'0': 1, '1': 2, '2': 3},
    ...overrides};
}

function learningRecord(history, overrides = {}) {
  const model = {seed: 42, steps: 64, updates: 2, trained: true, actorWeightsChanged: true,
    policySha256: 'fixture-policy-42'};
  if (history !== undefined) model.history = history;
  return {available: true, status: 'completed', motionPolicyTrained: true,
    learnedImprovementVerified: false, models: [model], evaluation: null, ...overrides};
}

function evaluationModel(passed, policySha256 = 'fixture-policy-42', seed = 42) {
  return {seed, policySha256, baseline: {success: 6, count: 24}, untrained: {success: 7, count: 24},
    learned: {success: passed ? 24 : 7, count: 24}, acceptance: {passed}};
}

async function createDashboard(record) {
  let saved = record;
  const elements = new Map();
  const makeElement = () => ({children: [], listeners: {}, textContent: '', hidden: false,
    disabled: false, value: '',
    addEventListener(name, callback) { this.listeners[name] = callback; },
    replaceChildren(...children) { this.children = children; },
    appendChild(child) { this.children.push(child); return child; }});
  const element = id => {
    if (!elements.has(id)) elements.set(id, makeElement());
    return elements.get(id);
  };
  element('motion-mode').value = 'scripted';
  const requests = [];
  const realm = vm.createContext({document: {getElementById: element, createElement: makeElement},
    location: {protocol: 'http:', hostname: '127.0.0.1'}, AbortController, console,
    fetch: async (url, options = {}) => {
      requests.push({url, options});
      assert.equal(options.method, undefined, 'Diagnostics may only read saved artifacts');
      assert(['/api/motion/status', '/api/motion/learning'].includes(url), `Unexpected request: ${url}`);
      const response = url === '/api/motion/learning' ? saved : {readyForBoundedCpuExperiment: false};
      return {ok: true, json: async () => JSON.parse(JSON.stringify(response))};
    }});
  vm.runInContext(dashboardSource, realm);
  await settle();
  assert.deepEqual(requests.map(request => request.url), ['/api/motion/status']);
  const click = async id => {
    await element(id).listeners.click();
    await settle();
  };
  return {element, requests, click,
    async refresh(next = saved) {
      saved = next;
      await click('motion-learning-refresh');
    },
    row(index = 0) {
      return element('motion-learning-table').children[0].children[index + 1];
    },
    diagnostics(index = 0) {
      const table = element('motion-learning-table').children[0];
      assert(table, 'Available learning records must render a table');
      const column = table.children[0].children.findIndex(cell => /diagnostics/i.test(cell.textContent));
      assert(column >= 0, 'Learning table must identify its diagnostics column');
      return table.children[index + 1].children[column].textContent;
    },
    details() { return JSON.parse(element('motion-learning-details').textContent); }};
}

function assertReadOnly(dashboard) {
  assert(dashboard.requests.every(request => request.options.method === undefined));
  assert(dashboard.requests.every(request => ['/api/motion/status', '/api/motion/learning'].includes(request.url)));
}

test('absent and old learning histories do not fabricate zero PPO diagnostics', async context => {
  const cases = [
    ['history omitted', undefined],
    ['history null', null],
    ['empty history', []],
    ['empty history entry', [{}]],
    ['legacy loss-only history', [{update: 1, meanLoss: 0.25, maximumGradientNorm: 0.5}]],
    ['latest entry has no diagnostics', [ppoDiagnostics(), {update: 3, meanLoss: 0.25}]]
  ];
  for (const [name, history] of cases) {
    await context.test(name, async () => {
      const record = learningRecord(history);
      const dashboard = await createDashboard(record);
      await dashboard.refresh();
      assert.match(dashboard.diagnostics(), /not recorded|unavailable|older run/i);
      assert.doesNotMatch(dashboard.diagnostics(), /\b\d+(?:\.\d+)?\b/);
      assert.deepEqual(dashboard.details(), JSON.parse(JSON.stringify(record)));
      assert.equal(dashboard.element('motion-use-verified').disabled, true);
      assert.equal(dashboard.element('motion-mode').value, 'scripted');
      assertReadOnly(dashboard);
    });
  }
});

test('missing or null individual PPO metrics stay unknown instead of becoming zero', async context => {
  for (const missing of [undefined, null]) {
    await context.test(missing === null ? 'null metrics' : 'omitted metrics', async () => {
      const record = learningRecord([{meanApproxKL: 0.0123, clipFraction: missing, meanEntropy: missing,
        meanActorLoss: missing, meanCriticLoss: missing, valueExplainedVariance: missing}]);
      const dashboard = await createDashboard(record);
      await dashboard.refresh();
      const diagnostics = dashboard.diagnostics();
      assert.match(diagnostics, /\bKL\s+0\.0123\b/);
      for (const label of ['clip', 'entropy', 'actor', 'critic']) {
        assert.match(diagnostics, new RegExp(`\\b${label}\\s+(?:not recorded|unavailable|n/a)\\b`, 'i'));
      }
      assert.doesNotMatch(diagnostics, /\b0\.0000\b|\bNaN\b|\bInfinity\b/);
      assert.deepEqual(dashboard.details(), JSON.parse(JSON.stringify(record)));
      assertReadOnly(dashboard);
    });
  }
});

test('negative continuous entropy is valid and genuinely recorded zero metrics remain zero', async () => {
  const recorded = ppoDiagnostics({meanApproxKL: 0, clipFraction: 0, meanActorLoss: 0, meanCriticLoss: 0});
  const dashboard = await createDashboard(learningRecord([recorded]));
  await dashboard.refresh();
  const diagnostics = dashboard.diagnostics();
  assert.match(diagnostics, /\bentropy\s+-1\.1250\b/i);
  for (const label of ['KL', 'clip', 'actor', 'critic']) {
    assert.match(diagnostics, new RegExp(`\\b${label}\\s+0\\.0000\\b`, 'i'));
  }
  assert.doesNotMatch(diagnostics, /\b(?:entropy|KL|clip|actor|critic)\s+(?:invalid|nonfinite|not recorded|unavailable)\b/i);
  assert.deepEqual(dashboard.details().models[0].history[0], recorded);
  assertReadOnly(dashboard);
});

test('null explained variance stays explicitly unknown in the diagnostics summary and saved data', async () => {
  const recorded = ppoDiagnostics({valueExplainedVariance: null});
  const dashboard = await createDashboard(learningRecord([recorded]));
  await dashboard.refresh();
  assert.equal(dashboard.details().models[0].history[0].valueExplainedVariance, null);
  const diagnostics = dashboard.diagnostics();
  assert.match(diagnostics,
    /\b(?:explained(?:[ -]+variance)?|EV)\s*(?:[:=]\s*)?(?:not recorded|unavailable|undefined|not defined|n\/a|null)\b/i);
  assert.doesNotMatch(diagnostics, /\b(?:explained(?:[ -]+variance)?|EV)\s*(?:[:=]\s*)?-?0(?:\.0+)?\b/i);
  assert.equal(dashboard.element('motion-use-verified').disabled, true);
  assertReadOnly(dashboard);
});

test('sampled action spread and boundary diagnostics are not mislabeled as applied actions', async () => {
  const recorded = ppoDiagnostics();
  const dashboard = await createDashboard(learningRecord([recorded]));
  await dashboard.refresh();
  const history = dashboard.details().models[0].history[0];
  assert.deepEqual(history.sampledActionMean, recorded.sampledActionMean);
  assert.deepEqual(history.sampledActionStd, recorded.sampledActionStd);
  assert.equal(history.sampledActionBoundaryFraction, recorded.sampledActionBoundaryFraction);
  for (const field of ['appliedActionMean', 'appliedActionStd', 'appliedActionBoundaryFraction']) {
    assert.equal(Object.hasOwn(history, field), false, `${field} was not recorded`);
  }
  const diagnostics = dashboard.diagnostics();
  const sampledSpread = diagnostics.split('·').find(part =>
    /\bsampled\b/i.test(part) && /\b(?:std|standard[ -]+deviation)\b/i.test(part));
  assert(sampledSpread, 'Action spread must explicitly identify sampled, not applied, actions');
  assert.match(sampledSpread, /\b0\.1250[, /]+0\.2500[, /]+0\.3750[, /]+0\.5000\b/);
  const sampledBoundary = diagnostics.split('·').find(part =>
    /\bsampled\b/i.test(part) && /\bboundary\b/i.test(part));
  assert(sampledBoundary, 'Action boundary fraction must explicitly identify sampled actions');
  assert.match(sampledBoundary, /\b0\.3750\b|\b37\.5(?:0+)?%/);
  assert.doesNotMatch(diagnostics,
    /\b(?:applied|executed)(?:[ -]+actions?)?[ -]+(?:std|standard deviation|boundary(?:[ -]+fraction)?)\s*(?:[:=]\s*)?\[?-?\d/i);
  assertReadOnly(dashboard);
});

test('refreshing an older or unavailable record clears richer diagnostics instead of retaining stale metrics', async () => {
  const dashboard = await createDashboard(learningRecord([ppoDiagnostics()]));
  await dashboard.refresh();
  assert.match(dashboard.diagnostics(), /\bentropy\s+-1\.1250\b/i);
  const older = learningRecord([{update: 1, meanLoss: 0.5}]);
  await dashboard.refresh(older);
  assert.match(dashboard.diagnostics(), /not recorded|unavailable|older run/i);
  assert.doesNotMatch(dashboard.diagnostics(), /-1\.1250|0\.0123|0\.0000/);
  assert.deepEqual(dashboard.details(), older);
  const unavailable = {available: false, learnedImprovementVerified: false, message: 'No saved motion learner.'};
  await dashboard.refresh(unavailable);
  assert.equal(dashboard.element('motion-learning-results').hidden, true);
  assert.equal(dashboard.element('motion-learning-table').children.length, 0);
  assert.equal(dashboard.element('motion-learning-status').textContent, unavailable.message);
  assert.deepEqual(dashboard.details(), unavailable);
  assert.equal(dashboard.element('motion-use-verified').disabled, true);
  assert.equal(dashboard.element('motion-compare-learned').disabled, true);
  assert.equal(dashboard.element('motion-mode').value, 'scripted');
  assertReadOnly(dashboard);
});

test('favorable training diagnostics never promote a policy without independent verification', async context => {
  const favorable = ppoDiagnostics({meanApproxKL: 0, clipFraction: 0, meanActorLoss: 0,
    meanCriticLoss: 0, valueExplainedVariance: 1, trainingSuccessRate: 1, meanTrainingReward: 10});
  const cases = [
    ['no evaluation', {}],
    ['failed frozen gate', {evaluation: {models: [evaluationModel(false)]}}],
    ['one passing seed is not independent verification', {evaluation: {models: [evaluationModel(true)]}}],
    ['training still running', {status: 'running', evaluation: {models: [evaluationModel(true)]}}],
    ['verification flag absent', {learnedImprovementVerified: undefined,
      evaluation: {models: [evaluationModel(true)]}}]
  ];
  for (const [name, overrides] of cases) {
    await context.test(name, async () => {
      const dashboard = await createDashboard(learningRecord([favorable], overrides));
      await dashboard.refresh();
      assert.match(dashboard.diagnostics(), /\bentropy\s+-1\.1250\b/i);
      assert.match(dashboard.element('motion-learning-status').textContent, /IMPROVEMENT NOT VERIFIED/);
      assert.equal(dashboard.element('motion-use-verified').disabled, true);
      assert.equal(dashboard.element('motion-compare-learned').disabled, overrides.status === 'running');
      assert.equal(dashboard.element('motion-mode').value, 'scripted');
      if (overrides.evaluation) {
        assert.equal(dashboard.row().children[5].textContent,
          overrides.evaluation.models[0].acceptance.passed ? 'Pass' : 'Failed');
      }
      const requestCount = dashboard.requests.length;
      await dashboard.click('motion-use-verified');
      assert.equal(dashboard.element('motion-mode').value, 'scripted');
      assert.equal(dashboard.requests.length, requestCount);
      assertReadOnly(dashboard);
    });
  }
});

test('independent verification permits only explicit selection even with null variance and negative entropy', async () => {
  const record = learningRecord([ppoDiagnostics({valueExplainedVariance: null})]);
  record.models = [42, 43, 44].map(seed => ({...record.models[0], seed, policySha256: `fixture-policy-${seed}`}));
  record.learnedImprovementVerified = true;
  record.evaluation = {learnedImprovementVerified: true,
    models: record.models.map(model => evaluationModel(true, model.policySha256, model.seed))};
  const dashboard = await createDashboard(record);
  await dashboard.refresh();
  assert.equal(dashboard.element('motion-use-verified').disabled, false);
  assert.match(dashboard.element('motion-learning-status').textContent, /FROZEN IMPROVEMENT GATE PASSED/);
  assert.equal(dashboard.element('motion-mode').value, 'scripted');
  const requestCount = dashboard.requests.length;
  await dashboard.click('motion-use-verified');
  assert.equal(dashboard.element('motion-mode').value, 'learned-experiment');
  assert.equal(dashboard.requests.length, requestCount);
  assert.equal(dashboard.details().models[0].history[0].valueExplainedVariance, null);
  assertReadOnly(dashboard);
});
