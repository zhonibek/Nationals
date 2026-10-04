'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const harness = require('../roboproof/pedro_reference/run');

const root = path.resolve(__dirname, '..');
const cli = path.join(root, 'roboproof/pedro_reference/run.js');
const reference = process.env.PEDRO_REFERENCE || path.resolve(root, '../Nationals/PedroPathing-main/PedroPathing-main');
const hasReference = harness.SOURCE_FILES.every(file => fs.existsSync(path.join(reference, 'core/src/main/java/com/pedropathing', file)));

function temporary(context) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'roboproof-pedro-test-'));
  context.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

test('seeded corpus is reproducible, float-aligned, bounded, and includes signed/degenerate fixtures', () => {
  const cases = harness.generateCases(12345, 2000);
  assert.equal(cases.length, 2077);
  assert.deepEqual(cases, harness.generateCases(12345, 2000));
  assert.notDeepEqual(cases, harness.generateCases(12346, 2000));
  for (const [index, input] of cases.entries()) {
    assert.equal(input.id, index);
    assert(input.t >= 0 && input.t <= 1);
    assert.equal(input.t, Math.fround(input.t));
    assert.equal(input.points.length, 4);
    for (const point of input.points) for (const value of point) {
      assert.equal(value, Math.fround(value));
      assert(Math.abs(value) <= 144);
    }
  }
  for (const label of ['cw', 'ccw', 'constant', 'cusp', 'stationary-start', 'stationary-end', 'small-speed-cutoff']) {
    assert(cases.some(input => input.label === label));
  }
  assert.equal(harness.generateCases(0, 10000).length, 10077);
  for (const count of [0, -1, 10001, 1.5, NaN, Infinity]) assert.throws(() => harness.generateCases(1, count), /count/);
  for (const seed of [-1, 2 ** 32, 0.1, NaN]) assert.throws(() => harness.generateCases(seed, 1), /seed/);
});

test('Java protocol rejects missing, duplicate, reordered, extra and malformed results', () => {
  const cases = [{ id: 0 }, { id: 1 }];
  assert.deepEqual(harness.parseJavaOutput('0 1 2 3 4 5 6 -7\r\n1 1e-3 2 3 4 5 6 7\r\n', cases), [
    [1, 2, 3, 4, 5, 6, -7], [0.001, 2, 3, 4, 5, 6, 7],
  ]);
  for (const output of ['', '0 1 2 3 4 5 6 7', '0 1 2 3 4 5 6 7\n0 1 2 3 4 5 6 7',
    '1 1 2 3 4 5 6 7\n0 1 2 3 4 5 6 7', '0 1 2 3 4 5 6 nope\n1 1 2 3 4 5 6 7',
    '0 1 2 3 4 5 6 7 8\n1 1 2 3 4 5 6 7']) {
    assert.throws(() => harness.parseJavaOutput(output, cases));
  }
  const nonFinite = harness.parseJavaOutput('0 NaN Infinity -Infinity 4 5 6 7', [{ id: 0 }]);
  assert(Number.isNaN(nonFinite[0][0]));
});

test('comparison uses absolute plus relative tolerance and records replayable failures/statistics', () => {
  const cases = harness.generateCases(1, 1).slice(0, 1);
  const referenceRows = [[2, 3, 4, 5, 6, 7, 0.25]];
  const within = [[2 + 1e-6, 3, 4, 5, 6, 7, 0.25]];
  assert.equal(harness.compare(cases, referenceRows, within).status, 'PASS');
  const changed = [[2.1, 3, 4, 5, 6, 7, -0.25]];
  const result = harness.compare(cases, referenceRows, changed);
  assert.equal(result.status, 'FAIL');
  assert.equal(result.comparedCases, 1);
  assert.deepEqual(result.discrepancies[0].input, cases[0]);
  assert.deepEqual(result.discrepancies[0].java, referenceRows[0]);
  assert.deepEqual(result.discrepancies[0].wasm, changed[0]);
  assert.deepEqual(result.discrepancies[0].failures.map(failure => failure.field), ['x', 'curvature']);
  assert.equal(result.statistics.curvature.failures, 1);
  assert.equal(result.statistics.curvature.maxAbsoluteError, 0.5);
  assert.equal(result.statistics.curvature.rmsError, 0.5);
  assert.equal(result.statistics.curvature.worstCaseId, 0);
  assert(result.statistics.curvature.maxToleranceRatio > 1);
  assert.throws(() => harness.compare(cases, [], changed), /Incomplete/);
  assert.throws(() => harness.compare([], [], []), /Incomplete/);
  assert.throws(() => harness.compare(cases, [[0]], changed), /seven/);
});

test('non-finite outputs fail even when both implementations return the same non-finite value', () => {
  const cases = harness.generateCases(1, 1).slice(0, 1);
  for (const bad of [NaN, Infinity, -Infinity]) {
    for (let index = 0; index < 7; index++) {
      const row = [1, 2, 3, 4, 5, 6, 0.25];
      row[index] = bad;
      const result = harness.compare(cases, [row], [row]);
      assert.equal(result.status, 'FAIL');
      assert.equal(result.discrepancies[0].failures[0].reason, 'non-finite');
    }
  }
});

test('only documented small-speed cutoff is normalized; sign and arbitrary zeros still fail', () => {
  const cases = harness.generateCases(1, 1).slice(0, 1);
  const java = [[0, 0, 0.003, 0, 0, 0.006, 666.666]];
  const wasm = [[0, 0, 0.003, 0, 0, 0.006, 0]];
  const result = harness.compare(cases, java, wasm);
  assert.equal(result.status, 'PASS');
  assert.equal(result.intentionalDifferences.length, 1);
  assert.equal(result.statistics.curvature.maxRawAbsoluteError, 666.666);
  assert.equal(result.statistics.curvature.maxAbsoluteError, 0);
  wasm[0][6] = 1e-10;
  assert.equal(harness.compare(cases, java, wasm).status, 'FAIL');
  java[0][2] = wasm[0][2] = 0.1;
  wasm[0][6] = 0;
  assert.equal(harness.compare(cases, java, wasm).status, 'FAIL');
  wasm[0][6] = -java[0][6];
  assert.equal(harness.compare(cases, java, wasm).status, 'FAIL');
});

test('actual compiled C++ primitive returns values, derivatives and signed mirrored curvature', () => {
  const { core, provenance } = harness.loadWasm();
  assert.equal(provenance.wasm.length, 64);
  const cases = harness.generateCases(1, 1).filter(input => ['cw', 'ccw'].includes(input.label) && input.t === 0.5);
  const [ccw, cw] = harness.evaluateWasm(core, cases);
  assert.deepEqual(ccw.slice(0, 6), [0.75, 0.5, 0, 1.5, -6, 0]);
  assert.deepEqual(cw.slice(0, 6), [0.75, -0.5, 0, -1.5, -6, 0]);
  assert(Math.abs(ccw[6] - 8 / 3) < 1e-6);
  assert.equal(cw[6], -ccw[6]);
});

test('WASM provenance rejects altered binaries and stale C++ sources', context => {
  const directory = temporary(context);
  const simulator = path.join(directory, 'simulator');
  fs.mkdirSync(simulator);
  const bytes = fs.readFileSync(path.join(root, 'simulator/control.wasm'));
  fs.writeFileSync(path.join(simulator, 'control.wasm'), bytes);
  const manifest = { wasm: 'wrong', sources: {} };
  fs.writeFileSync(path.join(simulator, 'control-build.json'), JSON.stringify(manifest));
  assert.throws(() => harness.loadWasm(directory), /WASM hash/);
  manifest.wasm = crypto.createHash('sha256').update(bytes).digest('hex');
  manifest.sources['source.cpp'] = 'wrong';
  fs.writeFileSync(path.join(directory, 'source.cpp'), 'changed');
  fs.writeFileSync(path.join(simulator, 'control-build.json'), JSON.stringify(manifest));
  assert.throws(() => harness.loadWasm(directory), /Stale WASM source: source.cpp/);
});

test('CLI validates bounded arguments, rejects source output and never overwrites existing output', context => {
  assert.deepEqual(harness.parseArgs(['--seed', '0', '--count', '100']), { seed: 0, count: 100 });
  for (const args of [['--bad', 'x'], ['--out'], ['--count', '1', '--count', '2']]) {
    assert.throws(() => harness.parseArgs(args));
  }
  assert.throws(() => harness.run({ reference, out: path.join(root, 'roboproof/pedro_reference/generated') }), /outside/);
  assert.throws(() => harness.run({ reference, out: path.join(reference, 'generated') }), /outside/);
  assert.throws(() => harness.run({ reference, out: 'unused', count: 10001 }), /count/);
  const existing = temporary(context);
  fs.writeFileSync(path.join(existing, 'sentinel'), 'keep');
  assert.throws(() => harness.run({ reference, out: existing }), /EEXIST/);
  assert.equal(fs.readFileSync(path.join(existing, 'sentinel'), 'utf8'), 'keep');
});

test('missing Java executable reports BLOCKED with exit 2 and zero comparisons, never substitutes a port', { skip: !hasReference }, context => {
  const directory = temporary(context);
  const out = path.join(directory, 'output with spaces');
  const result = spawnSync(process.execPath, [cli, '--reference', reference, '--out', out, '--count', '100', '--seed', '42', '--java', path.join(directory, 'missing-java')], { encoding: 'utf8', timeout: 15000, windowsHide: true });
  assert.equal(result.error, undefined);
  assert.equal(result.status, 2, result.stderr);
  assert.match(result.stdout, /^BLOCKED:/);
  const report = JSON.parse(fs.readFileSync(path.join(out, 'report.json'), 'utf8'));
  assert.equal(report.status, 'BLOCKED');
  assert.equal(report.comparedCases, 0);
  assert.match(report.reason, /Java reference unavailable/);
  assert(report.javaAttempts.some(attempt => attempt.error));
  assert.equal(report.javaRuntime, undefined);
  assert.equal(fs.existsSync(path.join(out, 'java-results.txt')), false);
  assert.equal(JSON.parse(fs.readFileSync(path.join(out, 'inputs.json'), 'utf8')).length, 177);
  assert.equal(JSON.parse(fs.readFileSync(path.join(out, 'wasm-results.json'), 'utf8')).length, 177);
});

test('missing Pedro source reports BLOCKED and retains the reproducible corpus', context => {
  const directory = temporary(context);
  const report = harness.run({ reference: path.join(directory, 'missing-reference'), out: path.join(directory, 'output'), count: 1 });
  assert.equal(report.status, 'BLOCKED');
  assert.equal(report.comparedCases, 0);
  assert.match(report.reason, /ENOENT/);
  assert.equal(fs.existsSync(path.join(report.out, 'inputs.json')), true);
});

test('actual surviving Java against actual C++ WASM: 2,077 cases (BLOCKED is an explicit skip)', { skip: !hasReference, timeout: 140000 }, context => {
  const directory = temporary(context);
  const report = harness.run({ reference, out: path.join(directory, 'integration'), count: 2000,
    java: process.env.PEDRO_JAVA, javac: process.env.PEDRO_JAVAC });
  if (report.status === 'BLOCKED') {
    assert.equal(report.comparedCases, 0);
    context.diagnostic(JSON.stringify({ status: report.status, reason: report.reason, attempts: report.javaAttempts }));
    context.skip(`BLOCKED: ${report.reason}`);
    return;
  }
  assert.equal(report.status, 'PASS', JSON.stringify(report.discrepancies.slice(0, 3)));
  assert.equal(report.comparedCases, 2077);
  assert.equal(Object.keys(report.javaSources).length, 16);
  assert(['javac', 'source-launcher'].includes(report.javaRuntime.mode));
  assert(report.intentionalDifferences.length >= 7);
  for (const stats of Object.values(report.statistics)) assert.equal(stats.compared, 2077);
  const cases = JSON.parse(fs.readFileSync(path.join(report.out, 'inputs.json'), 'utf8'));
  const javaRows = harness.parseJavaOutput(fs.readFileSync(path.join(report.out, 'java-results.txt'), 'utf8'), cases);
  const wasmRows = JSON.parse(fs.readFileSync(path.join(report.out, 'wasm-results.json'), 'utf8'));
  const changedIndex = cases.findIndex(input => input.label === 'cw' && input.t === 0.5);
  wasmRows[changedIndex][6] *= -1;
  const changed = harness.compare(cases, javaRows, wasmRows);
  assert.equal(changed.status, 'FAIL');
  assert(changed.discrepancies.some(discrepancy => discrepancy.input.id === changedIndex));
});
