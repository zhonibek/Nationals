'use strict';

const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const Control = require('../../simulator/control-runtime');

const ROOT = path.resolve(__dirname, '../..');
const SOURCE_FILES = [
  'math/Matrix.java', 'math/Vector.java', 'math/Vector2D.java', 'math/Pose.java',
  'math/Twist.java', 'math/Velocity.java', 'utils/Angle.java', 'utils/Pair.java',
  'utils/Utils.java', 'utils/BijectiveMap.java', 'paths/TValue.java', 'paths/curves/Curve.java',
  'paths/curves/bezier/BasisMatrixSupplier.java', 'paths/curves/bezier/PolynomialMatrix.java',
  'paths/curves/bezier/BezierCurve.java',
];
const FIELDS = ['x', 'y', 'dx', 'dy', 'ddx', 'ddy', 'curvature'];
const TOLERANCES = Object.freeze({
  point: Object.freeze({ absolute: 5e-5, relative: 3e-6 }),
  first: Object.freeze({ absolute: 2e-4, relative: 4e-6 }),
  second: Object.freeze({ absolute: 5e-4, relative: 4e-6 }),
  curvature: Object.freeze({ absolute: 5e-5, relative: 5e-4 }),
});
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const json = value => JSON.stringify(value, (key, item) =>
  typeof item === 'number' && !Number.isFinite(item) ? String(item) : item, 2) + '\n';

function integer(value, name, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer in [${minimum}, ${maximum}]`);
  }
  return value;
}

function generateCases(seed = 0x50454452, count = 2000) {
  integer(seed, 'seed', 0, 0xffffffff);
  integer(count, 'count', 1, 10000);
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), state | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  const cases = [];
  const append = (label, points, parameter) => cases.push({
    id: cases.length, label, points: points.map(point => point.map(Math.fround)), t: Math.fround(parameter),
  });
  const fixtures = [
    ['line', [[0, 0], [1, 1], [2, 2], [3, 3]]],
    ['ccw', [[0, 0], [1, 0], [1, 1], [0, 1]]],
    ['cw', [[0, 0], [1, 0], [1, -1], [0, -1]]],
    ['s-curve', [[0, 0], [0, 16], [24, 8], [24, 24]]],
    ['loop', [[0, 0], [4, 8], [-4, 8], [0, 0]]],
    ['constant', [[3, -2], [3, -2], [3, -2], [3, -2]]],
    ['stationary-start', [[0, 0], [0, 0], [1, 2], [3, 0]]],
    ['stationary-end', [[0, 0], [1, 2], [3, 0], [3, 0]]],
    ['cusp', [[0, 0], [1, 1], [1, 1], [0, 0]]],
    ['small-speed-cutoff', [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001]]],
    ['large-offset', [[143, -143], [144, -143], [144, -142], [143, -142]]],
  ];
  for (const [label, points] of fixtures) {
    for (const parameter of [0, 1 / 1024, 0.125, 0.5, 0.875, 1 - 1 / 1024, 1]) {
      append(label, points, parameter);
    }
  }
  for (let index = 0; index < count; index++) {
    const scale = [0.01, 1, 144][index % 3];
    const points = Array.from({ length: 4 }, () => [scale * (2 * random() - 1), scale * (2 * random() - 1)]);
    append('seeded', points, index % 11 === 0 ? 0 : index % 11 === 1 ? 1 : random());
  }
  return cases;
}

function parseJavaOutput(output, cases) {
  const lines = output.trim() ? output.trim().split(/\r?\n/) : [];
  if (lines.length !== cases.length) throw new Error(`Java returned ${lines.length}/${cases.length} cases`);
  return lines.map((line, index) => {
    const fields = line.trim().split(/\s+/);
    if (fields.length !== 8 || fields[0] !== String(cases[index].id)) {
      throw new Error(`Invalid Java protocol at case ${index}`);
    }
    return fields.slice(1).map(field => {
      if (!/^(?:NaN|-?Infinity|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$/.test(field)) {
        throw new Error(`Invalid Java number: ${field}`);
      }
      return Number(field);
    });
  });
}

function compare(cases, javaRows, wasmRows) {
  if (javaRows.length !== cases.length || wasmRows.length !== cases.length || !cases.length) {
    throw new Error('Incomplete comparison');
  }
  const statistics = Object.fromEntries(FIELDS.map(field => [field, {
    compared: 0, nonFinite: 0, failures: 0, maxAbsoluteError: 0, maxRawAbsoluteError: 0,
    maxToleranceRatio: 0, worstCaseId: null, sumSquaredError: 0,
  }]));
  const discrepancies = [];
  const intentionalDifferences = [];
  for (let index = 0; index < cases.length; index++) {
    const reference = javaRows[index];
    const actual = wasmRows[index];
    if (reference.length !== 7 || actual.length !== 7) throw new Error('Expected seven primitive outputs');
    const javaSpeed = Math.hypot(reference[2], reference[3]);
    const wasmSpeed = Math.hypot(actual[2], actual[3]);
    const cutoff = javaSpeed >= 1e-9 && javaSpeed < 0.00999 && wasmSpeed < 0.00999;
    const failures = [];
    for (let fieldIndex = 0; fieldIndex < FIELDS.length; fieldIndex++) {
      const field = FIELDS[fieldIndex];
      const stats = statistics[field];
      const expected = fieldIndex === 6 && cutoff ? 0 : reference[fieldIndex];
      const observed = actual[fieldIndex];
      const tolerance = TOLERANCES[fieldIndex < 2 ? 'point' : fieldIndex < 4 ? 'first' : fieldIndex < 6 ? 'second' : 'curvature'];
      const limit = tolerance.absolute + tolerance.relative * Math.max(Math.abs(expected), Math.abs(observed));
      const error = Math.abs(observed - expected);
      const finite = Number.isFinite(observed) && Number.isFinite(reference[fieldIndex]);
      stats.compared++;
      if (finite) {
        stats.maxRawAbsoluteError = Math.max(stats.maxRawAbsoluteError, Math.abs(observed - reference[fieldIndex]));
        stats.maxAbsoluteError = Math.max(stats.maxAbsoluteError, error);
        stats.sumSquaredError += error * error;
        if (error / limit > stats.maxToleranceRatio) {
          stats.maxToleranceRatio = error / limit;
          stats.worstCaseId = cases[index].id;
        }
      } else {
        stats.nonFinite++;
      }
      if (!finite || error > limit || (fieldIndex === 6 && cutoff && observed !== 0)) {
        stats.failures++;
        failures.push({ field, expected, observed, absoluteError: error, limit, reason: finite ? 'tolerance' : 'non-finite' });
      }
    }
    if (cutoff && reference[6] !== 0 && Number.isFinite(reference[6]) && actual[6] === 0) {
      intentionalDifferences.push({ input: cases[index], reason: 'C++ speed-cubed cutoff < 1e-6; Java speed cutoff < 1e-9', java: reference, wasm: actual });
    }
    if (failures.length) discrepancies.push({ input: cases[index], java: reference, wasm: actual, failures });
  }
  for (const stats of Object.values(statistics)) {
    stats.rmsError = Math.sqrt(stats.sumSquaredError / Math.max(1, stats.compared - stats.nonFinite));
    delete stats.sumSquaredError;
  }
  return { status: discrepancies.length ? 'FAIL' : 'PASS', comparedCases: cases.length, statistics, discrepancies, intentionalDifferences };
}

function loadWasm(root = ROOT) {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'simulator/control-build.json'), 'utf8').replace(/^\uFEFF/, ''));
  const bytes = fs.readFileSync(path.join(root, 'simulator/control.wasm'));
  if (hash(bytes) !== manifest.wasm) throw new Error('WASM hash differs from control-build.json; rebuild required');
  for (const [file, digest] of Object.entries(manifest.sources)) {
    const contents = fs.readFileSync(path.join(root, file), 'utf8').replace(/\r\n/g, '\n');
    if (hash(contents) !== digest) throw new Error(`Stale WASM source: ${file}; rebuild required`);
  }
  const core = Control.fromModule(new WebAssembly.Module(bytes));
  if (typeof core.e.control_curve_eval !== 'function') throw new Error('WASM lacks control_curve_eval');
  return { core, provenance: { wasm: manifest.wasm, sources: manifest.sources, compiler: manifest.compiler } };
}

function evaluateWasm(core, cases) {
  return cases.map(input => {
    core.buffer.set(input.points.flat());
    core.e.control_curve_eval(input.t);
    return Array.from(core.buffer.slice(32, 39));
  });
}

function executable(name, environment = process.env) {
  if (path.isAbsolute(name) || name.includes('/') || name.includes('\\')) return path.resolve(name);
  const suffix = process.platform === 'win32' && !name.endsWith('.exe') ? '.exe' : '';
  for (const directory of (environment.PATH || '').split(path.delimiter)) {
    const candidate = path.join(directory.replace(/^"|"$/g, ''), name + suffix);
    if (fs.existsSync(candidate)) return candidate;
  }
  return name;
}

function javaCandidates(options, environment = process.env) {
  const extension = process.platform === 'win32' ? '.exe' : '';
  if (options.java) {
    const java = executable(options.java, environment);
    return [{ java, javac: options.javac ? executable(options.javac, environment) : path.join(path.dirname(java), `javac${extension}`) }];
  }
  if (options.javac) throw new Error('--javac requires --java to select the matching runtime');
  const candidates = [];
  const addHome = home => {
    if (home) candidates.push({ java: path.join(home, 'bin', `java${extension}`), javac: path.join(home, 'bin', `javac${extension}`) });
  };
  addHome(environment.JAVA_HOME);
  addHome(environment.JDK_HOME);
  for (const name of ['javac', 'java']) {
    const found = executable(name, environment);
    if (fs.existsSync(found)) {
      addHome(path.dirname(path.dirname(fs.realpathSync(found))));
    }
  }
  const roots = process.platform === 'win32' ? [
    ...['Java', 'Eclipse Adoptium', 'Microsoft', 'Amazon Corretto', 'Android/Android Studio'].map(vendor => path.join(environment.ProgramFiles || 'C:/Program Files', vendor)),
    path.join(os.homedir(), '.jdks'),
  ] : ['/usr/lib/jvm', '/Library/Java/JavaVirtualMachines'];
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    addHome(root);
    addHome(path.join(root, 'jbr'));
    for (const entry of fs.readdirSync(root, { withFileTypes: true }).sort((left, right) => left.name.localeCompare(right.name))) {
      if (!entry.isDirectory()) continue;
      addHome(path.join(root, entry.name));
      addHome(path.join(root, entry.name, 'Contents/Home'));
    }
  }
  const unique = new Map(candidates.filter(candidate => fs.existsSync(candidate.java)).map(candidate => [candidate.java, candidate]));
  return [...unique.values()].slice(0, 16);
}

function invoke(command, args, input, timeout, attempts, stage) {
  const result = spawnSync(command, args, { input, encoding: 'utf8', timeout, maxBuffer: 16 * 1024 * 1024, windowsHide: true, shell: false });
  attempts.push({ stage, command, args, exitCode: result.status, signal: result.signal,
    error: result.error ? result.error.message : null, stderr: (result.stderr || '').slice(0, 16000),
    stdout: stage === 'java-run' ? undefined : (result.stdout || '').slice(0, 16000) });
  return result;
}

function runJava(options, sources, classes, cases, attempts) {
  const deadline = Date.now() + 120000;
  const invokeBounded = (command, args, input, stage) => {
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error('Java preparation/execution exceeded 120 seconds');
    return invoke(command, args, input, Math.min(stage === 'probe' ? 3000 : 30000, remaining), attempts, stage);
  };
  const input = cases.map(item => [item.id, item.t, ...item.points.flat()].join(' ')).join('\n') + '\n';
  for (const candidate of javaCandidates(options)) {
    const version = invokeBounded(candidate.java, ['-version'], undefined, 'probe');
    if (version.error || version.status !== 0) continue;
    const versionText = (version.stderr || '') + (version.stdout || '');
    const majorMatch = versionText.match(/version\s+"(?:1\.)?(\d+)/);
    const major = majorMatch ? Number(majorMatch[1]) : 0;
    let args;
    let mode;
    if (fs.existsSync(candidate.javac)) {
      mode = 'javac';
      const compile = invokeBounded(candidate.javac, ['-J-Xmx256m', '-encoding', 'UTF-8', '-proc:none', '-classpath', classes, '-sourcepath', classes, '-d', classes, ...sources], undefined, 'compile');
      if (compile.error || compile.status !== 0) continue;
      args = ['-Xmx256m', '-cp', classes, 'PedroProbe'];
    } else if (major >= 11) {
      mode = 'source-launcher';
      args = ['-Xmx256m', path.join(__dirname, 'CompileAndRun.java'), classes, ...sources];
    } else {
      attempts.push({ stage: 'capability', command: candidate.java, reason: `Java ${major || 'unknown'} has no adjacent javac or Java 11+ source launcher` });
      continue;
    }
    const result = invokeBounded(candidate.java, args, input, 'java-run');
    if (result.error || result.status !== 0) continue;
    return { rows: parseJavaOutput(result.stdout, cases), output: result.stdout, runtime: { ...candidate, mode, version: versionText.trim() } };
  }
  throw new Error('Java reference unavailable: no usable compiler/runtime, compilation failed, or execution failed; inspect javaAttempts');
}

function canonicalTarget(target) {
  if (fs.existsSync(target)) return fs.realpathSync(target);
  const parent = path.dirname(target);
  return parent === target ? target : path.join(canonicalTarget(parent), path.basename(target));
}

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}

function run(options) {
  if (!options.reference || !options.out) throw new Error('--reference and --out are required');
  const seed = integer(options.seed ?? 0x50454452, 'seed', 0, 0xffffffff);
  const count = integer(options.count ?? 2000, 'count', 1, 10000);
  if (options.javac && !options.java) throw new Error('--javac requires --java');
  const reference = canonicalTarget(path.resolve(options.reference));
  const out = canonicalTarget(path.resolve(options.out));
  if (isInside(fs.realpathSync(ROOT), out) || isInside(reference, out)) {
    throw new Error('Output must be outside the source checkout and Java reference tree');
  }
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.mkdirSync(out);
  const cases = generateCases(seed, count);
  const write = (file, value) => fs.writeFileSync(path.join(out, file), json(value));
  write('inputs.json', cases);
  const report = { schemaVersion: 1, status: 'BLOCKED', seed, randomCases: count, totalCases: cases.length,
    comparedCases: 0, reference, out, tolerances: TOLERANCES, javaAttempts: [],
    convention: 'Signed Cartesian curvature in both primitives (positive CCW); not clockwise robot heading rate',
    scope: 'Cubic point, first/second parameter derivatives and curvature only; no follower equivalence claim' };
  try {
    const sourceRoot = path.join(reference, 'core/src/main/java/com/pedropathing');
    const sources = SOURCE_FILES.map(file => path.join(sourceRoot, file));
    sources.push(path.join(__dirname, 'PedroProbe.java'));
    report.javaSources = Object.fromEntries(sources.map(file => [file, hash(fs.readFileSync(file))]));
    report.launcherSha256 = hash(fs.readFileSync(path.join(__dirname, 'CompileAndRun.java')));
    report.harnessSha256 = hash(fs.readFileSync(__filename));
    const loaded = loadWasm();
    report.wasm = loaded.provenance;
    const wasmRows = evaluateWasm(loaded.core, cases);
    write('wasm-results.json', wasmRows);
    const classes = path.join(out, 'classes');
    fs.mkdirSync(classes);
    const java = runJava(options, sources, classes, cases, report.javaAttempts);
    report.javaRuntime = java.runtime;
    fs.writeFileSync(path.join(out, 'java-results.txt'), java.output);
    Object.assign(report, compare(cases, java.rows, wasmRows));
  } catch (error) {
    report.status = 'BLOCKED';
    report.reason = error.message;
  }
  write('report.json', report);
  return report;
}

function parseArgs(args) {
  const options = {};
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index];
    if (!['--reference', '--out', '--java', '--javac', '--seed', '--count'].includes(flag) || args[index + 1] === undefined) {
      throw new Error(`Unknown option or missing value: ${flag}`);
    }
    const key = flag.slice(2);
    if (Object.hasOwn(options, key)) throw new Error(`Duplicate option: ${flag}`);
    options[key] = ['seed', 'count'].includes(key) ? Number(args[index + 1]) : args[index + 1];
  }
  return options;
}

if (require.main === module) {
  try {
    if (process.argv.includes('--help')) {
      console.log('node roboproof/pedro_reference/run.js --reference <PedroPathing-root> --out <new-external-directory> [--count 2000] [--seed 1346716754] [--java <java>] [--javac <javac>]');
    } else {
      const report = run(parseArgs(process.argv.slice(2)));
      console.log(`${report.status}: ${report.comparedCases}/${report.totalCases} cases compared; ${path.join(report.out, 'report.json')}`);
      if (report.reason) console.log(report.reason);
      process.exitCode = { PASS: 0, FAIL: 1, BLOCKED: 2 }[report.status];
    }
  } catch (error) {
    console.error(`BLOCKED: ${error.message}`);
    process.exitCode = 2;
  }
}

module.exports = { generateCases, parseJavaOutput, compare, loadWasm, evaluateWasm, run, parseArgs, javaCandidates, TOLERANCES, SOURCE_FILES };
