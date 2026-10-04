'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');
const {ingest, args} = require('../roboproof/ingest');

const repository = path.resolve(__dirname, '..');
const executable = path.join(repository, 'roboproof/ingest.js');
const actual = ingest(repository);

function temporary(context, copy = false) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'roboproof-ingest-'));
  context.after(() => {
    const resolved = fs.realpathSync(directory);
    const parent = fs.realpathSync(os.tmpdir());
    assert.equal(path.dirname(resolved), parent);
    assert(path.basename(resolved).startsWith('roboproof-ingest-'));
    fs.rmSync(resolved, {recursive: true, force: true});
  });
  if (copy) {
    for (const file of actual.files) {
      const destination = path.join(directory, file.path);
      fs.mkdirSync(path.dirname(destination), {recursive: true});
      fs.copyFileSync(path.join(repository, file.path), destination);
    }
  }
  return directory;
}

function replace(directory, filename, before, after) {
  const destination = path.join(directory, filename);
  const text = fs.readFileSync(destination, 'utf8');
  assert(text.includes(before));
  fs.writeFileSync(destination, text.replace(before, after));
}

test('actual supported repository is ingested deterministically with verifiable evidence', () => {
  assert.equal(actual.status, 'SUPPORTED_SOURCE_SNAPSHOT');
  assert.deepEqual(ingest(repository), actual);
  assert(actual.active_confirmed.includes('cascade_control'));
  assert(actual.active_confirmed.includes('encoder_localization'));
  assert(actual.active_confirmed.includes('simulator_cascade'));
  assert.equal(actual.sensors.localization, 'four_drive_motor_encoders');
  assert.deepEqual(actual.sensors.external_localization_sensors, []);
  assert.deepEqual(actual.configuration.declared, JSON.parse(fs.readFileSync(path.join(repository, 'config/robot.json'), 'utf8')));
  for (const filename of actual.files) {
    const bytes = fs.readFileSync(path.join(repository, filename.path));
    assert.equal(filename.sha256, crypto.createHash('sha256').update(bytes).digest('hex'));
  }
  for (const check of actual.checks) {
    assert.equal(check.status, 'SOURCE_CONFIRMED', check.id);
    for (const observation of check.observations) {
      assert(observation.found);
      const evidence = observation.evidence;
      const lines = fs.readFileSync(path.join(repository, evidence.path), 'utf8').replace(/\r\n/g, '\n').split('\n');
      assert.equal(evidence.snippet, lines.slice(evidence.line - 1, evidence.end_line).join('\n'));
      assert.equal(evidence.sha256, actual.files.find(file => file.path === evidence.path).sha256);
    }
  }
});

test('available legacy and optional modules are not forged into the active controller', () => {
  for (const id of ['legacy_pedro_follower', 'ekf', 'mcl', 'fuzzy', 'odom_reset']) {
    const module = actual.modules.find(candidate => candidate.id === id);
    assert.equal(module.activation, 'NOT_ACTIVE_IN_RECOGNIZED_MAIN');
    assert(actual.available_modules.includes(id));
    assert(!actual.active_confirmed.includes(id));
  }
});

test('unsupported root and a forged repository map cannot fabricate architecture', context => {
  const directory = temporary(context);
  fs.mkdirSync(path.join(directory, 'docs'));
  fs.writeFileSync(path.join(directory, 'docs/repository_map.json'), JSON.stringify(actual));
  const report = ingest(directory);
  assert.equal(report.status, 'UNVERIFIED');
  assert.deepEqual(report.active_confirmed, []);
  assert.deepEqual(report.available_modules, []);
  assert.equal(report.sensors.localization, null);
  assert.equal(report.sensors.external_localization_sensors, null);
  assert.equal(report.configuration.declared, null);
  assert(report.files.every(file => file.reason === 'MISSING'));
});

test('copied inspected sources give identical reports without consulting the host repo', context => {
  const directory = temporary(context, true);
  assert.deepEqual(ingest(directory), actual);
  fs.unlinkSync(path.join(directory, 'src/subsystems/control/HolonomicMotion.cpp'));
  const report = ingest(directory);
  assert.equal(report.status, 'UNVERIFIED');
  assert.deepEqual(report.active_confirmed, []);
  assert.notEqual(report.fingerprint.value, actual.fingerprint.value);
});

test('changed core source with the same symbols is unverified, not silently recognized', context => {
  const directory = temporary(context, true);
  replace(directory, 'include/subsystems/control/Cascade.hpp', 'double kS=0.40', 'double kS=9.40');
  const report = ingest(directory);
  const check = report.checks.find(candidate => candidate.id === 'cascade');
  assert(check.observations.every(observation => observation.found));
  assert(check.reasons.includes('CHANGED_REVIEWED_SOURCE'));
  assert.deepEqual(report.active_confirmed, []);
});

test('changed sensor wiring is detected even when the original text remains in a comment', context => {
  const directory = temporary(context, true);
  const original = 'lemlib::OdomSensors sensors(nullptr,nullptr,nullptr,nullptr,nullptr);';
  replace(directory, 'src/main.cpp', original, `// ${original}\nlemlib::OdomSensors sensors(nullptr,nullptr,nullptr,nullptr,&imu);`);
  const report = ingest(directory);
  assert.equal(report.sensors.status, 'UNVERIFIED');
  assert.equal(report.sensors.localization, null);
  assert.deepEqual(report.checks.find(check => check.id === 'sensors').observations.map(observation => observation.found), [false]);
  assert.deepEqual(report.active_confirmed, []);
});

test('optional module removal does not claim it is available or disable known core wiring', context => {
  const directory = temporary(context, true);
  fs.unlinkSync(path.join(directory, 'src/subsystems/ekf/EKF.cpp'));
  const report = ingest(directory);
  assert.equal(report.status, 'SUPPORTED_SOURCE_SNAPSHOT');
  assert.equal(report.modules.find(module => module.id === 'ekf').activation, 'UNVERIFIED');
  assert(!report.available_modules.includes('ekf'));
});

test('configuration is extracted from the supplied root and disagreements are explicit', context => {
  const directory = temporary(context, true);
  const configPath = path.join(directory, 'config/robot.json');
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  config.wheelDiameterIn = 4;
  config.imuPort = 3;
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
  const report = ingest(directory);
  assert.equal(report.configuration.declared.wheelDiameterIn, 4);
  assert.equal(report.configuration.declared.imuPort, 3);
  assert(report.configuration.reasons.includes('CONFIG_HEADER_MISMATCH_wheelDiameterIn'));
  assert(report.configuration.reasons.includes('DECLARED_IMU_DIFFERS_FROM_REVIEWED_NULL_WIRING'));
  assert.deepEqual(report.active_confirmed, []);
  for (const invalid of ['{broken', 'null', '[]', '{"ports":{},"drive":"TANK"}']) {
    fs.writeFileSync(configPath, invalid);
    assert.equal(ingest(directory).configuration.status, 'UNVERIFIED');
  }
});

test('hash recognition tolerates CRLF but byte fingerprint records the actual bytes', context => {
  const directory = temporary(context, true);
  const filename = path.join(directory, 'src/main.cpp');
  const original = fs.readFileSync(filename, 'utf8');
  const normalized = original.replace(/\r\n/g, '\n');
  fs.writeFileSync(filename, original.includes('\r\n') ? normalized : normalized.replace(/\n/g, '\r\n'));
  const report = ingest(directory);
  assert.equal(report.status, 'SUPPORTED_SOURCE_SNAPSHOT');
  assert.notEqual(report.fingerprint.value, actual.fingerprint.value);
});

test('CLI writes stable reports only to requested output and returns 2 for unknown roots', context => {
  const directory = temporary(context);
  const output = path.join(directory, 'reports/repository.json');
  const run = repo => spawnSync(process.execPath, [executable, '--repo', repo, '--out', output], {encoding: 'utf8', cwd: directory});
  assert.equal(run(repository).status, 0);
  const first = fs.readFileSync(output, 'utf8');
  assert.deepEqual(JSON.parse(first), actual);
  assert.equal(run(repository).status, 0);
  assert.equal(fs.readFileSync(output, 'utf8'), first);
  assert.equal(run(directory).status, 2);
  assert.deepEqual(JSON.parse(fs.readFileSync(output, 'utf8')).active_confirmed, []);
});

test('CLI refuses invalid options and source overwrite', context => {
  for (const argv of [['--repo'], ['--other', '.'], ['--repo', '.', '--repo', '.'], ['--out', '--repo']]) assert.throws(() => args(argv));
  const directory = temporary(context, true);
  const configPath = path.join(directory, 'config/robot.json');
  const before = fs.readFileSync(configPath, 'utf8');
  const result = spawnSync(process.execPath, [executable, '--repo', directory, '--out', configPath], {encoding: 'utf8'});
  assert.equal(result.status, 1);
  assert.match(result.stderr, /not an inspected source/);
  assert.equal(fs.readFileSync(configPath, 'utf8'), before);
});

test('source directory links are not traversed to borrow evidence from another repository', context => {
  const directory = temporary(context);
  fs.symlinkSync(path.join(repository, 'src'), path.join(directory, 'src'), process.platform === 'win32' ? 'junction' : 'dir');
  const report = ingest(directory);
  assert.equal(report.files.find(file => file.path === 'src/main.cpp').reason, 'SYMLINK_NOT_INSPECTED');
  assert.deepEqual(report.active_confirmed, []);
});
