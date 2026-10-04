'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const PROFILE = 'nationals-xdrive-source-v2';
const MAIN = 'src/main.cpp';
const CONFIG = 'config/robot.json';
const HEADER = 'include/robot_config.hpp';
const CASCADE = 'include/subsystems/control/Cascade.hpp';
const BOUNDARY = 'src/subsystems/control/HolonomicMotion.cpp';
const ODOM = 'src/lemlib/chassis/odom.cpp';
const SAFETY = 'src/lemlib/safety.cpp';
const REVIEWED = {
  'src/main.cpp': '24361265605b8e917748953786cfada0bf0806ef1ddea823bdec28c2166d6c27',
  'include/robot_config.hpp': 'a398014bb612cb1d13d61ea8fe9930d4136562feaeac22f6c22356d778456e5b',
  'include/subsystems/control/Cascade.hpp': '1afa2a237881005035cd1ef74e8cdbe191cfcdb73c3f27471d57e0b71c291e63',
  'include/subsystems/control/BezierReference.hpp': '35425902da85f2812e804f7655d62bb4a12ab472f8cc19a1d8840fc596a65680',
  'src/subsystems/control/HolonomicMotion.cpp': 'fc42bd6223cf92fa46ed0278239f9077ed1240ebc5da764da650fdccceb7c5be',
  'src/subsystems/ltv/ltv.cpp': '6a7e5f2ff565898da0f66968802924d55bdadf815bf7bbc59e48dcc9e2a26960',
  'src/subsystems/trajectory/QuinticSpline.cpp': '1e7ec2a644bf58065fa05c0dc2306f65c6207918421dd55dc688c9d81049c672',
  'src/subsystems/pedro/BezierCurve.cpp': '58bc2bc6873abf9f45d898255632ee1ec9fb8bd573b0e11161d68d283b5643f2',
  'src/lemlib/chassis/odom.cpp': '66670c7583ce09bb00ada679c436871e84e94d441981962af462165a284f0233',
  'src/lemlib/chassis/chassis.cpp': '2fbf10296659e20a1425d014e36ab339547f4da2dae5abd6b1c710790e599f91',
  'src/lemlib/safety.cpp': '0e413eac3c8ea076c5c0a0c28c6878941d89dab4f4ab29413e24672ea948d19d',
  'src/subsystems/MotorMonitor.cpp': 'af5578b041610ea392bbaa8a58d979dc5d66d50cbe04b0c7f732113a8a8d972d',
  'src/subsystems/pedro/PedroFollower.cpp': '4be8e2f2d104c4ddf9f20e105550731a9f08f39188da1d72ccdabadb36000f7a',
  'src/subsystems/ekf/EKF.cpp': 'fc12e326f33ec112c99641ddd9f177bdccb928b63fa45264a33ad071dd2e2ac0',
  'src/subsystems/mcl/MCL.cpp': '196b3519c2bcd1d9c07fd7605d7638c4093a6becd5cc0519b90988df902a10e3',
  'src/subsystems/flc/FuzzyLogic.cpp': 'a230cf42f819746cf707bbfc695c6a0e119a2b125486454a1445f00cf6d08e2d',
  'src/subsystems/OdomReset.cpp': 'c3b20356311e0a886c715ca97458450e89288c0a970b855990eb8db85d572448',
  'simulator/native/control.cpp': '3f632d72f56847876f8357374a0f292b5bf07ea5000e5f59c4f620f9e8b37b1a',
  'simulator/control-runtime.js': 'dad5517b5acbffe37fad87dda99978a1634d96cc3b9567079edcdd9978706bec',
  'simulator/engine.js': '042e6c84fc8e78269409f22aa7a53c1f05c16e46afeb931bf1bc8cee43a801b3'
};
const CHECKS = [
  ['entry', MAIN, 'void initialize()', 'void autonomous()', 'void opcontrol()'],
  ['drive', MAIN, 'pros::MotorGroup leftMotors({frontLeft,backLeft},pros::MotorGearset::green)', 'pros::MotorGroup rightMotors({frontRight,backRight},pros::MotorGearset::green)', 'chassis.setDrivebaseType(lemlib::DrivebaseType::XDRIVE)'],
  ['sensors', MAIN, 'lemlib::OdomSensors sensors(nullptr,nullptr,nullptr,nullptr,nullptr)'],
  ['point_wiring', MAIN, 'lemlib::MotionResult autoPose(', 'return nationals::followReference(', 'nationals::pointReference(start,end,time,duration)'],
  ['boundary', BOUNDARY, 'Feedback readFeedback(', 'getOdomSnapshot()', 'get_actual_velocity_all()', 'Cascade control;', 'control.update(target,feedback,dt)', 'output.wheelVoltages(token,command.volts)'],
  ['cascade', CASCADE, 'class Cascade', 'class WheelVelocity', 'bool solve(', 'inverseX(s,f,w,c.radius,o.target)', 'return wheels.update(s,f,w,fb,dt,config)'],
  ['odom', ODOM, 'if(type==lemlib::DrivebaseType::XDRIVE)', 'get_position_all()', 'const float mid=pose.theta+dh/2', 'driveOutput().watchdog()'],
  ['calibration', 'src/lemlib/chassis/chassis.cpp', 'void lemlib::Chassis::calibrate(', 'setSensors(sensors, drivetrain)', 'init()'],
  ['output', SAFETY, 'DriveOutput::acquire(', 'DriveOutput::wheelVoltages(', 'pros::c::motor_move_voltage(', 'DriveOutput::watchdog()'],
  ['bezier_wiring', MAIN, 'nationals::BezierReference path(', 'path.at(t)'],
  ['bezier_reference', 'include/subsystems/control/BezierReference.hpp', 'class BezierReference', 'Reference at(double time)const'],
  ['bezier_geometry', 'src/subsystems/pedro/BezierCurve.cpp', 'Point BezierCurve::getPoint(', 'float BezierCurve::getCurvature('],
  ['spline_wiring', MAIN, 'ltvFollower.followTrajectory(', 'QuinticSplineGenerator::generateTrajectory('],
  ['spline', 'src/subsystems/trajectory/QuinticSpline.cpp', 'QuinticSplineGenerator::generateTrajectory('],
  ['ltv', 'src/subsystems/ltv/ltv.cpp', 'if(chassis.getDrivebaseType()==DrivebaseType::XDRIVE)', 'nationals::followReference('],
  ['manual', MAIN, 'driveOutput().holonomic(', 'driveOutput().wheels('],
  ['monitor_wiring', MAIN, 'monitor.startTask(500)'],
  ['monitor', 'src/subsystems/MotorMonitor.cpp', 'void MotorMonitor::startTask('],
  ['legacy_pedro', 'src/subsystems/pedro/PedroFollower.cpp', 'PedroFollower::follow('],
  ['ekf', 'src/subsystems/ekf/EKF.cpp', 'RobotEKF::predict('],
  ['mcl', 'src/subsystems/mcl/MCL.cpp', 'MCL::update()'],
  ['fuzzy', 'src/subsystems/flc/FuzzyLogic.cpp', 'FuzzyLogicController::computeMultipliers('],
  ['odom_reset', 'src/subsystems/OdomReset.cpp', 'OdomReset::wallReset('],
  ['wasm_bridge', 'simulator/native/control.cpp', 'static nationals::Cascade controller;', 'int control_step()'],
  ['wasm_runtime', 'simulator/control-runtime.js', 'class ProductionControl', 'this.e.control_step()'],
  ['simulator', 'simulator/engine.js', 'class VexRobotSimulator', 'this.productionControl.step(ref,feedback,dt)']
];
const CORE_CHECKS = ['entry', 'drive', 'sensors', 'point_wiring', 'boundary', 'cascade', 'odom', 'calibration', 'output'];
const MODULES = [
  ['cascade_control', 'Outer Riccati pose feedback and four wheel PI/feedforward loops', ['point_wiring', 'boundary', 'cascade'], true],
  ['encoder_localization', 'Four drive encoder odometry; no external localization sensors wired in recognized main', ['sensors', 'odom', 'calibration'], true],
  ['leased_motor_output', 'Leased PROS voltage output and watchdog', ['output', 'boundary'], true],
  ['manual_drive', 'Driver holonomic and diagnostic wheel power bypass the cascade', ['manual', 'output'], true],
  ['bezier_reference', 'Bezier time reference into cascade, not Foresight', ['bezier_wiring', 'bezier_reference', 'bezier_geometry'], true],
  ['spline_adapter', 'Quintic spline and XDRIVE LTV adapter into cascade', ['spline_wiring', 'spline', 'ltv'], true],
  ['motor_monitor', 'Motor monitoring task', ['monitor_wiring', 'monitor'], true],
  ['legacy_pedro_follower', 'Legacy PedroFollower implementation; no follow call in recognized main', ['legacy_pedro'], false],
  ['ekf', 'Optional C++ EKF, not wired into recognized main', ['ekf'], false],
  ['mcl', 'Optional C++ MCL, not wired into recognized main', ['mcl'], false],
  ['fuzzy', 'Optional fuzzy control helper', ['fuzzy'], false],
  ['odom_reset', 'Optional external-sensor reset helpers', ['odom_reset'], false],
  ['simulator_cascade', 'JS motion execution calls C++ cascade bridge; not the full PROS runtime', ['wasm_bridge', 'wasm_runtime', 'simulator', 'cascade'], true]
];

function hash(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  return value;
}

function maskedCode(text) {
  return text.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*'/g, match => match.replace(/[^\n]/g, ' '));
}

function readSource(root, filename) {
  const record = {path: filename, status: 'UNVERIFIED', sha256: null, normalized_sha256: null, reviewed_sha256: REVIEWED[filename] || null};
  try {
    let current = root;
    for (const part of filename.split('/')) {
      current = path.join(current, part);
      if (fs.lstatSync(current).isSymbolicLink()) return {...record, reason: 'SYMLINK_NOT_INSPECTED'};
    }
    const stat = fs.statSync(current);
    if (!stat.isFile() || stat.size > 2 * 1024 * 1024) return {...record, reason: 'NOT_A_BOUNDED_SOURCE_FILE'};
    const bytes = fs.readFileSync(current);
    const text = bytes.toString('utf8').replace(/\r\n/g, '\n');
    return {...record, status: 'READ', sha256: hash(bytes), normalized_sha256: hash(text), text, code: maskedCode(text)};
  } catch (error) {
    return {...record, reason: error.code === 'ENOENT' ? 'MISSING' : `READ_ERROR_${error.code || 'UNKNOWN'}`};
  }
}

function evidence(source, offset, length) {
  const line = source.text.slice(0, offset).split('\n').length;
  const endLine = line + source.text.slice(offset, offset + length).split('\n').length - 1;
  return {path: source.path, line, end_line: endLine, snippet: source.text.split('\n').slice(line - 1, endLine).join('\n'), sha256: source.sha256};
}

function inspectCheck(definition, sources) {
  const [id, filename, ...symbols] = definition;
  const source = sources.get(filename);
  const observations = symbols.map(symbol => {
    const offset = source.code === undefined ? -1 : source.code.indexOf(symbol);
    return {symbol, found: offset >= 0, evidence: offset < 0 ? null : evidence(source, offset, symbol.length)};
  });
  const reasons = [];
  if (source.status !== 'READ') reasons.push(source.reason);
  else if (source.normalized_sha256 !== source.reviewed_sha256) reasons.push('CHANGED_REVIEWED_SOURCE');
  if (observations.some(observation => !observation.found)) reasons.push('EXPECTED_SYMBOL_NOT_OBSERVED');
  return {id, path: filename, status: reasons.length ? 'UNVERIFIED' : 'SOURCE_CONFIRMED', reasons, observations};
}

function inspectConfiguration(sources) {
  const source = sources.get(CONFIG);
  const header = sources.get(HEADER);
  const result = {status: 'UNVERIFIED', declared: null, fields: [], header_constants: {}, reasons: []};
  if (source.status !== 'READ') {
    result.reasons.push(`CONFIG_${source.reason}`);
    return result;
  }
  let config;
  try { config = JSON.parse(source.text.replace(/^\uFEFF/, '')); }
  catch { result.reasons.push('INVALID_CONFIG_JSON'); return result; }
  result.declared = canonical(config);
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    result.reasons.push('CONFIG_NOT_AN_OBJECT');
    return result;
  }
  for (const key of Object.keys(config).sort()) {
    const offset = source.text.indexOf(JSON.stringify(key));
    result.fields.push({key, value: canonical(config[key]), evidence: offset < 0 ? null : evidence(source, offset, key.length + 2)});
  }
  const geometry = ['cartridgeRpm', 'wheelDiameterIn', 'widthIn', 'wheelbaseIn', 'externalGearRatio'];
  const portNames = ['frontLeft', 'backLeft', 'frontRight', 'backRight'];
  for (const key of geometry) if (!Number.isFinite(config[key]) || config[key] <= 0) result.reasons.push(`INVALID_${key}`);
  const ports = portNames.map(key => config.ports?.[key]);
  if (ports.some(port => !Number.isInteger(port) || port === 0 || Math.abs(port) > 21) || new Set(ports.map(Math.abs)).size !== 4) result.reasons.push('INVALID_MOTOR_PORTS');
  if (config.drive !== 'XDRIVE') result.reasons.push('UNSUPPORTED_DECLARED_DRIVE');
  if (config.imuPort !== null) result.reasons.push('DECLARED_IMU_DIFFERS_FROM_REVIEWED_NULL_WIRING');
  if (config.cartridgeRpm !== 200) result.reasons.push('DECLARED_RPM_DIFFERS_FROM_REVIEWED_GREEN_MOTORS');
  const expected = Object.fromEntries([...geometry, ...portNames].map(key => [key, config[key] ?? config.ports?.[key]]));
  for (const group of ['linear', 'angular']) {
    for (const key of ['kP', 'kI', 'kD', 'kV']) {
      const value = config[group]?.[key];
      if (!Number.isFinite(value)) result.reasons.push(`INVALID_${group}_${key}`);
      expected[`${group}_${key}`] = value;
    }
  }
  if (header.status !== 'READ' || header.normalized_sha256 !== header.reviewed_sha256) result.reasons.push('GENERATED_HEADER_MISSING_OR_CHANGED');
  if (header.code) {
    for (const match of header.code.matchAll(/inline\s+constexpr\s+(?:int|float)\s+(\w+)\s*=\s*([-+]?\d+(?:\.\d+)?)(?:f)?\s*;/g)) {
      result.header_constants[match[1]] = {value: Number(match[2]), evidence: evidence(header, match.index, match[0].length)};
    }
  }
  for (const [key, value] of Object.entries(expected)) {
    if (result.header_constants[key]?.value !== value) result.reasons.push(`CONFIG_HEADER_MISMATCH_${key}`);
  }
  result.status = result.reasons.length ? 'UNVERIFIED' : 'SOURCE_CONFIRMED';
  return result;
}

function ingest(repo = '.') {
  const root = fs.realpathSync(path.resolve(repo));
  if (!fs.statSync(root).isDirectory()) throw Error('--repo must be a directory');
  const paths = [CONFIG, ...Object.keys(REVIEWED)].sort();
  const sources = new Map(paths.map(filename => [filename, readSource(root, filename)]));
  const checks = CHECKS.map(definition => inspectCheck(definition, sources));
  const checkById = new Map(checks.map(check => [check.id, check]));
  const configuration = inspectConfiguration(sources);
  const coreConfirmed = configuration.status === 'SOURCE_CONFIRMED' && CORE_CHECKS.every(id => checkById.get(id).status === 'SOURCE_CONFIRMED');
  const modules = MODULES.map(([id, recognizedRole, required, active]) => {
    const confirmed = coreConfirmed && required.every(key => checkById.get(key).status === 'SOURCE_CONFIRMED');
    const available = required.some(key => checkById.get(key).observations.some(observation => observation.found));
    return {
      id,
      recognized_profile_role: recognizedRole,
      availability: available ? 'SYMBOLS_OBSERVED' : 'UNVERIFIED',
      activation: confirmed ? (active ? 'ACTIVE_SOURCE_CONFIRMED' : 'NOT_ACTIVE_IN_RECOGNIZED_MAIN') : 'UNVERIFIED',
      check_ids: required,
      reasons: confirmed ? [] : [...(!coreConfirmed ? ['CORE_WIRING_OR_CONFIG_UNVERIFIED'] : []), ...required.filter(key => checkById.get(key).status !== 'SOURCE_CONFIRMED').map(key => `CHECK_${key}_UNVERIFIED`)]
    };
  });
  const files = [...sources.values()].map(({text, code, ...record}) => ({...record, matches_reviewed: record.status === 'READ' && record.reviewed_sha256 !== null ? record.normalized_sha256 === record.reviewed_sha256 : null}));
  const sensorCheck = checkById.get('sensors');
  return {
    schema_version: 1,
    format: 'roboproof-repository-ingestion',
    profile: PROFILE,
    status: coreConfirmed ? 'SUPPORTED_SOURCE_SNAPSHOT' : 'UNVERIFIED',
    scope: 'Conservative recognizer for the reviewed Nationals XDRIVE sources, not a general C++ analyzer or framework importer.',
    method: 'Inspect only supplied repo; require LF-normalized reviewed hashes and comment/string-masked symbol evidence. Any changed reviewed file requires re-audit, even a harmless edit. No repository code, Git hooks, or documentation claims are executed/trusted.',
    fingerprint: {algorithm: 'sha256', scope: 'Listed input file bytes and read statuses only; not the entire repository or build', value: hash(JSON.stringify(files.map(file => [file.path, file.sha256, file.status, file.reason || null])))},
    files,
    configuration,
    checks,
    modules,
    active_confirmed: modules.filter(module => module.activation === 'ACTIVE_SOURCE_CONFIRMED').map(module => module.id),
    available_modules: modules.filter(module => module.availability === 'SYMBOLS_OBSERVED').map(module => module.id),
    sensors: {
      status: coreConfirmed ? 'SOURCE_CONFIRMED' : 'UNVERIFIED',
      localization: coreConfirmed ? 'four_drive_motor_encoders' : null,
      external_localization_sensors: coreConfirmed ? [] : null,
      evidence: sensorCheck.observations.filter(observation => observation.found).map(observation => observation.evidence),
      check_ids: ['drive', 'sensors', 'odom', 'boundary']
    },
    limitations: [
      'SOURCE_CONFIRMED describes recognized source wiring, not a successful build, runtime reachability proof, hardware wiring or measured accuracy.',
      'Only listed sources are inspected; build flags, transitive dependencies, added files, runtime overrides and concurrent edits are not verified.',
      'Available symbol evidence does not establish activation; absent symbols do not prove that an arbitrary repository lacks a capability.',
      'Optional changed/missing modules remain UNVERIFIED independently. Changed core wiring suppresses all active architecture claims.',
      'Configuration values are declared inputs, not calibrated measurements. Linear/angular JSON gains are not all cascade gains.',
      'Simulator source recognition does not verify the WASM artifact, sensor parity, deterministic execution or Pedro/Foresight equivalence.'
    ]
  };
}

function args(argv) {
  const options = {repo: '.', out: 'roboproof/runs/repository.json'};
  const seen = new Set();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!['--repo', '--out'].includes(key) || seen.has(key) || !value || value.startsWith('--')) throw Error('Usage: node roboproof/ingest.js --repo <directory> --out <report.json>');
    options[key.slice(2)] = value;
    seen.add(key);
  }
  return options;
}

function main(argv = process.argv.slice(2)) {
  const options = args(argv);
  const report = ingest(options.repo);
  const destination = path.resolve(options.out);
  const repo = fs.realpathSync(path.resolve(options.repo));
  fs.mkdirSync(path.dirname(destination), {recursive: true});
  const resolvedOutput = path.join(fs.realpathSync(path.dirname(destination)), path.basename(destination));
  if (report.files.some(file => path.relative(path.join(repo, file.path), resolvedOutput) === '') || path.extname(destination) !== '.json') throw Error('--out must be a report .json, not an inspected source');
  if (fs.existsSync(destination) && (!fs.lstatSync(destination).isFile() || fs.lstatSync(destination).isSymbolicLink() || fs.statSync(destination).nlink > 1)) throw Error('--out must not be a link or non-file');
  fs.writeFileSync(destination, JSON.stringify(report, null, 2) + '\n');
  process.stdout.write(JSON.stringify({status: report.status, active_confirmed: report.active_confirmed, out: options.out}) + '\n');
  return report.status === 'SUPPORTED_SOURCE_SNAPSHOT' ? 0 : 2;
}

if (require.main === module) {
  try { process.exitCode = main(); }
  catch (error) { process.stderr.write(`Repository ingestion failed: ${error.message}\n`); process.exitCode = 1; }
}

module.exports = {ingest, args, main};
