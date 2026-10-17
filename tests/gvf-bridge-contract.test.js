'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {EventEmitter} = require('node:events');
const Policy = require('../simulator/policy');
const Gvf = require('../simulator/gvf-policy');
const filename = path.resolve(__dirname, '../roboproof/motion-bridge.js');
const source = fs.readFileSync(filename, 'utf8');

function fixture(args, cli = false) {
  const identity = {canonicalSimulator: true, sourceHash: 'original-runner-fixture'};
  const calls = [];
  const observation = {contractName: Gvf.CONTRACT_NAME, contractVersion: Gvf.CONTRACT_VERSION,
    observationSchema: Gvf.OBSERVATION_SCHEMA,
    ...Object.fromEntries(Gvf.OBSERVATION_FIELDS.map(([name, width]) => [name, width === 1 ? 0 : Array(width).fill(0)]))};
  const legacy = {legacy: true};
  let terminal = false;
  const transition = current => ({observation: current, vector: current === legacy ? Array(34).fill(7) : Gvf.vector(current),
    reward: 1, rewardComponents: {progress: 1}, terminated: terminal, truncated: false, info: {physicsTicks: 5}});
  const episode = {
    effortProxyVAs: 12, sim: {contactSeconds: 0.2},
    referenceAdapter: {duration: 9}, gvfResidualAdapter: {},
    reset(seed, configuration, task, options) {
      calls.push(['reset', seed, configuration, task, options]);
      this.options = options;
      return options.controlMode === 'gvf-residual' ? observation : legacy;
    },
    policyStep(action) { calls.push(['policyStep', action]); return transition(legacy); },
    gvfResidualStep(action) { calls.push(['gvfResidualStep', action]); return transition(observation); },
    observe() { calls.push(['observe']); return this.options.controlMode === 'gvf-residual' ? observation : legacy; },
    applyAction(action) { calls.push(['applyAction', action]); },
    neutral() { calls.push(['neutral']); },
    step() { calls.push(['step']); return transition(observation); },
    report() { return {options: this.options}; }
  };
  const moduleStub = {exports: {}};
  let loads = 0;
  const stdin = new EventEmitter();
  stdin.setEncoding = () => {};
  stdin.destroy = () => {};
  let stderr = '';
  let stdout = '';
  const processStub = {argv: ['node', filename, ...(args || [])], stdin,
    stderr: {write: value => { stderr += value; }}, stdout: {write: value => { stdout += value; }}};
  const requireStub = name => {
    if (name === '../simulator/headless') return {loadHeadless: () => {
      loads++;
      return {identity, createEpisode: () => episode};
    }};
    if (name === '../simulator/motion') return {FIXED_DT: 0.01, SCHEMA_VERSION: 2};
    if (name === '../simulator/policy') return {...Policy, vector: current => {
      assert.equal(current, legacy);
      return Array(34).fill(7);
    }};
    if (name === '../simulator/gvf-policy') return Gvf;
    throw Error(`Unexpected dependency: ${name}`);
  };
  requireStub.main = cli ? moduleStub : {};
  vm.runInNewContext(source, {require: requireStub, module: moduleStub, process: processStub, Buffer}, {filename});
  let sequence = 0;
  return {createBridge: moduleStub.exports.createBridge, calls, identity, episode, observation, stdin,
    request: (bridge, op, fields = {}) => bridge.handle({protocolVersion: 1, id: ++sequence, op, ...fields}).result,
    terminal: () => { terminal = true; },
    state: () => ({loads, stderr, stdout, exitCode: processStub.exitCode})};
}

const plain = value => JSON.parse(JSON.stringify(value));

test('default process retains legacy actions, options and reference duration', () => {
  const mock = fixture();
  const bridge = mock.createBridge();
  const contract = mock.request(bridge, 'contract');
  assert.equal(contract.actionSize, 4);
  assert.equal(contract.policyLimits, Policy.LIMITS);
  assert.equal(contract.contractName, undefined);
  const options = {policyIntervalTicks: 3, actionTimeoutTicks: 8, recordTransitions: false};
  const reset = mock.request(bridge, 'reset', {seed: 8, options});
  assert.equal(reset.referenceDurationSeconds, 9);
  assert.deepEqual(plain(reset.options), {...options, controlMode: 'policy'});
  const action = [1, 0, 0, 0];
  mock.request(bridge, 'step', {action});
  assert.equal(mock.calls.at(-1)[0], 'policyStep');
  assert.equal(mock.calls.at(-1)[1], action);
  assert.throws(() => mock.request(bridge, 'reset', {options: {controlMode: 'gvf-residual'}}), /Process-fixed/);
});

test('GVF contract declares distinct identity, all features, limits and cadence', () => {
  const mock = fixture();
  const bridge = mock.createBridge({controlMode: 'gvf-residual'});
  const contract = mock.request(bridge, 'contract');
  for (const [key, value] of Object.entries(Gvf.CONTRACT)) assert.equal(contract[key], value);
  assert.equal(contract.controlMode, 'gvf-residual');
  assert.equal(contract.policyContractVersion, 2);
  assert.equal(contract.policyContractName, 'gvf-residual');
  assert.equal(contract.actionSize, 3);
  assert.equal(contract.observationSize, 34);
  assert.equal(contract.observationFields, Gvf.OBSERVATION_FIELDS);
  assert.equal(contract.policyLimits, Gvf.LIMITS);
  assert.notEqual(contract.policyLimits, Policy.LIMITS);
  assert.deepEqual(plain(contract.zeroAction), [0, 0, 0]);
  assert.equal(contract.policyIntervalTicks, 5);
  assert.equal(contract.actionTimeoutTicks, 10);
  assert.equal(contract.identity, mock.identity);
});

test('GVF reset injects identity without altering caller options or policy provenance', () => {
  const mock = fixture();
  const bridge = mock.createBridge({controlMode: 'gvf-residual'});
  const policyIdentity = {kind: 'untrained', id: 'gvf-fixture'};
  const options = {policyIdentity, recordTransitions: false};
  const reset = mock.request(bridge, 'reset', {seed: 3, options});
  assert.deepEqual(plain(reset.options), {...options, controlMode: 'gvf-residual', policyContractVersion: 2,
    policyContractName: 'gvf-residual'});
  assert.deepEqual(options, {policyIdentity, recordTransitions: false});
  assert.equal(reset.options.policyIdentity, policyIdentity);
  assert.equal(reset.contractName, 'gvf-residual');
  assert.equal(reset.observationSchema, Gvf.OBSERVATION_SCHEMA);
  assert.equal(reset.referenceDurationSeconds, undefined);
  assert.equal(reset.vector.length, 34);
  const action = [0.1, -0.2, 0.3];
  const result = mock.request(bridge, 'step', {action});
  assert.deepEqual(plain(mock.calls.at(-1)), ['gvfResidualStep', action]);
  assert.equal(result.info.cumulativeEffortProxyVAs, 12);
  assert.equal(result.info.cumulativeContactSeconds, 0.2);
  assert.deepEqual(plain(result.info.rewardComponents), {progress: 1});
  assert.equal(result.contractVersion, 2);
  const observed = mock.request(bridge, 'observe');
  assert.equal(observed.observation, mock.observation);
  assert.equal(mock.calls.filter(call => call[0] === 'observe').length, 1);
  assert.equal(mock.request(bridge, 'report').identity, mock.identity);
});

test('GVF reset rejects every explicit identity mismatch before episode reset', () => {
  for (const options of [{controlMode: 'policy'}, {controlMode: null}, {policyContractVersion: 1},
    {policyContractVersion: null}, {policyContractName: 'policy'}, {policyContractName: null}, [], null]) {
    const mock = fixture();
    const bridge = mock.createBridge({controlMode: 'gvf-residual'});
    assert.throws(() => mock.request(bridge, 'reset', {options}), /mismatch|options/);
    assert.deepEqual(plain(mock.calls), [['neutral']]);
    assert.throws(() => mock.request(bridge, 'step', {action: [0, 0, 0]}), /Reset/);
  }
});

test('bad GVF actions stop and latch reset without slicing a legacy pace action', () => {
  for (const action of [[1, 0, 0, 0], [true, 0, 0], [NaN, 0, 0], [2, 0, 0], [0, 0], null, Array(3)]) {
    const mock = fixture();
    const bridge = mock.createBridge({controlMode: 'gvf-residual'});
    assert.throws(() => mock.request(bridge, 'step', {action: [0, 0, 0]}), /Reset/);
    mock.request(bridge, 'reset');
    assert.throws(() => mock.request(bridge, 'step', {action}), /action/);
    assert.deepEqual(plain(mock.calls.slice(-2)), [['applyAction', {type: 'stop'}], ['step']]);
    assert.equal(mock.calls.filter(call => call[0] === 'gvfResidualStep').length, 0);
    assert.throws(() => mock.request(bridge, 'step', {action: [0, 0, 0]}), /Reset/);
    mock.request(bridge, 'reset');
    mock.request(bridge, 'step', {action: [0, 0, 0]});
  }
});

test('GVF vectors require tags rather than equal length, and terminal/stop requires reset', () => {
  const mock = fixture();
  const bridge = mock.createBridge({controlMode: 'gvf-residual'});
  mock.request(bridge, 'reset');
  mock.request(bridge, 'stop');
  assert.throws(() => mock.request(bridge, 'step', {action: [0, 0, 0]}), /Reset/);
  mock.request(bridge, 'reset');
  mock.terminal();
  assert.equal(mock.request(bridge, 'step', {action: [0, 0, 0]}).terminated, true);
  assert.throws(() => mock.request(bridge, 'step', {action: [0, 0, 0]}), /Reset/);
  mock.observation.contractName = 'legacy-policy';
  assert.throws(() => mock.request(bridge, 'reset'), /Explicit GVF/);
  assert.throws(() => mock.request(bridge, 'step', {action: [0, 0, 0]}), /Reset/);
});

test('unknown selectors/CLI arguments reject before loading the mocked runner', () => {
  const mock = fixture();
  for (const controlMode of ['gvf', '', true, null]) assert.throws(() => mock.createBridge({controlMode}), /selector/);
  assert.throws(() => mock.createBridge({extra: true}), /selector/);
  assert.equal(mock.state().loads, 0);
  for (const args of [['--contract=gvf'], ['--secret=value'], ['--contract=gvf-residual', '--contract=policy']]) {
    const invalid = fixture(args, true);
    assert.equal(invalid.state().loads, 0);
    assert.equal(invalid.state().exitCode, 1);
    assert.match(invalid.state().stderr, /Unknown bridge argument/);
  }
  for (const args of [[], ['--contract=policy'], ['--contract=gvf-residual']]) {
    const valid = fixture(args, true);
    valid.stdin.emit('data', '{"protocolVersion":1,"id":1,"op":"contract"}\n');
    const reply = JSON.parse(valid.state().stdout);
    assert.equal(reply.ok, true);
    assert.equal(reply.result.actionSize, args[0] === '--contract=gvf-residual' ? 3 : 4);
    assert.equal(valid.state().loads, 1);
  }
});
