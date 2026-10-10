'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {loadHeadless} = require('../simulator/headless');
const {DEFAULT_TASK, FIXED_DT} = require('../simulator/motion');
const Policy = require('../simulator/policy');
const runner = loadHeadless();
const componentNames = ['progress', 'time', 'effort', 'contact', 'success', 'fault'];
const sum = values => values.reduce((total, value) => total + value, 0);
const distance = (goal, pose) => Math.hypot(goal.xIn - pose[0], goal.yIn - pose[1]) * 0.0254;

function assertNear(actual, expected, label) {
  assert(Number.isFinite(actual), `${label} must be finite`);
  assert(Number.isFinite(expected), `${label} expectation must be finite`);
  assert(Math.abs(actual - expected) <= 1e-12 * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} differs from ${expected}`);
}

function assertReportAccounting(report) {
  assert.equal(report.transitions.length, report.ticks);
  for (const transition of report.transitions) {
    assert.deepEqual(Object.keys(transition.rewardComponents), componentNames);
    assert(Object.values(transition.rewardComponents).every(Number.isFinite));
    assert.equal(transition.reward, sum(Object.values(transition.rewardComponents)));
  }
  assert.equal(report.totalReward, sum(report.transitions.map(transition => transition.reward)));
  const components = Object.fromEntries(componentNames.map(name =>
    [name, sum(report.transitions.map(transition => transition.rewardComponents[name]))]));
  assertNear(report.totalReward, sum(Object.values(components)), 'episode component sum');
  assertNear(components.time, -0.01 * report.metrics.elapsedSeconds, 'elapsed-time penalty');
  assertNear(components.effort, -0.0001 * report.metrics.effortProxyVAs, 'effort penalty');
  assertNear(components.contact, -0.1 * report.metrics.contactSeconds, 'contact penalty');
  assert.equal(components.success, report.reason === 'success' ? 1 : 0);
  return components;
}

test('original MotionEpisode accounts for each measured physics-tick component and report total', () => {
  const episode = runner.createEpisode();
  const task = {...DEFAULT_TASK, deadlineSeconds: 0.08};
  episode.reset(42, {}, task, {recordTransitions: true});
  episode.applyAction({type: 'pose', ...task.goal});
  let previousDistance = distance(task.goal, [task.start.xIn, task.start.yIn]);
  let previousContact = 0;
  let totalReward = 0;
  for (let tick = 0; tick < 8; tick++) {
    const result = episode.step();
    const measuredDistance = distance(task.goal, [episode.sim.x, episode.sim.y]);
    const effort = sum(episode.sim.motorVolts.map((voltage, index) =>
      Math.abs(voltage * episode.sim.motorCurrents[index]))) * FIXED_DT;
    assert.deepEqual(result.rewardComponents, {
      progress: previousDistance - measuredDistance,
      time: -0.01 * FIXED_DT,
      effort: -0.0001 * effort,
      contact: -0.1 * (episode.sim.contactSeconds - previousContact),
      success: 0,
      fault: 0
    });
    assert.equal(result.reward, sum(Object.values(result.rewardComponents)));
    assert.deepEqual(episode.transitions[tick].rewardComponents, result.rewardComponents);
    assert.equal(episode.transitions[tick].reward, result.reward);
    totalReward += result.reward;
    assert.equal(episode.totalReward, totalReward);
    previousDistance = measuredDistance;
    previousContact = episode.sim.contactSeconds;
  }
  const report = episode.report();
  const components = assertReportAccounting(report);
  assert(components.effort < 0, 'the real controller must incur measured motor effort');
  assert.equal(report.reason, 'time_limit');
  assert.equal(report.terminated, false);
  assert.equal(report.truncated, true);
});

test('real wall contact is charged only for newly measured contact time', () => {
  const episode = runner.createEpisode();
  const task = {start: {xIn: 60, yIn: 0, headingDeg: 0},
    goal: {xIn: 60, yIn: 0, headingDeg: 0}, deadlineSeconds: 0.03};
  episode.reset(42, {}, task, {recordTransitions: true});
  episode.sim.setPose(66, 0, 0);
  let previousContact = 0;
  for (let tick = 0; tick < 3; tick++) {
    const result = episode.step();
    const measuredContact = episode.sim.contactSeconds;
    assert.equal(result.rewardComponents.contact, -0.1 * (measuredContact - previousContact));
    if (tick === 0) {
      assert(measuredContact > 0, 'the original engine must resolve the perimeter overlap');
      assert(result.rewardComponents.contact < 0);
    } else {
      assert.equal(measuredContact, previousContact);
      assert.equal(Math.abs(result.rewardComponents.contact), 0);
    }
    previousContact = measuredContact;
  }
  const report = episode.report();
  const components = assertReportAccounting(report);
  assert(components.contact < 0);
  assert.equal(report.reason, 'time_limit');
  assert.equal(components.success, 0);
});

test('policyStep sums every component across decisions and a shortened terminal interval', () => {
  const episode = runner.createEpisode();
  episode.reset(42, {}, {...DEFAULT_TASK, deadlineSeconds: 0.08},
    {controlMode: 'policy', policyIntervalTicks: 5});
  const decisions = [];
  for (const expectedTicks of [5, 3]) {
    const firstTick = episode.ticks;
    const result = episode.policyStep([...Policy.DEFAULT_ACTION]);
    const transitions = episode.transitions.slice(firstTick);
    assert.equal(result.info.physicsTicks, expectedTicks);
    assert.equal(result.info.elapsedSeconds, expectedTicks * FIXED_DT);
    assert.equal(transitions.length, expectedTicks);
    assert.equal(result.reward, sum(transitions.map(transition => transition.reward)));
    assert.deepEqual(Object.keys(result.rewardComponents), componentNames);
    for (const name of componentNames) {
      assert.equal(result.rewardComponents[name], sum(transitions.map(transition => transition.rewardComponents[name])));
    }
    assertNear(result.reward, sum(Object.values(result.rewardComponents)), 'decision component sum');
    assert.equal(result.terminated, false);
    assert.equal(result.truncated, expectedTicks === 3);
    decisions.push(result);
  }
  const report = episode.report();
  assertReportAccounting(report);
  assertNear(sum(decisions.map(result => result.reward)), report.totalReward, 'decision reward total');
  const terminalReport = episode.report();
  assert.throws(() => episode.policyStep([...Policy.DEFAULT_ACTION]), /ended/);
  assert.deepEqual(episode.report(), terminalReport);
});

test('success bonus occurs once after 15 actual settled ticks and wins an exact deadline tie', () => {
  const episode = runner.createEpisode();
  const task = {...DEFAULT_TASK, goal: {...DEFAULT_TASK.start}, deadlineSeconds: 0.15};
  for (let reset = 0; reset < 2; reset++) {
    episode.reset(42, {}, task, {recordTransitions: true});
    assert.equal(episode.totalReward, 0);
    assert.equal(episode.settledTicks, 0);
    assert.equal(episode.transitions.length, 0);
    for (let tick = 0; tick < 15; tick++) {
      const result = episode.step();
      assert.equal(result.rewardComponents.progress, 0);
      assert.equal(result.rewardComponents.success, tick === 14 ? 1 : 0);
      assert.equal(result.terminated, tick === 14);
      assert.equal(result.truncated, false);
      assert.equal(result.info.reason, tick === 14 ? 'success' : 'running');
      if (tick < 14) assert(result.reward < 0, 'being physically close does not create an early bonus');
    }
    const report = episode.report();
    const components = assertReportAccounting(report);
    assert.equal(report.ticks, 15);
    assert.equal(report.reason, 'success');
    assert.equal(report.transitions.filter(transition => transition.rewardComponents.success !== 0).length, 1);
    assert.equal(components.success, 1);
    assertNear(report.totalReward, 1 - 0.01 * 15 * FIXED_DT, 'settled success reward');
    assert.throws(() => episode.step(), /ended/);
    assert.throws(() => episode.applyAction({type: 'stop'}), /ended/);
    assert.deepEqual(episode.report(), report);
  }
});

test('a deadline before the settling window pays no bonus even when already at the goal', () => {
  const episode = runner.createEpisode();
  const task = {...DEFAULT_TASK, goal: {...DEFAULT_TASK.start}, deadlineSeconds: 0.14};
  episode.reset(42, {}, task, {recordTransitions: true});
  for (let tick = 0; tick < 14; tick++) episode.step();
  const report = episode.report();
  const components = assertReportAccounting(report);
  assert.equal(episode.settledTicks, 14);
  assert.equal(report.metrics.positionErrorMeters, 0);
  assert.equal(report.metrics.headingErrorRadians, 0);
  assert.equal(report.metrics.speedMetersPerSecond, 0);
  assert.equal(report.metrics.yawRateRadiansPerSecond, 0);
  assert.equal(report.reason, 'time_limit');
  assert.equal(report.terminated, false);
  assert.equal(report.truncated, true);
  assert.equal(components.success, 0);
  assert(report.totalReward < 0);
});

test('policyStep includes one success bonus and stops before the remaining cadence ticks', () => {
  const episode = runner.createEpisode();
  const task = {...DEFAULT_TASK, goal: {...DEFAULT_TASK.start}, deadlineSeconds: 0.2};
  episode.reset(42, {}, task, {controlMode: 'policy', policyIntervalTicks: 20});
  const result = episode.policyStep([...Policy.DEFAULT_ACTION]);
  assert.equal(result.info.physicsTicks, 15);
  assert.equal(result.info.elapsedSeconds, 15 * FIXED_DT);
  assert.equal(result.info.reason, 'success');
  assert.equal(result.terminated, true);
  assert.equal(result.truncated, false);
  assert.equal(result.rewardComponents.success, 1);
  const report = episode.report();
  const components = assertReportAccounting(report);
  assert.deepEqual(result.rewardComponents, components);
  assert.equal(result.reward, report.totalReward);
});

test('distance progress is signed measured translation and telescopes, not traveled distance', () => {
  const task = {...DEFAULT_TASK, deadlineSeconds: 0.12};
  for (const direction of [1, -1]) {
    const episode = runner.createEpisode();
    episode.reset(42, {}, task, {recordTransitions: true});
    episode.applyAction({type: 'pose', xIn: 0, yIn: direction * task.goal.yIn, headingDeg: 0});
    for (let tick = 0; tick < 12; tick++) episode.step();
    const report = episode.report();
    const components = assertReportAccounting(report);
    const initialDistance = distance(task.goal, [task.start.xIn, task.start.yIn]);
    let previousDistance = initialDistance;
    for (const transition of report.transitions) {
      const measuredDistance = distance(task.goal, transition.evaluator.poseInchesDegrees);
      assert.equal(transition.info.positionErrorMeters, measuredDistance);
      assert.equal(transition.rewardComponents.progress, previousDistance - measuredDistance);
      previousDistance = measuredDistance;
    }
    assertNear(components.progress, initialDistance - report.metrics.positionErrorMeters, 'net distance progress');
    assert(direction * components.progress > 0, 'moving away must produce negative, unclipped progress');
    assert(report.metrics.pathLengthMeters > 0);
    if (direction < 0) assert.notEqual(components.progress, report.metrics.pathLengthMeters);
    assert.equal(report.reason, 'time_limit');
    assert.equal(components.success, 0);
  }
});

test('recording transitions does not change distance bookkeeping or cumulative reward', () => {
  const task = {...DEFAULT_TASK, deadlineSeconds: 0.08};
  const reports = [true, false].map(recordTransitions => {
    const episode = runner.createEpisode();
    episode.reset(42, {}, task, {recordTransitions});
    episode.applyAction({type: 'pose', ...task.goal});
    for (let tick = 0; tick < 8; tick++) episode.step();
    assert.equal(episode.previousDistance, episode.metrics().positionErrorMeters);
    return episode.report();
  });
  assertReportAccounting(reports[0]);
  assert.equal(reports[1].transitions.length, 0);
  assert.equal(reports[0].totalReward, reports[1].totalReward);
  assert.deepEqual(reports[0].metrics, reports[1].metrics);
  assert.equal(reports[0].reason, reports[1].reason);
});

test('positive shaped reward on an unfinished movement is not physical success', () => {
  const episode = runner.createEpisode();
  const task = {...DEFAULT_TASK, deadlineSeconds: 0.3};
  episode.reset(42, {}, task, {recordTransitions: true});
  episode.applyAction({type: 'pose', ...task.goal});
  for (let tick = 0; tick < 30; tick++) episode.step();
  const report = episode.report();
  const components = assertReportAccounting(report);
  assert(components.progress > 0);
  assert(report.totalReward > 0, 'translation shaping can pay without reaching the goal');
  assert(report.metrics.positionErrorMeters >= 0.02032);
  assert.equal(report.reason, 'time_limit');
  assert.equal(report.terminated, false);
  assert.equal(report.truncated, true);
  assert.equal(components.success, 0);
  assert.equal(components.fault, 0);
});

test('neither perfect sensor pose nor translation-only proximity can claim physical success', () => {
  const episode = runner.createEpisode();
  episode.reset(42, {}, {...DEFAULT_TASK, deadlineSeconds: 0.15}, {recordTransitions: true});
  episode.sim.odom.y = DEFAULT_TASK.goal.yIn;
  assert.equal(episode.observe().targetError[1], 0);
  for (let tick = 0; tick < 15; tick++) episode.step();
  let report = episode.report();
  let components = assertReportAccounting(report);
  assert.equal(components.progress, 0);
  assert.equal(components.success, 0);
  assert.equal(report.reason, 'time_limit');
  assert.equal(report.metrics.positionErrorMeters, DEFAULT_TASK.goal.yIn * 0.0254);
  const task = {...DEFAULT_TASK, goal: {...DEFAULT_TASK.start, headingDeg: 90}, deadlineSeconds: 0.15};
  episode.reset(42, {}, task, {recordTransitions: true});
  for (let tick = 0; tick < 15; tick++) episode.step();
  report = episode.report();
  components = assertReportAccounting(report);
  assert.equal(report.metrics.positionErrorMeters, 0);
  assert.equal(report.metrics.headingErrorRadians, Math.PI / 2);
  assert.equal(components.progress, 0);
  assert.equal(components.success, 0);
  assert.equal(report.reason, 'time_limit');
  assert.equal(report.terminated, false);
  assert.equal(report.truncated, true);
});

test('a stale policy terminal charges its fault once without a success bonus', () => {
  const episode = runner.createEpisode();
  episode.reset(42, {}, {...DEFAULT_TASK, deadlineSeconds: 0.01}, {controlMode: 'policy'});
  const result = episode.step();
  assert.equal(result.info.reason, 'stale_policy_action');
  assert.equal(result.terminated, true);
  assert.equal(result.truncated, false);
  assert.equal(result.rewardComponents.fault, -1);
  assert.equal(result.rewardComponents.success, 0);
  const report = episode.report();
  const components = assertReportAccounting(report);
  assert.equal(components.fault, -1);
  assert.throws(() => episode.step(), /ended/);
  assert.deepEqual(episode.report(), report);
});
