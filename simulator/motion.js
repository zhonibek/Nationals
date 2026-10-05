(function(root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./engine'), require('./policy'));
  else root.NationalsMotion = factory(root.NationalsEngine, root.NationalsPolicy);
})(globalThis, function(Engine, Policy) {
  'use strict';

  const SCHEMA_VERSION = 2;
  const FIXED_DT = 0.01;
  const METERS_PER_INCH = 0.0254;
  const RADIANS_PER_DEGREE = Math.PI / 180;
  const PARAMETERS = Object.freeze({
    massKg: [3.4, 13.6], moiKgM2: [0.0725, 0.58], muLong: [0.3, 1.1],
    muLat: [0.02, 0.15], kSlip: [325, 975], batteryInternalR: [0, 0.12]
  });
  const DEFAULT_TASK = Object.freeze({
    start: Object.freeze({xIn: 0, yIn: 0, headingDeg: 0}),
    goal: Object.freeze({xIn: 0, yIn: 24, headingDeg: 0}), deadlineSeconds: 10
  });
  const wrap = radians => Math.atan2(Math.sin(radians), Math.cos(radians));
  const copy = value => JSON.parse(JSON.stringify(value));

  function keys(value, allowed, label) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error(`${label} must be an object`);
    for (const key of Object.keys(value)) if (!allowed.includes(key)) throw Error(`Unsupported ${label}.${key}`);
  }

  function bounded(value, minimum, maximum, label) {
    if (!Number.isFinite(value) || value < minimum || value > maximum) throw Error(`${label} must be in [${minimum}, ${maximum}]`);
    return value;
  }

  function pose(value) {
    keys(value, ['xIn', 'yIn', 'headingDeg'], 'pose');
    return {
      xIn: bounded(value.xIn, -60, 60, 'xIn'), yIn: bounded(value.yIn, -60, 60, 'yIn'),
      headingDeg: bounded(value.headingDeg, -180, 180, 'headingDeg')
    };
  }

  function normalizeTask(task) {
    keys(task, ['start', 'goal', 'deadlineSeconds'], 'task');
    const normalized = {start: pose(task.start), goal: pose(task.goal),
      deadlineSeconds: bounded(task.deadlineSeconds, 0.01, 60, 'deadlineSeconds')};
    if (Math.abs(Math.round(normalized.deadlineSeconds / FIXED_DT) * FIXED_DT - normalized.deadlineSeconds) > 1e-9) throw Error('Deadline must be a multiple of 0.01 seconds');
    return normalized;
  }

  function configuration(seed, requested) {
    keys(requested, Object.keys(PARAMETERS), 'configuration');
    let state = seed;
    const resolved = {};
    for (const [key, limits] of Object.entries(PARAMETERS)) {
      const value = requested[key];
      if (value === undefined) continue;
      if (Array.isArray(value)) {
        if (value.length !== 2) throw Error(`${key} range must contain two values`);
        const low = bounded(value[0], ...limits, key);
        const high = bounded(value[1], low, limits[1], key);
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        resolved[key] = low + (high - low) * (state / 4294967296);
      } else resolved[key] = bounded(value, ...limits, key);
    }
    return resolved;
  }

  class MotionEpisode {
    constructor(controlFactory) {
      if (typeof controlFactory !== 'function') throw Error('An isolated production-controller factory is required');
      this.controlFactory = controlFactory;
    }

    reset(seed = 42, requested = {}, task = DEFAULT_TASK, requestedOptions = {}) {
      if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw Error('Seed must be a uint32');
      const normalized = normalizeTask(task);
      const deadlineTicks = Math.round(normalized.deadlineSeconds / FIXED_DT);
      const resolved = configuration(seed, requested);
      const episodeOptions = Policy.options(requestedOptions);
      const sim = new Engine.VexRobotSimulator(this.controlFactory());
      if (!sim.productionControl || !sim.override) throw Error('Original Simulator and production controller are required');
      Object.assign(sim, resolved);
      sim.resetSimulation();
      sim.setPose(normalized.start.xIn, normalized.start.yIn, normalized.start.headingDeg);
      this.sim = sim;
      this.seed = seed;
      this.task = normalized;
      this.configuration = Object.fromEntries(Object.keys(PARAMETERS).map(key => [key, sim[key]]));
      this.deadlineTicks = deadlineTicks;
      this.ticks = 0;
      this.settledTicks = 0;
      this.terminated = this.truncated = false;
      this.reason = 'running';
      this.previousDistance = this.metrics().positionErrorMeters;
      this.previousContact = 0;
      this.totalReward = this.effortProxyVAs = this.pathLengthMeters = 0;
      this.previousPosition = [sim.x, sim.y];
      this.previousAction = null;
      this.actions = [];
      this.transitions = [];
      this.options = episodeOptions;
      this.lastPolicyTick = null;
      this.policyCommand = null;
      this.pendingObservation = null;
      this.referenceAdapter = episodeOptions.controlMode === 'policy' ? new Policy.ReferenceAdapter(sim.productionControl,
        [normalized.start.xIn * METERS_PER_INCH, normalized.start.yIn * METERS_PER_INCH, normalized.start.headingDeg * RADIANS_PER_DEGREE],
        [normalized.goal.xIn * METERS_PER_INCH, normalized.goal.yIn * METERS_PER_INCH, normalized.goal.headingDeg * RADIANS_PER_DEGREE]) : null;
      const observation = this.observe();
      this.lastValidObservation = copy(observation);
      return observation;
    }

    requireActive() {
      if (!this.sim) throw Error('Reset the episode first');
      if (this.terminated || this.truncated) throw Error('Episode has ended; reset before stepping');
    }

    neutral() {
      const sim = this.sim;
      sim.isRunning = false;
      sim.currentAction = null;
      sim.routineQueue = [];
      sim.commandedWheelVoltages = null;
      sim.actionThrottle = sim.actionStrafe = sim.actionTurn = 0;
      sim.manualThrottle = sim.manualStrafe = sim.manualTurn = 0;
      this.policyCommand = null;
    }

    applyAction(action) {
      this.requireActive();
      const actionObservation = this.options.recordTransitions ? this.observe() : null;
      if (this.actions.length >= 6001) { this.neutral(); throw Error('Motion command budget exhausted'); }
      let normalized;
      try {
        if (action?.type === 'stop') {
          keys(action, ['type'], 'action');
          normalized = {type: 'stop'};
        } else if (this.options.controlMode === 'policy') {
          keys(action, ['type', 'values'], 'action');
          if (action.type !== 'policy') throw Error('Policy mode requires a bounded policy action or stop');
          if (this.ticks % this.options.policyIntervalTicks !== 0 || this.lastPolicyTick === this.ticks) throw Error('Policy action must occur once at a policy cadence boundary');
          normalized = {type: 'policy', values: Policy.normalizeAction(action.values)};
        } else {
          keys(action, ['type', 'xIn', 'yIn', 'headingDeg'], 'action');
          if (action.type !== 'pose') throw Error('Only existing pose commands and stop are supported in benchmark v1');
          normalized = {type: 'pose', ...pose({xIn: action.xIn, yIn: action.yIn, headingDeg: action.headingDeg})};
        }
      } catch (error) {
        this.neutral();
        if (this.options.controlMode === 'policy') {
          this.pendingObservation ??= actionObservation;
          this.previousAction = {type: 'stop'};
          this.actions.push({tick: this.ticks, action: {type: 'stop'}});
        }
        throw error;
      }
      if (normalized.type === 'stop') this.neutral();
      else if (normalized.type === 'policy') {
        if (!this.policyCommand) {
          this.policyCommand = {type: 'policyReference', referenceSI: [...this.referenceAdapter.reference],
            timeout: this.task.deadlineSeconds * 1000};
          this.sim.queueAction(this.policyCommand);
          this.sim.isRunning = true;
        }
        this.lastPolicyTick = this.ticks;
      }
      else {
        if (this.sim.isRunning) throw Error('A scripted motion is already active; do not reset the controller on policy ticks');
        this.sim.queueAction({type: 'pose', targetX: normalized.xIn, targetY: normalized.yIn, targetTheta: normalized.headingDeg});
        this.sim.isRunning = true;
      }
      this.previousAction = normalized;
      this.pendingObservation ??= actionObservation;
      this.actions.push({tick: this.ticks, action: copy(normalized)});
    }

    observe() {
      if (!this.sim) throw Error('Reset the episode first');
      const sim = this.sim;
      const heading = sim.odom.theta * RADIANS_PER_DEGREE;
      const deltaX = (this.task.goal.xIn - sim.odom.x) * METERS_PER_INCH;
      const deltaY = (this.task.goal.yIn - sim.odom.y) * METERS_PER_INCH;
      const headingError = wrap(this.task.goal.headingDeg * RADIANS_PER_DEGREE - heading);
      const observation = {
        pose: [sim.odom.x * METERS_PER_INCH, sim.odom.y * METERS_PER_INCH, heading],
        velocity: [...sim.odomVelocity], wheelRimSpeeds: sim.wheelOmega.map(speed => speed * sim.wheelRadiusM),
        targetError: [deltaX * Math.cos(heading) - deltaY * Math.sin(heading),
          deltaX * Math.sin(heading) + deltaY * Math.cos(heading), Math.sin(headingError), Math.cos(headingError)],
        previousAction: this.previousAction ? copy(this.previousAction) : null,
        remainingSeconds: (this.deadlineTicks - this.ticks) * FIXED_DT
      };
      if (this.referenceAdapter) observation.policyState = this.referenceAdapter.observe(this.lastPolicyTick === null ? 0 : this.ticks - this.lastPolicyTick);
      return observation;
    }

    metrics() {
      const sim = this.sim;
      return {
        positionErrorMeters: Math.hypot(this.task.goal.xIn - sim.x, this.task.goal.yIn - sim.y) * METERS_PER_INCH,
        headingErrorRadians: Math.abs(wrap((this.task.goal.headingDeg - sim.theta) * RADIANS_PER_DEGREE)),
        speedMetersPerSecond: Math.hypot(sim.Vx, sim.Vy), yawRateRadiansPerSecond: Math.abs(sim.w)
      };
    }

    step(dt = FIXED_DT) {
      this.requireActive();
      if (dt !== FIXED_DT) { this.neutral(); throw Error('The original Simulator benchmark requires fixed dt = 0.01'); }
      const before = this.options.recordTransitions ? this.pendingObservation ?? this.observe() : null;
      this.pendingObservation = null;
      let policyFault = null;
      if (this.options.controlMode === 'policy') {
        if (this.previousAction?.type === 'stop') { policyFault = 'stopped'; this.neutral(); }
        else if (this.lastPolicyTick === null || this.ticks - this.lastPolicyTick >= this.options.actionTimeoutTicks) {
          policyFault = 'stale_policy_action';
          this.neutral();
        } else if (this.policyCommand) this.policyCommand.referenceSI = this.referenceAdapter.advance(this.previousAction.values, FIXED_DT);
      }
      this.sim.update(FIXED_DT);
      this.ticks++;
      const measured = this.metrics();
      const finite = Object.values(measured).every(Number.isFinite) &&
        [...this.sim.motorVolts, ...this.sim.motorCurrents, ...this.observe().pose, ...this.sim.odomVelocity, ...this.sim.wheelOmega].every(Number.isFinite);
      const close = finite && measured.positionErrorMeters < 0.02032 && measured.headingErrorRadians < 0.035 &&
        measured.speedMetersPerSecond < 0.0254 && measured.yawRateRadiansPerSecond < 0.0873;
      this.settledTicks = close ? this.settledTicks + 1 : 0;
      const faults = ['InvalidInput', 'InvalidDt', 'SensorFault', 'ControlUnavailable'];
      if (policyFault) {
        this.terminated = true;
        this.reason = policyFault;
      } else if (!finite || faults.includes(this.sim.lastMotionResult)) {
        this.terminated = true;
        this.reason = 'controller_or_sensor_fault';
      } else if (this.settledTicks >= 15) {
        this.terminated = true;
        this.reason = 'success';
      } else if (this.ticks >= this.deadlineTicks) {
        this.truncated = true;
        this.reason = 'time_limit';
      }
      const effort = this.sim.motorVolts.reduce((sum, voltage, index) => sum + Math.abs(voltage * this.sim.motorCurrents[index]), 0) * FIXED_DT;
      const contact = this.sim.contactSeconds - this.previousContact;
      const rewardComponents = {
        progress: finite ? this.previousDistance - measured.positionErrorMeters : 0,
        time: -0.01 * FIXED_DT, effort: Number.isFinite(effort) ? -0.0001 * effort : 0,
        contact: -0.1 * contact, success: this.reason === 'success' ? 1 : 0,
        fault: ['controller_or_sensor_fault', 'stale_policy_action'].includes(this.reason) ? -1 : 0
      };
      const reward = Object.values(rewardComponents).reduce((sum, value) => sum + value, 0);
      this.totalReward += reward;
      this.effortProxyVAs += Number.isFinite(effort) ? effort : 0;
      this.pathLengthMeters += finite ? Math.hypot(this.sim.x - this.previousPosition[0], this.sim.y - this.previousPosition[1]) * METERS_PER_INCH : 0;
      this.previousDistance = measured.positionErrorMeters;
      this.previousPosition = [this.sim.x, this.sim.y];
      this.previousContact = this.sim.contactSeconds;
      if (this.terminated || this.truncated) this.neutral();
      const result = {observation: this.observe(), reward, rewardComponents, terminated: this.terminated,
        truncated: this.truncated, info: {reason: this.reason, ...measured}};
      if (this.referenceAdapter) {
        try {
          Policy.vector(result.observation);
          this.lastValidObservation = copy(result.observation);
          result.info.observationValid = true;
        } catch (error) {
          if (!this.terminated) throw error;
          result.observation = {...copy(this.lastValidObservation), remainingSeconds: (this.deadlineTicks - this.ticks) * FIXED_DT};
          result.info.observationValid = false;
          result.info.observationFallback = 'last valid sensor observation; unsuccessful terminal';
        }
      }
      if (this.options.recordTransitions) this.transitions.push({tick: this.ticks - 1, observation: copy(before),
        action: this.previousAction ? copy(this.previousAction) : null, reward, rewardComponents: {...rewardComponents},
        nextObservation: copy(result.observation), terminated: this.terminated, truncated: this.truncated,
        info: {...result.info}, evaluator: {poseInchesDegrees: [this.sim.x, this.sim.y, this.sim.theta],
          velocity: [this.sim.Vx, this.sim.Vy, this.sim.w], motorVolts: [...this.sim.motorVolts]}});
      return result;
    }

    policyStep(values) {
      if (this.options?.controlMode !== 'policy') throw Error('Reset in policy mode before policyStep');
      this.applyAction({type: 'policy', values});
      let result;
      let reward = 0;
      const rewardComponents = {};
      let ticks = 0;
      for (; ticks < this.options.policyIntervalTicks; ticks++) {
        result = this.step();
        reward += result.reward;
        for (const [name, value] of Object.entries(result.rewardComponents)) rewardComponents[name] = (rewardComponents[name] || 0) + value;
        if (result.terminated || result.truncated) { ticks++; break; }
      }
      return {...result, reward, rewardComponents, info: {...result.info, physicsTicks: ticks,
        elapsedSeconds: ticks * FIXED_DT}, vector: Policy.vector(result.observation)};
    }

    report() {
      if (!this.sim) throw Error('Reset the episode first');
      return {schemaVersion: SCHEMA_VERSION, taskType: 'reach-pose-benchmark', seed: this.seed, fixedDt: FIXED_DT,
        task: copy(this.task), configuration: {...this.configuration}, actions: copy(this.actions), ticks: this.ticks,
        options: copy(this.options), policyContractVersion: Policy.CONTRACT_VERSION, transitions: copy(this.transitions),
        terminated: this.terminated, truncated: this.truncated, reason: this.reason, totalReward: this.totalReward,
        metrics: {...this.metrics(), elapsedSeconds: this.ticks * FIXED_DT, effortProxyVAs: this.effortProxyVAs,
          contactSeconds: this.sim.contactSeconds, pathLengthMeters: this.pathLengthMeters},
        learningStatus: this.options.policyIdentity.kind === 'learned'
          ? 'learned checkpoint action stream; improvement requires independent frozen evaluation'
          : 'scripted or untrained benchmark; this episode does not train a motion policy'};
    }
  }

  return {MotionEpisode, SCHEMA_VERSION, FIXED_DT, DEFAULT_TASK, PARAMETERS, normalizeTask};
});
