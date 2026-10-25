'use strict';

const OverrideGame = require('../simulator/override');

const MATCH_SECONDS = OverrideGame.constants.MATCH_SECONDS;
const MAX_FIELD_OBJECTS_PER_ROBOT = 6;
const FEATURE_COUNT = 23;
// Nominal seconds per mechanism action. `pickup` is a loose field object, `loaderPickup` an object at the loader outlet.
const ACTION_SECONDS = Object.freeze({pickup: 1.25, loaderPickup: 1.25, place: 1.5, toggle: 0.6, park: 0.5, load: 1.0});
// The robot waits this far in front of its loader (towards field centre) for a match load.
const LOADER_APPROACH_INCHES = 10;

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function cloneGame(game) {
  const copy = new OverrideGame({world: game.world, physical: false});
  Object.assign(copy, structuredClone(game));
  copy.physical = false;
  return copy;
}

function margin(game, alliance) {
  const score = game.score();
  return score[alliance] - score[alliance === 'red' ? 'blue' : 'red'];
}

function goalAllows(game, kind, goal) {
  if (goal.alliance && goal.alliance !== game._tacticalAlliance) return false;
  if (game.mode === 'match' && game.clock >= MATCH_SECONDS - OverrideGame.constants.ENDGAME_SECONDS && goal.id === 'g-neutral-tall') return false;
  const top = goal.stack.at(-1);
  return kind === 'pin' ? !top || top.type === 'cup' : Boolean(top && top.type === 'pin');
}

function loaderApproach(loader) {
  const length = Math.hypot(loader.x, loader.y) || 1;
  return {x: loader.x - loader.x / length * LOADER_APPROACH_INCHES, y: loader.y - loader.y / length * LOADER_APPROACH_INCHES};
}

function nextMatchLoad(game, alliance, kind) {
  return game[kind + 's'].find(object => object.location === 'alliance-station' && (object.alliance || object.allianceColor) === alliance);
}

function loaderOutletBusy(game, loader) {
  return [...game.pins, ...game.cups].some(object => object.location === 'field' && Math.hypot(object.x - loader.x, object.y - loader.y) < 3);
}

// Straight-line top speed of the X-drive from config/robot.json: wheel rim speed × √2.
function xDriveTopSpeed(robot = require('../config/robot.json')) {
  const rim = robot.cartridgeRpm * (robot.externalGearRatio || 1) * Math.PI * robot.wheelDiameterIn / 60;
  return robot.drive === 'XDRIVE' ? rim * Math.SQRT2 : rim;
}

// Fits mechanism times so that a typical cycle in this field geometry takes the measured time on the robot.
// A cycle is a round trip between a goal and its nearest pickup point plus handling; `placeSeconds` is shared.
function calibrateTiming({loaderCycleSeconds, fieldCycleSeconds, speed = xDriveTopSpeed(), placeSeconds = 0.4} = {}) {
  const game = new OverrideGame({world: 'head-to-head', physical: false});
  const distance = (first, second) => Math.hypot(first.x - second.x, first.y - second.y);
  const loaderTrip = Math.min(...game.loaders.map(loader => Math.min(...game.goals
    .filter(goal => !goal.alliance || goal.alliance === loader.alliance).map(goal => distance(loaderApproach(loader), goal)))));
  const fieldTrips = [...game.pins, ...game.cups].filter(object => object.location === 'field')
    .map(object => Math.min(...game.goals.map(goal => distance(game.objectPose(object), goal)))).sort((first, second) => first - second);
  const fieldTrip = fieldTrips[Math.floor(fieldTrips.length / 2)];
  const loaderPickup = loaderCycleSeconds - placeSeconds - 2 * loaderTrip / speed;
  const pickup = fieldCycleSeconds - placeSeconds - 2 * fieldTrip / speed;
  if (!(loaderPickup >= 0) || !(pickup >= 0)) throw Error('Measured cycle times are shorter than the travel time at this speed');
  return {movementSpeedInchesPerSecond: Number(speed.toFixed(3)),
    actionSeconds: {...ACTION_SECONDS, load: 0, loaderPickup: Number(loaderPickup.toFixed(3)), pickup: Number(pickup.toFixed(3)), place: placeSeconds},
    calibration: {loaderCycleSeconds, fieldCycleSeconds, loaderTripInches: Number(loaderTrip.toFixed(2)),
      fieldTripInches: Number(fieldTrip.toFixed(2)), placeSeconds}};
}

// `measured`: the team's robot timing (2026-10-25): about 2 s per loading-zone cycle and about 4 s per field cycle.
// `nominal`: the earlier uncalibrated assumptions (24 in/s, 1.25 s pickups), kept for comparison.
const TIMING_PROFILES = Object.freeze({
  measured: Object.freeze({source: 'team measurement on the competition robot, 2026-10-25',
    ...calibrateTiming({loaderCycleSeconds: 2, fieldCycleSeconds: 4})}),
  nominal: Object.freeze({source: 'uncalibrated nominal assumptions', movementSpeedInchesPerSecond: 24, actionSeconds: ACTION_SECONDS})
});

function bounded(value, denominator) {
  return Math.max(-1, Math.min(1, value / denominator));
}

function advanceTime(game, seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return false;
  let remaining = seconds;
  while (remaining > 1e-8) {
    const duration = Math.min(0.5, remaining);
    const previousClock = game.clock;
    game.tick(duration);
    if (game.clock <= previousClock) return false;
    remaining -= duration;
  }
  return true;
}

class TacticalEnvironment {
  constructor({timing = 'measured', movementSpeedInchesPerSecond, maximumDecisions = 60, actionSeconds = {}} = {}) {
    if (!Object.hasOwn(TIMING_PROFILES, timing)) throw Error(`Unknown tactical timing profile ${timing}`);
    const profile = TIMING_PROFILES[timing];
    this.timing = timing;
    movementSpeedInchesPerSecond ??= profile.movementSpeedInchesPerSecond;
    actionSeconds = {...profile.actionSeconds, ...actionSeconds};
    if (!Number.isFinite(movementSpeedInchesPerSecond) || movementSpeedInchesPerSecond < 1 || movementSpeedInchesPerSecond > 60) {
      throw Error('Tactical movement-speed assumption must be between 1 and 60 inches per second');
    }
    if (!Number.isInteger(maximumDecisions) || maximumDecisions < 1 || maximumDecisions > 100) {
      throw Error('Tactical decision budget must be between 1 and 100');
    }
    this.movementSpeedInchesPerSecond = movementSpeedInchesPerSecond;
    for (const [key, value] of Object.entries(actionSeconds)) {
      if (!(key in ACTION_SECONDS) || !Number.isFinite(value) || value < 0 || value > 20) throw Error(`Invalid tactical action time ${key}`);
    }
    this.actionSeconds = Object.freeze({...ACTION_SECONDS, ...actionSeconds});
    this.maximumDecisions = maximumDecisions;
    this.reset(1, 'red');
  }

  reset(seed = 1, alliance = 'red') {
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw Error('Tactical seed must be uint32');
    if (!['red', 'blue'].includes(alliance)) throw Error('Tactical alliance must be red or blue');
    const random = seededRandom(seed);
    this.game = new OverrideGame({world: 'head-to-head', physical: false});
    this.alliance = alliance;
    this.seed = seed;
    this.decisions = 0;
    this.done = false;
    for (const robot of this.game.robots) {
      const sign = robot.alliance === 'red' ? -1 : 1;
      const slot = robot.id.endsWith('1') ? -1 : 1;
      this.game.setRobotPose(robot.id, {x: sign * (61.2 - random() * 2), y: slot * (40 + (random() - 0.5) * 6), theta: sign < 0 ? 0 : 180});
    }
    this.game._tacticalAlliance = alliance;
    this.game.startMatch('match');
    if (!advanceTime(this.game, OverrideGame.constants.AUTO_SECONDS)) throw Error('Tactical environment failed to advance autonomous period');
    if (this.game.phase !== 'driver') throw Error('Tactical environment failed to enter driver period');
    this._prepared = new Map();
    return this.observe();
  }

  score() {
    const score = this.game.score();
    return {red: score.red, blue: score.blue, margin: margin(this.game, this.alliance)};
  }

  _travel(game, robotId, destination) {
    const robot = game.robots.find(candidate => candidate.id === robotId);
    if (!robot) return {ok: false, seconds: 0};
    const distance = Math.hypot(destination.x - robot.x, destination.y - robot.y);
    const seconds = distance / this.movementSpeedInchesPerSecond;
    if (game.clock + seconds >= MATCH_SECONDS) return {ok: false, seconds};
    if (!advanceTime(game, seconds)) return {ok: false, seconds};
    if (game.phase !== 'driver') return {ok: false, seconds};
    game.setRobotPose(robotId, {x: destination.x, y: destination.y, theta: robot.theta});
    return {ok: true, seconds};
  }

  _preview(action) {
    const game = cloneGame(this.game);
    game._tacticalAlliance = this.alliance;
    const beforeMargin = margin(game, this.alliance);
    const beforeClock = game.clock;
    let target = null;
    let goal = null;
    let robot = game.robots.find(candidate => candidate.id === action.robotId);
    if (!robot || robot.alliance !== this.alliance || game.phase !== 'driver') return null;

    if (action.kind === 'park') {
      const index = Number(robot.id.endsWith('2'));
      const destination = {x: robot.alliance === 'red' ? -9 : 9, y: index ? 9 : -9};
      const travel = this._travel(game, robot.id, destination);
      if (!travel.ok) return null;
      const moved = game.robots.find(candidate => candidate.id === robot.id);
      if (!moved.midfield) return null;
      if (!advanceTime(game, this.actionSeconds.park)) return null;
    } else if (action.kind === 'toggle') {
      target = game.toggles.find(candidate => candidate.id === action.targetId);
      if (!target || target.state === this.alliance) return null;
      const travel = this._travel(game, robot.id, target);
      if (!travel.ok) return null;
      const result = game.interact(robot.id, 'toggle', 'pin');
      if (!result.ok) return null;
      if (!advanceTime(game, this.actionSeconds.toggle)) return null;
    } else if (action.kind === 'score-pin' || action.kind === 'score-cup') {
      const kind = action.kind === 'score-pin' ? 'pin' : 'cup';
      const key = `${kind}Id`;
      target = game[kind](action.targetId);
      goal = game.goal(action.goalId);
      if (!target || !goal || !goalAllows(game, kind, goal)) return null;
      if (robot.possession[key] !== target.id) {
        if (target.status !== 'field' || target.location !== 'field') return null;
        const support = kind === 'cup' && game.pins.find(pin => pin.supportId === target.id && pin.status === 'field');
        if (support && robot.possession.pinId) return null;
        const targetPose = game.objectPose(target);
        const travel = this._travel(game, robot.id, targetPose);
        if (!travel.ok) return null;
        const pickup = game.interact(robot.id, 'pickup', kind);
        if (!pickup.ok || game.robots.find(candidate => candidate.id === robot.id).possession[key] !== target.id) return null;
        if (!advanceTime(game, this.actionSeconds.pickup)) return null;
      }
      robot = game.robots.find(candidate => candidate.id === action.robotId);
      const destination = {x: goal.x, y: goal.y};
      const travel = this._travel(game, robot.id, destination);
      if (!travel.ok) return null;
      const placement = game.interact(robot.id, 'place', kind);
      if (!placement.ok) return null;
      if (!advanceTime(game, this.actionSeconds.place)) return null;
    } else if (action.kind === 'load-score-pin' || action.kind === 'load-score-cup') {
      // Match load: drive to the alliance loader, receive the object through the rules engine,
      // pick it up at the outlet, carry it to the goal and place it.
      const kind = action.kind === 'load-score-pin' ? 'pin' : 'cup';
      const key = `${kind}Id`;
      const loader = game.loaders.find(candidate => candidate.id === action.loaderId && candidate.alliance === this.alliance);
      goal = game.goal(action.goalId);
      if (!loader || !goal || robot.possession[key] || !goalAllows(game, kind, goal)) return null;
      const approach = this._travel(game, robot.id, loaderApproach(loader));
      if (!approach.ok) return null;
      const onField = new Set(game[kind + 's'].filter(object => object.location === 'field').map(object => object.id));
      if (!game.interact(robot.id, 'load', kind).ok) return null;
      target = game[kind + 's'].find(object => object.location === 'field' && !onField.has(object.id));
      if (!target || !advanceTime(game, this.actionSeconds.load)) return null;
      // The intake reaches the outlet from the approach point, so pickup takes the loaded object, not a nearby field one.
      const waiting = game.robots.find(candidate => candidate.id === robot.id);
      const waitingPose = {x: waiting.x, y: waiting.y, theta: waiting.theta};
      game.setRobotPose(robot.id, {x: target.x, y: target.y, theta: waiting.theta});
      const pickup = game.interact(robot.id, 'pickup', kind);
      game.setRobotPose(robot.id, waitingPose);
      if (!pickup.ok || game.robots.find(candidate => candidate.id === robot.id).possession[key] !== target.id) return null;
      if (!advanceTime(game, this.actionSeconds.loaderPickup)) return null;
      const travel = this._travel(game, robot.id, {x: goal.x, y: goal.y});
      if (!travel.ok) return null;
      const placement = game.interact(robot.id, 'place', kind);
      if (!placement.ok) return null;
      if (!advanceTime(game, this.actionSeconds.place)) return null;
    } else {
      return null;
    }

    if (game.clock >= MATCH_SECONDS && !game.finalScore) game.tick(0.01);
    return {game, reward: margin(game, this.alliance) - beforeMargin, durationSeconds: game.clock - beforeClock,
      score: game.score(), margin: margin(game, this.alliance)};
  }

  _rawActions() {
    const actions = [];
    const goals = this.game.goals.filter(goal => !goal.alliance || goal.alliance === this.alliance);
    const robots = this.game.robots.filter(robot => robot.alliance === this.alliance && !this.game.disqualifiedRobots.includes(robot.id));
    for (const robot of robots) {
      for (const kind of ['pin', 'cup']) {
        const heldId = robot.possession[`${kind}Id`];
        let targets = heldId ? [this.game[kind](heldId)].filter(Boolean) :
          this.game[kind + 's'].filter(object => object.status === 'field' && object.location === 'field')
            .sort((first, second) => Math.hypot(first.x - robot.x, first.y - robot.y) - Math.hypot(second.x - robot.x, second.y - robot.y))
            .slice(0, MAX_FIELD_OBJECTS_PER_ROBOT);
        if (kind === 'cup' && robot.possession.pinId) {
          targets = targets.filter(cup => !this.game.pins.some(pin => pin.supportId === cup.id && pin.status === 'field'));
        }
        for (const target of targets) {
          const targetPose = this.game.objectPose(target);
          for (const goal of goals) {
            if (!goalAllows(this.game, kind, goal)) continue;
            const action = {id: `score-${kind}:${robot.id}:${target.id}:${goal.id}`, kind: `score-${kind}`,
              robotId: robot.id, targetId: target.id, goalId: goal.id};
            const distanceToTarget = heldId ? 0 : Math.hypot(targetPose.x - robot.x, targetPose.y - robot.y);
            const distanceToGoal = Math.hypot(goal.x - targetPose.x, goal.y - targetPose.y);
            if (this.game.clock + (distanceToTarget + distanceToGoal) / this.movementSpeedInchesPerSecond +
                this.actionSeconds.pickup + this.actionSeconds.place >= MATCH_SECONDS) continue;
            if (!heldId && this.game[kind + 's'].some(other => other.id !== target.id && other.status === 'field' &&
                other.location === 'field' && Math.hypot(other.x - targetPose.x, other.y - targetPose.y) < 1e-6)) continue;
            action._features = this._features(action, {distanceToTarget, distanceToGoal, target, goal, robot, kind});
            actions.push(action);
          }
        }
      }
      for (const kind of ['pin', 'cup']) {
        if (robot.possession[`${kind}Id`]) continue;
        const load = nextMatchLoad(this.game, this.alliance, kind);
        if (!load) continue;
        for (const loader of this.game.loaders.filter(candidate => candidate.alliance === this.alliance)) {
          if (loaderOutletBusy(this.game, loader)) continue;
          const approach = loaderApproach(loader);
          const distanceToTarget = Math.hypot(approach.x - robot.x, approach.y - robot.y);
          for (const goal of goals) {
            if (!goalAllows(this.game, kind, goal)) continue;
            const distanceToGoal = Math.hypot(goal.x - approach.x, goal.y - approach.y);
            if (this.game.clock + (distanceToTarget + distanceToGoal) / this.movementSpeedInchesPerSecond +
                this.actionSeconds.load + this.actionSeconds.loaderPickup + this.actionSeconds.place >= MATCH_SECONDS) continue;
            const action = {id: `load-score-${kind}:${robot.id}:${loader.id}:${goal.id}`, kind: `load-score-${kind}`,
              robotId: robot.id, loaderId: loader.id, goalId: goal.id};
            action._features = this._features(action, {distanceToTarget, distanceToGoal, target: load, goal, robot, kind, fromLoader: true});
            actions.push(action);
          }
        }
      }
      for (const toggle of this.game.toggles) {
        if (toggle.state === this.alliance) continue;
        const action = {id: `toggle:${robot.id}:${toggle.id}`, kind: 'toggle', robotId: robot.id, targetId: toggle.id};
        if (this.game.clock + Math.hypot(toggle.x - robot.x, toggle.y - robot.y) / this.movementSpeedInchesPerSecond + this.actionSeconds.toggle >= MATCH_SECONDS) continue;
        action._features = this._features(action, {distanceToTarget: Math.hypot(toggle.x - robot.x, toggle.y - robot.y), robot, toggle});
        actions.push(action);
      }
      if (!robot.midfield) {
        const action = {id: `park:${robot.id}`, kind: 'park', robotId: robot.id};
        const destination = {x: robot.alliance === 'red' ? -9 : 9, y: robot.id.endsWith('2') ? 9 : -9};
        if (this.game.clock + Math.hypot(destination.x - robot.x, destination.y - robot.y) / this.movementSpeedInchesPerSecond + this.actionSeconds.park >= MATCH_SECONDS) continue;
        action._features = this._features(action, {distanceToTarget: Math.hypot(destination.x - robot.x, destination.y - robot.y), robot});
        actions.push(action);
      }
    }
    return actions;
  }

  _features(action, details) {
    const game = this.game;
    const robot = details.robot;
    const score = game.score();
    const controlled = score[this.alliance] - score[this.alliance === 'red' ? 'blue' : 'red'];
    const type = action.kind.endsWith('score-pin') ? 0 : action.kind.endsWith('score-cup') ? 1 : action.kind === 'toggle' ? 2 : 3;
    const features = Array(FEATURE_COUNT).fill(0);
    features[type] = 1;
    features[4] = this.alliance === 'red' ? -1 : 1;
    features[5] = details.target?.halves?.filter(half => half === 'red').length / 2 || 0;
    features[6] = details.target?.halves?.filter(half => half === 'blue').length / 2 || 0;
    features[7] = details.target?.halves?.filter(half => half === 'yellow').length / 2 || 0;
    features[8] = details.target?.up === 'opaque' ? 1 : details.target?.up === 'transparent' ? -1 : 0;
    const distanceToTarget = Number.isFinite(details.distanceToTarget) ? details.distanceToTarget : 0;
    const distanceToGoal = Number.isFinite(details.distanceToGoal) ? details.distanceToGoal : 0;
    features[9] = Math.min(1, distanceToTarget / 144);
    features[10] = Math.min(1, distanceToGoal / 144);
    features[11] = Math.min(1, (distanceToTarget + distanceToGoal) / this.movementSpeedInchesPerSecond / 105);
    features[12] = Math.max(0, Math.min(1, (MATCH_SECONDS - game.clock) / OverrideGame.constants.DRIVER_SECONDS));
    features[13] = bounded(controlled, 100);
    features[14] = robot.possession.pinId ? 1 : 0;
    features[15] = robot.possession.cupId ? 1 : 0;
    features[16] = details.goal ? Math.min(1, details.goal.stack.length / 8) : 0;
    features[17] = details.goal?.alliance === this.alliance ? 1 : details.goal ? 0 : -1;
    features[18] = details.toggle ? (details.toggle.state === this.alliance ? 1 : details.toggle.state === 'yellow' ? 0 : -1) : 0;
    features[19] = details.goal ? bounded(details.goal.x, 72) : 0;
    features[20] = details.goal ? bounded(details.goal.y, 72) : 0;
    features[21] = game.clock >= MATCH_SECONDS - OverrideGame.constants.ENDGAME_SECONDS ? 1 : 0;
    features[22] = details.fromLoader ? 1 : 0;
    return features;
  }

  observe() {
    const score = this.score();
    const raw = this._rawActions();
    const actions = !this.done && this.decisions < this.maximumDecisions && this.game.phase === 'driver'
      ? raw.map(action => ({id: action.id, features: action._features})) : [];
    const terminal = this.done || this.decisions >= this.maximumDecisions || this.game.phase !== 'driver' || actions.length === 0;
    if (terminal && !this.done && this.game.phase === 'driver') {
      advanceTime(this.game, Math.max(0.01, MATCH_SECONDS - this.game.clock));
    }
    this.done = terminal;
    this._prepared = new Map(raw.map(action => [action.id, action]));
    return {schemaVersion: 1, seed: this.seed, alliance: this.alliance, clock: this.game.clock,
      score, decisions: this.decisions, done: this.done, timing: this.timing, movementModel: 'teleport macro-actions with explicit nominal travel-time proxy; not a motion/physics result',
      actions};
  }

  // Engine preview of every current candidate: points gained (score-margin change) and seconds used.
  // Read-only: each preview runs on a cloned game.
  previewCandidates() {
    const rows = [];
    for (const action of this._prepared.values()) {
      const preview = this._preview(action);
      rows.push({id: action.id, kind: action.kind, robotId: action.robotId, targetId: action.targetId ?? null,
        loaderId: action.loaderId ?? null, goalId: action.goalId ?? null, legal: Boolean(preview),
        points: preview ? preview.reward : null, seconds: preview ? Number(preview.durationSeconds.toFixed(3)) : null,
        pointsPerSecond: preview && preview.durationSeconds > 0 ? Number((preview.reward / preview.durationSeconds).toFixed(4)) : null});
    }
    return rows;
  }

  matchLoadsRemaining() {
    const count = kind => this.game[kind + 's'].filter(object => object.location === 'alliance-station' &&
      (object.alliance || object.allianceColor) === this.alliance).length;
    return {pins: count('pin'), cups: count('cup')};
  }

  step(actionId) {
    if (this.done) throw Error('Tactical episode is already complete');
    if (typeof actionId !== 'string' || !this._prepared.has(actionId)) throw Error('Action is not in the current legal candidate set');
    const before = this.score();
    const action = this._prepared.get(actionId);
    const result = this._preview(action);
    if (!result) {
      advanceTime(this.game, 0.5);
      this.decisions += 1;
      const observation = this.observe();
      return {observation, reward: -2, durationSeconds: 0.5, score: this.score(), done: observation.done, rejectedAction: true};
    }
    this.game = result.game;
    this.decisions += 1;
    const after = this.score();
    const observation = this.observe();
    return {observation, reward: after.margin - before.margin, durationSeconds: result.durationSeconds,
      score: after, done: observation.done};
  }
}

module.exports = {TacticalEnvironment, FEATURE_COUNT, MAX_FIELD_OBJECTS_PER_ROBOT, ACTION_SECONDS, TIMING_PROFILES, calibrateTiming,
  xDriveTopSpeed, cloneGame, loaderApproach};
