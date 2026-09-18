/*
 * V5RC Override rules engine for the simulator.
 * Geometry is expressed in field inches with (0,0) at field center.
 * The scoring model follows the public VEX manual: 2026-2027, v2.0.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.OverrideGame = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const FIELD_WIDTH_IN = 140.40;
  const HALF = FIELD_WIDTH_IN / 2;
  const AUTO_SECONDS = 15;
  const DRIVER_SECONDS = 105;
  const MATCH_SECONDS = AUTO_SECONDS + DRIVER_SECONDS;
  const ENDGAME_SECONDS = 10;
  const MIDFIELD_HALF = 23.11;
  const GOAL_RADIUS_IN = 4.25;
  const POINTS = Object.freeze({ alliancePin: 5, yellowPin: 10, midfieldRobot: 8, autonomousBonus: 12 });

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const finite = (v, fallback = 0) => Number.isFinite(v) ? v : fallback;
  const centered = (x, y) => ({ x: x - HALF, y: y - HALF });
  const clone = value => JSON.parse(JSON.stringify(value));

  const GOAL_LAYOUT = [
    { id: "g-red-sw", type: "alliance", alliance: "red", x: 23.11, y: 23.11, height: 12 },
    { id: "g-red-w", type: "alliance", alliance: "red", x: 46.66, y: 46.66, height: 12 },
    { id: "g-neutral-se", type: "neutral", alliance: null, x: 93.75, y: 46.66, height: 12 },
    { id: "g-blue-se", type: "alliance", alliance: "blue", x: 117.30, y: 23.11, height: 12 },
    { id: "g-blue-e", type: "alliance", alliance: "blue", x: 117.30, y: 93.75, height: 12 },
    { id: "g-neutral-ne", type: "neutral", alliance: null, x: 93.75, y: 117.30, height: 12 },
    { id: "g-red-n", type: "alliance", alliance: "red", x: 46.66, y: 117.30, height: 12 },
    { id: "g-neutral-sw", type: "neutral", alliance: null, x: 23.11, y: 93.75, height: 12 },
    { id: "g-neutral-tall", type: "neutral", alliance: null, x: 70.20, y: 70.20, height: 24 }
  ];

  const TOGGLE_LAYOUT = [
    { id: "toggle-west", quadrant: "west", x: 0, y: HALF - 4.0, wall: "north" },
    { id: "toggle-east", quadrant: "east", x: 0, y: -HALF + 4.0, wall: "south" },
    { id: "toggle-south", quadrant: "south", x: -HALF + 4.0, y: 0, wall: "west" },
    { id: "toggle-north", quadrant: "north", x: HALF - 4.0, y: 0, wall: "east" }
  ];

  const LOADER_LAYOUT = [
    { id: "loader-red-west", alliance: "red", x: -58, y: 64, wall: "north" },
    { id: "loader-red-east", alliance: "red", x: 58, y: 64, wall: "north" },
    { id: "loader-blue-west", alliance: "blue", x: -58, y: -64, wall: "south" },
    { id: "loader-blue-east", alliance: "blue", x: 58, y: -64, wall: "south" }
  ];

  function makePins() {
    const pins = [];
    let sequence = 1;
    const add = (kind, allianceColor, location, x, y, visibleHalf) => {
      pins.push({
        id: "pin-" + String(sequence++).padStart(2, "0"),
        kind,
        color: kind === "yellow-yellow" ? "yellow" : allianceColor,
        allianceColor: allianceColor || null,
        halves: kind === "red-yellow" ? ["red", "yellow"] : kind === "blue-yellow" ? ["blue", "yellow"] : kind === "yellow-yellow" ? ["yellow", "yellow"] : ["red", "blue"],
        visibleHalf: visibleHalf || allianceColor || "yellow",
        visibleHalves: [visibleHalf || allianceColor || "yellow"],
        location,
        x, y,
        status: "field",
        placed: false,
        goalId: null,
        owner: null,
        stackIndex: 0
      });
    };
    for (let i = 0; i < 20; i++) {
      const x = -60 + (i % 5) * 8;
      const y = i < 10 ? 62 : 52 - Math.floor(i / 5) * 9;
      add("red-yellow", "red", i < 12 ? "alliance-station" : "field", x, y, i % 2 ? "yellow" : "red");
    }
    for (let i = 0; i < 20; i++) {
      const x = 60 - (i % 5) * 8;
      const y = i < 10 ? -62 : -52 + Math.floor(i / 5) * 9;
      add("blue-yellow", "blue", i < 12 ? "alliance-station" : "field", x, y, i % 2 ? "yellow" : "blue");
    }
    for (let i = 0; i < 19; i++) {
      const a = (i / 19) * Math.PI * 2;
      const radius = i < 4 ? 26 : 48;
      const p = centered(70.2 + Math.cos(a) * radius, 70.2 + Math.sin(a) * radius);
      add("yellow-yellow", null, i < 4 ? "goal" : "field", p.x, p.y, "yellow");
    }
    // Four red/blue pins are pre-placed around the midfield cup.
    for (let i = 0; i < 4; i++) {
      const a = i * Math.PI / 2;
      const p = { x: Math.cos(a) * 18, y: Math.sin(a) * 18 };
      add("red-blue", i % 2 === 0 ? "red" : "blue", "midfield", p.x, p.y, i % 2 === 0 ? "red" : "blue");
    }
    return pins;
  }

  function makeCups() {
    const cups = [];
    let sequence = 1;
    const add = (kind, location, x, y) => cups.push({
      id: "cup-" + String(sequence++).padStart(2, "0"),
      kind, location, x, y, status: "field", placed: false, goalId: null, stackIndex: 0
    });
    for (let i = 0; i < 20; i++) add("opaque", "alliance-station", -60 + (i % 10) * 12, i < 10 ? 66 : -66);
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      add("opaque", "field", Math.cos(a) * 42, Math.sin(a) * 42);
    }
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      add("transparent", "field", Math.cos(a) * 18, Math.sin(a) * 18);
    }
    return cups;
  }

  function makeRobots() {
    return [
      { id: "red-1", alliance: "red", x: -54, y: -54, theta: 0, width: 18, length: 18, height: 18, perimeterContact: true, midfield: false, possession: { pinId: null, cupId: null } },
      { id: "red-2", alliance: "red", x: -54, y: 54, theta: 0, width: 18, length: 18, height: 18, perimeterContact: true, midfield: false, possession: { pinId: null, cupId: null } },
      { id: "blue-1", alliance: "blue", x: 54, y: 54, theta: 180, width: 18, length: 18, height: 18, perimeterContact: true, midfield: false, possession: { pinId: null, cupId: null } },
      { id: "blue-2", alliance: "blue", x: 54, y: -54, theta: 180, width: 18, length: 18, height: 18, perimeterContact: true, midfield: false, possession: { pinId: null, cupId: null } }
    ];
  }

  class OverrideGame {
    constructor(options = {}) {
      this.world = options.world || "head-to-head";
      this.reset();
    }

    reset() {
      this.clock = 0;
      this.phase = "pre_match";
      this.matchEnded = false;
      this.postMatchSeconds = 0;
      this.violations = { red: false, blue: false };
      this.autonomousScores = { red: 0, blue: 0 };
      this.autonomousBonus = { red: 0, blue: 0 };
      this.awp = { red: false, blue: false };
      this.goals = GOAL_LAYOUT.map(g => {
        const p = centered(g.x, g.y);
        return { ...g, x: p.x, y: p.y, stack: [] };
      });
      this.toggles = TOGGLE_LAYOUT.map(t => ({ ...t, state: "yellow", seated: true, robotContact: false }));
      this.loaders = LOADER_LAYOUT.map(l => ({ ...l }));
      this.cups = makeCups();
      this.pins = makePins();
      this.robots = makeRobots();
      this.events = [];
      return this.getState();
    }

    startMatch() {
      if (this.phase !== "pre_match") return false;
      this.clock = 0;
      this.phase = "autonomous";
      this.matchEnded = false;
      return true;
    }

    startAutonomous() { return this.startMatch(); }

    stopMatch() {
      if (this.matchEnded) return;
      this.clock = MATCH_SECONDS;
      this.phase = "post_match";
      this.matchEnded = true;
      this.postMatchSeconds = 0;
      this.evaluateAutonomousBonus();
    }

    tick(dt) {
      dt = clamp(finite(dt), 0, 1);
      if (this.phase === "pre_match" || this.phase === "post_match") {
        if (this.phase === "post_match") this.postMatchSeconds = clamp(this.postMatchSeconds + dt, 0, 5);
        return this.getState();
      }
      this.clock = clamp(this.clock + dt, 0, MATCH_SECONDS);
      if (this.clock >= AUTO_SECONDS && this.phase === "autonomous") this.phase = "driver";
      if (this.clock >= MATCH_SECONDS) this.stopMatch();
      return this.getState();
    }

    setViolation(alliance, value = true) {
      if (alliance === "red" || alliance === "blue") this.violations[alliance] = Boolean(value);
    }

    setRobotPose(id, pose = {}) {
      const robot = this.robots.find(r => r.id === id);
      if (!robot) return { ok: false, error: "unknown robot" };
      robot.x = finite(pose.x, robot.x);
      robot.y = finite(pose.y, robot.y);
      robot.theta = finite(pose.theta, robot.theta);
      robot.perimeterContact = Boolean(pose.perimeterContact);
      robot.midfield = this.isInMidfield(robot);
      return { ok: true, robot: clone(robot) };
    }

    setRobotSize(id, size = {}) {
      const robot = this.robots.find(r => r.id === id);
      if (!robot) return { ok: false, error: "unknown robot" };
      const max = this.phase === "pre_match" ? 18 : 24;
      const width = finite(size.width, robot.width);
      const length = finite(size.length, robot.length);
      const height = finite(size.height, robot.height);
      if (width > max || length > max || height > (this.phase === "pre_match" ? 18 : 50)) return { ok: false, error: "robot envelope exceeds Override limit" };
      robot.width = Math.max(0, width);
      robot.length = Math.max(0, length);
      robot.height = Math.max(0, height);
      robot.midfield = this.isInMidfield(robot);
      return { ok: true, robot: clone(robot) };
    }

    isInMidfield(robot) {
      const half = MIDFIELD_HALF + Math.max(robot.width, robot.length) / 2;
      return Math.abs(robot.x) <= half && Math.abs(robot.y) <= half;
    }

    setPossession(robotId, possession = {}) {
      const robot = this.robots.find(r => r.id === robotId);
      if (!robot) return { ok: false, error: "unknown robot" };
      const pinId = possession.pinId || null;
      const cupId = possession.cupId || null;
      if (pinId && this.robots.some(r => r.id !== robotId && r.possession.pinId === pinId)) return { ok: false, error: "pin already possessed" };
      if (cupId && this.robots.some(r => r.id !== robotId && r.possession.cupId === cupId)) return { ok: false, error: "cup already possessed" };
      robot.possession = { pinId, cupId };
      return { ok: true, possession: { ...robot.possession } };
    }

    setToggle(id, state, options = {}) {
      const toggle = this.toggles.find(t => t.id === id);
      if (!toggle || !["red", "blue", "yellow"].includes(state)) return false;
      toggle.seated = options.seated !== false;
      toggle.robotContact = Boolean(options.robotContact);
      toggle.state = toggle.seated && !toggle.robotContact ? state : "yellow";
      return true;
    }

    goal(id) { return this.goals.find(g => g.id === id) || null; }
    pin(id) { return this.pins.find(p => p.id === id) || null; }
    cup(id) { return this.cups.find(c => c.id === id) || null; }

    placePin(pinId, goalId, options = {}) {
      const pin = this.pin(pinId);
      const goal = this.goal(goalId);
      if (!pin || !goal) return { ok: false, error: "unknown scoring object or goal" };
      if (pin.status === "placed" && !options.replace) return { ok: false, error: "pin already placed" };
      const stackIndex = goal.stack.length;
      pin.status = "placed";
      pin.placed = true;
      pin.goalId = goalId;
      pin.owner = options.owner || this.ownerForPin(pin, goal);
      if (options.visibleHalves) {
        pin.visibleHalves = options.visibleHalves.filter(half => pin.halves.includes(half));
        pin.visibleHalf = pin.visibleHalves[0] || pin.visibleHalf;
      } else if (options.visibleHalf) {
        pin.visibleHalf = options.visibleHalf;
        pin.visibleHalves = [options.visibleHalf];
      }
      pin.stackIndex = stackIndex;
      goal.stack.push({ type: "pin", id: pin.id });
      this.events.push({ type: "place-pin", pinId, goalId, clock: this.clock });
      return { ok: true, pin: clone(pin) };
    }

    placeCup(cupId, goalId, options = {}) {
      const cup = this.cup(cupId);
      const goal = this.goal(goalId);
      if (!cup || !goal) return { ok: false, error: "unknown scoring object or goal" };
      if (cup.status === "placed" && !options.replace) return { ok: false, error: "cup already placed" };
      cup.status = "placed";
      cup.placed = true;
      cup.goalId = goalId;
      cup.stackIndex = goal.stack.length;
      goal.stack.push({ type: "cup", id: cup.id });
      this.events.push({ type: "place-cup", cupId, goalId, clock: this.clock });
      return { ok: true, cup: clone(cup) };
    }

    ownerForPin(pin, goal) {
      if (!pin.halves.includes("yellow")) return pin.allianceColor || null;
      if (Math.abs(goal.x) <= MIDFIELD_HALF && Math.abs(goal.y) <= MIDFIELD_HALF) {
        const red = this.robots.filter(r => r.alliance === "red" && this.isInMidfield(r)).length;
        const blue = this.robots.filter(r => r.alliance === "blue" && this.isInMidfield(r)).length;
        return red === blue ? null : red > blue ? "red" : "blue";
      }
      const quadrant = goal.x < 0 ? (goal.y < 0 ? "west" : "north") : (goal.y < 0 ? "south" : "east");
      const toggle = this.toggles.find(t => t.quadrant === quadrant);
      return toggle && (toggle.state === "red" || toggle.state === "blue") ? toggle.state : null;
    }

    pinScore(pin, goal) {
      const totals = { red: 0, blue: 0 };
      if (!pin || !pin.placed || !goal) return { alliance: null, points: 0, totals };
      const visibleHalves = Array.isArray(pin.visibleHalves) ? pin.visibleHalves : [pin.visibleHalf];
      const yellowOwner = pin.halves.includes("yellow") ? this.ownerForPin(pin, goal) : null;
      for (const half of visibleHalves) {
        if (half === "red" || half === "blue") totals[half] += POINTS.alliancePin;
        else if (half === "yellow" && yellowOwner) totals[yellowOwner] += POINTS.yellowPin;
      }
      const alliances = Object.keys(totals).filter(alliance => totals[alliance] > 0);
      return {
        alliance: alliances.length === 1 ? alliances[0] : null,
        points: totals.red + totals.blue,
        totals
      };
    }
    scorePins(includeMidfield = true) {
      const score = { red: 0, blue: 0 };
      for (const goal of this.goals) {
        for (const item of goal.stack) {
          if (item.type !== "pin") continue;
          const result = this.pinScore(this.pin(item.id), goal);
          score.red += result.totals.red;
          score.blue += result.totals.blue;
        }
      }
      if (!includeMidfield) {
        for (const goal of this.goals.filter(g => Math.abs(g.x) <= MIDFIELD_HALF && Math.abs(g.y) <= MIDFIELD_HALF)) {
          for (const item of goal.stack) {
            if (item.type !== "pin") continue;
            const result = this.pinScore(this.pin(item.id), goal);
            score.red -= result.totals.red;
            score.blue -= result.totals.blue;
          }
        }
      }
      return score;
    }

    score() {
      const score = this.scorePins(true);
      for (const robot of this.robots) if (robot.midfield) score[robot.alliance] += POINTS.midfieldRobot;
      score.red += this.autonomousBonus.red;
      score.blue += this.autonomousBonus.blue;
      return score;
    }

    evaluateAutonomousBonus() {
      const score = this.scorePins(false);
      if (score.red === score.blue) this.autonomousBonus = { red: 6, blue: 6 };
      else this.autonomousBonus = score.red > score.blue ? { red: POINTS.autonomousBonus, blue: 0 } : { red: 0, blue: POINTS.autonomousBonus };
      if (this.violations.red && this.violations.blue) this.autonomousBonus = { red: 0, blue: 0 };
      else if (this.violations.red) this.autonomousBonus = { red: 0, blue: POINTS.autonomousBonus };
      else if (this.violations.blue) this.autonomousBonus = { red: POINTS.autonomousBonus, blue: 0 };
      return { ...this.autonomousBonus };
    }

    qualifiesAWP(alliance, worlds = false) {
      const requiredPins = worlds ? 7 : 6;
      const requiredGoals = worlds ? 3 : 2;
      const allianceRobots = this.robots.filter(r => r.alliance === alliance);
      const scoredPins = this.goals.reduce((n, goal) => n + goal.stack.filter(item => item.type === "pin" && this.pinScore(this.pin(item.id), goal).totals[alliance] > 0).length, 0);
      const goals = this.goals.filter(goal => goal.stack.filter(item => item.type === "pin" && this.pinScore(this.pin(item.id), goal).totals[alliance] > 0).length >= 2).length;
      const clearPerimeter = allianceRobots.every(r => !r.perimeterContact);
      const qualifies = scoredPins >= requiredPins && goals >= requiredGoals && clearPerimeter && !this.violations[alliance];
      this.awp[alliance] = qualifies;
      return qualifies;
    }

    getState() {
      return {
        field: { width: FIELD_WIDTH_IN, half: HALF, midfieldHalf: MIDFIELD_HALF },
        rules: { season: "2026-2027", manualVersion: "2.0", autoSeconds: AUTO_SECONDS, driverSeconds: DRIVER_SECONDS, endgameSeconds: ENDGAME_SECONDS, matchSeconds: MATCH_SECONDS, points: POINTS },
        phase: this.phase,
        clock: this.clock,
        endgame: this.clock >= MATCH_SECONDS - ENDGAME_SECONDS && this.clock < MATCH_SECONDS,
        matchEnded: this.matchEnded,
        goals: clone(this.goals),
        toggles: clone(this.toggles),
        loaders: clone(this.loaders),
        cups: clone(this.cups),
        pins: clone(this.pins),
        robots: clone(this.robots),
        score: this.score(),
        autonomousBonus: { ...this.autonomousBonus },
        awp: { ...this.awp }
      };
    }
  }

  OverrideGame.constants = Object.freeze({
    FIELD_WIDTH_IN, FIELD_HALF: HALF, AUTO_SECONDS, DRIVER_SECONDS, MATCH_SECONDS,
    ENDGAME_SECONDS, MIDFIELD_HALF, GOAL_RADIUS_IN, POINTS
  });
  OverrideGame.layouts = Object.freeze({
    goals: clone(GOAL_LAYOUT), toggles: clone(TOGGLE_LAYOUT), loaders: clone(LOADER_LAYOUT)
  });
  return OverrideGame;
});
