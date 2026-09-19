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
  const MIDFIELD_HALF = 23.11; // Diamond vertex distance, not axis-aligned half-width.
  const GOAL_RADIUS_IN = 4.25;
  const POINTS = Object.freeze({ alliancePin: 5, yellowPin: 10, midfieldRobot: 8, autonomousBonus: 12 });

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const finite = (v, fallback = 0) => Number.isFinite(v) ? v : fallback;
  const centered = (x, y) => ({ x: x - HALF, y: y - HALF });
  const clone = value => JSON.parse(JSON.stringify(value));

  // FO-1/FO-2 arrangement. Tile coordinates approximate Appendix A tolerances.
  const GOAL_LAYOUT = [
    {id:"g-red-sw",alliance:"red",x:23.11,y:46.66,height:3.25},
    {id:"g-red-s",alliance:"red",x:46.66,y:23.11,height:3.25},
    {id:"g-blue-ne",alliance:"blue",x:117.29,y:93.74,height:3.25},
    {id:"g-blue-n",alliance:"blue",x:93.74,y:117.29,height:3.25},
    {id:"g-neutral-nw",alliance:null,x:46.66,y:117.29,height:5.8},
    {id:"g-neutral-w",alliance:null,x:23.11,y:93.74,height:5.8},
    {id:"g-neutral-se",alliance:null,x:93.74,y:23.11,height:5.8},
    {id:"g-neutral-e",alliance:null,x:117.29,y:46.66,height:5.8},
    {id:"g-neutral-tall",alliance:null,x:70.2,y:70.2,height:8.7}
  ].map(g=>({...g,type:g.alliance?"alliance":"neutral"}));
  const TOGGLE_LAYOUT = [
    {id:"toggle-north",quadrant:"north",x:0,y:HALF,wall:"north"},
    {id:"toggle-south",quadrant:"south",x:0,y:-HALF,wall:"south"},
    {id:"toggle-west",quadrant:"west",x:-HALF,y:0,wall:"west"},
    {id:"toggle-east",quadrant:"east",x:HALF,y:0,wall:"east"}
  ];
  const LOADER_LAYOUT=[
    {id:"loader-red-n",alliance:"red",x:-64,y:60},
    {id:"loader-red-s",alliance:"red",x:-64,y:-60},
    {id:"loader-blue-n",alliance:"blue",x:64,y:60},
    {id:"loader-blue-s",alliance:"blue",x:64,y:-60}
  ];
  function inventory(){
    const pins=[],cups=[];
    const pin=(halves,x,y,location="field",alliance=null,goalId=null)=>{
      const id="pin-"+String(pins.length+1).padStart(2,"0");
      pins.push({id,kind:halves.join("-"),halves,color:halves[0],allianceColor:alliance,
        x,y,location,status:goalId?"placed":"field",placed:!!goalId,goalId,upIndex:0,visibleHalves:null});return id;
    };
    const cup=(x,y,up="opaque",location="field",alliance=null)=>{
      cups.push({id:"cup-"+String(cups.length+1).padStart(2,"0"),kind:"cup",halves:["opaque","transparent"],
        up,x,y,location,alliance,status:"field",placed:false,goalId:null});
    };
    for(const [color,side] of [["red",-1],["blue",1]]){
      for(let i=0;i<12;i++)pin([color,"yellow"],side*78,55-i*10,i<2?"preload":"alliance-station",color);
      pin(["yellow","yellow"],side*84,0,"alliance-station",color);
      for(let i=0;i<10;i++)cup(side*88,45-i*10,"opaque","alliance-station",color);
    }
    for(const [x,y] of [[-47.09,47.09],[-23.54,23.54],[23.54,-23.54],[47.09,-47.09]]){
      cup(x,y,"transparent");
      for(const [dx,dy,color] of [[0,5,"blue"],[5,0,"blue"],[0,-5,"red"],[-5,0,"red"]])pin([color,"yellow"],x+dx,y+dy);
    }
    for(const a of [-23.54,23.54])for(const side of [-1,1]){
      for(const d of [-8,0,8]){cup(a+d,side*68);cup(side*68,a+d);}
      pin(["yellow","yellow"],a,side*68);pin(["yellow","yellow"],side*68,a);
    }
    for(const a of [-47.09,-23.54,23.54,47.09]){cup(a,a,"transparent");pin(["yellow","yellow"],a,a);}
    for(const [x,y,color] of [[-23.54,0,"red"],[0,-23.54,"red"],[23.54,0,"blue"],[0,23.54,"blue"]]){
      cup(x,y,"transparent");pin([color,color==="red"?"blue":"red"],x,y);
    }
    for(const g of GOAL_LAYOUT.filter(g=>!g.alliance)){const p=centered(g.x,g.y);pin(["yellow","yellow"],p.x,p.y,"goal",null,g.id);}
    return {pins,cups};
  }

  function makeRobots() {
    return [
      { id: "red-1", alliance: "red", x: -61.2, y: -40, theta: 0, width: 18, length: 18, height: 18, perimeterContact: true, midfield: false, possession: { pinId: null, cupId: null } },
      { id: "red-2", alliance: "red", x: -61.2, y: 40, theta: 0, width: 18, length: 18, height: 18, perimeterContact: true, midfield: false, possession: { pinId: null, cupId: null } },
      { id: "blue-1", alliance: "blue", x: 61.2, y: 40, theta: 180, width: 18, length: 18, height: 18, perimeterContact: true, midfield: false, possession: { pinId: null, cupId: null } },
      { id: "blue-2", alliance: "blue", x: 61.2, y: -40, theta: 180, width: 18, length: 18, height: 18, perimeterContact: true, midfield: false, possession: { pinId: null, cupId: null } }
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
      const objects=inventory();this.cups=objects.cups;this.pins=objects.pins;
      for(const pin of this.pins.filter(p=>p.placed))this.goal(pin.goalId).stack.push({type:"pin",id:pin.id});
      this.autoFrozen=false;this.finalScore=null;
      this.robots = makeRobots();
      for(const robot of this.robots){
        const preload=this.pins.find(p=>p.location==="preload"&&p.allianceColor===robot.alliance&&p.status!=="held");
        preload.status="held";robot.possession.pinId=preload.id;
      }
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
      if(!this.autoFrozen)this.freezeAutonomous();
      this.finalScore=this.score();
    }

    tick(dt) {
      dt = clamp(finite(dt), 0, 1);
      if (this.phase === "pre_match" || this.phase === "post_match") {
        if (this.phase === "post_match") this.postMatchSeconds = clamp(this.postMatchSeconds + dt, 0, 5);
        return this.getState();
      }
      this.clock = clamp(this.clock + dt, 0, MATCH_SECONDS);
      if (this.clock >= AUTO_SECONDS && this.phase === "autonomous") {this.freezeAutonomous();this.phase = "driver";}
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
      robot.perimeterContact = this.touchesPerimeter(robot);
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
      if (width <= 0 || length <= 0 || height <= 0 || width > max || length > max || height > (this.phase === "pre_match" ? 18 : 50)) return { ok: false, error: "invalid robot envelope" };
      robot.width = Math.max(0, width);
      robot.length = Math.max(0, length);
      robot.height = Math.max(0, height);
      robot.midfield = this.isInMidfield(robot);
      robot.perimeterContact = this.touchesPerimeter(robot);
      return { ok: true, robot: clone(robot) };
    }

    touchesPerimeter(robot) {
      const a=robot.theta*Math.PI/180,c=Math.abs(Math.cos(a)),s=Math.abs(Math.sin(a));
      return Math.abs(robot.x)+(robot.width*c+robot.length*s)/2>=HALF-1e-6 ||
        Math.abs(robot.y)+(robot.width*s+robot.length*c)/2>=HALF-1e-6;
    }

    isInMidfield(robot) {
      // Separating axis test: rotated robot rectangle against the midfield diamond.
      const a=robot.theta*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
      const body=[[-1,-1],[-1,1],[1,1],[1,-1]].map(([u,v])=>({x:robot.x+u*robot.width/2*c+v*robot.length/2*s,y:robot.y-u*robot.width/2*s+v*robot.length/2*c}));
      const diamond=[{x:MIDFIELD_HALF,y:0},{x:0,y:MIDFIELD_HALF},{x:-MIDFIELD_HALF,y:0},{x:0,y:-MIDFIELD_HALF}];
      for(const [nx,ny] of [[1,1],[1,-1],[c,-s],[s,c]]){
        const b=body.map(p=>p.x*nx+p.y*ny),d=diamond.map(p=>p.x*nx+p.y*ny);
        if(Math.max(...b)<=Math.min(...d)||Math.max(...d)<=Math.min(...b))return false;
      }return true;
    }

    setPossession(robotId, possession = {}) {
      const robot = this.robots.find(r => r.id === robotId);
      if (!robot) return { ok: false, error: "unknown robot" };
      const pinId = possession.pinId || null;
      const cupId = possession.cupId || null;
      if (pinId && this.robots.some(r => r.id !== robotId && r.possession.pinId === pinId)) return { ok: false, error: "pin already possessed" };
      if (cupId && this.robots.some(r => r.id !== robotId && r.possession.cupId === cupId)) return { ok: false, error: "cup already possessed" };
      if(pinId&&(!this.pin(pinId)||this.pin(pinId).placed))return {ok:false,error:"pin unavailable"};
      if(cupId&&(!this.cup(cupId)||this.cup(cupId).placed))return {ok:false,error:"cup unavailable"};
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

    interact(robotId,action,kind="pin"){
      const robot=this.robots.find(r=>r.id===robotId);
      if(!robot||this.matchEnded)return {ok:false,error:"match ended or unknown robot"};
      if(!["pin","cup"].includes(kind))return {ok:false,error:"unknown object type"};
      const key=kind+"Id",objects=kind==="pin"?this.pins:this.cups;
      const held=objects.find(o=>o.id===robot.possession[key]);
      const near=(list,range)=>list.filter(o=>Math.hypot(o.x-robot.x,o.y-robot.y)<=range).sort((a,b)=>Math.hypot(a.x-robot.x,a.y-robot.y)-Math.hypot(b.x-robot.x,b.y-robot.y))[0];
      if(action==="pickup"){
        if(held)return {ok:false,error:"already holding this object type"};
        const object=near(objects.filter(o=>o.status==="field"&&o.location!=="alliance-station"&&o.location!=="preload"),14);
        if(!object)return {ok:false,error:"no reachable object"};
        object.status="held";robot.possession[key]=object.id;return {ok:true};
      }
      if(action==="load"){
        if(this.phase!=="driver")return {ok:false,error:"match loads are driver-period only"};
        const loader=near(this.loaders.filter(l=>l.alliance===robot.alliance),18);
        if(!loader)return {ok:false,error:"approach an alliance loader"};
        const object=objects.find(o=>o.location==="alliance-station"&&(o.alliance||o.allianceColor)===robot.alliance);
        if(!object)return {ok:false,error:"no match loads remain"};
        if([...this.pins,...this.cups].some(o=>o.location==="field"&&Math.hypot(o.x-loader.x,o.y-loader.y)<3))return {ok:false,error:"loader outlet is occupied"};
        object.location="field";object.status="field";object.x=loader.x;object.y=loader.y;return {ok:true};
      }
      if(action==="flip"){
        if(!held)return {ok:false,error:"nothing held"};
        if(kind==="pin")held.upIndex=1-held.upIndex;else held.up=held.up==="opaque"?"transparent":"opaque";
        return {ok:true};
      }
      if(action==="drop"){
        if(!held)return {ok:false,error:"nothing held"};
        held.x=robot.x+10*Math.sin(robot.theta*Math.PI/180);held.y=robot.y+10*Math.cos(robot.theta*Math.PI/180);
        held.status="field";held.location="field";robot.possession[key]=null;return {ok:true};
      }
      if(action==="place"){
        if(!held)return {ok:false,error:"nothing held"};
        const goal=near(this.goals,16);
        if(!goal)return {ok:false,error:"no goal within reach"};
        if(goal.alliance&&goal.alliance!==robot.alliance)return {ok:false,error:"opposing alliance goal is protected"};
        const result=kind==="pin"?this.placePin(held.id,goal.id):this.placeCup(held.id,goal.id);
        if(result.ok){held.x=goal.x;held.y=goal.y;robot.possession[key]=null;}return result;
      }
      if(action==="toggle"){
        const toggle=near(this.toggles,16);if(!toggle)return {ok:false,error:"no toggle within reach"};
        this.setToggle(toggle.id,robot.alliance);return {ok:true};
      }
      return {ok:false,error:"unknown action"};
    }

    placePin(pinId, goalId, options = {}) {
      const pin = this.pin(pinId);
      const goal = this.goal(goalId);
      if (!pin || !goal) return { ok: false, error: "unknown scoring object or goal" };
      if (pin.status === "placed") return { ok: false, error: "pin already placed" };
      if(goal.stack.length&&goal.stack.at(-1).type!=="cup")return {ok:false,error:"a cup is needed above the previous pin"};
      const visible=options.visibleHalves||(options.visibleHalf?[options.visibleHalf]:null);
      if(visible){const available=[...pin.halves];for(const half of visible){const i=available.indexOf(half);if(i<0)return {ok:false,error:"invalid visible half"};available.splice(i,1);}}
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
      if (cup.status === "placed") return { ok: false, error: "cup already placed" };
      if(!goal.stack.length||goal.stack.at(-1).type!=="pin")return {ok:false,error:"cup needs a supporting pin"};
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
      if (Math.abs(goal.x)+Math.abs(goal.y) < MIDFIELD_HALF) {
        const red = this.robots.filter(r => r.alliance === "red" && this.isInMidfield(r)).length;
        const blue = this.robots.filter(r => r.alliance === "blue" && this.isInMidfield(r)).length;
        return red === blue ? null : red > blue ? "red" : "blue";
      }
      const quadrant = Math.abs(goal.x)>Math.abs(goal.y)?(goal.x<0?"west":"east"):(goal.y<0?"south":"north");
      const toggle = this.toggles.find(t => t.quadrant === quadrant);
      return toggle && (toggle.state === "red" || toggle.state === "blue") ? toggle.state : null;
    }

    pinScore(pin, goal, includeMidfield=true) {
      const totals = { red: 0, blue: 0 };
      if (!pin || !pin.placed || !goal) return { alliance: null, points: 0, totals };
      const index=goal.stack.findIndex(item=>item.id===pin.id),below=goal.stack[index-1],above=goal.stack[index+1];
      const visibleHalves = Array.isArray(pin.visibleHalves) ? pin.visibleHalves : pin.halves.filter((half,i)=>{
        const top=i===pin.upIndex;
        return top?(!above||this.cup(above.id)?.up==="opaque"):(!below||this.cup(below.id)?.up==="transparent");
      });
      const yellowOwner = pin.halves.includes("yellow") && (includeMidfield||Math.abs(goal.x)+Math.abs(goal.y)>=MIDFIELD_HALF) ? this.ownerForPin(pin, goal) : null;
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
          const result = this.pinScore(this.pin(item.id), goal, includeMidfield);
          score.red += result.totals.red;
          score.blue += result.totals.blue;
        }
      }
      return score;
    }

    score() {
      if(this.finalScore)return {...this.finalScore};
      const score = this.scorePins(true);
      for (const robot of this.robots) if (robot.midfield) score[robot.alliance] += POINTS.midfieldRobot;
      score.red += this.autonomousBonus.red;
      score.blue += this.autonomousBonus.blue;
      return score;
    }

    evaluateAutonomousBonus() {
      if(this.autoFrozen)return {...this.autonomousBonus};
      const score = this.scorePins(false);
      this.autonomousScores={...score};
      if (score.red === score.blue) this.autonomousBonus = { red: 6, blue: 6 };
      else this.autonomousBonus = score.red > score.blue ? { red: POINTS.autonomousBonus, blue: 0 } : { red: 0, blue: POINTS.autonomousBonus };
      if (this.violations.red && this.violations.blue) this.autonomousBonus = { red: 0, blue: 0 };
      else if (this.violations.red) this.autonomousBonus = { red: 0, blue: POINTS.autonomousBonus };
      else if (this.violations.blue) this.autonomousBonus = { red: POINTS.autonomousBonus, blue: 0 };
      return { ...this.autonomousBonus };
    }

    freezeAutonomous(){
      if(this.autoFrozen)return;
      this.evaluateAutonomousBonus();this.qualifiesAWP("red",this.world==="worlds");this.qualifiesAWP("blue",this.world==="worlds");this.autoFrozen=true;
    }
    qualifiesAWP(alliance, worlds = false) {
      if(this.autoFrozen)return this.awp[alliance];
      const requiredPins = worlds ? 7 : 6;
      const requiredGoals = worlds ? 3 : 2;
      const allianceRobots = this.robots.filter(r => r.alliance === alliance);
      const eligible=this.goals.filter(g=>alliance==="red"?g.x+g.y<=0:g.x+g.y>=0);
      // SC3 defines a scored Pin as a visible half, not a physical two-half object.
      const count=goal=>goal.stack.reduce((n,item)=>{
        if(item.type!=="pin")return n;
        const p=this.pin(item.id),v=this.pinScore(p,goal,false).totals[alliance];
        const copy={...p,halves:p.halves.map(h=>h==="yellow"?"none":h),visibleHalves:p.visibleHalves?.map(h=>h==="yellow"?"none":h)};
        const colored=this.pinScore(copy,goal,false).totals[alliance];return n+colored/5+(v-colored)/10;
      },0);
      const scoredPins=eligible.reduce((n,g)=>n+count(g),0);
      const goals=eligible.filter(g=>count(g)>=2).length;
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
