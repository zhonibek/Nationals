/*
 * V5RC Override rules engine for the simulator.
 * Geometry is expressed in field inches with (0,0) at field center.
 * The scoring model follows the public VEX manual: 2026-2027, v2.0.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require('./override-geometry'),require('./override-dynamics'));
  else root.OverrideGame = factory(root.OverrideGeometry,root.OverrideDynamics);
})(typeof globalThis !== "undefined" ? globalThis : this, function (Geometry,Dynamics) {
  "use strict";

  const FIELD_WIDTH_IN = Geometry.FIELD_HALF*2;
  const HALF = FIELD_WIDTH_IN / 2;
  const AUTO_SECONDS = 15;
  const DRIVER_SECONDS = 105;
  const MATCH_SECONDS = AUTO_SECONDS + DRIVER_SECONDS;
  const ENDGAME_SECONDS = 10;
  const MIDFIELD_HALF = 23.11; // Diamond vertex distance, not axis-aligned half-width.
  const GOAL_RADIUS_IN = Geometry.OBJECTS.goalRadius;
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
        x,y,location,status:goalId?"placed":"field",placed:!!goalId,goalId,upIndex:0,visibleHalves:null,supportId:null,lying:false,yaw:0,autoShared:false});return id;
    };
    const cup=(x,y,up="opaque",location="field",alliance=null)=>{
      cups.push({id:"cup-"+String(cups.length+1).padStart(2,"0"),kind:"cup",halves:["opaque","transparent"],
        up,x,y,location,alliance,status:"field",placed:false,goalId:null});return cups.at(-1).id;
    };
    for(const [color,side] of [["red",-1],["blue",1]]){
      for(let i=0;i<12;i++)pin([color,"yellow"],side*78,55-i*10,i<2?"preload":"alliance-station",color);
      pin(["yellow","yellow"],side*84,0,"alliance-station",color);
      for(let i=0;i<10;i++)cup(side*88,45-i*10,"opaque","alliance-station",color);
    }
    for(const [x,y] of [[-47.09,47.09],[-23.54,23.54],[23.54,-23.54],[47.09,-47.09]]){
      cup(x,y,"transparent");cups.at(-1).autoShared=true;
      for(const [dx,dy,color] of [[0,5,"blue"],[5,0,"blue"],[0,-5,"red"],[-5,0,"red"]]){
        pin([color,"yellow"],x+dx,y+dy);
        Object.assign(pins.at(-1),{lying:true,yaw:Math.atan2(dx,dy)*180/Math.PI,autoShared:true});
      }
    }
    for(const a of [-23.54,23.54])for(const side of [-1,1]){
      for(const d of [-8,0,8]){cup(a+d,side*68);cup(side*68,a+d);}
      for(const [x,y] of [[a,side*68],[side*68,a]]){
        pin(["yellow","yellow"],x,y);pins.at(-1).supportId=cups.find(c=>c.x===x&&c.y===y).id;
      }
    }
    for(const a of [-47.09,-23.54,23.54,47.09]){const supportId=cup(a,a,"transparent");pin(["yellow","yellow"],a,a);pins.at(-1).supportId=supportId;}
    for(const [x,y,color] of [[-23.54,0,"red"],[0,-23.54,"red"],[23.54,0,"blue"],[0,23.54,"blue"]]){
      const supportId=cup(x,y,"transparent");pin([color,color==="red"?"blue":"red"],x,y);pins.at(-1).supportId=supportId;pins.at(-1).autoShared=true;cups.at(-1).autoShared=true;
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
      this.physical = options.physical !== false;
      this.reset();
    }

    reset() {
      this.mode='match';
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
      this.events = [];this.ruleEvents=[];this.awpAward={red:false,blue:false};this.disqualifiedRobots=[];this.ruleContacts=new Set();
      for(const robot of this.robots){robot.manipulator={height:0,target:0,reach:15,speed:24};robot.velocity={vx:0,vy:0,omega:0};}
      return this.getState();
    }

    startMatch(mode='match') {
      if (this.phase !== "pre_match") return false;
      if(!['match','practice'].includes(mode))return false;
      this.mode=mode;
      this.clock = 0;
      this.phase = mode==='practice'?'driver':"autonomous";
      this.matchEnded = false;
      return true;
    }

    startAutonomous() { return this.startMatch(); }

    stopMatch() {
      if (this.matchEnded) return;
      if(this.mode==='match')this.clock = MATCH_SECONDS;
      this.phase = "post_match";
      this.matchEnded = true;
      this.postMatchSeconds = 0;
      if(this.mode==='match'&&!this.autoFrozen)this.freezeAutonomous();
      if(!this.physical||Dynamics.resting(this))this.finalScore=this.score();
    }

    tick(dt) {
      if(!Number.isFinite(dt)||dt<=0||dt>1)return this.getState();
      // Split exactly at phase boundaries so no autonomous action leaks into driver time.
      let remaining=dt;
      while(remaining>1e-8){
        if(this.phase==='pre_match'||this.finalScore)break;
        const boundary=this.mode==='practice'?Infinity:this.phase==='autonomous'?AUTO_SECONDS:MATCH_SECONDS;
        const h=Math.min(.01,remaining,this.phase==='post_match'?remaining:boundary-this.clock);
        if(h<=1e-8){if(this.phase==='autonomous'){this.freezeAutonomous();this.phase='driver';}else this.stopMatch();continue;}
        if(this.physical){
          if(!this.matchEnded)for(const r of this.robots){const m=r.manipulator;m.height+=clamp(m.target-m.height,-m.speed*h,m.speed*h);}
          Dynamics.step(this,h);
        }
        if(this.phase==='post_match'){
          this.postMatchSeconds=Math.min(5,this.postMatchSeconds+h);
          if(this.postMatchSeconds>=5-1e-8){this.postMatchSeconds=5;this.finalScore=this.score();}
          else if(Dynamics.resting(this))this.finalScore=this.score();
        }else{
          this.clock+=h;
          if(this.mode==='match'&&this.clock>=AUTO_SECONDS-1e-8&&this.phase==='autonomous'){this.clock=AUTO_SECONDS;this.freezeAutonomous();this.phase='driver';}
          if(this.mode==='match'&&this.clock>=MATCH_SECONDS-1e-8)this.stopMatch();
        }
        remaining-=h;
      }
      return this.getState();
    }

    recordRule(robotId,rule,detail,severity='review'){
      const robot=this.robots.find(r=>r.id===robotId);
      if(!robot)return {ok:false,error:'unknown robot'};
      const event={robotId,alliance:robot.alliance,rule,detail,severity,clock:this.clock,phase:this.phase};
      this.ruleEvents.push(event);return {ok:true,event:clone(event)};
    }

    adjudicate(robotId,rule,severity,detail,{autonomous=false,awardAWP=false}={}){
      if(!['SC1','SC2','SC3','SC4','SC5','SC6','SC7','SC8','SG1','SG2','SG3','SG4','SG5','SG6','SG7','SG8','SG9','SG10','SG11','SG12','SG13','GG13','GG14','GG15','S1'].includes(rule)||!['warning','minor','major'].includes(severity)||typeof detail!=='string'||!detail.trim())return {ok:false,error:'Choose a rule, severity and reason'};
      const result=this.recordRule(robotId,rule,detail.trim().slice(0,500),severity);if(!result.ok)return result;
      Object.assign(this.ruleEvents.at(-1),{autonomous,awardAWP});Object.assign(result.event,{autonomous,awardAWP});
      const robot=this.robots.find(r=>r.id===robotId),other=robot.alliance==='red'?'blue':'red';
      if(severity==='major'&&!this.disqualifiedRobots.includes(robotId))this.disqualifiedRobots.push(robotId);
      if(autonomous){
        const previousBonus={...this.autonomousBonus};
        this.violations[robot.alliance]=true;
        if(awardAWP)this.awpAward[other]=true;
        this.awp[robot.alliance]=false;
        if(this.autoFrozen){
          this.autonomousBonus=this.violations.red&&this.violations.blue?{red:0,blue:0}:this.violations.red?{red:0,blue:12}:{red:12,blue:0};
          this.awp.red=!this.violations.red&&(this.awp.red||this.awpAward.red);this.awp.blue=!this.violations.blue&&(this.awp.blue||this.awpAward.blue);
          if(this.finalScore)for(const a of ["red","blue"])this.finalScore[a]+=this.autonomousBonus[a]-previousBonus[a];
        }
      }
      return result;
    }

    auditAutonomousContact(robot,object){
      if(this.phase!=='autonomous'||object.autoShared)return;
      const side=object.x+object.y;
      if(robot.alliance==='red'?side>1:side<-1){
        this.adjudicate(robot.id,'SG7','minor','Contact with a scoring object from the opposing autonomous side',{autonomous:true,awardAWP:true});
      }
    }

    setLiftTarget(robotId,height,source='manual'){
      const robot=this.robots.find(r=>r.id===robotId);
      if(!robot||!Number.isFinite(height)||height<0||height>40)return {ok:false,error:'Lift range: 0–40 inches'};
      if(this.disqualifiedRobots.includes(robotId))return {ok:false,error:'Robot disqualified'};
      if(this.matchEnded||this.phase==='pre_match'||(this.phase==='autonomous'&&source!=='autonomous'))return {ok:false,error:'Lift disabled in this phase'};
      robot.manipulator.target=height;return {ok:true};
    }

    gripPose(robot,kind='cup'){
      const m=robot.manipulator,offset=Geometry.rotate(kind==='pin'&&robot.possession.cupId?4:0,m.reach,robot.theta);
      return {x:robot.x+offset.x,y:robot.y+offset.y,z:m.height,centerZ:m.height+3.25,yaw:robot.theta,lying:false};
    }

    captureObject(object,goal){
      const passenger=this.pins.find(p=>p.supportId===object.id),actor=object.lastActor;
      const result=object.kind==='cup'?this.placeCup(object.id,goal.id,{settling:true}):this.placePin(object.id,goal.id);
      if(!result.ok)return false;
      delete object.body;
      if(passenger){passenger.supportId=null;this.placePin(passenger.id,goal.id);delete passenger.body;}
      if(actor&&this.mode==='match'&&this.clock>=110&&goal.id==='g-neutral-tall')this.recordRule(actor,'SG12','Object nested on midfield goal during endgame; referee review required');
      this.events.push({type:'capture',objectId:object.id,goalId:goal.id,clock:this.clock});return true;
    }

    releaseHeld(robot,kind){
      const key=kind+'Id',o=this[kind](robot.possession[key]);if(!o)return {ok:false,error:'nothing held'};
      const p=this.objectPose(o),passenger=kind==='cup'?this.pins.find(pin=>pin.supportId===o.id):null;
      robot.possession[key]=null;o.lastActor=robot.id;
      if(kind==='pin')o.supportId=null;
      Dynamics.release(this,o,p,robot.velocity);
      if(passenger){Object.assign(passenger,{status:'field',location:'field',lastActor:robot.id});robot.possession.pinId=null;}
      return {ok:true};
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
      if(this.phase==='autonomous'&&Geometry.corners(robot).some(p=>Math.abs(p.x)+Math.abs(p.y)>MIDFIELD_HALF+1&&(robot.alliance==='red'?p.x+p.y>1:p.x+p.y<-1))){
        const key=robot.id+':SG7';if(!this.ruleContacts.has(key)){this.ruleContacts.add(key);this.recordRule(robot.id,'SG7','Robot footprint crossed autonomous-side boundary; check shared-line exceptions');}
      }
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
      return Geometry.touchesPerimeter(robot);
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
      const passenger=cupId?this.pins.find(p=>p.supportId===cupId):null;
      if(passenger&&passenger.id!==pinId)return {ok:false,error:"supported Pin must move with its Cup"};
      for(const kind of ['pin','cup']){
        const old=this[kind](robot.possession[kind+'Id']);
        const next=kind==='pin'?pinId:cupId;
        if(old&&old.id!==next){Object.assign(old,{status:'field',location:'field',x:robot.x,y:robot.y,supportId:null});}
        const object=this[kind](next);
        if(object){object.status='held';object.lying=false;delete object.body;}
      }
      if(pinId&&this.pin(pinId).supportId!==cupId)this.pin(pinId).supportId=null;
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

    objectPose(object){
      const holder=this.robots.find(r=>r.possession.pinId===object.id||r.possession.cupId===object.id);
      const support=this.cup(object.supportId);
      if(support&&(support.status==='field'||(holder&&holder.possession.cupId===support.id))){
        const p=this.objectPose(support);
        const tilt=p.tilt||0,a=(p.yaw||0)*Math.PI/180,h=Geometry.OBJECTS.halfHeight;
        return {...p,x:p.x+h*Math.sin(tilt)*Math.sin(a),y:p.y+h*Math.sin(tilt)*Math.cos(a),z:p.z+h*Math.cos(tilt),centerZ:p.centerZ===undefined?undefined:p.centerZ+h*Math.cos(tilt),lying:tilt>Math.PI/4};
      }
      if(holder&&this.physical)return this.gripPose(holder,object.kind==='cup'?'cup':'pin');
      if(holder){
        const offset=Geometry.rotate(holder.possession.cupId&&object.id===holder.possession.pinId?4:0,10,holder.theta);
        return {x:holder.x+offset.x,y:holder.y+offset.y,z:10,lying:false,yaw:holder.theta};
      }
      if(object.body)return Dynamics.pose(object);
      const goal=this.goal(object.goalId);
      if(goal){
        const index=goal.stack.findIndex(i=>i.id===object.id);
        return {x:goal.x,y:goal.y,z:goal.height-Geometry.OBJECTS.halfHeight+index*Geometry.OBJECTS.halfHeight,lying:false,yaw:0};
      }
      return {x:object.x,y:object.y,z:object.lying?Geometry.OBJECTS.pinRadius:0,lying:!!object.lying,yaw:object.yaw||0};
    }

    resolveRobotContact(robotId,pose,withObstacles=true){
      const robot=this.robots.find(r=>r.id===robotId);
      const obstacles=withObstacles?[
        ...this.goals.map(g=>({x:g.x,y:g.y,radius:GOAL_RADIUS_IN})),
        ...this.robots.filter(r=>r.id!==robotId)
      ]:[];
      return Geometry.resolveRobot({...robot,...pose},obstacles);
    }

    interact(robotId,action,kind="pin",options={}){
      const robot=this.robots.find(r=>r.id===robotId);
      if(!robot||this.matchEnded)return {ok:false,error:"match ended or unknown robot"};
      if(this.disqualifiedRobots.includes(robotId))return {ok:false,error:"Robot disqualified"};
      if(this.physical&&(this.phase==='pre_match'||(this.phase==='autonomous'&&options.source!=='autonomous')))return {ok:false,error:"Manual actions disabled in this phase"};
      if(!["pin","cup"].includes(kind))return {ok:false,error:"unknown object type"};
      const key=kind+"Id",objects=kind==="pin"?this.pins:this.cups;
      const held=objects.find(o=>o.id===robot.possession[key]);
      const near=(list,range)=>list.filter(o=>Math.hypot(o.x-robot.x,o.y-robot.y)<=range).sort((a,b)=>Math.hypot(a.x-robot.x,a.y-robot.y)-Math.hypot(b.x-robot.x,b.y-robot.y))[0];
      if(action==="pickup"){
        if(held)return {ok:false,error:"already holding this object type"};
        const grip=this.gripPose(robot,kind);
        const candidates=objects.filter(o=>o.status==='field'&&o.location!=='alliance-station'&&o.location!=='preload');
        const object=this.physical?candidates.filter(o=>{const p=this.objectPose(o);return Math.hypot(p.x-grip.x,p.y-grip.y)<=3&&Math.abs((p.centerZ??(p.lying?p.z:p.z+3.25))-grip.centerZ)<=3;}).sort((a,b)=>{const pa=this.objectPose(a),pb=this.objectPose(b);return Math.hypot(pa.x-grip.x,pa.y-grip.y)-Math.hypot(pb.x-grip.x,pb.y-grip.y);})[0]:near(candidates,14);
        if(!object)return {ok:false,error:"no reachable object"};
        const passenger=kind==='cup'?this.pins.find(p=>p.supportId===object.id):null;
        if(passenger&&robot.possession.pinId)return {ok:false,error:"free the Pin slot before lifting this stack"};
        this.auditAutonomousContact(robot,object);
        object.status="held";object.lying=false;delete object.body;robot.possession[key]=object.id;
        if(kind==='pin')object.supportId=null;
        if(passenger){passenger.status='held';delete passenger.body;robot.possession.pinId=passenger.id;}
        return {ok:true};
      }
      if(action==="load"){
        if(this.phase!=="driver")return {ok:false,error:"match loads are driver-period only"};
        const loader=near(this.loaders.filter(l=>l.alliance===robot.alliance),18);
        if(!loader)return {ok:false,error:"approach an alliance loader"};
        const object=objects.find(o=>o.location==="alliance-station"&&(o.alliance||o.allianceColor)===robot.alliance);
        if(!object)return {ok:false,error:"no match loads remain"};
        if([...this.pins,...this.cups].some(o=>o.location==="field"&&Math.hypot(o.x-loader.x,o.y-loader.y)<3))return {ok:false,error:"loader outlet is occupied"};
        object.location="field";object.status="field";object.x=loader.x;object.y=loader.y;if(this.physical)Dynamics.release(this,object,{x:loader.x,y:loader.y,z:9,yaw:0},{});return {ok:true};
      }
      if(action==="flip"){
        if(!held)return {ok:false,error:"nothing held"};
        if(kind==='cup'&&this.pins.some(p=>p.supportId===held.id))return {ok:false,error:"remove the supported Pin before flipping the Cup"};
        if(kind==="pin")held.upIndex=1-held.upIndex;else held.up=held.up==="opaque"?"transparent":"opaque";
        return {ok:true};
      }
      if(action==="drop"&&this.physical)return this.releaseHeld(robot,kind);
      if(action==="drop"){
        if(!held)return {ok:false,error:"nothing held"};
        held.x=robot.x+10*Math.sin(robot.theta*Math.PI/180);held.y=robot.y+10*Math.cos(robot.theta*Math.PI/180);
        held.status="field";held.location="field";held.lying=false;robot.possession[key]=null;
        if(kind==='pin')held.supportId=null;
        else{
          const passenger=this.pins.find(p=>p.supportId===held.id);
          if(passenger){Object.assign(passenger,{x:held.x,y:held.y,status:'field',location:'field'});robot.possession.pinId=null;}
        }
        return {ok:true};
      }
      if(action==="place"){
        if(!held)return {ok:false,error:"nothing held"};
        const goal=near(this.goals,16);
        if(!goal)return {ok:false,error:"no goal within reach"};
        if(goal.alliance&&goal.alliance!==robot.alliance)return {ok:false,error:"opposing alliance goal is protected"};
        if(this.mode==='match'&&this.clock>=110&&goal.id==='g-neutral-tall')return {ok:false,error:'SG12: no midfield placement during endgame'};
        if(this.physical){
          const p=this.objectPose(held),base=goal.height-3.25+goal.stack.length*3.25;
          if(Math.hypot(p.x-goal.x,p.y-goal.y)>1.1)return {ok:false,error:'Align gripper with goal (within 1.1 in)'};
          if(p.z<base||p.z>base+8)return {ok:false,error:`Set lift between ${base.toFixed(1)} and ${(base+8).toFixed(1)} in`};
          const top=goal.stack.at(-1);if(kind==='cup'?top?.type!=='pin':top&&top.type!=='cup')return {ok:false,error:'Stack must alternate Pin / Cup'};
          return this.releaseHeld(robot,kind);
        }
        const passenger=kind==='cup'?this.pins.find(p=>p.supportId===held.id&&p.status==='held'):null;
        const result=kind==="pin"?this.placePin(held.id,goal.id):this.placeCup(held.id,goal.id);
        if(result.ok&&passenger)this.placePin(passenger.id,goal.id);
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
      delete pin.body;
      pin.status = "placed";
      pin.location='goal';
      pin.supportId=null;pin.lying=false;
      for(const r of this.robots)if(r.possession.pinId===pin.id)r.possession.pinId=null;
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
      if(!options.settling&&this.pins.some(p=>p.supportId===cup.id&&p.status!=='held'))return {ok:false,error:"pick up or remove the supported Pin first"};
      if(!goal.stack.length||goal.stack.at(-1).type!=="pin")return {ok:false,error:"cup needs a supporting pin"};
      delete cup.body;
      cup.status = "placed";
      cup.location='goal';
      for(const r of this.robots)if(r.possession.cupId===cup.id)r.possession.cupId=null;
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
      this.evaluateAutonomousBonus();this.qualifiesAWP("red",this.world==="worlds");this.qualifiesAWP("blue",this.world==="worlds");for(const a of ["red","blue"])this.awp[a]=!this.violations[a]&&(this.awp[a]||this.awpAward[a]);this.autoFrozen=true;
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
        mode: this.mode,
        clock: this.clock,
        endgame: this.mode==='match'&&this.clock >= MATCH_SECONDS - ENDGAME_SECONDS && this.clock < MATCH_SECONDS,
        matchEnded: this.matchEnded,
        scoreFinal: !!this.finalScore,
        settlingSeconds: this.postMatchSeconds,
        ruleEvents: clone(this.ruleEvents),
        disqualifiedRobots: [...this.disqualifiedRobots],
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
