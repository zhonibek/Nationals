'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const Game=require('./override'),G=require('./override-geometry');
const close=(a,b)=>assert(Math.abs(a-b)<1e-7,`${a} != ${b}`);
const initial=new Game({physical:false});
assert.equal(initial.pins.filter(p=>p.lying).length,16);
assert.equal(initial.pins.filter(p=>p.supportId).length,16);
for(const pin of initial.pins.filter(p=>p.supportId)){
  const cup=initial.cup(pin.supportId),pose=initial.objectPose(pin);
  assert(cup&&cup.status==='field');close(pin.x,cup.x);close(pin.y,cup.y);close(pose.z,3.25);
}
for(const pin of initial.pins.filter(p=>p.placed))close(initial.objectPose(pin).z,initial.goal(pin.goalId).height-3.25);
for(const pin of initial.pins.filter(p=>p.lying)){
  const outward=G.rotate(0,1,pin.yaw);
  const cup=initial.cups.find(c=>Math.abs(Math.hypot(c.x-pin.x,c.y-pin.y)-5)<1e-7);
  assert(cup);close(pin.x-cup.x,5*outward.x);close(pin.y-cup.y,5*outward.y);
  assert.equal(pin.halves[pin.upIndex],outward.x+outward.y>0?'blue':'red');
  close(initial.objectPose(pin).z,.8);
}
// Picking up a loaded Cup moves the pair; taking the Pin separately detaches it.
const game=new Game({physical:false}),robot=game.robots[0];
const pin=game.pins.find(p=>p.supportId&&p.x===-23.54&&p.y===0),cup=game.cup(pin.supportId);
game.setRobotPose(robot.id,{x:cup.x,y:cup.y});
assert.equal(game.interact(robot.id,'pickup','cup').ok,false,'Preload occupies the Pin slot');
assert(game.setPossession(robot.id,{}).ok);
assert(game.interact(robot.id,'pickup','cup').ok);
assert.equal(robot.possession.pinId,pin.id);assert.equal(robot.possession.cupId,cup.id);
close(game.objectPose(pin).z-game.objectPose(cup).z,3.25);
assert.equal(game.interact(robot.id,'flip','cup').ok,false);
assert(game.interact(robot.id,'drop','cup').ok);
assert.deepEqual(robot.possession,{pinId:null,cupId:null});
assert.equal(pin.status,'field');assert.equal(pin.supportId,cup.id);close(pin.y,cup.y);
game.setRobotPose(robot.id,{x:pin.x,y:pin.y});
assert(game.interact(robot.id,'pickup','pin').ok);assert.equal(pin.supportId,null);
assert(game.setPossession(robot.id,{pinId:pin.id}).ok,'Pin-only possession must be valid');
assert(game.interact(robot.id,'pickup','cup').ok);
assert(game.interact(robot.id,'flip','cup').ok);
// Place a held pair above the existing Pin; both slots must clear atomically.
const pair=new Game({physical:false}),p=pair.pins.find(p=>p.supportId),c=pair.cup(p.supportId);
assert(pair.setPossession('red-1',{pinId:p.id,cupId:c.id}).ok);
const goal=pair.goal('g-neutral-tall');pair.setRobotPose('red-1',{x:goal.x,y:goal.y});
assert(pair.interact('red-1','place','cup').ok);
assert.deepEqual(goal.stack.slice(-2),[{type:'cup',id:c.id},{type:'pin',id:p.id}]);
assert.equal(p.supportId,null);assert.equal(c.location,'goal');assert.equal(p.location,'goal');
assert.deepEqual(pair.robots[0].possession,{pinId:null,cupId:null});
close(pair.objectPose(p).z-pair.objectPose(c).z,3.25);
const preload=pair.robots[1].possession.pinId;
assert(pair.placePin(preload,'g-red-sw').ok);assert.equal(pair.pin(preload).location,'goal');
assert.equal(pair.robots[1].possession.pinId,null);
const practice=new Game({physical:false});assert(practice.startMatch('practice'));
for(let i=0;i<130;i++)practice.tick(1);
close(practice.clock,130);assert.equal(practice.phase,'driver');assert(!practice.matchEnded);
assert(!practice.getState().endgame);assert.deepEqual(practice.autonomousBonus,{red:0,blue:0});
practice.stopMatch();const score=practice.score();practice.tick(1);close(practice.clock,130);
practice.setRobotPose('red-1',{x:0,y:0});assert.deepEqual(practice.score(),score);
practice.reset();assert(practice.startMatch());assert.equal(practice.mode,'match');
// Analytic wall boundary for a 45-degree square, including all corners.
const body={x:68,y:68,theta:45,width:18,length:18};
const wall=G.resolveRobot(body,[]);close(wall.x,70.2-9*Math.SQRT2);close(wall.y,wall.x);
for(const p of G.corners({...body,...wall}))assert(Math.abs(p.x)<=70.2+1e-7&&Math.abs(p.y)<=70.2+1e-7);
// Obstacle at the rotated corner: old axis-aligned/radius approximations missed it.
const a={x:0,y:0,theta:45,width:18,length:18},circle={x:14,y:0,radius:4.25};
assert(G.circleContact(a,circle));const resolved=G.resolveRobot(a,[circle]);
assert((G.circleContact({...a,...resolved},circle)?.depth||0)<1e-7);
for(const obstacle of [{x:0,y:0,radius:4.25},{x:0,y:0,theta:17,width:4,length:4},{x:12,y:2,theta:31,width:18,length:18}]){
  const result=G.resolveRobot(a,[obstacle]);
  const contact=obstacle.radius?G.circleContact({...a,...result},obstacle):G.rectangleContact({...a,...result},obstacle);
  assert((contact?.depth||0)<1e-7,'Containment and rotated edges must separate');
}
assert.equal(G.rectangleContact({x:0,y:0,theta:0,width:18,length:18},{x:18,y:0,theta:0,width:18,length:18}),null);
// Actual simulator integration: blocked truth cannot masquerade as encoder feedback.
const Control=require('./control-runtime');
const core=new WebAssembly.Module(fs.readFileSync(__dirname+'/control.wasm'));
const context={console,Math,setTimeout(){},document:{addEventListener(){}},window:{},OverrideGame:Game};
vm.createContext(context);
vm.runInContext(fs.readFileSync(__dirname+'/robot-config.js','utf8'),context);
vm.runInContext(fs.readFileSync(__dirname+'/simulator.js','utf8')+'\nthis.Simulator=VexRobotSimulator;',context);
const sim=new context.Simulator(Control.fromModule(core));
sim.setPose(0,58,0);sim.queueAction({type:'drive',targetInches:24,heading:0});sim.isRunning=true;
for(let i=0;i<1200&&sim.isRunning;i++)sim.update(.01);
assert(sim.y<=70.2-sim.trackWidthInches/2+1e-7);assert(sim.odom.y-sim.y>1,'Wall obstruction must cause odometry drift');
assert.equal(sim.lastMotionResult,'TimedOut','Obstructed motion must not claim settlement');
sim.matchMode=true;sim.setPose(0,0,45);sim.updateMobileGoalsPhysics(.01);
const active=sim.override.robots.find(r=>r.id===sim.activeRobotId),pose={...active,x:sim.x,y:sim.y,theta:sim.theta};
for(const goal of sim.override.goals)assert((G.circleContact(pose,{...goal,radius:G.OBJECTS.goalRadius})?.depth||0)<1e-7);
console.log('PASS: FO-2 orientations/nesting, pair manipulation, practice clock, rotated wall/goal/robot contacts and collision-induced encoder drift.');
