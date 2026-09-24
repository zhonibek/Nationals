'use strict';
const assert=require('node:assert/strict'),Game=require('./override'),D=require('./override-dynamics');
const near=(a,b,tolerance=1e-6)=>assert(Math.abs(a-b)<tolerance,`${a} != ${b}`);
function isolated(mode='practice'){
  const game=new Game();game.startMatch(mode);const r=game.robots[0],p=game.pin(r.possession.pinId);
  game.robots=[r];game.pins=[p];game.cups=[];game.goals=[];game.setRobotPose(r.id,{x:0,y:0,theta:0});return {game,r,p};
}
const {game,r,p}=isolated();
assert(!game.setLiftTarget(r.id,41).ok);assert(!game.setLiftTarget(r.id,NaN).ok);
assert(game.setLiftTarget(r.id,24).ok);game.tick(.5);near(r.manipulator.height,12);game.tick(.5);near(r.manipulator.height,24);
assert(game.interact(r.id,'drop').ok);const start=p.body.z;
game.tick(.1);near(p.body.z,start-D.GRAVITY*.005**2*20*21/2);near(p.body.vz,-D.GRAVITY*.1);
for(let i=0;i<200;i++)game.tick(.01);
near(p.body.z,.8);near(p.body.tilt,Math.PI/2);assert(D.resting(game));
game.setLiftTarget(r.id,0);game.tick(1);game.setRobotPose(r.id,{x:0,y:30});
assert(!game.interact(r.id,'pickup').ok,'No pickup behind the robot');game.setRobotPose(r.id,{x:0,y:0});
assert(game.interact(r.id,'pickup').ok);assert(!p.body);assert.equal(p.status,'held');
// Actuator limits are checked by the rules API, not only disabled UI controls.
const autonomous=isolated('match');assert(!autonomous.game.interact(autonomous.r.id,'drop').ok);
assert(!autonomous.game.setLiftTarget(autonomous.r.id,10).ok);
assert(autonomous.game.setLiftTarget(autonomous.r.id,10,'autonomous').ok);
assert(autonomous.game.interact(autonomous.r.id,'drop','pin',{source:'autonomous'}).ok);
// Correctly aligned release settles into the opening; a near miss stays unscored.
const score=isolated();score.game.goals=[{id:'own',x:40,y:0,height:3.25,alliance:'red',stack:[]}];
score.game.setRobotPose(score.r.id,{x:42,y:-15});score.r.manipulator.height=2;
assert(!score.game.interact(score.r.id,'place').ok);score.game.setRobotPose(score.r.id,{x:40,y:-15});
assert(score.game.interact(score.r.id,'place').ok);assert(!score.p.placed);score.game.tick(.3);
assert(score.p.placed);assert.equal(score.game.scorePins().red,5);assert(!score.p.body);
// Physical SC1 grace: falling scoring object resolves after 0:00, then final score freezes.
const grace=isolated('match');grace.game.goals=[{id:'own',x:40,y:0,height:3.25,alliance:'red',stack:[]}];
grace.game.phase='driver';grace.game.clock=119.95;grace.game.autoFrozen=true;
grace.game.setRobotPose(grace.r.id,{x:40,y:-15});grace.r.manipulator.height=4;
assert(grace.game.interact(grace.r.id,'place').ok);grace.game.tick(.05);
assert(grace.game.matchEnded);assert(!grace.game.finalScore);assert(!grace.game.interact(grace.r.id,'drop').ok);
grace.game.tick(.3);assert(grace.game.finalScore);assert.equal(grace.game.finalScore.red,5);
const final={...grace.game.finalScore};grace.game.setToggle('toggle-north','blue');grace.game.tick(1);assert.deepEqual(grace.game.score(),final);
const end=isolated('match');end.game.phase='driver';end.game.clock=110;end.game.goals=[{id:'g-neutral-tall',x:0,y:15,height:8.7,alliance:null,stack:[]}];
assert.match(end.game.interact(end.r.id,'place').error,/SG12/);
// Recorded decisions change awards/DQ, never invent negative scoring points.
const referee=new Game({physical:false});referee.startMatch();referee.tick(1);referee.adjudicate('red-1','SG7','minor','Opposing-side object',{autonomous:true,awardAWP:true});
for(let i=0;i<14;i++)referee.tick(1);
assert.deepEqual(referee.autonomousBonus,{red:0,blue:12});assert(referee.awp.blue);assert(!referee.awp.red);
referee.adjudicate('blue-1','SG7','minor','Second alliance autonomous violation',{autonomous:true,awardAWP:true});assert.deepEqual(referee.autonomousBonus,{red:0,blue:0});assert(!referee.awp.blue);
assert(referee.adjudicate('blue-2','SG9','major','Referee confirmed intentional interference').ok);assert(referee.disqualifiedRobots.includes('blue-2'));
assert(!referee.adjudicate('red-1','invented','minor','x').ok);assert(!referee.adjudicate('red-1','SG9','major','').ok);
assert.equal(new Game().pins.filter(p=>p.autoShared).length+new Game().cups.filter(c=>c.autoShared).length,28);
console.log('PASS: lift limits/rate, ballistic fall, impact/toppling, front-only pickup, physical nested scoring, SC1 settling, endgame prohibition and referee awards.');

// Zero initial velocity is not rest when an unsupported body is above the floor.
const instant=isolated('match');instant.game.phase='driver';instant.game.clock=119.99;
instant.r.manipulator.height=20;assert(instant.game.interact(instant.r.id,'drop').ok);
assert(!D.resting(instant.game));instant.game.stopMatch();assert(!instant.game.finalScore);
for(let i=0;i<500&&!instant.game.finalScore;i++)instant.game.tick(.01);
assert(instant.game.finalScore);assert(instant.game.postMatchSeconds>0);
// A ruling after scoring changes the bonus alone; its meaning survives report export.
const late=new Game({physical:false});late.startMatch();for(let i=0;i<15;i++)late.tick(1);late.stopMatch();
const before=late.score();assert(late.adjudicate('red-1','SG7','minor','Confirmed after match',{autonomous:true,awardAWP:true}).ok);
assert.equal(late.score().red,before.red-6);assert.equal(late.score().blue,before.blue+6);
assert.equal(late.ruleEvents.at(-1).autonomous,true);assert.equal(late.ruleEvents.at(-1).awardAWP,true);

// Newly dropped Pins can nest into floor Cups, then fall when their support tips.
const floorPair=isolated(),cup={id:'test-cup',kind:'cup',up:'opaque',x:0,y:15,status:'field',location:'field',placed:false,goalId:null};
floorPair.game.cups=[cup];floorPair.r.manipulator.height=5;
assert(floorPair.game.interact(floorPair.r.id,'drop').ok);floorPair.game.tick(.25);
assert.equal(floorPair.p.supportId,cup.id);assert(!floorPair.p.placed);assert(!floorPair.p.body);
cup.body.spin=3;floorPair.game.tick(.1);assert.equal(floorPair.p.supportId,null);assert(floorPair.p.body);
for(let i=0;i<200;i++)floorPair.game.tick(.01);assert(D.resting(floorPair.game));
// A dropped pair breaks apart on a hard landing; it cannot remain suspended.
const fallingPair=isolated();fallingPair.game.cups=[{...cup,id:'falling-cup',body:undefined,status:'held'}];
fallingPair.r.possession.cupId='falling-cup';fallingPair.p.supportId='falling-cup';fallingPair.r.manipulator.height=24;
assert(fallingPair.game.interact(fallingPair.r.id,'drop','cup').ok);fallingPair.game.tick(1);fallingPair.game.tick(1);
assert.equal(fallingPair.p.supportId,null);assert(fallingPair.p.body);assert(D.resting(fallingPair.game));
// SC1 has a hard five-second limit even if a robot never reports rest.
const moving=isolated('match');moving.r.velocity.vx=1;moving.game.stopMatch();
for(let i=0;i<500;i++)moving.game.tick(.01);
assert(moving.game.finalScore);near(moving.game.postMatchSeconds,5);
console.log('PASS: floor nesting, loss of support, dropped pair separation and five-second scoring deadline.');

// Moving robot transfers normal velocity to an unheld object.
const push=isolated();push.r.possession.pinId=null;push.r.velocity.vx=10;
D.release(push.game,push.p,{x:8.5,y:0,z:.8,lying:true});push.game.tick(.01);
assert(push.p.x>=9.8-1e-7);assert(push.p.body.vx>0);
// Contact between free objects cannot inject normal kinetic energy.
const pair=isolated();pair.game.robots=[];
const other={...pair.p,id:'other',body:undefined};pair.game.pins.push(other);
D.release(pair.game,pair.p,{x:-.7,y:20,z:.8,lying:true},{vx:2});D.release(pair.game,other,{x:.7,y:20,z:.8,lying:true},{vx:-2});
pair.game.tick(.01);assert(Math.hypot(pair.p.x-other.x,pair.p.y-other.y)>=1.6-1e-7);
assert(pair.p.body.vx**2+other.body.vx**2<=8+1e-7);
// A robot pinning an object against the perimeter cannot push it outside the field.
const wall=isolated();wall.r.possession.pinId=null;wall.r.x=61.2;wall.r.velocity.vx=20;
D.release(wall.game,wall.p,{x:69.1,y:0,z:.8,lying:true});wall.game.tick(.1);
assert(wall.p.x<=69.4+1e-7);assert(Number.isFinite(wall.p.body.vx));
console.log('PASS: robot/object momentum transfer, bounded pair collision and perimeter containment.');
