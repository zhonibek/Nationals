'use strict';

const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const Game=require('../simulator/override');
const Geometry=require('../simulator/override-geometry');
const Dynamics=require('../simulator/override-dynamics');
const {scoringFixture}=require('./helpers/override-fixtures');
const {VexRobotSimulator}=require('../simulator/engine');
const Control=require('../simulator/control-runtime');
const compiled=new WebAssembly.Module(fs.readFileSync(path.join(__dirname,'../simulator/control.wasm')));

function isolated(){
  const game=new Game();game.startMatch('practice');
  const robot=game.robots[0],pin=game.pin(robot.possession.pinId);
  game.robots=[];game.pins=[pin];game.cups=[];game.goals=[];robot.possession.pinId=null;
  return {game,pin,robot};
}

test('official drawing flange diameter and Cup waist replace the undersized proxies',()=>{
  assert.equal(Geometry.OBJECTS.pinRadius*2,3.16);
  assert.equal(Geometry.OBJECTS.pinTipRadius*2,1.4);
  assert.equal(Geometry.OBJECTS.cupWaistRadius*2,2.32);
  assert.equal(Geometry.OBJECTS.cupHeight,6.48);
  const footprint=Geometry.objectBounds({kind:'red-yellow'},{x:0,y:0,lying:true,yaw:90});
  assert.equal(footprint.x,3.25);assert.equal(footprint.y,1.58);
});

test('lying Pin tips collide with robot and perimeter, not only their center',()=>{
  const {game,pin,robot}=isolated();
  Object.assign(robot,{x:0,y:0});game.robots=[robot];
  Dynamics.release(game,pin,{x:11,y:0,z:1.58,lying:true,yaw:90},{vx:-2});
  game.tick(.01);
  for(const sample of Geometry.objectFootprint(pin,game.objectPose(pin)))assert((Geometry.circleContact(robot,sample)?.depth||0)<1e-7);
  game.robots=[];Dynamics.release(game,pin,{x:69,y:25,z:1.58,lying:true,yaw:90},{vx:10});
  game.tick(.1);
  for(const sample of Geometry.objectFootprint(pin,game.objectPose(pin)))assert(sample.x+sample.radius<=Geometry.FIELD_HALF+1e-7);
});

test('unequal-mass object collisions conserve normal momentum without adding energy',()=>{
  const {game,pin}=isolated(),cup={id:'cup-test',kind:'cup',status:'field',location:'field',up:'opaque'};
  game.cups=[cup];Dynamics.release(game,pin,{x:-1.4,y:10,z:0},{vx:5});Dynamics.release(game,cup,{x:1.4,y:10,z:0},{vx:-3});
  const before=Dynamics.MASS_KG.pin*5+Dynamics.MASS_KG.cup*-3;
  const energy=Dynamics.MASS_KG.pin*25+Dynamics.MASS_KG.cup*9;
  game.tick(.005);
  const damping=Math.exp(-8*.005),after=Dynamics.MASS_KG.pin*pin.body.vx+Dynamics.MASS_KG.cup*cup.body.vx;
  assert(Math.abs(after-before*damping)<1e-10);
  assert(Dynamics.MASS_KG.pin*pin.body.vx**2+Dynamics.MASS_KG.cup*cup.body.vx**2<=energy+1e-10);
  assert(Math.hypot(pin.x-cup.x,pin.y-cup.y)>=3.16-1e-7);
});

test('pickup prioritizes reachable height over a closer unreachable elevated Pin',()=>{
  const {game,pin,robot}=isolated();game.robots=[robot];Object.assign(robot,{x:0,y:0});
  const elevated={...pin,id:'elevated'};game.pins.push(elevated);
  Dynamics.release(game,elevated,{x:0,y:15,z:20});Dynamics.release(game,pin,{x:2,y:15,z:0});
  const hint=game.pickupHint(robot.id,'pin');assert(hint.reachable);assert.equal(hint.id,pin.id);
  assert(game.interact(robot.id,'pickup','pin').ok);assert.equal(robot.possession.pinId,pin.id);
});

test('scoring fixture uses original four WASM controllers; manual movement, pickup and ballistic scoring work',()=>{
  const sim=new VexRobotSimulator(Control.fromModule(compiled));
  assert(scoringFixture(sim).ok);
  const game=sim.override,robot=game.robots.find(candidate=>candidate.id===sim.activeRobotId);
  assert.equal(game.phase,'driver');assert.equal(sim.fleet.entries.size,4);
  assert(game.pickupHint(robot.id,'pin').reachable);
  assert(game.interact(robot.id,'pickup','pin').ok);
  assert(!game.placementHint(robot.id,'pin').ready);
  for(let tick=0;tick<1500&&!game.placementHint(robot.id,'pin').ready;tick++){
    sim.holonomicArcade(.35,0,0);sim.update(.01);
  }
  const hint=game.placementHint(robot.id,'pin');assert(hint.ready,JSON.stringify(hint));
  sim.holonomicArcade(0,0,0);const pin=game.pin(robot.possession.pinId);
  assert(game.interact(robot.id,'place','pin').ok);assert(!pin.placed);
  for(let tick=0;tick<100&&!pin.placed;tick++)sim.update(.01);
  assert(pin.placed);assert.equal(game.goal(pin.goalId).id,'g-red-sw');
  assert.equal(game.scorePins().red,5);assert.equal(robot.possession.pinId,null);
});

test('live breakdown matches rules, opaque occlusion, yellow ownership and frozen final score',()=>{
  const game=new Game({physical:false});
  game.pins=game.pins.filter(pin=>!pin.placed);for(const goal of game.goals)goal.stack=[];
  const pin=game.pins.find(object=>object.location==='field'&&object.halves.includes('red'));
  assert(game.placePin(pin.id,'g-red-sw').ok);game.setToggle('toggle-west','red');
  let breakdown=game.scoreBreakdown();assert.equal(breakdown.red.coloredHalves,1);assert.equal(breakdown.red.yellowHalves,1);
  assert.equal(breakdown.red.total,15);
  const cup=game.cups.find(object=>object.location==='alliance-station');cup.up='transparent';
  assert(game.placeCup(cup.id,'g-red-sw').ok);
  breakdown=game.scoreBreakdown();assert.equal(breakdown.red.coloredHalves,0);
  assert.equal(breakdown.red.yellowHalves,1);assert.equal(breakdown.red.total,10);
  game.setRobotPose('red-1',{x:0,y:0});game.startMatch('practice');
  assert.deepEqual({red:game.scoreBreakdown().red.total,blue:game.scoreBreakdown().blue.total},game.score());
  game.stopMatch();const frozen=game.scoreBreakdown();assert(frozen.final);
  game.setToggle('toggle-west','blue');game.setRobotPose('red-1',{x:50,y:50});
  assert.deepEqual(game.scoreBreakdown(),frozen);assert.deepEqual(game.score(),{red:frozen.red.total,blue:frozen.blue.total});
});

test('manual calculator counts visible halves, rejects invalid values and never edits game state',()=>{
  const game=new Game(),before=game.getState();
  assert.equal(Game.calculatePoints({coloredHalves:3,yellowHalves:2,midfieldRobots:1,autonomousBonus:12}),55);
  for(const values of [{coloredHalves:-1},{coloredHalves:.5},{yellowHalves:NaN},{midfieldRobots:3},{autonomousBonus:5},{coloredHalves:127}])assert.throws(()=>Game.calculatePoints(values));
  assert.deepEqual(game.getState(),before);
});

test('game UI shows hints, disables impossible actions and updates measured calculator',()=>{
  const elements=new Map(),buttons=['pickup','place','drop','flip','load','toggle'].map(name=>({dataset:{gameAction:name},addEventListener(){}}));
  function element(id){
    if(!elements.has(id))elements.set(id,{value:id==='gameObject'?'pin':id==='calc-coloredHalves'?'3':'0',textContent:'',listeners:{},children:[],
      addEventListener(name,callback){this.listeners[name]=callback;},replaceChildren(){this.children=[];},append(child){this.children.push(child);}});
    return elements.get(id);
  }
  const context=vm.createContext({module:{exports:{}},require:filename=>filename==='./override'?Game:Dynamics,
    document:{getElementById:element,querySelectorAll:selector=>selector==='[data-game-action]'?buttons:[],createElement:()=>({textContent:'',children:[],append(child){this.children.push(child);}})},
    window:{addEventListener(){}},performance:{now:()=>1000}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../simulator/override-ui.js'),'utf8'),context);
  const sim=new VexRobotSimulator(Control.fromModule(compiled));scoringFixture(sim);
  const attached=context.module.exports.attach(sim);attached.update();
  assert.match(element('gameModeStatus').textContent,/физика включена/);
  assert.match(element('pickupHint').textContent,/Можно взять/);
  assert.equal(buttons.find(button=>button.dataset.gameAction==='pickup').disabled,false);
  assert.equal(buttons.find(button=>button.dataset.gameAction==='place').disabled,true);
  assert.equal(element('goalPoints').children.length,9);
  element('pointCalculator').listeners.input();assert.equal(element('manualPoints').textContent,'15 очков');
  const html=fs.readFileSync(path.join(__dirname,'../simulator/index.html'),'utf8');
  assert.match(html,/override-ui.js/);assert.doesNotMatch(html,/gameTutorial|gameCupTutorial|Учебная сцена/);assert.match(html,/не меняет поле/);
});
