'use strict';

const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const Game=require('../simulator/override');
const Dynamics=require('../simulator/override-dynamics');
const UI=require('../simulator/override-ui');
const {scoringFixture}=require('./helpers/override-fixtures');
const {VexRobotSimulator}=require('../simulator/engine');
const Control=require('../simulator/control-runtime');
const compiled=new WebAssembly.Module(fs.readFileSync(path.join(__dirname,'../simulator/control.wasm')));

function cupsAtGripper(){
  const game=new Game();game.startMatch('practice');
  const robot=game.robots[0],loaded=game.cups.find(cup=>cup.location==='field'&&game.pins.some(pin=>pin.supportId===cup.id));
  const empty=game.cups.find(cup=>cup.location==='field'&&!game.pins.some(pin=>pin.supportId===cup.id));
  game.setRobotPose(robot.id,{x:0,y:0,theta:0});Object.assign(loaded,{x:-1.5,y:15});Object.assign(empty,{x:1.75,y:15});
  return {game,robot,loaded,empty,passenger:game.pins.find(pin=>pin.supportId===loaded.id)};
}

function positionAtStack(sim,offset=0){
  const game=sim.override,robot=game.robots[0];
  const cup=game.cups.find(candidate=>candidate.x===0&&candidate.y===-23.54),pin=game.pins.find(candidate=>candidate.supportId===cup.id);
  sim.fleet.entries.get(robot.id).sim.setPose(cup.x+offset,cup.y-15,0);
  game.setRobotPose(robot.id,{x:cup.x+offset,y:cup.y-15,theta:0});sim.fleet.syncView();
  return {game,robot,cup,pin};
}

function uiFixture({replay=false,kind='auto',setup='field'}={}){
  const elements=new Map(),keyboard={};let now=1000;
  const buttons=['use','place','drop','flip','load'].map(gameAction=>({dataset:{gameAction},listeners:{},addEventListener(name,callback){this.listeners[name]=callback;}}));
  const liftButtons=[-2,2].map(lift=>({dataset:{lift:String(lift)},listeners:{},addEventListener(name,callback){this.listeners[name]=callback;}}));
  function element(id){
    if(!elements.has(id))elements.set(id,{value:id==='gameObject'?kind:id==='gameMode'?'practice':'0',textContent:'',listeners:{},children:[],
      addEventListener(name,callback){this.listeners[name]=callback;},replaceChildren(){this.children=[];},append(child){this.children.push(child);}});
    return elements.get(id);
  }
  const context=vm.createContext({module:{exports:{}},require:filename=>filename==='./override'?Game:Dynamics,
    document:{getElementById:element,querySelectorAll:selector=>selector==='[data-game-action]'?buttons:selector==='[data-lift]'?liftButtons:[],
      createElement:()=>({textContent:'',children:[],append(child){this.children.push(child);}})},
    window:{addEventListener(name,callback){keyboard[name]=callback;}},performance:{now:()=>now}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../simulator/override-ui.js'),'utf8'),context);
  const sim=new VexRobotSimulator(Control.fromModule(compiled));
  if(setup==='cup')assert(scoringFixture(sim,'cup').ok);
  else {assert(UI.startGame(sim,'practice').ok);positionAtStack(sim);}
  const attached=context.module.exports.attach(sim,{replay});
  const update=()=>{now+=200;attached.update();};update();
  return {sim,element,buttons,liftButtons,keyboard,attached,update};
}

test('blocked nearest loaded Cup cannot hide a reachable empty Cup beside it',()=>{
  const {game,robot,loaded,empty,passenger}=cupsAtGripper(),preload=robot.possession.pinId;
  assert(Math.hypot(loaded.x-empty.x,loaded.y-empty.y)>=3.16);
  const hint=game.pickupHint(robot.id,'cup');assert(hint.reachable);assert.equal(hint.id,empty.id);
  assert(game.interact(robot.id,'pickup','cup').ok);
  assert.equal(robot.possession.cupId,empty.id);assert.equal(robot.possession.pinId,preload);
  assert.equal(loaded.status,'field');assert.equal(passenger.status,'field');assert.equal(passenger.supportId,loaded.id);
});

test('empty Cup is compatible with preload Pin and follows robot pose and finite lift',()=>{
  const {game,robot,empty}=cupsAtGripper();game.cups=[empty];
  const preload=robot.possession.pinId;assert(game.interact(robot.id,'pickup','cup').ok);
  game.setRobotPose(robot.id,{x:20,y:10,theta:90});
  const pose=game.objectPose(empty);assert(Math.abs(pose.x-35)<1e-8);assert(Math.abs(pose.y-10)<1e-8);
  assert.equal(robot.possession.pinId,preload);
  assert(game.setLiftTarget(robot.id,4).ok);assert.equal(game.objectPose(empty).z,0);
  game.tick(.01);assert(Math.abs(game.objectPose(empty).z-.24)<1e-8);
});

test('loaded Cup correctly requires a free Pin slot and reports why without mutation',()=>{
  const {game,robot,loaded}=cupsAtGripper();game.cups=[loaded];
  const before=game.getState(),hint=game.pickupHint(robot.id,'cup');
  assert.equal(hint.reachable,false);assert.match(hint.reason,/Cup.*Pin.*поставьте/);
  const result=game.interact(robot.id,'pickup','cup');assert.equal(result.ok,false);assert.equal(result.error,hint.reason);
  assert.deepEqual(game.getState(),before);
});

test('Cup with supported Pin moves and releases both when possession slots are free',()=>{
  const {game,robot,loaded,passenger}=cupsAtGripper();game.cups=[loaded];
  assert(game.placePin(robot.possession.pinId,'g-red-sw').ok);
  assert(game.interact(robot.id,'pickup','cup').ok);
  assert.equal(robot.possession.cupId,loaded.id);assert.equal(robot.possession.pinId,passenger.id);
  assert.equal(passenger.status,'held');assert.equal(passenger.supportId,loaded.id);
  assert.equal(game.objectPose(passenger).z,game.objectPose(loaded).z+3.25);
  assert(game.interact(robot.id,'drop','cup').ok);
  assert.equal(robot.possession.pinId,null);assert.equal(robot.possession.cupId,null);
  assert.equal(loaded.status,'field');assert.equal(passenger.status,'field');assert(loaded.body);
});

test('test-only Cup fixture takes, lifts, drives and physically nests Cup through the original four controllers',()=>{
  const sim=new VexRobotSimulator(Control.fromModule(compiled));assert(scoringFixture(sim,'cup').ok);
  const game=sim.override,robot=game.robots.find(candidate=>candidate.id===sim.activeRobotId),goal=game.goal('g-red-sw');
  assert.equal(sim.fleet.entries.size,4);assert.equal(game.phase,'driver');assert.equal(sim.gameObjectKind,'cup');
  assert.equal(robot.possession.pinId,null);assert.equal(goal.stack.length,1);assert.equal(goal.stack[0].type,'pin');
  assert(game.pickupHint(robot.id,'cup').reachable);assert(game.interact(robot.id,'pickup','cup').ok);
  const cup=game.cup(robot.possession.cupId),initial=game.objectPose(cup);
  const height=game.placementHint(robot.id,'cup').minimumLift;assert.equal(height,3.25);
  assert(game.setLiftTarget(robot.id,height).ok);assert.equal(game.objectPose(cup).z,initial.z);
  sim.update(.01);assert(game.objectPose(cup).z>initial.z);assert(game.objectPose(cup).z<height);
  for(let tick=0;tick<1500&&!game.placementHint(robot.id,'cup').ready;tick++){
    sim.holonomicArcade(.35,0,0);sim.update(.01);
  }
  const hint=game.placementHint(robot.id,'cup');assert(hint.ready,JSON.stringify(hint));assert(game.objectPose(cup).y>initial.y+10);
  sim.holonomicArcade(0,0,0);assert(game.interact(robot.id,'place','cup').ok);assert(!cup.placed);
  for(let tick=0;tick<100&&!cup.placed;tick++)sim.update(.01);
  assert(cup.placed);assert.equal(cup.goalId,goal.id);assert.equal(robot.possession.cupId,null);
  assert.deepEqual(goal.stack.map(item=>item.type),['pin','cup']);
});

test('single auto button captures Cup and updates hints and finite lift controls without changing selection',()=>{
  const {sim,element,buttons,update}=uiFixture({setup:'cup'});
  assert.equal(element('gameObject').value,'auto');
  assert.match(element('cupPickupHint').textContent,/Можно взять/);
  const useButton=buttons.find(button=>button.dataset.gameAction==='use');assert.equal(useButton.disabled,false);
  assert.match(useButton.textContent,/Взять Cup \[F\]/);useButton.listeners.click();
  const robot=sim.override.robots.find(candidate=>candidate.id===sim.activeRobotId);
  assert(robot.possession.cupId);assert.equal(element('gameObject').value,'auto');assert.equal(sim.gameObjectKind,'cup');
  assert.match(element('gameFeedback').textContent,/Cup в захвате/);assert.equal(useButton.disabled,true);
  assert.equal(element('gameLiftToGoal').disabled,false);
  element('gameLiftToGoal').listeners.click();assert.equal(robot.manipulator.height,0);assert.equal(robot.manipulator.target,3.25);
  sim.update(.01);update();assert(robot.manipulator.height>0);assert(robot.manipulator.height<3.25);
  const html=fs.readFileSync(path.join(__dirname,'../simulator/index.html'),'utf8');
  assert.equal((html.match(/data-game-action="use"/g)||[]).length,1);assert.match(html,/<option value="auto">/);
  assert.doesNotMatch(html,/data-game-kind|gameCupTutorial|gameTutorial|Учебная сцена/);
  assert.equal(UI.startTutorial,undefined);
});

test('F grabs an official-field Cup with Pin together and auto R releases both; phase and replay remain guarded',()=>{
  const current=uiFixture(),useButton=current.buttons.find(button=>button.dataset.gameAction==='use');
  assert.match(useButton.textContent,/Cup \+ Pin/);
  let prevented=false;current.keyboard.keydown({key:'а',code:'KeyF',preventDefault(){prevented=true;}});
  assert(prevented);assert(current.sim.override.robots[0].possession.cupId);assert(current.sim.override.robots[0].possession.pinId);
  assert.match(current.element('gameFeedback').textContent,/Cup \+ Pin в захвате/);
  current.attached.action('drop');current.keyboard.keydown({key:'f',preventDefault(){}});
  assert(current.sim.override.robots[0].possession.cupId);
  current.attached.action('drop');assert.equal(current.sim.override.robots[0].possession.cupId,null);assert.equal(current.sim.override.robots[0].possession.pinId,null);
  current.sim.override.phase='autonomous';current.update();assert.equal(useButton.disabled,true);
  const phaseState=current.sim.override.getState();current.keyboard.keydown({key:'f',preventDefault(){}});assert.deepEqual(current.sim.override.getState(),phaseState);
  const replay=uiFixture({replay:true}),before=replay.sim.override.getState();
  assert.equal(replay.element('gameLiftToGoal').disabled,true);
  replay.buttons.find(button=>button.dataset.gameAction==='use').listeners.click();
  replay.keyboard.keydown({key:'f',preventDefault(){throw Error('replay must not handle key');}});
  assert.deepEqual(replay.sim.override.getState(),before);
});

test('normal practice preserves inventory and field layout with free hands; match and optional preloads stay original',()=>{
  const sim=new VexRobotSimulator(Control.fromModule(compiled)),original=new Game();
  const field=original.getState(),preloadIds=original.robots.map(robot=>robot.possession.pinId);
  assert(UI.startGame(sim,'practice').ok);
  assert.equal(sim.fleet.entries.size,4);assert.equal(sim.override.pins.length,63);assert.equal(sim.override.cups.length,56);
  assert(sim.override.robots.every(robot=>!robot.possession.pinId&&!robot.possession.cupId));
  for(const id of preloadIds){const pin=sim.override.pin(id);assert.equal(pin.location,'alliance-station');assert.equal(pin.status,'field');}
  assert.deepEqual(sim.override.pins.filter(pin=>!preloadIds.includes(pin.id)),field.pins.filter(pin=>!preloadIds.includes(pin.id)));
  assert.deepEqual(sim.override.cups,field.cups);assert.deepEqual(sim.override.goals,field.goals);
  assert(UI.startGame(sim,'practice',{}, {preload:true}).ok);
  assert.deepEqual(sim.override.robots.map(robot=>robot.possession.pinId),preloadIds);
  assert(UI.startGame(sim,'match',{}, {preload:false}).ok);assert.equal(sim.override.phase,'autonomous');
  assert.deepEqual(sim.override.robots.map(robot=>robot.possession.pinId),preloadIds);
});

test('auto pickup on unchanged field follows original WASM strafe, lifts Cup plus Pin and physically drops both',()=>{
  const sim=new VexRobotSimulator(Control.fromModule(compiled));assert(UI.startGame(sim,'practice').ok);
  const {game,robot,cup,pin}=positionAtStack(sim,-12),initial=game.objectPose(cup);
  assert.equal(game.pickupHint(robot.id,'auto').reachable,false);
  for(let tick=0;tick<1000&&!game.pickupHint(robot.id,'auto').reachable;tick++){sim.holonomicArcade(0,.35,0);sim.update(.01);}
  sim.holonomicArcade(0,0,0);
  const hint=game.pickupHint(robot.id,'auto');assert(hint.reachable,JSON.stringify(hint));assert.equal(hint.kind,'cup');assert.equal(hint.id,cup.id);
  assert.equal(cup.x,initial.x);assert.equal(cup.y,initial.y);assert(game.interact(robot.id,'use','auto').ok);
  assert.equal(robot.possession.cupId,cup.id);assert.equal(robot.possession.pinId,pin.id);
  assert(game.setLiftTarget(robot.id,6).ok);sim.update(.01);assert(game.objectPose(cup).z>0&&game.objectPose(cup).z<6);
  for(let tick=0;tick<30;tick++)sim.update(.01);
  assert.equal(game.objectPose(cup).z,6);assert.equal(game.objectPose(pin).z,9.25);
  assert(game.interact(robot.id,'drop','auto').ok);assert.equal(pin.supportId,cup.id);assert(cup.body);
  assert.equal(robot.possession.pinId,null);assert.equal(robot.possession.cupId,null);
  sim.update(.01);assert.equal(pin.supportId,cup.id);assert(game.objectPose(cup).z<6);
  for(let tick=0;tick<100;tick++)sim.update(.01);
  assert.equal(game.objectPose(cup).z,0);assert.equal(game.objectPose(pin).z,0);
  assert.equal(pin.supportId,null);assert(cup.lying&&pin.lying);
});

test('auto R sets down an upright Cup plus Pin at floor height without splitting the bundle',()=>{
  const current=uiFixture(),game=current.sim.override,robot=game.robots[0];current.attached.action('use');
  const cup=game.cup(robot.possession.cupId),pin=game.pin(robot.possession.pinId);
  current.keyboard.keydown({code:'KeyR',key:'к',preventDefault(){}});
  for(let tick=0;tick<10;tick++)current.sim.update(.01);
  assert.equal(robot.possession.cupId,null);assert.equal(robot.possession.pinId,null);
  assert.equal(pin.supportId,cup.id);assert.equal(game.objectPose(cup).z,0);assert.equal(game.objectPose(pin).z,3.25);
});

test('auto placement releases a held bundle together and captures both only after descending into a valid stack',()=>{
  const sim=new VexRobotSimulator(Control.fromModule(compiled));assert(UI.startGame(sim,'practice').ok);
  const {game,robot,cup,pin}=positionAtStack(sim),goal=game.goal('g-red-sw');
  assert(game.interact(robot.id,'use','auto').ok);
  const basePin=game.pins.find(candidate=>candidate.location==='alliance-station'&&candidate.allianceColor==='red');
  assert(game.placePin(basePin.id,goal.id).ok);
  sim.fleet.entries.get(robot.id).sim.setPose(goal.x,goal.y-15,0);game.setRobotPose(robot.id,{x:goal.x,y:goal.y-15,theta:0});sim.fleet.syncView();
  assert.equal(game.interactionKind(robot.id,'place','auto'),'cup');assert(game.setLiftTarget(robot.id,5).ok);
  for(let tick=0;tick<25;tick++)sim.update(.01);
  assert(game.placementHint(robot.id,'cup').ready);assert(game.interact(robot.id,'place','auto').ok);
  assert.equal(cup.placed,false);assert.equal(pin.placed,false);
  for(let tick=0;tick<100&&!cup.placed;tick++)sim.update(.01);
  assert.deepEqual(goal.stack.map(item=>item.id),[basePin.id,cup.id,pin.id]);assert(cup.placed&&pin.placed);
  assert.equal(robot.possession.pinId,null);assert.equal(robot.possession.cupId,null);
});

test('auto controls keep possession limits and choose the legal layer for independent Pin and Cup slots',()=>{
  const {game,robot,empty}=cupsAtGripper();game.cups=[empty];assert(game.interact(robot.id,'use','auto').ok);
  const goal=game.goal('g-red-sw');game.setRobotPose(robot.id,{x:goal.x-4,y:goal.y-15,theta:0});
  assert.equal(game.interactionKind(robot.id,'place','auto'),'pin');assert(game.placementHint(robot.id,'pin').ready);
  const before=game.getState();assert.equal(game.interact(robot.id,'pickup','auto').ok,false);assert.deepEqual(game.getState(),before);
  assert(game.interact(robot.id,'place','auto').ok);assert.equal(robot.possession.pinId,null);assert.equal(robot.possession.cupId,empty.id);
});

test('auto pickup handles a lone lying Pin while explicit type selection remains available',()=>{
  const sim=new VexRobotSimulator(Control.fromModule(compiled));assert(UI.startGame(sim,'practice').ok);
  const game=sim.override,robot=game.robots[0],pin=game.pins.find(candidate=>candidate.location==='field'&&candidate.lying&&candidate.x===-47.09&&candidate.y===42.09);
  sim.fleet.entries.get(robot.id).sim.setPose(pin.x,pin.y-15,0);game.setRobotPose(robot.id,{x:pin.x,y:pin.y-15,theta:0});sim.fleet.syncView();
  const hint=game.interactionHint(robot.id,'auto');assert.equal(hint.kind,'pin');assert(hint.reachable);
  assert(game.interact(robot.id,'use','auto').ok);assert.equal(robot.possession.pinId,pin.id);assert.equal(robot.possession.cupId,null);
  assert(game.interact(robot.id,'drop','pin').ok);assert(game.interact(robot.id,'pickup','pin').ok);
});

test('F switches nearby Toggle with the same button and keeps an already-held bundle intact',()=>{
  const current=uiFixture(),game=current.sim.override,robot=game.robots[0];
  current.attached.action('use');const possession={...robot.possession};
  current.sim.fleet.entries.get(robot.id).sim.setPose(0,55,0);game.setRobotPose(robot.id,{x:0,y:55,theta:0});current.sim.fleet.syncView();current.update();
  const useButton=current.buttons.find(button=>button.dataset.gameAction==='use');assert.equal(useButton.disabled,false);assert.match(useButton.textContent,/Toggle \[F\]/);
  current.keyboard.keydown({key:'а',code:'KeyF',preventDefault(){}});
  assert.equal(game.toggles.find(toggle=>toggle.id==='toggle-north').state,'red');assert.deepEqual(robot.possession,possession);
  assert.match(current.element('gameFeedback').textContent,/Toggle переключён/);
  game.setRobotPose(robot.id,{x:0,y:50,theta:0});assert.equal(game.toggleHint(robot.id).reachable,false);
  const before=game.getState();assert.equal(game.interact(robot.id,'use','auto').ok,false);assert.deepEqual(game.getState(),before);
});

test('context F chooses the nearer reachable target when both a Cup and Toggle are nearby',()=>{
  const {game,robot,empty}=cupsAtGripper();game.cups=[empty];Object.assign(empty,{x:0,y:15});
  game.toggles=[{id:'toggle-test',x:0,y:12,state:'yellow',seated:true,robotContact:false}];
  assert.equal(game.interactionHint(robot.id,'auto').action,'toggle');assert(game.interact(robot.id,'use','auto').ok);assert.equal(robot.possession.cupId,null);
  game.toggles[0].y=15.5;assert.equal(game.interactionHint(robot.id,'auto').action,'pickup');assert(game.interact(robot.id,'use','auto').ok);assert.equal(robot.possession.cupId,empty.id);
});

test('held E/Q smoothly raise/lower lift and held objects; release, bounds, pause and blur stop movement',()=>{
  const current=uiFixture(),robot=current.sim.override.robots[0];current.attached.action('use');
  const cup=current.sim.override.cup(robot.possession.cupId),pin=current.sim.override.pin(robot.possession.pinId);
  const press=code=>current.keyboard.keydown({code,key:code==='KeyE'?'у':'й',preventDefault(){}});
  const step=count=>{for(let tick=0;tick<count;tick++){current.attached.stepControls(.01);current.sim.update(.01);}};
  press('KeyE');assert.equal(robot.manipulator.height,0);step(25);assert(Math.abs(robot.manipulator.height-6)<1e-8);
  assert.equal(current.sim.override.objectPose(cup).z,robot.manipulator.height);assert.equal(current.sim.override.objectPose(pin).z,robot.manipulator.height+3.25);
  current.keyboard.keyup({code:'KeyE',key:'e'});const stopped=robot.manipulator.height;step(10);assert.equal(robot.manipulator.height,stopped);
  press('KeyQ');step(10);assert(Math.abs(robot.manipulator.height-3.6)<1e-8);step(30);assert.equal(robot.manipulator.height,0);
  current.keyboard.keyup({code:'KeyQ'});press('KeyE');step(200);assert.equal(robot.manipulator.height,40);
  current.keyboard.blur();assert.equal(robot.manipulator.target,40);
  press('KeyQ');step(5);current.keyboard.blur();const blurred=robot.manipulator.height;step(10);assert.equal(robot.manipulator.height,blurred);
  press('KeyQ');current.sim.isPaused=true;step(10);assert.equal(robot.manipulator.height,blurred);
  current.sim.isPaused=false;step(10);assert.equal(robot.manipulator.height,blurred);
});

test('lift buttons and keys preserve phase, replay, focused-input and modifier locks',()=>{
  const current=uiFixture(),robot=current.sim.override.robots[0];
  current.liftButtons[1].listeners.click();assert.equal(robot.manipulator.target,2);assert.equal(robot.manipulator.height,0);
  for(const blocked of [{ctrlKey:true},{target:{matches:()=>true}}]){
    current.keyboard.keydown({code:'KeyE',...blocked,preventDefault(){throw Error('ignored key');}});current.attached.stepControls(.01);
    assert.equal(robot.manipulator.target,2);
  }
  current.sim.override.phase='autonomous';current.update();const before=current.sim.override.getState();
  current.liftButtons[1].listeners.click();current.keyboard.keydown({code:'KeyE',preventDefault(){}});current.attached.stepControls(.01);
  assert.deepEqual(current.sim.override.getState(),before);assert(current.liftButtons.every(button=>button.disabled));
  const replay=uiFixture({replay:true}),replayState=replay.sim.override.getState();
  replay.liftButtons[1].listeners.click();replay.keyboard.keydown({code:'KeyE',preventDefault(){throw Error('replay key');}});replay.attached.stepControls(.01);
  assert.deepEqual(replay.sim.override.getState(),replayState);
  const source=fs.readFileSync(path.join(__dirname,'../simulator/simulator.js'),'utf8');
  assert.match(source,/!sim.matchMode&&keysDown\['e'\]/);assert.match(source,/!sim.matchMode&&keysDown\['q'\]/);assert.match(source,/gameUI.stepControls\(PHYSICS_DT\)/);
});
