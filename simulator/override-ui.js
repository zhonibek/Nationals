(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory(require('./override'));
  else root.OverrideUI=factory(root.OverrideGame);
})(globalThis,function(Game){
  'use strict';

  function describe(game,robotId,selection='auto'){
    const robot=game.robots.find(candidate=>candidate.id===robotId),kind=game.interactionKind(robotId,'place',selection);
    const pickup=game.pickupHint(robotId,selection),placement=game.placementHint(robotId,kind),interaction=game.interactionHint(robotId,selection);
    const held=robot?.possession[kind+'Id'];
    const pickupText=pickup?`${pickup.reason}${pickup.id?` · ${pickup.id}: ${pickup.distance.toFixed(1)}″ to the fork, height difference ${pickup.heightError.toFixed(1)}″`:''}`:'No robot selected';
    const target=interaction?.action==='toggle'?'Toggle':interaction?.kind==='cup'?(interaction.passengerId?'Cup + Pin':'Cup'):'Pin';
    return {kind,pickup,placement,held,interaction,target,pickupText,
      interactionText:interaction?`F → ${target}: ${interaction.reason}`:'No robot selected',
      placementText:placement?`${placement.reason}${placement.goalId?` · ${placement.goalId}: error ${placement.distance.toFixed(1)}″, lift ${placement.minimumLift.toFixed(1)}–${placement.maximumLift.toFixed(1)}″`:''}`:'No robot selected',
      possessionText:robot?`Pin: ${robot.possession.pinId||'none'} · Cup: ${robot.possession.cupId||'none'}`:'No robot selected'};
  }

  function startGame(sim,mode,choices={}, {preload=false}={}){
    const result=sim.startGame(mode,choices);
    if(!result.ok)return result;
    if(mode==='practice'&&!preload){
      for(const robot of sim.override.robots){
        const pin=sim.override.pin(robot.possession.pinId);
        if(pin?.location!=='preload')continue;
        pin.location='alliance-station';pin.status='field';robot.possession.pinId=null;
      }
      sim.fleet.syncView();
    }
    return result;
  }

  function attach(sim,{replay=false}={}){
    const element=id=>document.getElementById(id);
    sim.solidGameObjects=true;sim.showGameColliders=false;sim.gameObjectKind='pin';
    let nextUpdate=0;
    const liftKeys={up:false,down:false};
    const feedback=message=>{element('gameFeedback').textContent=message;};
    const selectedKind=()=>element('gameObject').value;
    const active=()=>!replay&&sim.matchMode&&sim.override.phase==='driver'&&!sim.override.matchEnded&&!sim.override.disqualifiedRobots.includes(sim.activeRobotId);
    const translations={
      'Start Override first':'Press Start game first',
      'Manual actions disabled in this phase':'During the first 15 seconds of a match the autonomous routine is in control',
      'no reachable object':'No object at the front fork: check the distance and height',
      'already holding this object type':'This grabber slot is already in use. Release or place the object',
      'nothing held':'The grabber is empty',
      'no toggle within reach':'Drive to the Toggle at the middle of the wall',
      'approach an alliance loader':'Drive to your alliance loader',
      'Stack must alternate Pin / Cup':'A stack must alternate Pin → Cup → Pin',
      'remove the supported Pin before flipping the Cup':'Take the Pin off the Cup first: choose Pin and release it',
      'opposing alliance goal is protected':'The opponent goal is protected',
      'Align gripper with goal (within 1.1 in)':'Line the fork up with the goal opening: error at most 1.1″'
    };
    function action(name,selection=selectedKind()){
      if(replay)return;
      const hint=name==='use'?sim.override.interactionHint(sim.activeRobotId,selection):null;
      const kind=hint?.kind||sim.override.interactionKind(sim.activeRobotId,name,selection);
      const result=!sim.matchMode?{ok:false,error:'Start Override first'}:sim.override.interact(sim.activeRobotId,name,selection);
      const target=kind==='cup'?(hint?.passengerId?'Cup + Pin':'Cup'):'Pin';
      feedback(result.ok?(hint?.action==='toggle'||name==='toggle'?'Toggle switched':name==='place'?'Object released. Points appear after it lands in the goal':
        ['use','pickup'].includes(name)?`${target} held`:name==='drop'?'Object released: falling and collisions continue':'Action done'):(translations[result.error]||result.error));
      nextUpdate=0;update();
    }
    function liftBy(delta){
      if(replay)return;
      const robot=sim.override.robots.find(candidate=>candidate.id===sim.activeRobotId);
      const result=!sim.matchMode?{ok:false,error:'Start Override first'}:sim.override.setLiftTarget(robot.id,Math.max(0,Math.min(40,robot.manipulator.target+delta)));
      feedback(result.ok?`Lift moving to ${robot.manipulator.target.toFixed(1)}″`:(translations[result.error]||result.error));
      nextUpdate=0;update();
    }
    function stopLift(){
      const moving=liftKeys.up||liftKeys.down;liftKeys.up=false;liftKeys.down=false;
      if(moving&&active()){
        const robot=sim.override.robots.find(candidate=>candidate.id===sim.activeRobotId);
        sim.override.setLiftTarget(robot.id,robot.manipulator.height);
      }
    }
    function stepControls(dt){
      if(!Number.isFinite(dt)||dt<=0||dt>1||!active()||sim.isPaused){stopLift();return;}
      const direction=Number(liftKeys.up)-Number(liftKeys.down);
      if(!direction)return;
      const robot=sim.override.robots.find(candidate=>candidate.id===sim.activeRobotId),lift=robot.manipulator;
      sim.override.setLiftTarget(robot.id,Math.max(0,Math.min(40,lift.height+direction*lift.speed*dt)));
    }
    element('gameLiftToGoal').addEventListener('click',()=>{
      if(replay||!sim.matchMode)return;
      const kind=sim.override.interactionKind(sim.activeRobotId,'place',selectedKind()),placement=sim.override.placementHint(sim.activeRobotId,kind);
      if(!Number.isFinite(placement?.minimumLift)){feedback('Take an object first');return;}
      const result=sim.override.setLiftTarget(sim.activeRobotId,placement.minimumLift);
      feedback(result.ok?`Lift moving to ${placement.minimumLift.toFixed(2)}″. Wait for the right height`:result.error);
      nextUpdate=0;update();
    });
    element('gameFinish').addEventListener('click',()=>{
      if(replay||!sim.matchMode)return;
      stopLift();sim.holonomicArcade(0,0,0);for(const entry of sim.fleet.entries.values())sim.fleet.stop(entry,'Cancelled');
      sim.override.stopMatch();feedback('Match over. Waiting for objects to settle, then the score is frozen');
    });
    element('gameSolid').addEventListener('change',event=>{sim.solidGameObjects=event.target.checked;});
    element('gameColliders').addEventListener('change',event=>{sim.showGameColliders=event.target.checked;});
    element('gameRobot').addEventListener('change',stopLift);
    element('gameMode').addEventListener('change',()=>{nextUpdate=0;update();});
    element('gameObject').addEventListener('change',()=>{nextUpdate=0;update();});
    for(const button of document.querySelectorAll('[data-game-action]'))button.addEventListener('click',()=>action(button.dataset.gameAction));
    for(const button of document.querySelectorAll('[data-lift]'))button.addEventListener('click',()=>liftBy(Number(button.dataset.lift)));
    const liftKey=event=>({KeyE:'up',KeyQ:'down'}[event.code]||{e:'up',q:'down'}[event.key?.toLowerCase()]);
    window.addEventListener('keydown',event=>{
      if(replay||event.ctrlKey||event.altKey||event.metaKey||event.target?.matches?.('input,select,textarea,[contenteditable]'))return;
      const direction=liftKey(event);
      if(direction&&sim.matchMode){
        event.preventDefault();if(active()&&!sim.isPaused)liftKeys[direction]=true;return;
      }
      if(event.repeat)return;
      const commands={f:'use',g:'place',r:'drop',t:'flip'},physical={KeyF:'use',KeyG:'place',KeyR:'drop',KeyT:'flip'};
      const name=physical[event.code]||commands[event.key?.toLowerCase()];
      if(name){event.preventDefault();action(name);}
    });
    window.addEventListener('keyup',event=>{
      const direction=liftKey(event);
      if(!direction||!liftKeys[direction])return;
      liftKeys[direction]=false;
      if(!liftKeys.up&&!liftKeys.down&&active()){
        const robot=sim.override.robots.find(candidate=>candidate.id===sim.activeRobotId);
        sim.override.setLiftTarget(robot.id,robot.manipulator.height);
      }
    });
    window.addEventListener('blur',stopLift);
    function updateCalculator(){
      try{
        const values=Object.fromEntries(['coloredHalves','yellowHalves','midfieldRobots','autonomousBonus'].map(name=>{
          const raw=String(element('calc-'+name).value??'').trim();
          if(!raw)throw Error('Fill in all counts; enter 0 for missing objects');
          return [name,Number(raw)];
        }));
        const total=Game.calculatePoints(values);
        element('manualPoints').textContent=String(total)+' points';
        element('manualFormula').textContent=`${values.coloredHalves} × 5 + ${values.yellowHalves} × 10 + ${values.midfieldRobots} × 8 + ${values.autonomousBonus} = ${total}`;
        element('calculatorError').textContent='';
      }catch(error){
        element('manualPoints').textContent='—';element('manualFormula').textContent='Calculation unavailable: fix the fields';
        element('calculatorError').textContent=error.message;
      }
    }
    element('pointCalculator').addEventListener('input',updateCalculator);
    element('pointCalculator').addEventListener('change',updateCalculator);
    for(const color of ['red','blue'])element('calculator-'+color).addEventListener('click',()=>{
      const row=sim.override.scoreBreakdown()[color];
      for(const name of ['coloredHalves','yellowHalves','midfieldRobots','autonomousBonus'])element('calc-'+name).value=String(row[name]);
      updateCalculator();
    });
    element('calculatorReset').addEventListener('click',()=>{
      for(const name of ['coloredHalves','yellowHalves','midfieldRobots','autonomousBonus'])element('calc-'+name).value='0';
      updateCalculator();
    });
    updateCalculator();
    function update(){
      const now=performance.now();if(now<nextUpdate)return;nextUpdate=now+100;
      const game=sim.override,details=describe(game,sim.activeRobotId,selectedKind()),enabled=active();
      if(!enabled)stopLift();
      sim.gameObjectKind=details.interaction?.action==='pickup'&&details.interaction.reachable?details.interaction.kind:details.kind;
      element('gameModeStatus').textContent=replay?'Viewing a recording: game controls are disabled':!sim.matchMode?'Motion test mode: game objects are not involved. Press Start game':game.phase==='autonomous'?'Autonomous: manual control starts in '+Math.max(0,15-game.clock).toFixed(1)+' s':game.matchEnded?'Game over':'Game physics on · WASD — move · ←/→ — turn · E/Q — lift up/down · F take / Toggle · G place · R release · T flip';
      element('actionHint').textContent=details.interactionText;
      element('possessionStatus').textContent=details.possessionText;
      element('pickupHint').textContent='Pin: '+describe(game,sim.activeRobotId,'pin').pickupText;
      element('cupPickupHint').textContent='Cup: '+describe(game,sim.activeRobotId,'cup').pickupText;
      element('placementHint').textContent=details.placementText;
      element('gameFinish').disabled=replay||!sim.matchMode||game.matchEnded;
      element('practicePreload').disabled=replay||element('gameMode').value==='match';
      element('gameLiftToGoal').disabled=!enabled||!details.held||!Number.isFinite(details.placement?.minimumLift);
      for(const button of document.querySelectorAll('[data-game-action]')){
        const name=button.dataset.gameAction;
        button.disabled=!enabled||(name==='use'&&!details.interaction?.reachable)||(name==='place'&&!details.placement?.ready)||(['flip','drop'].includes(name)&&!details.held);
        if(name==='use'){
          button.textContent=details.interaction?.reachable?(details.interaction.action==='toggle'?'Switch Toggle [F]':`Take ${details.target} [F]`):'Take / Toggle [F]';
          button.title=details.interactionText;
        }
      }
      for(const button of document.querySelectorAll('[data-lift]'))button.disabled=!enabled;
      element('btnPlay').textContent=sim.matchMode?'▶ Continue game':'▶ Run route';
      const breakdown=game.scoreBreakdown();
      element('scoreState').textContent=breakdown.final?'FINAL SCORE':'CURRENT SCORE · preliminary';
      element('scoreDockState').textContent=breakdown.final?'Final score':'Current · preliminary';
      for(const color of ['red','blue']){
        const row=breakdown[color];element('points-'+color).textContent=String(row.total);
        element('scoreDock-'+color).textContent=String(row.total);
        element('formula-'+color).textContent=`${row.coloredHalves} × 5 + ${row.yellowHalves} × 10 + ${row.midfieldRobots} × 8 + ${row.autonomousBonus} = ${row.total}`;
      }
      element('scoreExplanation').textContent=`Unowned yellow halves: ${breakdown.unownedYellowHalves} (0 points). A Cup on its own scores nothing. Pin halves hidden by an opaque Cup half do not count. The final score is frozen after objects settle / no later than 5 seconds after the end of the match.`;
      const body=element('goalPoints');body.replaceChildren();
      for(const goal of breakdown.goals){
        const row=document.createElement('tr');
        for(const value of [goal.id,goal.stack.map(item=>item.type==='pin'?'Pin':'Cup').join(' → ')||'empty',goal.red,goal.blue]){
          const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);
        }
        body.append(row);
      }
    }
    return {update,action,stepControls,stopLift,updateCalculator};
  }

  return {attach,describe,startGame};
});
