'use strict';

const Dynamics=require('../../simulator/override-dynamics');

function scoringFixture(sim,kind='pin'){
  const result=sim.startGame('practice',Object.fromEntries(['red-1','red-2','blue-1','blue-2'].map(id=>[id,'idle'])));
  if(!result.ok)return result;
  sim.activeRobotId='red-1';
  const game=sim.override,robot=game.robots.find(candidate=>candidate.id===sim.activeRobotId),goal=game.goal('g-red-sw');
  const pin=game.pin(robot.possession.pinId);
  if(kind==='cup'){
    const placement=game.placePin(pin.id,goal.id);
    if(!placement.ok)return placement;
    const cup=game.cups.find(candidate=>candidate.location==='alliance-station'&&candidate.alliance===robot.alliance);
    Dynamics.release(game,cup,{x:goal.x,y:goal.y-15,z:0,yaw:0});
  }else{
    robot.possession.pinId=null;
    Dynamics.release(game,pin,{x:goal.x,y:goal.y-15,z:0,yaw:0});
  }
  const entry=sim.fleet.entries.get(robot.id);
  entry.sim.setPose(goal.x,goal.y-30,0);game.setRobotPose(robot.id,{x:goal.x,y:goal.y-30,theta:0});
  sim.fleet.syncView();sim.gameObjectKind=kind;
  return {ok:true};
}

module.exports={scoringFixture};
