(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory();else root.OverrideFleet=factory();
})(globalThis,function(){
  'use strict';
  class OverrideFleet {
    constructor(host,Simulator,choices={}){
      this.host=host;this.game=host.override;this.entries=new Map();this.trace=[];this.lastTrace=-1;
      for(const robot of this.game.robots){
        const sim=new Simulator(host.productionControl.fork());sim.override=this.game;sim.managedGame=true;sim.matchMode=true;sim.activeRobotId=robot.id;sim.setPose(robot.x,robot.y,robot.theta);
        const choice=choices[robot.id]||(robot.id.endsWith('1')?'preload':'taxi');
        const steps=Array.isArray(choice)?JSON.parse(JSON.stringify(choice)):this.program(robot,choice);
        this.entries.set(robot.id,{sim,steps,index:0,pending:false,status:this.game.phase==='autonomous'?(steps.length?'Ready':'Idle'):'Manual'});
      }
    }
    program(robot,name){
      if(name==='idle')return [];
      const sign=robot.alliance==='red'?-1:1;
      if(name==='preload'&&robot.id.endsWith('1'))return [
        {type:'pose',targetX:sign*47.09,targetY:sign*38.54,targetTheta:sign<0?0:180},
        {type:'lift',height:2},{type:'game',action:'place',kind:'pin'}
      ];
      return [{type:'pose',targetX:robot.x-sign*(robot.id.endsWith('2')?2:12),targetY:robot.y,targetTheta:robot.theta}];
    }
    stop(entry,reason){
      const s=entry.sim,robot=this.game.robots.find(r=>r.id===s.activeRobotId);robot.manipulator.target=robot.manipulator.height;
      s.isRunning=false;s.currentAction=null;s.routineQueue=[];s.commandedWheelVoltages=null;s.motorVolts=[0,0,0,0];s.manualThrottle=s.manualStrafe=s.manualTurn=0;s.productionControl.reset();
      if(entry.status==='Running'||entry.status==='Ready'||['Cancelled','InvalidDt'].includes(reason))entry.status=reason;
    }
    schedule(entry){
      const s=entry.sim,step=entry.steps[entry.index];
      if(!step||['Failed','PeriodEnded','Cancelled','InvalidDt'].includes(entry.status))return;
      entry.status='Running';
      if(step.type==='lift'){
        if(!entry.pending){const result=this.game.setLiftTarget(s.activeRobotId,step.height,'autonomous');if(!result.ok){entry.status='Failed';entry.error=result.error;return;}entry.pending=true;}
        const m=this.game.robots.find(r=>r.id===s.activeRobotId).manipulator;if(Math.abs(m.height-step.height)<.05){entry.index++;entry.pending=false;}
      }else if(step.type==='game'){
        const result=this.game.interact(s.activeRobotId,step.action,step.kind,{source:'autonomous'});
        if(result.ok){entry.index++;entry.pending=false;}else{entry.status='Failed';entry.error=result.error;}
      }else if(!entry.pending){
        if(!['pose','drive','strafe','turn','spline','bezier','diagonal'].includes(step.type)){entry.status='Failed';entry.error='Unknown autonomous step';return;}
        s.queueAction({...step});s.isRunning=true;entry.pending=true;
      }else if(!s.isRunning){
        if(s.lastMotionResult==='Settled'){entry.index++;entry.pending=false;}else{entry.status='Failed';entry.error=s.lastMotionResult;}
      }
      if(entry.index===entry.steps.length)entry.status='Complete';
    }
    update(dt){
      if(!Number.isFinite(dt)||dt<=0||dt>.1){for(const e of this.entries.values())this.stop(e,'InvalidDt');return;}
      for(let remaining=dt;remaining>1e-8;){
        if(this.game.finalScore)return;
        const boundary=this.game.mode==='practice'||this.game.phase==='post_match'?Infinity:this.game.phase==='autonomous'?15:120;
        const h=Math.min(.01,remaining,boundary-this.game.clock);
        if(h<=1e-8)return;
        this.step(h);remaining-=h;
      }
    }
    step(dt){
      if(!Number.isFinite(dt)||dt<=0||dt>.1){for(const e of this.entries.values())this.stop(e,'InvalidDt');return;}
      const phase=this.game.phase;
      for(const [id,e] of this.entries){
        if(this.game.disqualifiedRobots.includes(id)){this.stop(e,'Cancelled');}else if(phase==='autonomous')this.schedule(e);else if(e.sim.isRunning||e.status==='Running'||e.status==='Ready')this.stop(e,'PeriodEnded');
        const s=e.sim,selected=id===this.host.activeRobotId&&phase==='driver'&&!this.game.disqualifiedRobots.includes(id);
        s.manualThrottle=selected?this.host.manualThrottle||0:0;s.manualStrafe=selected?this.host.manualStrafe||0:0;s.manualTurn=selected?this.host.manualTurn||0:0;
        s.update(dt);
      }
      this.game.tick(dt);
      if(phase!==this.game.phase)for(const e of this.entries.values())this.stop(e,'PeriodEnded');
      this.host.simTime+=dt;
      if(this.host.simTime-this.lastTrace>=.5){this.lastTrace=this.host.simTime;this.trace.push({time:this.host.simTime,clock:this.game.clock,phase:this.game.phase,score:this.game.score(),robots:[...this.entries].map(([id,e])=>({id,x:e.sim.x,y:e.sim.y,theta:e.sim.theta,odom:{...e.sim.odom},volts:[...e.sim.motorVolts],status:e.status,step:e.index,error:e.error}))});if(this.trace.length>600)this.trace.shift();}
      this.syncView();
    }
    syncView(){
      const e=this.entries.get(this.host.activeRobotId);if(!e)return;
      for(const key of ['x','y','theta','v','vx','vy','Vx','Vy','w','odom','odomVelocity','ekfPose','wheelOmega','motorVolts','motorCurrents','wheelSlips','batteryVoltage','pathHistory','telemetry','brainLcdLines','activeTrajectory','plannedSplineVisual','lastMotionResult'])this.host[key]=e.sim[key];
      this.host.controllerLcdLines=[`Override ${this.game.phase}`,`${this.host.activeRobotId}: ${e.status}`,e.error||`Step ${e.index}/${e.steps.length}`];
    }
    report(){return {format:'nationals-match-v1',programs:Object.fromEntries([...this.entries].map(([id,e])=>[id,e.steps])),state:this.game.getState(),events:this.game.events,trace:this.trace};}
  }
  return OverrideFleet;
});
