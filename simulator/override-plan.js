(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory(require('./override-autonomy'));else root.OverridePlan=factory(root.OverrideFleet);
})(globalThis,function(OverrideFleet){
  'use strict';
  // Drives a Tactical AI plan (macro-action IDs from the abstract tactical model) through the original Simulator:
  // every move is a pose command to the production controller, every pickup and placement goes through the
  // physical Override rules (gripper within 3 in of the object, within 1.1 in of the goal, lift at stack height).
  // Nothing is teleported. The plan is open loop: it is not re-planned when physics differs from the abstract model.
  const REACH=15;                 // OverrideGame manipulator reach: the gripper is this far ahead of the robot centre
  const HALF=9;                   // the 18 x 18 in robot body
  const WALL=72;                  // field half-width
  const GOAL_RADIUS=4.25;
  const GOAL_PASS=14.25;          // body half-width (9 in) + goal radius (4.25 in) + 1 in: driving past a goal
  const GOAL_TURN=18;             // body half-diagonal (12.7 in) + goal radius + 1 in: turning next to a goal
  const ROBOT_PASS=9.5;           // clearance from another robot's square body: our half-width + 0.5 in
  const ROBOT_TURN=13.5;          // our half-diagonal + 0.8 in
  const ROUTE_LIMIT=62.5;         // a robot driving along a wall, square to it, stays inside the ±72 in walls
  const BACK_OFF=19.5;            // before turning, back away from a goal until a turning body cannot touch it
  const PRE_APPROACH=6;           // turn to the final heading this far before the final pose, in open space
  const MACRO_SECONDS=30;         // a macro-action that takes longer than this is abandoned
  const LOAD_SETTLE_SECONDS=0.8;  // a match load is released 9 in above the outlet and falls before pickup
  const PLACE_SETTLE_SECONDS=0.6;
  const LIFT_SECONDS=3;
  const TOGGLE_REACH=13;          // a Toggle is switched with the robot centre within 16 in of it
  const PARK={x:20,y:12};         // body overlaps the midfield diamond and stays clear of the tall goal

  const degrees=radians=>radians*180/Math.PI;
  const wrap=angle=>((angle+180)%360+360)%360-180;
  const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
  const headingTo=(from,to)=>degrees(Math.atan2(to.x-from.x,to.y-from.y));
  function centreFor(point,heading,reach=REACH){
    const a=heading*Math.PI/180;return {x:point.x-reach*Math.sin(a),y:point.y-reach*Math.cos(a)};
  }
  const square=heading=>Math.round(heading/90)*90;
  const inOpenField=p=>Math.abs(p.x)<=WALL-HALF*Math.SQRT2-.5&&Math.abs(p.y)<=WALL-HALF*Math.SQRT2-.5;
  // Clearance from a point to the robot body at centre c and heading (0 when they touch).
  function bodyDistance(c,heading,point){
    const a=heading*Math.PI/180,dx=point.x-c.x,dy=point.y-c.y;
    const lx=dx*Math.cos(a)-dy*Math.sin(a),ly=dx*Math.sin(a)+dy*Math.cos(a);
    return Math.hypot(Math.max(Math.abs(lx)-HALF,0),Math.max(Math.abs(ly)-HALF,0));
  }
  function corners(c,heading){
    const a=heading*Math.PI/180,cos=Math.cos(a),sin=Math.sin(a);
    return [[-1,-1],[-1,1],[1,1],[1,-1]].map(([u,v])=>({x:c.x+u*HALF*cos+v*HALF*sin,y:c.y-u*HALF*sin+v*HALF*cos}));
  }
  // Separating-axis test for two robot bodies, true when they are closer than margin.
  function squaresOverlap(c1,h1,c2,h2,margin=0){
    const p1=corners(c1,h1),p2=corners(c2,h2);
    for(const heading of [h1,h1+90,h2,h2+90]){
      const a=heading*Math.PI/180,nx=Math.sin(a),ny=Math.cos(a);
      const s1=p1.map(p=>p.x*nx+p.y*ny),s2=p2.map(p=>p.x*nx+p.y*ny);
      if(Math.max(...s1)+margin<=Math.min(...s2)||Math.max(...s2)+margin<=Math.min(...s1))return false;
    }
    return true;
  }
  function bodyFits(c,heading,goals,robots=[]){
    const a=heading*Math.PI/180,extent=HALF*(Math.abs(Math.cos(a))+Math.abs(Math.sin(a)));
    if(Math.abs(c.x)+extent>WALL-.5||Math.abs(c.y)+extent>WALL-.5)return false;
    if(goals.some(goal=>bodyDistance(c,heading,goal)<GOAL_RADIUS+.5))return false;
    return robots.every(robot=>!squaresOverlap(c,heading,robot,robot.theta,1));
  }
  // A robot pose that puts the gripper (reach ahead of the centre) on point: the direct heading if the body fits
  // there, otherwise the nearest of the square and diagonal headings that fits.
  function approach(point,from,reach,goals,robots){
    const direct=headingTo(from,point);
    const headings=[direct,...[0,90,180,-90,45,135,-45,-135].sort((p,q)=>Math.abs(wrap(p-direct))-Math.abs(wrap(q-direct)))];
    for(const heading of headings){
      const centre=centreFor(point,heading,reach);
      if(bodyFits(centre,heading,goals,robots))return {centre,heading,fits:true};
    }
    return {centre:centreFor(point,direct,reach),heading:direct,fits:false};
  }

  function parseAction(id){
    if(typeof id!=='string'||id.length>200)throw Error('Plan action IDs must be short strings');
    const [kind,robotId,first,second]=id.split(':');
    if(kind==='park'&&robotId)return {id,kind,robotId};
    if(kind==='toggle'&&robotId&&first)return {id,kind,robotId,targetId:first};
    if((kind==='score-pin'||kind==='score-cup')&&robotId&&first&&second)return {id,kind,robotId,targetId:first,goalId:second};
    if((kind==='load-score-pin'||kind==='load-score-cup')&&robotId&&first&&second)return {id,kind,robotId,loaderId:first,goalId:second};
    throw Error(`Unknown plan action ${id}`);
  }

  // Waypoints from a to b: a shortest path on a 2 in grid that keeps the robot `pass` clear of every obstacle
  // (driving facing the direction of travel) and inside the walls, with a cost for coming within `turn` clearance,
  // then shortened to the fewest straight legs that stay clear. Points near either end are always allowed, so a
  // robot that starts or must finish next to a goal can still leave or arrive. Returns [b] when no path exists.
  const CELL=2,CELLS=Math.round(2*WALL/CELL)+1;
  // Distance from an obstacle to a point: from a goal's centre, or from the nearest edge of a robot's body.
  const gap=(o,x,y)=>o.theta===undefined?Math.hypot(o.x-x,o.y-y):bodyDistance(o,o.theta,{x,y});
  function route(a,b,obstacles){
    const near=(x,y)=>Math.hypot(x-a.x,y-a.y)<6||Math.hypot(x-b.x,y-b.y)<6;
    const blocked=(x,y)=>!near(x,y)&&(Math.abs(x)>ROUTE_LIMIT||Math.abs(y)>ROUTE_LIMIT||obstacles.some(o=>gap(o,x,y)<o.pass));
    const clear=(p,q)=>{
      const steps=Math.max(1,Math.ceil(Math.hypot(q.x-p.x,q.y-p.y)));
      for(let k=0;k<=steps;k++){const x=p.x+(q.x-p.x)*k/steps,y=p.y+(q.y-p.y)*k/steps;if(blocked(x,y))return false;}
      return true;
    };
    if(clear(a,b))return [b];
    const coord=k=>-WALL+k*CELL,cell=v=>clamp(Math.round((v+WALL)/CELL),0,CELLS-1);
    const penalty=(x,y)=>obstacles.reduce((sum,o)=>{const d=gap(o,x,y);return sum+(d<o.turn?(o.turn-d):0);},0);
    const start=cell(a.x)*CELLS+cell(a.y),goal=cell(b.x)*CELLS+cell(b.y);
    const cost=new Float64Array(CELLS*CELLS).fill(Infinity),parent=new Int32Array(CELLS*CELLS).fill(-1),done=new Uint8Array(CELLS*CELLS);
    const heap=[];
    const push=(key,node)=>{heap.push([key,node]);let i=heap.length-1;while(i>0){const up=(i-1)>>1;if(heap[up][0]<=heap[i][0])break;[heap[up],heap[i]]=[heap[i],heap[up]];i=up;}};
    const pop=()=>{const top=heap[0],last=heap.pop();if(heap.length){heap[0]=last;let i=0;for(;;){const l=2*i+1,r=l+1;let m=i;
      if(l<heap.length&&heap[l][0]<heap[m][0])m=l;if(r<heap.length&&heap[r][0]<heap[m][0])m=r;if(m===i)break;[heap[m],heap[i]]=[heap[i],heap[m]];i=m;}}return top;};
    const bx=coord(cell(b.x)),by=coord(cell(b.y));
    cost[start]=0;push(Math.hypot(coord(cell(a.x))-bx,coord(cell(a.y))-by),start);
    while(heap.length){
      const [,node]=pop();
      if(done[node])continue;
      done[node]=1;
      if(node===goal)break;
      const i=Math.floor(node/CELLS),j=node%CELLS;
      for(let di=-1;di<=1;di++)for(let dj=-1;dj<=1;dj++){
        if(!di&&!dj)continue;
        const ni=i+di,nj=j+dj;
        if(ni<0||nj<0||ni>=CELLS||nj>=CELLS)continue;
        const next=ni*CELLS+nj,x=coord(ni),y=coord(nj);
        if(done[next]||blocked(x,y))continue;
        const step=CELL*Math.hypot(di,dj)*(1+penalty(x,y)/4);
        if(cost[node]+step<cost[next]){cost[next]=cost[node]+step;parent[next]=node;push(cost[next]+Math.hypot(x-bx,y-by),next);}
      }
    }
    if(!done[goal])return [b];
    const path=[];
    for(let node=goal;node!==-1&&node!==start;node=parent[node])path.push({x:coord(Math.floor(node/CELLS)),y:coord(node%CELLS)});
    path.reverse();path[path.length-1]=b;
    const points=[];
    for(let from=a,index=0;index<path.length;){
      let reach=index;
      for(let k=path.length-1;k>index;k--)if(clear(from,path[k])){reach=k;break;}
      points.push(path[reach]);from=path[reach];index=reach+1;
    }
    return points;
  }

  class PlanRunner {
    constructor(fleet,plan){
      this.fleet=fleet;this.game=fleet.game;
      this.plan=plan.map(parseAction);this.cursor=0;this.task=null;this.pending=null;this.macro=null;this.results=[];this.motions=0;
      this.alliance=this.plan[0]?this.robot(this.plan[0].robotId)?.alliance:null;
      for(const action of this.plan){
        const robot=this.robot(action.robotId);
        if(!robot||robot.alliance!==this.alliance)throw Error(`Plan action ${action.id} names a robot outside the planning alliance`);
      }
    }
    robot(id){return this.game.robots.find(robot=>robot.id===id);}
    get done(){return this.cursor>=this.plan.length&&!this.task;}

    tick(){
      if(this.game.phase!=='driver'||this.game.matchEnded){if(this.task)this.finish({ok:false,error:'Driver period ended'});return;}
      for(let guard=0;guard<8;guard++){
        if(!this.task){
          if(this.cursor>=this.plan.length)return;
          const action=this.plan[this.cursor++];
          this.macro={id:action.id,kind:action.kind,robotId:action.robotId,startClock:this.game.clock,scoreBefore:this.margin(),motions:0,steps:[]};
          this.task=this.macroSteps(action);this.advance(undefined);
          continue;
        }
        if(this.game.clock-this.macro.startClock>MACRO_SECONDS){this.abort('Macro-action took longer than 30 s');return;}
        if(!this.pending)return;
        const result=this.poll(this.pending);
        if(!result)return;
        this.advance(result);
      }
    }
    advance(result){
      if(this.pending&&this.macro){
        const p=this.pending,robot=this.robot(p.robotId),sim=this.fleet.entries.get(p.robotId)?.sim;
        if(this.macro.steps.length<40)this.macro.steps.push({type:p.type,...(p.action?{action:p.action}:{}),ok:Boolean(result?.ok),
          ...(result?.ok?{}:{error:String(result?.error||'failed')}),clock:Number(this.game.clock.toFixed(2)),
          ...(p.type==='move'?{target:{x:+p.x.toFixed(2),y:+p.y.toFixed(2),theta:+p.theta.toFixed(1)}}:{}),
          ...(p.type==='lift'?{height:+p.height.toFixed(2)}:{}),
          ...(robot&&sim?{pose:{x:+robot.x.toFixed(2),y:+robot.y.toFixed(2),theta:+robot.theta.toFixed(1)},odomError:+Math.hypot(sim.odom.x-sim.x,sim.odom.y-sim.y).toFixed(2)}:{})});
      }
      const next=this.task.next(result);
      if(next.done){this.finish(next.value||{ok:true});return;}
      this.pending=this.start(next.value);
    }
    finish(outcome){
      if(this.pending?.type==='move')this.fleet.stop(this.pending.entry,'Complete');
      this.pending=null;this.task=null;
      const macro=this.macro;
      this.results.push({...macro,ok:Boolean(outcome.ok),error:outcome.ok?null:String(outcome.error||'failed'),
        endClock:Number(this.game.clock.toFixed(2)),seconds:Number((this.game.clock-macro.startClock).toFixed(2)),
        startClock:Number(macro.startClock.toFixed(2)),marginGain:this.margin()-macro.scoreBefore});
      this.macro=null;
    }
    abort(error){this.finish({ok:false,error});}
    margin(){const score=this.game.score(),other=this.alliance==='red'?'blue':'red';return score[this.alliance]-score[other];}

    start(step){
      const pending={...step,startClock:this.game.clock};
      if(step.type==='move'){
        // Targets are field (true) positions; the controller closes the loop on odometry, so shift them into its frame.
        const entry=this.fleet.entries.get(step.robotId),sim=entry.sim;
        const heading=step.theta+wrap(sim.odom.theta-sim.theta),theta=sim.odom.theta+wrap(heading-sim.odom.theta);
        sim.queueAction({type:'pose',targetX:step.x+sim.odom.x-sim.x,targetY:step.y+sim.odom.y-sim.y,targetTheta:theta});sim.isRunning=true;entry.status='Running';
        this.motions++;this.macro.motions++;
        return {...pending,entry};
      }
      if(step.type==='lift'){
        const result=this.game.setLiftTarget(step.robotId,step.height,'driver');
        return result.ok?pending:{...pending,result};
      }
      if(step.type==='game')return {...pending,result:this.game.interact(step.robotId,step.action,step.kind,{source:'driver'})};
      if(step.type==='wait')return pending;
      return {...pending,result:{ok:false,error:`Unknown plan step ${step.type}`}};
    }
    poll(pending){
      if(pending.result)return pending.result;
      const elapsed=this.game.clock-pending.startClock;
      if(pending.type==='move'){
        if(pending.entry.sim.isRunning)return null;
        const result=pending.entry.sim.lastMotionResult;pending.entry.status='Idle';
        return result==='Settled'?{ok:true}:{ok:false,error:`Motion ${result}`};
      }
      if(pending.type==='lift'){
        const m=this.robot(pending.robotId).manipulator;
        if(Math.abs(m.height-pending.height)<.05)return {ok:true};
        return elapsed>LIFT_SECONDS?{ok:false,error:'Lift did not reach its target'}:null;
      }
      if(pending.type==='wait')return elapsed>=pending.seconds?{ok:true}:null;
      return {ok:false,error:'Unknown pending step'};
    }

    others(robotId){return this.game.robots.filter(robot=>robot.id!==robotId);}
    obstacles(robotId){
      return [...this.game.goals.map(goal=>({x:goal.x,y:goal.y,pass:GOAL_PASS,turn:GOAL_TURN})),
        ...this.others(robotId).map(robot=>({x:robot.x,y:robot.y,theta:robot.theta,pass:ROBOT_PASS,turn:ROBOT_TURN}))];
    }
    // Back away from a nearby goal, drive square to the field around goals and robots, turn to the final heading
    // a few inches short of the destination in open space, then make the final pose.
    *travel(robotId,centre,heading){
      let robot=this.robot(robotId);
      if(Math.hypot(centre.x-robot.x,centre.y-robot.y)>3){
        const near=this.game.goals.map(goal=>({goal,d:Math.hypot(goal.x-robot.x,goal.y-robot.y)}))
          .filter(row=>row.d<BACK_OFF-.5&&row.d>1e-6).sort((p,q)=>p.d-q.d)[0];
        if(near){
          const away={x:near.goal.x+(robot.x-near.goal.x)/near.d*BACK_OFF,y:near.goal.y+(robot.y-near.goal.y)/near.d*BACK_OFF};
          if(bodyFits(away,robot.theta,this.game.goals.filter(goal=>goal!==near.goal)))yield {type:'move',robotId,...away,theta:robot.theta};
        }
      }
      robot=this.robot(robotId);
      const a=heading*Math.PI/180;
      let pre={x:centre.x-PRE_APPROACH*Math.sin(a),y:centre.y-PRE_APPROACH*Math.cos(a)};
      const far=Math.hypot(centre.x-robot.x,centre.y-robot.y)>=2*PRE_APPROACH;
      if(!far||!bodyFits(pre,heading,this.game.goals))pre=centre;
      // In open field each leg is driven facing its direction of travel (or a quarter turn from it), the narrowest
      // body between goals; a leg that reaches into the band along a wall is driven square to the field, since a
      // diagonal body is too wide for it. The robot turns to the final heading only before the destination.
      const points=far?route({x:robot.x,y:robot.y},pre,this.obstacles(robotId)):[];
      let from={x:robot.x,y:robot.y},facing=robot.theta;
      for(const point of points){
        const direction=headingTo(from,point),open=inOpenField(from)&&inOpenField(point);
        const cruise=open?direction+90*Math.round(wrap(facing-direction)/90):square(facing);
        const result=yield {type:'move',robotId,...point,theta:cruise};
        if(!result.ok)return result;
        from=point;facing=cruise;
      }
      if(pre!==centre&&Math.abs(wrap(facing-heading))>2){
        const turned=yield {type:'move',robotId,...pre,theta:heading};
        if(!turned.ok)return turned;
      }
      return yield {type:'move',robotId,...centre,theta:heading};
    }
    // Gripper onto the object, then pick it up; a missed pickup is corrected by the measured gripper offset.
    *grab(robotId,kind,objectId){
      let last={ok:false,error:'Object is not on the field'};
      for(let attempt=0;attempt<3;attempt++){
        const object=this.game[kind](objectId);
        if(!object||object.status!=='field')return last;
        const pose=this.game.objectPose(object),centreZ=pose.centerZ??(pose.lying?pose.z:pose.z+3.25);
        const lift=yield {type:'lift',robotId,height:clamp(centreZ-3.25,0,40)};
        if(!lift.ok)return lift;
        const robot=this.robot(robotId);
        if(attempt===0){
          const plan=approach(pose,robot,REACH,this.game.goals,this.others(robotId));
          const moved=yield* this.travel(robotId,plan.centre,plan.heading);
          if(!moved.ok)last=moved;
        }else{
          const grip=this.game.gripPose(robot,kind);
          const moved=yield {type:'move',robotId,x:robot.x+pose.x-grip.x,y:robot.y+pose.y-grip.y,theta:robot.theta};
          if(!moved.ok)last=moved;
        }
        const taken=yield {type:'game',robotId,action:'pickup',kind};
        if(taken.ok)return {ok:true};
        last=taken;
      }
      return last;
    }
    // Lift to the stack height, put the held object over the goal opening and release it.
    *place(robotId,kind,goalId){
      const goal=this.game.goal(goalId);
      if(!goal)return {ok:false,error:`Unknown goal ${goalId}`};
      if(!this.robot(robotId).possession[kind+'Id'])return {ok:false,error:`Robot holds no ${kind}`};
      const base=goal.height-3.25+goal.stack.length*3.25;
      const lift=yield {type:'lift',robotId,height:clamp(base+2,0,40)};
      if(!lift.ok)return lift;
      let last={ok:false,error:'Not placed'};
      for(let attempt=0;attempt<3;attempt++){
        const robot=this.robot(robotId);
        if(attempt===0){
          const plan=approach(goal,robot,REACH,this.game.goals.filter(other=>other!==goal),this.others(robotId));
          const moved=yield* this.travel(robotId,plan.centre,plan.heading);
          if(!moved.ok)last=moved;
        }else{
          const held=this.game[kind](robot.possession[kind+'Id']);
          if(!held)return {ok:false,error:`Robot no longer holds the ${kind}`};
          const pose=this.game.objectPose(held);
          const moved=yield {type:'move',robotId,x:robot.x+goal.x-pose.x,y:robot.y+goal.y-pose.y,theta:robot.theta};
          if(!moved.ok)last=moved;
        }
        const placed=yield {type:'game',robotId,action:'place',kind};
        if(placed.ok){yield {type:'wait',seconds:PLACE_SETTLE_SECONDS};return {ok:true};}
        last=placed;
      }
      return last;
    }
    *macroSteps(action){
      const robotId=action.robotId,robot=this.robot(robotId);
      if(this.game.disqualifiedRobots.includes(robotId))return {ok:false,error:'Robot disqualified'};
      if(action.kind==='park'){
        const sign=robot.alliance==='red'?-1:1,destination={x:sign*PARK.x,y:(robotId.endsWith('2')?1:-1)*PARK.y};
        const moved=yield* this.travel(robotId,destination,square(robot.theta));
        return this.robot(robotId).midfield?{ok:true}:{ok:false,error:moved.ok?'Robot is not in midfield':moved.error};
      }
      if(action.kind==='toggle'){
        const toggle=this.game.toggles.find(candidate=>candidate.id===action.targetId);
        if(!toggle)return {ok:false,error:`Unknown toggle ${action.targetId}`};
        const plan=approach(toggle,robot,TOGGLE_REACH,this.game.goals,this.others(robotId));
        yield* this.travel(robotId,plan.centre,plan.heading);
        return yield {type:'game',robotId,action:'toggle',kind:'pin'};
      }
      const kind=action.kind.endsWith('pin')?'pin':'cup';
      if(action.kind.startsWith('load-')){
        if(robot.possession[kind+'Id'])return {ok:false,error:`Robot already holds a ${kind}`};
        const loader=this.game.loaders.find(candidate=>candidate.id===action.loaderId&&candidate.alliance===robot.alliance);
        if(!loader)return {ok:false,error:`Unknown loader ${action.loaderId}`};
        const plan=approach(loader,robot,REACH,this.game.goals,this.others(robotId));
        const moved=yield* this.travel(robotId,plan.centre,plan.heading);
        if(!moved.ok)return moved;
        const before=new Set(this.game[kind+'s'].filter(object=>object.location==='field').map(object=>object.id));
        const loaded=yield {type:'game',robotId,action:'load',kind};
        let object=loaded.ok?this.game[kind+'s'].find(candidate=>candidate.location==='field'&&!before.has(candidate.id)):null;
        // An object already lying in the outlet (pushed there or left by a missed pickup) blocks the next load;
        // a driver takes that object instead when it is the kind the plan wants.
        if(!loaded.ok&&/occupied/.test(loaded.error))object=this.game[kind+'s'].find(candidate=>candidate.status==='field'&&
          candidate.location==='field'&&Math.hypot(candidate.x-loader.x,candidate.y-loader.y)<3);
        if(!object)return loaded.ok?{ok:false,error:'The loaded object was not found'}:loaded;
        if(loaded.ok)yield {type:'wait',seconds:LOAD_SETTLE_SECONDS};
        const taken=yield* this.grab(robotId,kind,object.id);
        if(!taken.ok)return taken;
      }else if(robot.possession[kind+'Id']!==action.targetId){
        if(robot.possession[kind+'Id'])return {ok:false,error:`Robot holds a different ${kind}`};
        const taken=yield* this.grab(robotId,kind,action.targetId);
        if(!taken.ok)return taken;
      }
      return yield* this.place(robotId,kind,action.goalId);
    }
  }

  // An OverrideFleet whose robots stay idle in autonomous and carry out the plan in the driver period.
  function planFleet(plan){
    return class OverridePlanFleet extends OverrideFleet {
      constructor(host,Simulator){
        super(host,Simulator,Object.fromEntries(host.override.robots.map(robot=>[robot.id,'idle'])));
        this.runner=new PlanRunner(this,plan);
      }
      step(dt){
        if(!Number.isFinite(dt)||dt<=0||dt>.1){for(const e of this.entries.values())this.stop(e,'InvalidDt');return;}
        const phase=this.game.phase;
        if(phase==='driver')this.runner.tick();
        for(const [id,e] of this.entries){
          if(this.game.disqualifiedRobots.includes(id))this.stop(e,'Cancelled');
          else if(phase!=='driver'&&phase!=='autonomous'&&e.sim.isRunning)this.stop(e,'PeriodEnded');
          const s=e.sim;s.manualThrottle=s.manualStrafe=s.manualTurn=0;
          s.update(dt);
        }
        this.game.tick(dt);
        if(phase!==this.game.phase){for(const e of this.entries.values())if(e.sim.isRunning)this.stop(e,'PeriodEnded');this.runner.tick();}
        this.host.simTime+=dt;
        if(this.host.simTime-this.lastTrace>=.5){this.lastTrace=this.host.simTime;this.trace.push({time:this.host.simTime,clock:this.game.clock,phase:this.game.phase,score:this.game.score(),
          robots:[...this.entries].map(([id,e])=>({id,x:e.sim.x,y:e.sim.y,theta:e.sim.theta,status:e.status}))});if(this.trace.length>600)this.trace.shift();}
        this.syncView();
      }
    };
  }

  return {planFleet,parseAction,route,approach,bodyFits,squaresOverlap,centreFor,headingTo,constants:{REACH,HALF,GOAL_PASS,GOAL_TURN,ROBOT_PASS,ROBOT_TURN,MACRO_SECONDS}};
});
