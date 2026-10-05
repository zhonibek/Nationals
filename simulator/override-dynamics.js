/* Deterministic reduced rigid-body model, inches/seconds. Shapes are contact proxies. */
(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory(require('./override-geometry'));
  else root.OverrideDynamics=factory(root.OverrideGeometry);
})(globalThis,function(G){
  'use strict';
  const GRAVITY=386.09,HALF=3.25;
  const radius=G.objectRadius;
  const MASS_KG=Object.freeze({pin:.073,cup:.078});
  const mass=o=>MASS_KG[o.kind==='cup'?'cup':'pin'];
  const extent=(o,b)=>Math.abs(Math.cos(b.tilt))*HALF+Math.abs(Math.sin(b.tilt))*radius(o);
  function body(game,o){
    if(!o.body){const p=game.objectPose(o);o.body={z:p.centerZ??(p.lying?p.z:p.z+HALF),vx:0,vy:0,vz:0,tilt:p.lying?Math.PI/2:0,spin:0,yaw:p.yaw||0};}
    return o.body;
  }
  function release(game,o,pose,velocity={}){
    Object.assign(o,{status:'field',placed:false,goalId:null,location:'field',x:pose.x,y:pose.y});
    o.body={z:pose.centerZ??(pose.lying?pose.z:pose.z+HALF),vx:velocity.vx||0,vy:velocity.vy||0,vz:velocity.vz||0,tilt:pose.lying?Math.PI/2:0,spin:0,yaw:pose.yaw||0};
  }
  function pose(o){const b=o.body;return {x:o.x,y:o.y,z:b.z-extent(o,b),centerZ:b.z,tilt:b.tilt,lying:b.tilt>Math.PI/4,yaw:b.yaw};}
  function separate(o,b,other,r){
    const dx=o.x-other.x,dy=o.y-other.y,d=Math.hypot(dx,dy),limit=r;
    if(d>=limit)return;
    const nx=d>1e-8?dx/d:1,ny=d>1e-8?dy/d:0;
    o.x+=nx*(limit-d);o.y+=ny*(limit-d);
    const v=b.vx*nx+b.vy*ny;if(v<0){b.vx-=1.15*v*nx;b.vy-=1.15*v*ny;}
  }
  function detachPassenger(game,o,b,velocity){
    const passenger=game.pins.find(p=>p.supportId===o.id);if(!passenger)return;
    const p=game.objectPose(passenger);passenger.supportId=null;
    release(game,passenger,p,velocity);passenger.body.tilt=b.tilt;passenger.body.spin=3;
  }
  function confine(object){
    const bounds=G.objectBounds(object,pose(object));
    for(const axis of ['x','y']){
      const limit=G.FIELD_HALF-bounds[axis],sign=Math.sign(object[axis]);
      if(Math.abs(object[axis])>limit){object[axis]=sign*limit;if(object.body['v'+axis]*sign>0)object.body['v'+axis]*=-.15;}
    }
  }
  function robotContact(game,object){
    const state=object.body;
    if(state.z-extent(object,state)>10)return;
    for(const robot of game.robots){
      const envelope=G.halfExtents(robot);
      if(Math.abs(object.x-robot.x)>envelope.x+HALF||Math.abs(object.y-robot.y)>envelope.y+HALF)continue;
      const origin={x:object.x,y:object.y};
      for(const sample of G.objectFootprint(object,pose(object))){
        const hit=G.circleContact(robot,{...sample,x:sample.x+object.x-origin.x,y:sample.y+object.y-origin.y});if(!hit)continue;
        object.x-=hit.x*hit.depth;object.y-=hit.y*hit.depth;
        const velocity=robot.velocity||{vx:0,vy:0};
        const relative=(state.vx-velocity.vx)*-hit.x+(state.vy-velocity.vy)*-hit.y;
        if(relative<0){state.vx+=relative*hit.x;state.vy+=relative*hit.y;}
        if(Math.hypot(velocity.vx,velocity.vy)>4&&state.tilt<.1){
          state.spin=3;state.yaw=Math.atan2(velocity.vx,velocity.vy)*180/Math.PI;
          detachPassenger(game,object,state,velocity);
        }
      }
    }
  }
  function collisionPairs(objects){
    const cells=new Map(),pairs=[],cellSize=7;
    objects.forEach((object,index)=>{
      const key=Math.floor(object.x/cellSize)+','+Math.floor(object.y/cellSize);
      if(!cells.has(key))cells.set(key,[]);cells.get(key).push(index);
    });
    objects.forEach((object,index)=>{
      const column=Math.floor(object.x/cellSize),row=Math.floor(object.y/cellSize);
      for(let horizontal=-1;horizontal<=1;horizontal++)for(let vertical=-1;vertical<=1;vertical++){
        for(const other of cells.get((column+horizontal)+','+(row+vertical))||[])if(other>index)pairs.push([index,other]);
      }
    });
    return pairs;
  }
  function step(game,dt){
    const objects=[...game.cups,...game.pins];
    for(let remaining=dt;remaining>1e-8;){
      const h=Math.min(.005,remaining);remaining-=h;
      const free=objects.filter(o=>o.status==='field'&&o.location==='field'&&!o.supportId);
      for(const o of free){
        const b=body(game,o),previousZ=b.z;
        b.vz-=GRAVITY*h;o.x+=b.vx*h;o.y+=b.vy*h;b.z+=b.vz*h;
        b.tilt=Math.max(0,Math.min(Math.PI/2,b.tilt+b.spin*h));
        if(b.tilt>=Math.PI/2)b.spin=0;
        if(b.tilt>.1)detachPassenger(game,o,b,{vx:b.vx,vy:b.vy,vz:b.vz});
        // Nested capture only while descending, upright, aligned, and at the opening.
        if(b.vz<=0&&b.tilt<.15){
          const goal=game.goals.find(g=>Math.hypot(g.x-o.x,g.y-o.y)<1.1);
          if(goal){
            const landing=goal.height+goal.stack.length*HALF;
            const top=goal.stack.at(-1),valid=o.kind==='cup'?top?.type==='pin':!top||top.type==='cup';
            if(valid&&previousZ>=landing-1e-7&&b.z<=landing){
              if(game.captureObject(o,goal)){delete o.body;continue;}
            }
          }
        }
        // A Pin can nest in an upright floor Cup as well as in a scored stack.
        if(o.kind!=='cup'&&b.vz<=0&&b.tilt<.1){
          const cup=game.cups.find(c=>c.status==='field'&&c.location==='field'&&
            (!c.body||c.body.tilt<.1)&&!game.pins.some(p=>p.supportId===c.id)&&Math.hypot(c.x-o.x,c.y-o.y)<1.1);
          if(cup){const p=game.objectPose(cup),landing=(p.centerZ??p.z+HALF)+HALF;
            if(previousZ>=landing-1e-7&&b.z<=landing){o.supportId=cup.id;o.x=cup.x;o.y=cup.y;delete o.body;continue;}
          }
        }
        const floor=extent(o,b),impact=-b.vz;
        if(b.z<=floor){
          b.z=floor;b.vz=impact>12?impact*.12:0;
          const damping=Math.exp(-8*h);b.vx*=damping;b.vy*=damping;
          if(Math.hypot(b.vx,b.vy)<.05)b.vx=b.vy=0;
          if(impact>50&&b.tilt<.1){
            b.spin=4;b.yaw=Math.hypot(b.vx,b.vy)>.1?Math.atan2(b.vx,b.vy)*180/Math.PI:35;
            detachPassenger(game,o,b,{vx:b.vx+5,vy:b.vy+2,vz:impact*.1});
          }
        }
        const r=radius(o);
        confine(o);
        // Body/goal contacts at intersecting heights; no artificial magnetic pickup.
        robotContact(game,o);
        for(const goal of game.goals)if(b.z-extent(o,b)<goal.height){
          if(Math.hypot(o.x-goal.x,o.y-goal.y)>G.OBJECTS.goalRadius+HALF)continue;
          const top=goal.stack.at(-1),open=o.kind==='cup'?top?.type==='pin':!top||top.type==='cup';
          if(open&&b.tilt<.15&&Math.hypot(o.x-goal.x,o.y-goal.y)<1.1)continue;
          const origin={x:o.x,y:o.y};
          for(const sample of G.objectFootprint(o,pose(o))){
            const start={x:sample.x+o.x-origin.x,y:sample.y+o.y-origin.y},proxy={...start};
            separate(proxy,b,goal,r+G.OBJECTS.goalRadius);
            o.x+=proxy.x-start.x;o.y+=proxy.y-start.y;
          }
        }
        o.lying=b.tilt>Math.PI/4;o.yaw=b.yaw;
      }
      // Pair contacts exchange equal-mass normal impulses and correct overlap symmetrically.
      for(let pass=0;pass<4;pass++){
      let changed=false;
      for(const [firstIndex,secondIndex] of collisionPairs(free)){
        const a=free[firstIndex],b=free[secondIndex],av=a.body,bv=b.body;if(!av||!bv)continue;
        if(Math.hypot(a.x-b.x,a.y-b.y)>G.OBJECTS.height)continue;
        if(Math.abs(av.z-bv.z)>extent(a,av)+extent(b,bv))continue;
        const pin=a.kind==='cup'?b:a,cup=a.kind==='cup'?a:b;
        if(pin.kind!=='cup'&&cup.kind==='cup'&&pin.body.vz<=0&&pin.body.tilt<.1&&cup.body.tilt<.1&&
          pin.body.z>=cup.body.z+HALF-1e-7&&Math.hypot(pin.x-cup.x,pin.y-cup.y)<1.1&&!game.pins.some(p=>p.supportId===cup.id))continue;
        let contact=null;
        for(const first of G.objectFootprint(a,pose(a)))for(const second of G.objectFootprint(b,pose(b))){
          const dx=second.x-first.x,dy=second.y-first.y,distance=Math.hypot(dx,dy),depth=first.radius+second.radius-distance;
          if(depth>1e-8&&(!contact||depth>contact.depth))contact={x:distance>1e-8?dx/distance:1,y:distance>1e-8?dy/distance:0,depth};
        }
        if(!contact)continue;changed=true;
        const nx=contact.x,ny=contact.y,inverseA=1/mass(a),inverseB=1/mass(b),inverseSum=inverseA+inverseB;
        a.x-=nx*contact.depth*inverseA/inverseSum;a.y-=ny*contact.depth*inverseA/inverseSum;
        b.x+=nx*contact.depth*inverseB/inverseSum;b.y+=ny*contact.depth*inverseB/inverseSum;
        const rel=(bv.vx-av.vx)*nx+(bv.vy-av.vy)*ny;
        if(rel<0){const impulse=-1.1*rel/inverseSum;av.vx-=impulse*inverseA*nx;av.vy-=impulse*inverseA*ny;bv.vx+=impulse*inverseB*nx;bv.vy+=impulse*inverseB*ny;}
      }
      // Pair/robot projections can push bodies against the wall after the first clamp.
      for(const o of free){if(!o.body)continue;
        const previous={x:o.x,y:o.y};robotContact(game,o);confine(o);
        if(Math.hypot(o.x-previous.x,o.y-previous.y)>1e-8)changed=true;
      }
      if(!changed)break;
      }
    }
  }
  function resting(game){
    return [...game.cups,...game.pins].every(o=>!o.body||(o.body.z<=extent(o,o.body)+.001&&Math.hypot(o.body.vx,o.body.vy,o.body.vz)<.1&&Math.abs(o.body.spin)<.01))&&
      game.robots.every(r=>Math.hypot(r.velocity?.vx||0,r.velocity?.vy||0,r.velocity?.omega||0)<.1);
  }
  return {step,release,pose,resting,GRAVITY,MASS_KG};
});
