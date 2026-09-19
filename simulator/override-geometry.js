/* Shared scene/contact geometry in inches. Procedural proxies, not field CAD. */
(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory();
  else root.OverrideGeometry=factory();
})(globalThis,function(){
  'use strict';
  const FIELD_HALF=70.2;
  const OBJECTS=Object.freeze({height:6.5,halfHeight:3.25,pinRadius:.8,cupRadius:1.575,goalRadius:4.25});
  const radians=degrees=>degrees*Math.PI/180;
  function rotate(x,y,heading){
    const a=radians(heading),c=Math.cos(a),s=Math.sin(a);
    return {x:x*c+y*s,y:-x*s+y*c};
  }
  function corners(r){
    return [[-1,-1],[-1,1],[1,1],[1,-1]].map(([sx,sy])=>{
      const p=rotate(sx*r.width/2,sy*r.length/2,r.theta);
      return {x:r.x+p.x,y:r.y+p.y};
    });
  }
  function halfExtents(r){
    const a=radians(r.theta),c=Math.abs(Math.cos(a)),s=Math.abs(Math.sin(a));
    return {x:(r.width*c+r.length*s)/2,y:(r.width*s+r.length*c)/2};
  }
  function touchesPerimeter(r,half=FIELD_HALF){
    const e=halfExtents(r);
    return Math.abs(r.x)+e.x>=half-1e-6||Math.abs(r.y)+e.y>=half-1e-6;
  }
  // Minimum translation of rectangle A away from a fixed circle.
  function circleContact(a,b){
    const p=rotate(b.x-a.x,b.y-a.y,-a.theta),hx=a.width/2,hy=a.length/2;
    const dx=Math.max(-hx,Math.min(hx,p.x))-p.x;
    const dy=Math.max(-hy,Math.min(hy,p.y))-p.y;
    const distance=Math.hypot(dx,dy);
    if(distance>=b.radius)return null;
    let nx,ny,depth;
    if(distance>1e-9){nx=dx/distance;ny=dy/distance;depth=b.radius-distance;}
    else if(hx-Math.abs(p.x)<hy-Math.abs(p.y)){
      nx=p.x>=0?-1:1;ny=0;depth=b.radius+hx-Math.abs(p.x);
    }else{nx=0;ny=p.y>=0?-1:1;depth=b.radius+hy-Math.abs(p.y);}
    const normal=rotate(nx,ny,a.theta);
    return {...normal,depth};
  }
  // Separating axes also handle complete containment, not just intersecting edges.
  function rectangleContact(a,b){
    const ca=corners(a),cb=corners(b);
    let best=null;
    for(const heading of [a.theta,b.theta])for(const [x,y] of [[1,0],[0,1]]){
      const n=rotate(x,y,heading),pa=ca.map(p=>p.x*n.x+p.y*n.y),pb=cb.map(p=>p.x*n.x+p.y*n.y);
      const positive=Math.max(...pb)-Math.min(...pa),negative=Math.max(...pa)-Math.min(...pb);
      if(positive<=0||negative<=0)return null;
      const sign=positive<negative?1:-1,depth=Math.min(positive,negative);
      if(!best||depth<best.depth)best={x:n.x*sign,y:n.y*sign,depth};
    }
    return best;
  }
  function resolveRobot(robot,obstacles,half=FIELD_HALF){
    const r={...robot},normals=[];
    for(let pass=0;pass<8;pass++){
      let changed=false;
      const ext=halfExtents(r);
      for(const axis of ['x','y']){
        const limit=half-ext[axis],next=Math.max(-limit,Math.min(limit,r[axis]));
        if(Math.abs(next-r[axis])>1e-8){
          const n={x:0,y:0};n[axis]=Math.sign(next-r[axis]);normals.push(n);
          r[axis]=next;changed=true;
        }
      }
      for(const obstacle of obstacles){
        const hit=obstacle.radius===undefined?rectangleContact(r,obstacle):circleContact(r,obstacle);
        if(hit&&hit.depth>1e-8){
          r.x+=hit.x*hit.depth;r.y+=hit.y*hit.depth;normals.push(hit);changed=true;
        }
      }
      if(!changed)break;
    }
    return {x:r.x,y:r.y,normals};
  }
  return {FIELD_HALF,OBJECTS,rotate,corners,halfExtents,touchesPerimeter,circleContact,rectangleContact,resolveRobot};
});
