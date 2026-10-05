/* Both renderers use rules-state poses and shared inch dimensions. Procedural proxies, not field CAD. */
(function(root){
  const color={red:'#e12d3e',blue:'#2588dc',yellow:'#ffdf42',neutral:'#20242a',opaque:'#5e6570',transparent:'#c7ebff'};
  const d=root.OverrideGeometry.OBJECTS;
  const visible=o=>o.location!=='alliance-station'&&(o.location!=='preload'||o.status==='held');
  function draw2D(renderer){
    const game=renderer.sim.override,ctx=renderer.ctx,scale=renderer.scale;
    const polygon=(points,fill,stroke='#e2e8f0')=>{ctx.beginPath();points.forEach(([x,y],i)=>{const p=renderer.toCanvas(x,y);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y);});ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();};
    const h=root.OverrideGame.constants.MIDFIELD_HALF;
    polygon([[h,0],[0,h],[-h,0],[0,-h]],'#ffffff0a');
    for(const t of game.toggles){const w=t.x?1:4,l=t.x?4:1;polygon([[t.x-w,t.y-l],[t.x+w,t.y-l],[t.x+w,t.y+l],[t.x-w,t.y+l]],color[t.state]);}
    for(const l of game.loaders)polygon([[l.x-3,l.y-3],[l.x+3,l.y-3],[l.x+3,l.y+3],[l.x-3,l.y+3]],color[l.alliance]);
    for(const g of game.goals)polygon(Array.from({length:8},(_,i)=>[g.x+d.goalRadius*Math.cos(i*Math.PI/4),g.y+d.goalRadius*Math.sin(i*Math.PI/4)]),color[g.alliance||'neutral']);
    const objects=[...game.cups.map(o=>({o,type:'cup'})),...game.pins.map(o=>({o,type:'pin'}))]
      .filter(({o})=>visible(o)).map(item=>({...item,pose:game.objectPose(item.o)})).sort((a,b)=>a.pose.z-b.pose.z);
    for(const {o,type,pose} of objects){
      const p=renderer.toCanvas(pose.x,pose.y);ctx.save();ctx.translate(p.x,p.y);
      if(pose.lying){
        ctx.rotate(pose.yaw*Math.PI/180);
        for(const half of [-1,1]){
          const centerRadius=type==='pin'?d.pinRadius:d.cupWaistRadius,endRadius=type==='pin'?d.pinTipRadius:d.cupRadius;
          ctx.beginPath();ctx.moveTo(-centerRadius*scale,0);ctx.lineTo(-endRadius*scale,half*d.halfHeight*scale);
          ctx.lineTo(endRadius*scale,half*d.halfHeight*scale);ctx.lineTo(centerRadius*scale,0);ctx.closePath();
          ctx.fillStyle=type==='pin'?color[o.halves[half<0?o.upIndex:1-o.upIndex]]:color[half<0?o.up:o.up==='opaque'?'transparent':'opaque'];ctx.fill();ctx.strokeStyle='#26313e';ctx.stroke();
        }
        ctx.fillStyle='#26313e';ctx.fillRect(-d.pinRadius*scale,-.12*scale,2*d.pinRadius*scale,.24*scale);
      }else{
        ctx.beginPath();ctx.arc(0,0,(type==='cup'?d.cupRadius:d.pinRadius)*scale,0,Math.PI*2);
        ctx.fillStyle=type==='cup'?(o.up==='transparent'?color.transparent:color.opaque):color[o.halves[o.upIndex]];ctx.fill();ctx.strokeStyle='#26313e';ctx.lineWidth=2;ctx.stroke();
        ctx.beginPath();ctx.arc(0,0,(type==='cup'?d.cupWaistRadius:d.pinTipRadius)*scale,0,Math.PI*2);
        ctx.strokeStyle=type==='cup'?'#475569':'#ffffff88';ctx.lineWidth=1;ctx.stroke();
        if(type==='cup'){ctx.font='bold 9px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle=o.up==='transparent'?'#18384e':'#f8fafc';ctx.fillText(o.up==='transparent'?'C':'O',0,0);}
      }
      ctx.restore();
      if(renderer.sim.showGameColliders){
        ctx.strokeStyle='#facc15';ctx.lineWidth=1;ctx.setLineDash([2,2]);
        for(const sample of root.OverrideGeometry.objectFootprint(o,pose)){const point=renderer.toCanvas(sample.x,sample.y);ctx.beginPath();ctx.arc(point.x,point.y,sample.radius*scale,0,Math.PI*2);ctx.stroke();}
        ctx.setLineDash([]);
      }
    }
    if(renderer.sim.matchMode)for(const robot of game.robots){
      const p=renderer.toCanvas(robot.x,robot.y),g=game.gripPose(robot,renderer.sim.gameObjectKind||'pin'),q=renderer.toCanvas(g.x,g.y);
      ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.strokeStyle='#fbbf24';ctx.lineWidth=2;ctx.stroke();ctx.strokeRect(q.x-2*scale,q.y-scale,4*scale,2*scale);
      if(robot.id===renderer.sim.activeRobotId){
        polygon(root.OverrideGeometry.corners(robot).map(point=>[point.x,point.y]),null,'#fbbf24');
        const hint=game.pickupHint(robot.id,renderer.sim.gameObjectKind||'pin');
        ctx.beginPath();ctx.arc(q.x,q.y,3*scale,0,Math.PI*2);ctx.strokeStyle=hint?.reachable?'#34d399':'#fbbf24';ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);
        if(hint?.id){const object=game.pin(hint.id)||game.cup(hint.id),pose=game.objectPose(object),point=renderer.toCanvas(pose.x,pose.y);ctx.beginPath();ctx.arc(point.x,point.y,4*scale,0,Math.PI*2);ctx.stroke();}
      }
    }
    for(const g of game.goals){const p=renderer.toCanvas(g.x,g.y);ctx.font='10px sans-serif';ctx.fillStyle='#fff';ctx.fillText(String(g.stack.length),p.x+5*scale,p.y);}
    if(renderer.sim.matchMode)for(const robot of game.robots){
      if(robot.id===renderer.sim.activeRobotId)continue;
      polygon(root.OverrideGeometry.corners(robot).map(p=>[p.x,p.y]),color[robot.alliance]+'66',color[robot.alliance]);
    }
  }
  function build3D(view){
    const game=view.sim.override,THREE=root.THREE,items=[];
    const shellGeometries=new Map(),ribMaterial=new THREE.LineBasicMaterial({color:0x334155,transparent:true,opacity:.7});
    const material=c=>new THREE.MeshStandardMaterial({color:c,roughness:.65,side:THREE.DoubleSide});
    const cylinder=(top,bottom,height,c,open=false)=>{const m=new THREE.Mesh(new THREE.CylinderGeometry(top,bottom,height,16,1,open),material(c));m.rotation.x=Math.PI/2;return m;};
    const shell=(profile,thickness)=>{
      const points=profile.map(([radius,height])=>new THREE.Vector2(radius,height));
      if(thickness)for(const [radius,height] of [...profile].reverse())points.push(new THREE.Vector2(Math.max(.05,radius-thickness),height));
      else points.push(new THREE.Vector2(0,profile.at(-1)[1]),new THREE.Vector2(0,profile[0][1]));
      points.push(points[0].clone());
      const key=JSON.stringify([profile,thickness]);
      if(!shellGeometries.has(key))shellGeometries.set(key,new THREE.LatheGeometry(points,48));
      const mesh=new THREE.Mesh(shellGeometries.get(key),material('#ffffff'));mesh.rotation.x=Math.PI/2;mesh.castShadow=true;mesh.receiveShadow=true;return mesh;
    };
    for(const g of game.goals){const mesh=new THREE.Mesh(new THREE.CylinderGeometry(2.3,d.goalRadius,g.height,8,1,true),material(color[g.alliance||'neutral']));mesh.rotation.x=Math.PI/2;mesh.position.set(g.x,g.y,g.height/2);view.scene.add(mesh);}
    const tape=points=>{const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(([x,y])=>new THREE.Vector3(x,y,.1))),new THREE.LineBasicMaterial({color:0xffffff}));view.scene.add(line);};
    const h=root.OverrideGame.constants.MIDFIELD_HALF;tape([[h,0],[0,h],[-h,0],[0,-h],[h,0]]);tape([[-60,60],[-h/2,h/2]]);tape([[h/2,-h/2],[60,-60]]);tape([[-60,-60],[-h/2,-h/2]]);tape([[h/2,h/2],[60,60]]);
    for(const t of game.toggles){const mesh=new THREE.Mesh(new THREE.BoxGeometry(8,2,2),material(color[t.state]));mesh.position.set(t.x,t.y,11);if(t.x)mesh.rotation.z=Math.PI/2;view.scene.add(mesh);items.push({mesh,id:t.id,type:'toggle'});}
    for(const l of game.loaders){
      const sleeve=cylinder(2.3,2.3,9,color.transparent,true);sleeve.position.set(l.x,l.y,4.5);sleeve.material.transparent=true;sleeve.material.opacity=.25;sleeve.material.depthWrite=false;view.scene.add(sleeve);
      const base=new THREE.Mesh(new THREE.BoxGeometry(6,6,.5),material(color[l.alliance]));base.position.set(l.x,l.y,.25);view.scene.add(base);
      for(const dx of [-2.5,2.5]){const post=cylinder(.25,.25,11,color.neutral);post.position.set(l.x+dx,l.y,5.5);view.scene.add(post);}
    }
    for(const type of ['pin','cup'])for(const obj of type==='pin'?game.pins:game.cups){
      const mesh=new THREE.Group(),ribPoints=[];
      for(let half=0;half<2;half++){
        const direction=half?1:-1;
        const profile=type==='cup'?[[d.cupWaistRadius,0],[1.21,direction*2.85],[d.cupRadius,direction*3.08],[d.cupRadius,direction*d.cupHeight/2]]:
          [[d.pinRadius,0],[d.pinRadius,direction*.14],[d.pinShoulderRadius,direction*.32],[d.pinTipRadius,direction*2.61],[d.pinTipRadius,direction*d.halfHeight]];
        const part=shell(profile,type==='cup'?.06:0);mesh.add(part);
        for(let rib=0;rib<6;rib++){
          const heading=rib*Math.PI/3;
          const points=profile.map(([radius,height])=>new THREE.Vector3((radius+.015)*Math.cos(heading),(radius+.015)*Math.sin(heading),height));
          for(let segment=1;segment<points.length;segment++)ribPoints.push(points[segment-1],points[segment]);
        }
      }
      mesh.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(ribPoints),ribMaterial));
      view.scene.add(mesh);items.push({mesh,id:obj.id,type});
    }
    for(const r of game.robots){const mesh=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),material(color[r.alliance]));view.scene.add(mesh);items.push({mesh,id:r.id,type:'robot'});}
    for(const robot of game.robots){const mesh=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),new THREE.LineBasicMaterial({color:0xfbbf24}));view.scene.add(mesh);items.push({mesh,id:robot.id,type:'envelope'});}
    for(const r of game.robots){
      const mesh=new THREE.Group();
      const mast=new THREE.Mesh(new THREE.BoxGeometry(1,1,40),material('#64748b'));mast.position.set(0,7,20);mesh.add(mast);
      const fork=new THREE.Mesh(new THREE.BoxGeometry(4,2,1),material('#fbbf24'));mesh.add(fork);
      const arm=new THREE.Mesh(new THREE.BoxGeometry(1,8,1),material('#fbbf24'));mesh.add(arm);
      view.scene.add(mesh);items.push({mesh,id:r.id,type:'manipulator'});
    }
    view.overrideItems=items;
  }
  function update3D(view){
    const game=view.sim.override,THREE=root.THREE;
    for(const item of view.overrideItems||[]){
      if(item.type==='toggle'){item.mesh.material.color.set(color[game.toggles.find(t=>t.id===item.id).state]);continue;}
      if(item.type==='manipulator'){
        const r=game.robots.find(r=>r.id===item.id),m=r.manipulator;
        item.mesh.visible=view.sim.matchMode;item.mesh.position.set(r.x,r.y,0);item.mesh.rotation.z=-r.theta*Math.PI/180;
        item.mesh.children[0].scale.z=Math.max(.05,(m.height+6.5)/40);item.mesh.children[0].position.z=(m.height+6.5)/2;
        item.mesh.children[1].position.set(0,m.reach,m.height+3.25);item.mesh.children[2].position.set(0,m.reach-4,m.height+3.25);continue;
      }
      if(item.type==='robot'){const r=game.robots.find(r=>r.id===item.id);item.mesh.visible=view.sim.matchMode&&r.id!==view.sim.activeRobotId;item.mesh.position.set(r.x,r.y,5);item.mesh.scale.set(r.width,r.length,10);item.mesh.rotation.z=-r.theta*Math.PI/180;continue;}
      if(item.type==='envelope'){const robot=game.robots.find(candidate=>candidate.id===item.id);item.mesh.visible=view.sim.matchMode&&robot.id===view.sim.activeRobotId;item.mesh.position.set(robot.x,robot.y,5);item.mesh.scale.set(robot.width,robot.length,10);item.mesh.rotation.z=-robot.theta*Math.PI/180;continue;}
      const o=item.type==='pin'?game.pin(item.id):game.cup(item.id),p=game.objectPose(o);
      item.mesh.visible=visible(o);item.mesh.position.set(p.x,p.y,p.centerZ??(p.lying?p.z:p.z+d.halfHeight));
      if(p.lying||p.tilt){const a=p.yaw*Math.PI/180,t=p.tilt??Math.PI/2;item.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(Math.sin(a)*Math.sin(t),Math.cos(a)*Math.sin(t),Math.cos(t)));}
      else item.mesh.quaternion.identity();
      const halves=item.type==='pin'?[o.halves[1-o.upIndex],o.halves[o.upIndex]]:[o.up==='opaque'?'transparent':'opaque',o.up];
      item.mesh.children.filter(part=>part.isMesh).forEach((part,index)=>{
        part.material.color.set(color[halves[index]]);part.material.transparent=halves[index]==='transparent'&&!view.sim.solidGameObjects;
        part.material.opacity=part.material.transparent?.6:1;part.material.depthWrite=!part.material.transparent;
      });
    }
  }
  root.OverrideView={draw2D,build3D,update3D};
})(globalThis);
