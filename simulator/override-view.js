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
      if(type==='pin'&&pose.lying){
        ctx.rotate(pose.yaw*Math.PI/180);
        ctx.fillStyle=color[o.halves[o.upIndex]];ctx.fillRect(-d.pinRadius*scale,-d.halfHeight*scale,2*d.pinRadius*scale,d.halfHeight*scale);
        ctx.fillStyle=color[o.halves[1-o.upIndex]];ctx.fillRect(-d.pinRadius*scale,0,2*d.pinRadius*scale,d.halfHeight*scale);
        ctx.strokeStyle='#fff';ctx.strokeRect(-d.pinRadius*scale,-d.halfHeight*scale,2*d.pinRadius*scale,d.height*scale);
      }else{
        ctx.beginPath();ctx.arc(0,0,(type==='cup'?d.cupRadius:d.pinRadius)*scale,0,Math.PI*2);
        ctx.fillStyle=type==='cup'?(o.up==='transparent'?'#c7ebff55':color.opaque):color[o.halves[o.upIndex]];ctx.fill();ctx.strokeStyle='#e2e8f0';ctx.stroke();
      }
      ctx.restore();
    }
    if(renderer.sim.matchMode)for(const robot of game.robots){
      const p=renderer.toCanvas(robot.x,robot.y),g=game.gripPose(robot),q=renderer.toCanvas(g.x,g.y);
      ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.strokeStyle='#fbbf24';ctx.lineWidth=2;ctx.stroke();ctx.strokeRect(q.x-2*scale,q.y-scale,4*scale,2*scale);
    }
    for(const g of game.goals){const p=renderer.toCanvas(g.x,g.y);ctx.font='10px sans-serif';ctx.fillStyle='#fff';ctx.fillText(String(g.stack.length),p.x+5*scale,p.y);}
    if(renderer.sim.matchMode)for(const robot of game.robots){
      if(robot.id===renderer.sim.activeRobotId)continue;
      polygon(root.OverrideGeometry.corners(robot).map(p=>[p.x,p.y]),color[robot.alliance]+'66',color[robot.alliance]);
    }
  }
  function build3D(view){
    const game=view.sim.override,THREE=root.THREE,items=[];
    const material=c=>new THREE.MeshStandardMaterial({color:c,roughness:.65,side:THREE.DoubleSide});
    const cylinder=(top,bottom,height,c,open=false)=>{const m=new THREE.Mesh(new THREE.CylinderGeometry(top,bottom,height,16,1,open),material(c));m.rotation.x=Math.PI/2;return m;};
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
      const mesh=new THREE.Group();
      for(let half=0;half<2;half++){
        const outer=type==='cup'?d.cupRadius:.2,waist=type==='cup'?.45:d.pinRadius;
        const part=cylinder(half?outer:waist,half?waist:outer,d.halfHeight,'#fff',type==='cup');
        part.position.z=(half?1:-1)*d.halfHeight/2;mesh.add(part);
      }
      view.scene.add(mesh);items.push({mesh,id:obj.id,type});
    }
    for(const r of game.robots){const mesh=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),material(color[r.alliance]));view.scene.add(mesh);items.push({mesh,id:r.id,type:'robot'});}
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
      const o=item.type==='pin'?game.pin(item.id):game.cup(item.id),p=game.objectPose(o);
      item.mesh.visible=visible(o);item.mesh.position.set(p.x,p.y,p.centerZ??(p.lying?p.z:p.z+d.halfHeight));
      if(p.lying||p.tilt){const a=p.yaw*Math.PI/180,t=p.tilt??Math.PI/2;item.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(Math.sin(a)*Math.sin(t),Math.cos(a)*Math.sin(t),Math.cos(t)));}
      else item.mesh.quaternion.identity();
      const halves=item.type==='pin'?[o.halves[1-o.upIndex],o.halves[o.upIndex]]:[o.up==='opaque'?'transparent':'opaque',o.up];
      item.mesh.children.forEach((part,i)=>{part.material.color.set(color[halves[i]]);part.material.transparent=halves[i]==='transparent';part.material.opacity=part.material.transparent?.32:1;part.material.depthWrite=!part.material.transparent;});
    }
  }
  root.OverrideView={draw2D,build3D,update3D};
})(globalThis);
