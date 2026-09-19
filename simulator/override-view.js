/* Both renderers consume the same rules state. Shapes are procedural approximations. */
(function(root){
  const color={red:'#f43f5e',blue:'#38bdf8',yellow:'#fde047',neutral:'#303540',opaque:'#64748b',transparent:'#bae6fd'};
  function position(game,object){
    const holder=game.robots.find(r=>Object.values(r.possession).includes(object.id));
    if(holder)return {x:holder.x+7*Math.sin(holder.theta*Math.PI/180),y:holder.y+7*Math.cos(holder.theta*Math.PI/180),z:10};
    const goal=game.goal(object.goalId);
    return {x:goal?.x??object.x,y:goal?.y??object.y,z:goal?goal.height+Math.max(0,goal.stack.findIndex(i=>i.id===object.id))*3.25:0};
  }
  function draw2D(renderer){
    const game=renderer.sim.override,ctx=renderer.ctx,scale=renderer.scale;
    const polygon=(points,fill,stroke='#e2e8f0')=>{ctx.beginPath();points.forEach(([x,y],i)=>{const p=renderer.toCanvas(x,y);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y);});ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();};
    const h=OverrideGame.constants.MIDFIELD_HALF;
    polygon([[h,0],[0,h],[-h,0],[0,-h]],'#ffffff0a');
    for(const t of game.toggles)polygon([[t.x-3,t.y-3],[t.x+3,t.y-3],[t.x+3,t.y+3],[t.x-3,t.y+3]],color[t.state]);
    for(const l of game.loaders)polygon([[l.x-3,l.y-4],[l.x+3,l.y-4],[l.x+3,l.y+4],[l.x-3,l.y+4]],color[l.alliance]);
    for(const g of game.goals){
      polygon(Array.from({length:8},(_,i)=>[g.x+3*Math.cos(i*Math.PI/4),g.y+3*Math.sin(i*Math.PI/4)]),color[g.alliance||'neutral']);
      const p=renderer.toCanvas(g.x,g.y);ctx.font='10px sans-serif';ctx.fillStyle='#fff';ctx.fillText(String(g.stack.length),p.x+4*scale,p.y);
    }
    for(const cup of game.cups){
      if(cup.location==='alliance-station')continue;
      const p=renderer.toCanvas(position(game,cup).x,position(game,cup).y),r=1.575*scale;
      ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle=cup.up==='transparent'?'#bae6fd44':'#64748b';ctx.fill();ctx.strokeStyle='#e2e8f0';ctx.stroke();
    }
    for(const pin of game.pins){
      if(pin.location==='alliance-station'||(pin.location==='preload'&&pin.status!=='held'))continue;
      const o=position(game,pin),p=renderer.toCanvas(o.x,o.y),r=1.2*scale;
      ctx.fillStyle=color[pin.halves[pin.upIndex]];ctx.fillRect(p.x-r,p.y-r,r*2,r);
      ctx.fillStyle=color[pin.halves[1-pin.upIndex]];ctx.fillRect(p.x-r,p.y,r*2,r);ctx.strokeStyle='#fff';ctx.strokeRect(p.x-r,p.y-r,r*2,r*2);
    }
    if(renderer.sim.matchMode)for(const robot of game.robots){
      if(robot.id===renderer.sim.activeRobotId)continue;
      const p=renderer.toCanvas(robot.x,robot.y);ctx.save();ctx.translate(p.x,p.y);ctx.rotate(robot.theta*Math.PI/180);
      ctx.fillStyle=color[robot.alliance]+'66';ctx.strokeStyle=color[robot.alliance];ctx.fillRect(-9*scale,-9*scale,18*scale,18*scale);ctx.strokeRect(-9*scale,-9*scale,18*scale,18*scale);ctx.restore();
    }
  }
  function build3D(view){
    const game=view.sim.override,THREE=root.THREE,items=[];
    const material=c=>new THREE.MeshStandardMaterial({color:c,roughness:.65});
    const cylinder=(top,bottom,height,c)=>{const m=new THREE.Mesh(new THREE.CylinderGeometry(top,bottom,height,8,1,true),material(c));m.rotation.x=Math.PI/2;return m;};
    for(const g of game.goals){const mesh=cylinder(2.3,3,g.height,color[g.alliance||'neutral']);mesh.position.set(g.x,g.y,g.height/2);view.scene.add(mesh);}
    const tape=(points)=>{const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(([x,y])=>new THREE.Vector3(x,y,.1))),new THREE.LineBasicMaterial({color:0xffffff}));view.scene.add(line);};
    const h=OverrideGame.constants.MIDFIELD_HALF;tape([[h,0],[0,h],[-h,0],[0,-h],[h,0]]);tape([[-60,60],[-h/2,h/2]]);tape([[h/2,-h/2],[60,-60]]);tape([[-60,-60],[-h/2,-h/2]]);tape([[h/2,h/2],[60,60]]);
    for(const t of game.toggles){const mesh=new THREE.Mesh(new THREE.BoxGeometry(8,2,2),material(color[t.state]));mesh.position.set(t.x,t.y,11);if(t.x)mesh.rotation.z=Math.PI/2;view.scene.add(mesh);items.push({mesh,id:t.id,type:'toggle'});}
    for(const l of game.loaders){const mesh=cylinder(2.3,2.3,9,color[l.alliance]);mesh.position.set(l.x,l.y,4.5);view.scene.add(mesh);}
    for(const type of ['pin','cup'])for(const obj of type==='pin'?game.pins:game.cups){
      const mesh=new THREE.Group();
      for(let half=0;half<2;half++){
        const part=cylinder(type==='cup'?1.575:.5,type==='cup'?.5:.8,3.25,'#fff');
        part.position.z=1.625+half*3.25;if(half===0)part.rotation.x=-Math.PI/2;mesh.add(part);
      }
      view.scene.add(mesh);items.push({mesh,id:obj.id,type});
    }
    for(const r of game.robots){const mesh=new THREE.Mesh(new THREE.BoxGeometry(r.width,r.length,10),material(color[r.alliance]));view.scene.add(mesh);items.push({mesh,id:r.id,type:'robot'});}
    view.overrideItems=items;
  }
  function update3D(view){
    const game=view.sim.override;
    for(const item of view.overrideItems||[]){
      if(item.type==='toggle'){item.mesh.material.color.set(color[game.toggles.find(t=>t.id===item.id).state]);continue;}
      if(item.type==='robot'){const r=game.robots.find(r=>r.id===item.id);item.mesh.visible=view.sim.matchMode&&r.id!==view.sim.activeRobotId;item.mesh.position.set(r.x,r.y,5);item.mesh.rotation.z=-r.theta*Math.PI/180;continue;}
      const o=item.type==='pin'?game.pin(item.id):game.cup(item.id),p=position(game,o);
      item.mesh.visible=o.location!=='alliance-station'&&(o.location!=='preload'||o.status==='held');item.mesh.position.set(p.x,p.y,p.z);
      const halves=item.type==='pin'?[o.halves[1-o.upIndex],o.halves[o.upIndex]]:[o.up==='opaque'?'transparent':'opaque',o.up];
      item.mesh.children.forEach((part,i)=>{part.material.color.set(color[halves[i]]);part.material.transparent=halves[i]==='transparent';part.material.opacity=part.material.transparent?.28:1;});
    }
  }
  root.OverrideView={draw2D,build3D,update3D};
})(globalThis);
