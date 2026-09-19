'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const root=path.resolve(__dirname,'..'),Control=require('../simulator/control-runtime');
const crypto=require('node:crypto'),hash=data=>crypto.createHash('sha256').update(data).digest('hex');
const manifest=JSON.parse(fs.readFileSync(path.join(root,'simulator/control-build.json'),'utf8').replace(/^\uFEFF/,''));
for(const [file,digest] of Object.entries(manifest.sources))assert.equal(hash(fs.readFileSync(path.join(root,file),'utf8').replace(/\r\n/g,'\n')),digest,`Rebuild stale WASM: ${file}`);
assert.equal(hash(fs.readFileSync(path.join(root,'simulator/control.wasm'))),manifest.wasm);
const moduleCore=new WebAssembly.Module(fs.readFileSync(path.join(root,'simulator/control.wasm')));
const context={console,Math,setTimeout(){},document:{addEventListener(){}},window:{}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root,'simulator/robot-config.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(root,'simulator/simulator.js'),'utf8')+'\nthis.Simulator=VexRobotSimulator;',context);
const make=()=>new context.Simulator(Control.fromModule(moduleCore));
const report=[];
function motion(label,start,action,seconds=18,modify=()=>{}){
  const sim=make();sim.setPose(...start);modify(sim);sim.queueAction(action);sim.isRunning=true;
  let n=0;while(sim.isRunning&&n++<seconds*100)sim.update(.01);
  assert.equal(sim.lastMotionResult,'Settled',`${label}: ${sim.lastMotionResult}`);
  assert(sim.motorVolts.every(v=>Number.isFinite(v)&&Math.abs(v)<=12));
  report.push({label,seconds:sim.simTime,truth:[sim.x,sim.y,sim.theta],estimate:sim.odom});return sim;
}
for(const h of [0,90,180,270]){
  const rad=h*Math.PI/180;
  for(const sign of [-1,1]){
    const sim=motion(`strafe h=${h} sign=${sign}`,[0,0,h],{type:'strafe',targetInches:24*sign,heading:h});
    assert(Math.hypot(sim.x-24*sign*Math.cos(rad),sim.y+24*sign*Math.sin(rad))<1);
  }
}
motion('straight',[0,0,0],{type:'drive',targetInches:24,heading:0});
motion('reverse',[0,0,0],{type:'drive',targetInches:-24,heading:0});
motion('translation and independent heading',[0,0,0],{type:'pose',targetX:24,targetY:24,targetTheta:90});
const diagonal=motion('body-relative diagonal with independent heading',[0,0,0],{type:'diagonal',forwardInches:24,strafeInches:24,endHeading:90});
assert(Math.hypot(diagonal.x-24,diagonal.y-24)<1);
motion('359 to 1 degree wrap',[0,0,359],{type:'turn',targetHeading:1});
motion('spline',[0,0,0],{type:'spline',end:{x:20,y:40,theta:45},maxVel:.45,maxAccel:.8});
motion('Pedro Bezier',[0,0,0],{type:'bezier',points:[[0,0],[0,16],[24,8],[24,24]],startHeading:0,endHeading:90});
motion('extra load',[0,0,0],{type:'drive',targetInches:24,heading:0},18,s=>s.massKg=10);
const slip=motion('low traction',[0,0,0],{type:'drive',targetInches:24,heading:0},18,s=>s.muLong=.08);
assert(Math.abs(slip.y-slip.odom.y)>.2,'Wheel slip must cause odometry drift');
const halted=make();halted.startRoutine('testLinearDrive');for(let i=0;i<100;i++)halted.update(.01);halted.update(NaN);assert.equal(halted.lastMotionResult,'InvalidDt');assert(halted.motorVolts.every(v=>v===0));
for(const timeout of [50,-1,NaN,Infinity,120001]){
  const sim=make();sim.queueAction({type:'drive',targetInches:24,heading:0,timeout});sim.isRunning=true;
  for(let i=0;i<100&&sim.isRunning;i++)sim.update(.01);
  assert.equal(sim.lastMotionResult,timeout===50?'TimedOut':'InvalidInput');
  assert(sim.simTime<.1);assert(sim.motorVolts.every(v=>v===0));
}
const blocked=make();blocked.muLong=0;blocked.queueAction({type:'drive',targetInches:24,heading:0});blocked.isRunning=true;
for(let i=0;i<700&&blocked.isRunning;i++)blocked.update(.01);
assert(Math.abs(blocked.y)<.01);assert(blocked.odom.y>10,'Encoders must not be ground truth');
const core=Control.fromModule(moduleCore),zero=Array(10).fill(0);
for(const dt of [0,-.01,.101,NaN,Infinity]){
  const out=core.step([1,0,0,0,0,0],zero,dt);assert.equal(out.valid,false);assert.deepEqual(out.volts,[0,0,0,0]);
}
for(const index of [0,3,6,9]){
  const feedback=[...zero];feedback[index]=NaN;
  const out=core.step([0,0,0,0,0,0],feedback,.01);assert.equal(out.valid,false);assert(out.volts.every(v=>v===0));
}
for(let i=0;i<100;i++){const out=core.step([100,-100,3,20,20,10],zero,.01);assert(out.valid);assert(out.volts.every(v=>Math.abs(v)<=12));assert(out.targets.every(v=>Math.abs(v)<=.864463));}
// Independent Bernstein polynomial oracle, not a copy of the C++ de Casteljau implementation.
for(const points of [[[0,0],[0,16],[24,8],[24,24]],[[1,-2],[-9,14],[20,-30],[4,6]]])for(const t of [0,.1,.25,.5,.9,1]){
  core.buffer.set(points.flat());core.e.control_curve_eval(t);const got=core.buffer.slice(32,38),u=1-t;
  for(let axis=0;axis<2;axis++){
    const p=points.map(p=>p[axis]);const expected=[u*u*u*p[0]+3*u*u*t*p[1]+3*u*t*t*p[2]+t*t*t*p[3],3*(u*u*(p[1]-p[0])+2*u*t*(p[2]-p[1])+t*t*(p[3]-p[2])),6*(u*(p[2]-2*p[1]+p[0])+t*(p[3]-2*p[2]+p[1]))];
    expected.forEach((v,i)=>assert(Math.abs(got[i*2+axis]-v)<1e-4));
  }
}
core.reset();assert(core.step([0,0,0,0,0,0],zero,.01).volts.every(v=>v===0),'Reset must clear integral');
const spline=core.spline({x:0,y:0,theta:0},{x:24,y:24,theta:90},.45,.8,3.5);
assert(spline.length>2);for(let i=1;i<spline.length;i++){assert(spline[i].time>spline[i-1].time);assert(Math.abs(spline[i].linear_vel)<=.45);}
assert(Math.hypot(spline.at(-1).x-.6096,spline.at(-1).y-.6096)<1e-6);
assert.equal(core.spline({x:NaN,y:0,theta:0},{x:24,y:24,theta:90}).length,0);
fs.writeFileSync(path.join(root,'docs/recovery/control-results.json'),JSON.stringify(report,null,2)+'\n');
console.log(`PASS: ${report.length} closed-loop scenarios; faults, saturation, reset, production spline and slip observability.`);
