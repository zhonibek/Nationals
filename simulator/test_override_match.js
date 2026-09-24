'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const Game=require('./override'),Fleet=require('./override-autonomy'),Control=require('./control-runtime');
const moduleCore=new WebAssembly.Module(fs.readFileSync(__dirname+'/control.wasm'));
const context={console,Math,setTimeout(){},document:{addEventListener(){}},window:{},OverrideGame:Game,OverrideFleet:Fleet};
vm.createContext(context);vm.runInContext(fs.readFileSync(__dirname+'/robot-config.js','utf8'),context);vm.runInContext(fs.readFileSync(__dirname+'/simulator.js','utf8')+'\nthis.Simulator=VexRobotSimulator;',context);
const make=()=>new context.Simulator(Control.fromModule(moduleCore));
const near=(a,b)=>assert(Math.abs(a-b)<1e-6,`${a} != ${b}`);
const sim=make();assert(sim.startGame('match').ok);
const entries=[...sim.fleet.entries.values()];
assert.equal(new Set(entries.map(e=>e.sim.productionControl.e.memory)).size,4,'Each robot needs independent WASM memory/integrators');
assert.equal(new Set(entries.map(e=>e.sim.odom)).size,4);
for(let i=0;i<1500;i++)sim.update(.01);
assert.equal(sim.override.phase,'driver');near(sim.override.clock,15);
assert(entries.every(e=>e.status==='Complete'),JSON.stringify(entries.map(e=>[e.status,e.error])));
assert.deepEqual(sim.override.score(),{red:11,blue:11});
assert.deepEqual(sim.override.autonomousScores,{red:5,blue:5});
assert.equal(sim.override.events.filter(e=>e.type==='capture').length,2);
assert.equal(sim.override.ruleEvents.length,0,'Default routes must stay on their autonomous side');
assert(entries.every(e=>!e.sim.isRunning&&e.sim.motorVolts.every(v=>v===0)));
// Selecting another robot must preserve every other plant's pose/odometry.
const redPose={...entries[0].sim.odom},blue=sim.fleet.entries.get('blue-2');
sim.activeRobotId='blue-2';sim.fleet.syncView();near(sim.x,blue.sim.x);assert.deepEqual({...entries[0].sim.odom},redPose);
sim.holonomicArcade(.3,0,0);for(let i=0;i<50;i++)sim.update(.01);
assert(blue.sim.motorVolts.some(v=>Math.abs(v)>.1));assert(entries[0].sim.motorVolts.every(v=>Math.abs(v)<.01));sim.holonomicArcade(0,0,0);
// Run the entire remaining period, including the physical settling window.
while(!sim.override.finalScore&&sim.simTime<126)sim.update(.1);
assert(sim.override.matchEnded&&sim.override.finalScore);near(sim.override.clock,120);
assert(sim.override.postMatchSeconds<=5);assert(entries.every(e=>e.sim.motorVolts.every(v=>v===0)));
const final=sim.override.score(),time=sim.simTime;sim.holonomicArcade(1,1,1);sim.update(.1);
assert.deepEqual(sim.override.score(),final);near(sim.simTime,time);
const report=sim.fleet.report();assert.equal(report.format,'nationals-match-v1');assert(report.trace.length>200);assert.equal(Object.keys(report.programs).length,4);assert.doesNotThrow(()=>JSON.stringify(report));
// Boundary splitting with a long command and an in-progress lift at 15.000s.
const cutoff=make();cutoff.startGame('match',{'red-1':[{type:'lift',height:40}],'red-2':[{type:'pose',targetX:30,targetY:40,targetTheta:0,timeout:30000}],'blue-1':'idle','blue-2':'idle'});
cutoff.override.clock=14.985;cutoff.update(.03);
assert.equal(cutoff.override.phase,'driver');near(cutoff.override.clock,15.015);
for(const e of cutoff.fleet.entries.values())assert(!e.sim.isRunning&&e.sim.motorVolts.every(v=>v===0));
const lift=cutoff.override.robots[0].manipulator,height=lift.height;near(lift.target,height);cutoff.update(.1);near(lift.height,height);
assert.equal(cutoff.fleet.entries.get('red-1').status,'PeriodEnded');
// Invalid dt fails closed; disqualification disables drivetrain and mechanism commands.
const fault=make();fault.startGame('match');fault.update(.1);fault.update(NaN);
assert([...fault.fleet.entries.values()].every(e=>e.status==='InvalidDt'&&e.sim.motorVolts.every(v=>v===0)));near(fault.override.clock,.1);
const dq=make();dq.startGame('practice');dq.override.adjudicate('red-1','SG9','major','Test ruling');dq.holonomicArcade(1,1,1);dq.update(.1);
assert(dq.fleet.entries.get('red-1').sim.motorVolts.every(v=>v===0));assert(!dq.override.setLiftTarget('red-1',20).ok);assert(!dq.override.interact('red-1','drop').ok);
console.log('PASS: complete four-robot match, independent production controllers, two physically scored preloads, selection, 15/120s cutoffs, final freeze, lift stop, invalid dt, DQ and JSON report.');
