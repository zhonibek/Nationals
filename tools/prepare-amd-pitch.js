'use strict';

const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {loadHeadless}=require('../simulator/headless');
const {VexRobotSimulator}=require('../simulator/engine');
const Control=require('../simulator/control-runtime');
const UI=require('../simulator/override-ui');
const {scoringFixture}=require('../tests/helpers/override-fixtures');
const {evaluate,suite}=require('../roboproof/motion-evaluation');
const motion=require('../roboproof/motion');

const root=path.resolve(__dirname,'..');
const output=path.resolve(process.argv[2]||path.join(root,'roboproof/runs/pitch/amd-2026-10-05'));
fs.mkdirSync(output,{recursive:true});
const runtime=loadHeadless();
const compiled=new WebAssembly.Module(fs.readFileSync(path.join(root,'simulator/control.wasm')));
const createSimulator=()=>new VexRobotSimulator(Control.fromModule(compiled));

function sampleGame(sim){
  const game=sim.override;
  return {clock:game.clock,robots:game.robots.map(robot=>({id:robot.id,x:robot.x,y:robot.y,theta:robot.theta,lift:robot.manipulator.height,possession:{...robot.possession}})),
    objects:[...game.cups,...game.pins].filter(object=>object.location==='field'||object.status==='held'||object.placed).map(object=>({id:object.id,kind:object.kind,halves:object.halves,up:object.up,upIndex:object.upIndex,held:object.status==='held',placed:object.placed,pose:game.objectPose(object)})),
    goals:game.goals.map(goal=>({id:goal.id,x:goal.x,y:goal.y,alliance:goal.alliance,stack:goal.stack.length})),score:game.scoreBreakdown()};
}

function scoringTrace(){
  const sim=createSimulator();
  if(!scoringFixture(sim).ok)throw Error('Scoring fixture did not start');
  const game=sim.override,robot=game.robots[0],frames=[sampleGame(sim)];
  if(!game.interact(robot.id,'pickup','auto').ok)throw Error('Actual Pin pickup failed');
  game.setLiftTarget(robot.id,2);
  let released=false;
  for(let tick=0;tick<600;tick++){
    if(!released&&game.placementHint(robot.id,'pin').ready){
      sim.holonomicArcade(0,0,0);
      if(!game.interact(robot.id,'place','auto').ok)throw Error('Actual Pin placement failed');
      released=true;
    }
    sim.holonomicArcade(released?0:.35,0,0);sim.update(.01);
    if(tick%5===0)frames.push(sampleGame(sim));
  }
  if(!released||game.scorePins().red!==5)throw Error('Physical landing did not yield the expected five points');
  return {label:'Prepared scoring fixture; four original WASM controllers; not an official starting field',frames};
}

function bundleTrace(){
  const sim=createSimulator();
  if(!UI.startGame(sim,'practice').ok)throw Error('Practice did not start');
  const game=sim.override,robot=game.robots[0],cup=game.cups.find(object=>object.x===0&&object.y===-23.54);
  sim.fleet.entries.get(robot.id).sim.setPose(cup.x,cup.y-15,0);game.setRobotPose(robot.id,{x:cup.x,y:cup.y-15,theta:0});sim.fleet.syncView();
  if(!game.interact(robot.id,'use','auto').ok)throw Error('Actual Cup plus Pin pickup failed');
  game.setLiftTarget(robot.id,6);const frames=[sampleGame(sim)];
  for(let tick=0;tick<200;tick++){sim.holonomicArcade(0,tick<70?.2:0,0);sim.update(.01);if(tick%5===0)frames.push(sampleGame(sim));}
  return {label:'Original field; robot initial pose set for demonstration; genuine Cup plus Pin carry',frames};
}

const evaluation=evaluate(2);
fs.writeFileSync(path.join(output,'evaluation.json'),JSON.stringify(evaluation,null,2));
const worlds=suite(2).cases;
const failed=evaluation.results.find(row=>row.baseline.reason!=='success');
const failedWorld=worlds.find(world=>world.id===failed.id);
const failedReport=motion.baseline(failedWorld.seed,failedWorld.task,failedWorld.configuration,{recordTransitions:true});
motion.replay(failedReport);
const nominal=motion.baseline(42,undefined,{}, {recordTransitions:true});motion.replay(nominal);

const scenes=[
  {start:0,end:7,eyebrow:'THE PROBLEM',title:'AI can think.\nRobots must move.',narration:'AI can describe a task. A robot still has to move through slip, delay, and uncertainty.'},
  {start:7,end:16,eyebrow:'ONE CONNECTED PROJECT',title:'Simulation meets\nreal control.',narration:"RoboProof helps robotics teams test movement before risking hardware, connecting our VEX simulator to iraLIB's real C plus plus controllers."},
  {start:16,end:25,eyebrow:'WORKING GAMEPLAY',title:'Move. Grasp.\nPlace. Measure.',narration:'Robots drive, lift, grasp pins and cups, and score through physical placement. Experiments record seeds, telemetry, and replayable motion.'},
  {start:25,end:34,eyebrow:'EVIDENCE, NOT GUESSWORK',title:'Find the failure.\nReplay the proof.',narration:'We search stressful conditions, replay failures, and test proposed changes on unseen scenarios. Regressions stay visible, not hidden.'},
  {start:34,end:43,eyebrow:'AI + LEARNING FOUNDATION',title:'Coordinate today.\nLearn movement next.',narration:'Local Nemotron proposes reviewed tasks. An experimental neural predictor supports failure research. Movement-policy training is the next milestone.'},
  {start:43,end:52,eyebrow:'AMD ROADMAP',title:'Scale experiments\nwith AMD ROCm.',narration:'Next, AMD ROCm can power training and batched evaluation, measured against frozen baselines. GPU acceleration is not yet demonstrated.'},
  {start:52,end:60,eyebrow:'ROBOTICS DEVELOPERS + TEAMS',title:'RoboProof',narration:'For robotics developers: learn better movement, prove the improvement, then test on hardware. RoboProof. Evidence before motion.'}
];
const proof={createdAt:new Date().toISOString(),durationSeconds:60,language:'en',identity:runtime.identity,evaluation:{success:evaluation.baseline.success,count:evaluation.baseline.count,trainedPolicy:false},
  sources:{hackathon:'https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-iii',scoring:'https://www.vexrobotics.com/override-manual',repository:'https://github.com/zhonibek/Nationals'},
  disclosure:'Animated pitch rendered from source-checked engine data, not a browser screen recording. Gameplay includes labeled prepared scoring/pose fixtures. No measured GPU execution or learned movement improvement is claimed.',
  sourceFiles:Object.fromEntries(['simulator/override-ui.js','simulator/override.js','simulator/index.html','tools/prepare-amd-pitch.js'].map(filename=>[filename,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,filename))).digest('hex')])),
  scenes,scoring:scoringTrace(),bundle:bundleTrace(),nominal:{task:nominal.task,reason:nominal.reason,transitions:nominal.transitions},
  stress:{id:failedWorld.id,task:failedReport.task,reason:failedReport.reason,transitions:failedReport.transitions}};
fs.writeFileSync(path.join(output,'pitch-data.json'),JSON.stringify(proof));
fs.writeFileSync(path.join(output,'narration.txt'),scenes.map(scene=>scene.narration).join('\n\n'));
console.log(JSON.stringify({output,baseline:proof.evaluation,gameplayFrames:proof.scoring.frames.length+proof.bundle.frames.length,disclosure:proof.disclosure}));
