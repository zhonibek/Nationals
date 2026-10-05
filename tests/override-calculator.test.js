'use strict';

const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const Game=require('../simulator/override');

function calculatorFixture(initial={}){
  const elements=new Map();let now=1000;
  const element=id=>{
    if(!elements.has(id))elements.set(id,{value:initial[id]??(id==='gameObject'?'auto':'0'),textContent:'',listeners:{},children:[],
      addEventListener(name,callback){this.listeners[name]=callback;},replaceChildren(){this.children=[];},append(child){this.children.push(child);}});
    return elements.get(id);
  };
  const context=vm.createContext({module:{exports:{}},require:()=>Game,document:{getElementById:element,querySelectorAll:()=>[],
    createElement:()=>({children:[],append(child){this.children.push(child);}})},window:{addEventListener(){}},performance:{now:()=>now}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../simulator/override-ui.js'),'utf8'),context);
  const sim={override:new Game({physical:false}),activeRobotId:'red-1',matchMode:false};
  const ui=context.module.exports.attach(sim);
  return {sim,element,ui,update(){now+=100;ui.update();}};
}

test('calculator initializes restored inputs, recomputes input/change and never changes field',()=>{
  const {sim,element}=calculatorFixture({'calc-coloredHalves':'3','calc-yellowHalves':'2','calc-midfieldRobots':'1','calc-autonomousBonus':'12'}),before=sim.override.getState();
  assert.equal(element('manualPoints').textContent,'55 очков');assert.match(element('manualFormula').textContent,/3 × 5 \+ 2 × 10 \+ 1 × 8 \+ 12 = 55/);
  element('calc-coloredHalves').value='4';element('pointCalculator').listeners.input();assert.equal(element('manualPoints').textContent,'60 очков');
  element('calc-autonomousBonus').value='6';element('pointCalculator').listeners.change();assert.equal(element('manualPoints').textContent,'54 очков');
  assert.deepEqual(sim.override.getState(),before);
});

test('empty/invalid fields clear stale totals; correction and reset restore valid calculation',()=>{
  const {element}=calculatorFixture({'calc-coloredHalves':'2'});
  assert.equal(element('manualPoints').textContent,'10 очков');
  for(const invalid of ['', ' ', '-1', '0.5', 'NaN', 'Infinity','127']){
    element('calc-coloredHalves').value=invalid;element('pointCalculator').listeners.input();
    assert.equal(element('manualPoints').textContent,'—');assert(element('calculatorError').textContent);assert.doesNotMatch(element('manualFormula').textContent,/= 10/);
  }
  element('calc-coloredHalves').value='126';element('calc-yellowHalves').value='1';element('pointCalculator').listeners.input();assert.equal(element('manualPoints').textContent,'—');
  element('calc-yellowHalves').value='0';element('pointCalculator').listeners.input();assert.equal(element('manualPoints').textContent,'630 очков');assert.equal(element('calculatorError').textContent,'');
  element('calculatorReset').listeners.click();assert.equal(element('manualPoints').textContent,'0 очков');assert.equal(element('calc-coloredHalves').value,'0');
});

test('live snapshots, floating totals and frozen finals use the same rule-derived breakdown',()=>{
  const {sim,element,update}=calculatorFixture(),game=sim.override;
  const pin=game.pins.find(candidate=>candidate.location==='field'&&candidate.halves.includes('red'));
  assert(game.placePin(pin.id,'g-red-sw').ok);game.setToggle('toggle-west','red');game.setRobotPose('blue-1',{x:0,y:0});
  const before=game.getState();update();
  for(const color of ['red','blue']){
    const row=game.scoreBreakdown()[color];element('calculator-'+color).listeners.click();
    assert.equal(element('manualPoints').textContent,String(row.total)+' очков');assert.equal(element('points-'+color).textContent,String(row.total));
    assert.equal(element('scoreDock-'+color).textContent,String(row.total));
  }
  assert.deepEqual(game.getState(),before);
  game.startMatch('practice');game.stopMatch();const final=game.scoreBreakdown();game.setToggle('toggle-west','blue');update();
  assert.equal(element('scoreDockState').textContent,'Итоговый счёт');element('calculator-red').listeners.click();assert.equal(element('manualPoints').textContent,String(final.red.total)+' очков');
});

test('calculator presentation explains visible halves and preserves explicit bounds',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../simulator/index.html'),'utf8'),css=fs.readFileSync(path.join(__dirname,'../simulator/simulator.css'),'utf8');
  assert.match(html,/Считаются видимые половинки, не целые детали/);assert.match(html,/id="scoreDock-red"/);assert.match(html,/id="calculatorError" role="alert"/);
  assert.match(css,/\.score-dock \{ position: fixed/);assert.equal((html.match(/id="manualPoints"/g)||[]).length,1);
});
