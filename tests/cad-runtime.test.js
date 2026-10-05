'use strict';

const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const crypto=require('node:crypto');
const CAD=require('../simulator/cad-runtime');
const {mergeFaces}=require('../tools/prepare-cad');
const {createServer}=require('../roboproof/server');

function glb(document,binary){
  const json=Buffer.from(JSON.stringify(document)),jsonPadding=(4-json.length%4)%4,binaryPadding=(4-binary.length%4)%4;
  const header=Buffer.alloc(20),chunk=Buffer.alloc(8);
  header.writeUInt32LE(0x46546c67,0);header.writeUInt32LE(2,4);header.writeUInt32LE(28+json.length+jsonPadding+binary.length+binaryPadding,8);
  header.writeUInt32LE(json.length+jsonPadding,12);header.writeUInt32LE(0x4e4f534a,16);
  chunk.writeUInt32LE(binary.length+binaryPadding,0);chunk.writeUInt32LE(0x004e4942,4);
  return Buffer.concat([header,json,Buffer.alloc(jsonPadding,32),chunk,binary,Buffer.alloc(binaryPadding)]);
}

function fixture({differentMaterials=false}={}){
  const accessors=[],bufferViews=[],parts=[];
  let byteLength=0;
  function stream(values,type,componentType,min,max){
    const bytes=Buffer.from(componentType===5126?new Float32Array(values).buffer:new Uint32Array(values).buffer);
    bufferViews.push({buffer:0,byteOffset:byteLength,byteLength:bytes.length});byteLength+=bytes.length;parts.push(bytes);
    const index=accessors.length;accessors.push({bufferView:bufferViews.length-1,componentType,type,count:values.length/(type==='VEC3'?3:type==='VEC2'?2:1),...(min?{min,max}:{})});return index;
  }
  const primitives=[];
  for(const height of [0,1]){
    const positions=stream([0,0,height,1,0,height,0,1,height],'VEC3',5126,[0,0,height],[1,1,height]);
    const normals=stream([0,0,1,0,0,1,0,0,1],'VEC3',5126);
    const texture=stream([0,0,1,0,0,1],'VEC2',5126);
    const indices=stream([0,1,2],'SCALAR',5125);
    primitives.push({attributes:{POSITION:positions,NORMAL:normals,TEXCOORD_0:texture},indices,material:differentMaterials?height:0});
  }
  const document={asset:{version:'2.0'},scene:0,scenes:[{nodes:[0]}],nodes:[{name:'main',children:[1,2]},
    {name:'wheel',mesh:0,translation:[1,2,3]},{name:'lift',mesh:0,translation:[4,5,6]}],meshes:[{name:'part',primitives}],
    materials:[{name:'red',pbrMetallicRoughness:{baseColorFactor:[1,0,0,1]}},{name:'blue',pbrMetallicRoughness:{baseColorFactor:[0,0,1,1]}}],accessors,bufferViews,buffers:[{byteLength}]};
  return {document,binary:Buffer.concat(parts),bytes:glb(document,Buffer.concat(parts))};
}

function rendererRealm(extra={}){
  const elements=new Map();
  const element=id=>{
    if(!elements.has(id))elements.set(id,{textContent:'',style:{}});
    return elements.get(id);
  };
  const source=fs.readFileSync(path.join(__dirname,'../simulator/simulator.js'),'utf8').split("document.addEventListener('DOMContentLoaded'")[0];
  const context=vm.createContext({console,RobotCAD:CAD,TextDecoder,TextEncoder,ArrayBuffer,Uint8Array,setTimeout,
    document:{getElementById:element},window:{},performance:{now:()=>0},...extra});
  vm.runInContext(source+'\nglobalThis.Renderer=ThreeFieldRenderer;',context);
  return {view:Object.create(context.Renderer.prototype),elements,context};
}

test('GLB completeness, JSON, hierarchy and rendering budgets fail before model loading',()=>{
  const {bytes,document,binary}=fixture();
  assert.deepEqual(CAD.inspectGlb(bytes).stats,{bytes:bytes.length,nodes:3,meshes:1,meshInstances:2,drawCalls:4,triangles:4});
  assert.throws(()=>CAD.inspectGlb(bytes.subarray(0,10)),/header/);
  assert.throws(()=>CAD.inspectGlb(bytes.subarray(0,bytes.length-4)),/Incomplete/);
  const invalid=Buffer.from(bytes);invalid[20]=0;
  assert.throws(()=>CAD.inspectGlb(invalid),/JSON/);
  document.nodes[1].children=[0];assert.throws(()=>CAD.inspectGlb(glb(document,binary)),/hierarchy/);
  assert.throws(()=>CAD.checkSize(CAD.LIMITS.bytes+1),/32 МБ/);
  assert.throws(()=>CAD.checkCost({drawCalls:1501,triangles:1}),/3D/);
  assert.throws(()=>CAD.checkCost({drawCalls:1,triangles:1000001}),/3D/);
});

test('face merging preserves part transforms, colors, normals, UVs and triangle indices',()=>{
  const original=fixture(),merged=CAD.inspectGlb(mergeFaces(original.bytes));
  assert.equal(merged.stats.drawCalls,2);assert.equal(merged.stats.triangles,4);
  assert.deepEqual(merged.document.nodes,original.document.nodes);assert.deepEqual(merged.document.materials,original.document.materials);
  const primitive=merged.document.meshes[0].primitives[0],accessor=merged.document.accessors[primitive.indices],view=merged.document.bufferViews[accessor.bufferView];
  const indices=new DataView(merged.binary.buffer,merged.binary.byteOffset+view.byteOffset,view.byteLength);
  assert.deepEqual(Array.from({length:6},(_,index)=>indices.getUint32(index*4,true)),[0,1,2,3,4,5]);
  for(const name of ['POSITION','NORMAL','TEXCOORD_0']){
    const output=merged.document.accessors[primitive.attributes[name]],outputView=merged.document.bufferViews[output.bufferView];
    const expected=original.document.meshes[0].primitives.map(face=>{
      const input=original.document.accessors[face.attributes[name]],inputView=original.document.bufferViews[input.bufferView];
      return original.binary.subarray(inputView.byteOffset,inputView.byteOffset+inputView.byteLength);
    });
    assert.equal(output.count,6);assert.deepEqual(Buffer.from(merged.binary.subarray(outputView.byteOffset,outputView.byteOffset+outputView.byteLength)),Buffer.concat(expected));
  }
});

test('merging does not combine different materials or silently discard unsupported animation',()=>{
  const original=fixture({differentMaterials:true});
  assert.equal(CAD.inspectGlb(mergeFaces(original.bytes)).document.meshes[0].primitives.length,2);
  original.document.animations=[{}];
  assert.throws(()=>mergeFaces(glb(original.document,original.binary)),/static/);
});

test('merged CAD rejects invalid triangle indices instead of corrupting another component',()=>{
  const {document,binary}=fixture(),accessor=document.accessors[document.meshes[0].primitives[0].indices],view=document.bufferViews[accessor.bufferView];
  binary.writeUInt32LE(99,view.byteOffset);
  assert.throws(()=>mergeFaces(glb(document,binary)),/vertex count/);
});

test('optimized uploaded asset matches provenance and fits browser render limits',()=>{
  const bytes=fs.readFileSync(path.join(__dirname,'../simulator/models/robot.preview.glb'));
  const report=JSON.parse(fs.readFileSync(path.join(__dirname,'../simulator/models/robot.preview.json'),'utf8'));
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),report.preview.sha256);
  const {stats,document}=CAD.inspectGlb(bytes);
  for(const name of ['bytes','nodes','meshes','meshInstances','drawCalls','triangles'])assert.equal(stats[name],report.preview[name]);
  assert.equal(stats.nodes,report.source.nodes);assert.equal(stats.meshes,report.source.meshes);
  assert(stats.bytes<report.source.bytes/20);assert(stats.drawCalls<report.source.drawCalls/100);assert(stats.triangles<report.source.triangles/10);
  assert.equal(document.extensionsRequired?.length||0,0);
});

test('GPU resource disposal releases shared geometries, materials and textures exactly once',()=>{
  let geometries=0,materials=0,textures=0;
  const geometry={dispose(){geometries++;}},texture={isTexture:true,dispose(){textures++;}},material={map:texture,dispose(){materials++;}};
  CAD.disposeModel({traverse(callback){callback({geometry,material});callback({geometry,material:[material]});}});
  assert.deepEqual([geometries,materials,textures],[1,1,1]);
});

test('oversized file import never reads, parses or saves the original CAD',async()=>{
  const {view,elements}=rendererRealm();
  let touched=false;view.ensureCadLoaded=async()=>{touched=true;};
  await view.loadCadFile({name:'main.glb',size:166172592,arrayBuffer(){touched=true;throw Error('must not read');}});
  assert.equal(touched,false);assert.match(elements.get('cadStatusBadge').textContent,/32 МБ/);
});

test('CAD parse promises wait for asynchronous success and expose failures',async()=>{
  const input=fixture().bytes;
  const {view}=rendererRealm({THREE:{GLTFLoader:class{parse(buffer,base,loaded){setTimeout(()=>loaded({scene:{}}),5);}}}});
  let applied=false;view.applyCadModel=()=>{applied=true;};
  const pending=view.loadCadFromBuffer(input,'glb','fixture.glb');assert.equal(applied,false);await pending;assert.equal(applied,true);
  const broken=rendererRealm({THREE:{GLTFLoader:class{parse(buffer,base,loaded,failed){setTimeout(()=>failed(Error('parse failed')),5);}}}}).view;
  await assert.rejects(broken.loadCadFromBuffer(input,'glb','broken.glb'),/parse failed/);
});

test('auto-load awaits optimized preview, skips failed parses and fetches only once',async()=>{
  const files=[],{view}=rendererRealm({fetch:async filename=>{files.push(filename);return {ok:true,headers:{get(){return '10';}},arrayBuffer:async()=>new ArrayBuffer(10)};}});
  view.loadCadFromBuffer=async(buffer,ext,name)=>{if(name==='robot.preview.glb')throw Error('test parse failure');};
  await Promise.all([view.ensureCadLoaded(),view.ensureCadLoaded()]);
  assert.deepEqual(files,['models/robot.preview.glb','models/robot.glb']);
});

test('fast rendering caps graphics only and leaves production physics outside the renderer',()=>{
  let now=0,frames=0;const {view,elements}=rendererRealm({performance:{now:()=>now}});
  Object.assign(view,{isActive:true,nextRenderAt:0,renderQuality:'fast',performanceWindow:{start:0,frames:0},sim:{},
    renderer:{render(){frames++;},info:{render:{calls:366,triangles:406719}}}});
  for(now=0;now<=1000;now+=1000/60)view.render();
  assert(frames>=29&&frames<=31);now=1050;view.render();assert.match(elements.get('renderPerformance').textContent,/FPS/);
  let ratio;view.renderer.shadowMap={};view.renderer.setPixelRatio=value=>{ratio=value;};view.resize=()=>{};
  view.setRenderQuality('detail');assert.equal(view.renderer.shadowMap.enabled,true);assert.equal(ratio,1);
  view.setRenderQuality('fast');assert.equal(view.renderer.shadowMap.enabled,false);
  const html=fs.readFileSync(path.join(__dirname,'../simulator/index.html'),'utf8');assert.match(html,/cad-runtime.js/);assert.match(html,/renderQuality/);
});

test('server serves preview with correct MIME and does not expose original or tool cache',async context=>{
  const server=createServer({reportPath:'does-not-exist.json'});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  context.after(()=>new Promise(resolve=>{server.closeAllConnections();server.close(resolve);}));
  const base=`http://127.0.0.1:${server.address().port}/simulator/`;
  const asset=await fetch(base+'models/robot.preview.glb');assert.equal(asset.status,200);assert.equal(asset.headers.get('content-type'),'model/gltf-binary');
  assert.equal(CAD.inspectGlb(await asset.arrayBuffer()).stats.drawCalls,366);
  assert.equal((await fetch(base+'cad-runtime.js')).status,200);
  assert.equal((await fetch(base+'models/robot.preview.json')).status,200);
  assert.equal((await fetch(base+'models/main.glb')).status,404);
  assert.equal((await fetch(base+'.cache/cad-tools/gltfpack.exe')).status,404);
});
