'use strict';

const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {spawnSync}=require('node:child_process');
const CAD=require('../simulator/cad-runtime');
const COMPONENT_BYTES={5120:1,5121:1,5122:2,5123:2,5125:4,5126:4};
const COMPONENTS={SCALAR:1,VEC2:2,VEC3:3,VEC4:4};
const align=length=>(length+3)&~3;

function mergeFaces(input,parsed=CAD.inspectGlb(input,{enforceBudget:false})){
  const {document,binary}=parsed;
  if(document.animations?.length||document.skins?.length||document.images?.length||document.textures?.length||document.extensionsRequired?.length)throw Error('This preparation tool accepts static, untextured, self-contained CAD only');
  if(document.buffers?.length!==1||document.buffers[0].uri)throw Error('Expected one embedded CAD buffer');
  const parts=[],accessors=[],bufferViews=[];
  let byteLength=0;
  function accessorData(index){
    const accessor=document.accessors[index],view=document.bufferViews?.[accessor?.bufferView];
    const componentBytes=COMPONENT_BYTES[accessor?.componentType],components=COMPONENTS[accessor?.type];
    if(!view||view.buffer!==0||accessor.sparse||!componentBytes||!components||!Number.isInteger(accessor.count)||accessor.count<0)throw Error('Unsupported CAD accessor');
    const width=componentBytes*components,start=(view.byteOffset||0)+(accessor.byteOffset||0),stride=view.byteStride||width;
    const end=start+Math.max(0,accessor.count-1)*stride+width;
    if(stride<width||end>binary.length||end>(view.byteOffset||0)+view.byteLength)throw Error('CAD accessor exceeds buffer bounds');
    return {accessor,width,start,stride};
  }
  function append(data,accessor,target){
    const padding=align(byteLength)-byteLength;
    if(padding){parts.push(Buffer.alloc(padding));byteLength+=padding;}
    const index=accessors.length;
    bufferViews.push({buffer:0,byteOffset:byteLength,byteLength:data.length,target});
    accessors.push({...accessor,bufferView:bufferViews.length-1,byteOffset:0});
    parts.push(data);byteLength+=data.length;return index;
  }
  const meshes=document.meshes.map(mesh=>{
    const groups=new Map();
    for(const primitive of mesh.primitives){
      if((primitive.mode??4)!==4||primitive.targets||primitive.extensions)throw Error('Unsupported CAD primitive');
      const names=Object.keys(primitive.attributes).sort();
      if(!names.includes('POSITION'))throw Error('CAD primitive has no positions');
      const signature=JSON.stringify([primitive.material,names.map(name=>{
        const {accessor}=accessorData(primitive.attributes[name]);
        return [name,accessor.componentType,accessor.type,!!accessor.normalized];
      }),primitive.extras]);
      if(!groups.has(signature))groups.set(signature,[]);
      groups.get(signature).push(primitive);
    }
    const primitives=[...groups.values()].map(group=>{
      const counts=group.map(primitive=>accessorData(primitive.attributes.POSITION).accessor.count),vertices=counts.reduce((total,count)=>total+count,0);
      const attributes={};
      for(const name of Object.keys(group[0].attributes)){
        const first=accessorData(group[0].attributes[name]),data=Buffer.alloc(vertices*first.width);
        let cursor=0;
        const minimum=first.accessor.min?.map(()=>Infinity),maximum=first.accessor.max?.map(()=>-Infinity);
        group.forEach((primitive,index)=>{
          const source=accessorData(primitive.attributes[name]);
          if(source.accessor.count!==counts[index])throw Error('CAD vertex stream counts differ');
          if(source.stride===source.width){
            data.set(binary.subarray(source.start,source.start+source.accessor.count*source.width),cursor);
            cursor+=source.accessor.count*source.width;
          }else for(let vertex=0;vertex<source.accessor.count;vertex++){
            data.set(binary.subarray(source.start+vertex*source.stride,source.start+vertex*source.stride+source.width),cursor);cursor+=source.width;
          }
          if(minimum&&maximum){
            if(!source.accessor.min||!source.accessor.max)throw Error('CAD bounds missing');
            for(let component=0;component<minimum.length;component++){
              minimum[component]=Math.min(minimum[component],source.accessor.min[component]);maximum[component]=Math.max(maximum[component],source.accessor.max[component]);
            }
          }
        });
        attributes[name]=append(data,{componentType:first.accessor.componentType,type:first.accessor.type,count:vertices,
          ...(first.accessor.normalized?{normalized:true}:{}),...(minimum?{min:minimum,max:maximum}:{})},34962);
      }
      const indexCount=group.reduce((total,primitive,index)=>total+(primitive.indices===undefined?counts[index]:accessorData(primitive.indices).accessor.count),0),indices=Buffer.alloc(indexCount*4);
      let vertexOffset=0,cursor=0;
      group.forEach((primitive,index)=>{
        const source=primitive.indices===undefined?null:accessorData(primitive.indices),count=source?.accessor.count??counts[index];
        if(count%3||source&&(![5121,5123,5125].includes(source.accessor.componentType)||source.accessor.type!=='SCALAR'))throw Error('Unsupported CAD triangle indices');
        for(let offset=0;offset<count;offset++){
          const position=source?source.start+offset*source.stride:0;
          const vertex=!source?offset:source.width===1?binary[position]:source.width===2?new DataView(binary.buffer,binary.byteOffset+position,2).getUint16(0,true):new DataView(binary.buffer,binary.byteOffset+position,4).getUint32(0,true);
          if(vertex>=counts[index])throw Error('CAD triangle index exceeds vertex count');
          indices.writeUInt32LE(vertex+vertexOffset,cursor);cursor+=4;
        }
        vertexOffset+=counts[index];
      });
      return {...group[0],attributes,indices:append(indices,{componentType:5125,type:'SCALAR',count:indexCount},34963)};
    });
    return {...mesh,primitives};
  });
  const output={...document,accessors,bufferViews,buffers:[{byteLength}],meshes};
  const json=Buffer.from(JSON.stringify(output)),jsonLength=align(json.length),binaryLength=align(byteLength),header=Buffer.alloc(20),chunk=Buffer.alloc(8);
  header.writeUInt32LE(0x46546c67,0);header.writeUInt32LE(2,4);header.writeUInt32LE(28+jsonLength+binaryLength,8);header.writeUInt32LE(jsonLength,12);header.writeUInt32LE(0x4e4f534a,16);
  chunk.writeUInt32LE(binaryLength,0);chunk.writeUInt32LE(0x004e4942,4);
  return Buffer.concat([header,json,Buffer.alloc(jsonLength-json.length,32),chunk,...parts,Buffer.alloc(binaryLength-byteLength)]);
}

function prepare(filename,executable){
  const root=path.resolve(__dirname,'..'),directory=path.join(root,'.cache/cad-tools');
  fs.mkdirSync(directory,{recursive:true});
  const mergedPath=path.join(directory,'robot.merged.glb'),preview=path.join(directory,'robot.preview.glb'),metadata=path.join(directory,'merged.json');
  const worker=spawnSync(process.execPath,[__filename,'--merge',path.resolve(filename),mergedPath,metadata],{encoding:'utf8',maxBuffer:1024*1024});
  if(worker.error||worker.status!==0)throw Error(worker.error?.message||worker.stderr||'CAD merge worker failed');
  const result=spawnSync(executable,['-i',mergedPath,'-o',preview,'-si','0.05','-kn','-noq','-ke'],{encoding:'utf8',maxBuffer:1024*1024});
  if(result.error||result.status!==0)throw Error(result.error?.message||result.stderr||result.stdout||'gltfpack failed');
  const output=fs.readFileSync(preview),optimized=CAD.inspectGlb(output).stats;
  const report={schemaVersion:1,...JSON.parse(fs.readFileSync(metadata,'utf8')),
    preview:{filename:'robot.preview.glb',sha256:crypto.createHash('sha256').update(output).digest('hex'),...optimized},
    optimizer:'gltfpack 1.3',options:['-si','0.05','-kn','-noq','-ke'],scope:'Lossy visual CAD preview only; no collision, mass or joint calibration. Original file is not modified.'};
  fs.writeFileSync(path.join(root,'simulator/models/robot.preview.glb'),output);
  fs.writeFileSync(path.join(root,'simulator/models/robot.preview.json'),JSON.stringify(report,null,2)+'\n');
  return report;
}

if(require.main===module){
  try{
    if(process.argv[2]==='--merge'){
      const input=fs.readFileSync(process.argv[3]),parsed=CAD.inspectGlb(input,{enforceBudget:false}),merged=mergeFaces(input,parsed);
      fs.writeFileSync(process.argv[4],merged);
      fs.writeFileSync(process.argv[5],JSON.stringify({source:{filename:path.basename(process.argv[3]),sha256:crypto.createHash('sha256').update(input).digest('hex'),...parsed.stats},
        merged:CAD.inspectGlb(merged,{enforceBudget:false}).stats}));
    }else{
      if(!process.argv[2])throw Error('Usage: node tools/prepare-cad.js input.glb [gltfpack.exe]');
      const executable=process.argv[3]||path.join(__dirname,'../.cache/cad-tools/v1.3/gltfpack.exe');
      console.log(JSON.stringify(prepare(process.argv[2],executable),null,2));
    }
  }catch(error){console.error(error.message);process.exitCode=1;}
}
module.exports={mergeFaces,prepare};
