(function(root,factory){
  if(typeof module==='object'&&module.exports)module.exports=factory();
  else root.RobotCAD=factory();
})(globalThis,function(){
  'use strict';

  const LIMITS={bytes:32*1024*1024,drawCalls:1500,triangles:1000000};
  function checkSize(bytes){
    if(bytes>LIMITS.bytes)throw Error('CAD слишком тяжёлый: используйте оптимизированный GLB до 32 МБ');
  }
  function checkCost(stats){
    if(stats.drawCalls>LIMITS.drawCalls||stats.triangles>LIMITS.triangles)throw Error('CAD перегружает 3D: объедините грани по материалам и упростите сетку');
  }
  function inspectGlb(bytes,{enforceBudget=true}={}){
    const source=ArrayBuffer.isView(bytes)?new Uint8Array(bytes.buffer,bytes.byteOffset,bytes.byteLength):new Uint8Array(bytes);
    if(enforceBudget)checkSize(source.byteLength);
    if(source.byteLength<20)throw Error('Incomplete GLB header');
    const view=new DataView(source.buffer,source.byteOffset,source.byteLength);
    if(view.getUint32(0,true)!==0x46546c67||view.getUint32(4,true)!==2)throw Error('Expected GLB version 2');
    if(view.getUint32(8,true)!==source.byteLength)throw Error('Incomplete GLB file');
    let offset=12,document=null,binary=null;
    while(offset<source.byteLength){
      if(offset+8>source.byteLength)throw Error('Incomplete GLB chunk header');
      const length=view.getUint32(offset,true),type=view.getUint32(offset+4,true),end=offset+8+length;
      if(length%4||end>source.byteLength)throw Error('Invalid GLB chunk length');
      if(type===0x4e4f534a){
        if(document||offset!==12)throw Error('Invalid GLB JSON chunk');
        try{document=JSON.parse(new TextDecoder().decode(source.subarray(offset+8,end)));}
        catch{throw Error('Invalid GLB JSON');}
      }else if(type===0x004e4942){
        if(binary)throw Error('Duplicate GLB binary chunk');
        binary=source.subarray(offset+8,end);
      }
      offset=end;
    }
    if(!document||document.asset?.version!=='2.0')throw Error('Invalid glTF document');
    for(const [index,buffer] of (document.buffers||[]).entries()){
      if(!buffer.uri&&(index!==0||!binary||buffer.byteLength>binary.byteLength||binary.byteLength-buffer.byteLength>3))throw Error('Incomplete GLB binary data');
    }
    const scene=document.scenes?.[document.scene??0];
    if(!scene)throw Error('GLB has no default scene');
    let drawCalls=0,triangles=0,meshInstances=0;
    const visited=new Set(),pending=[...(scene.nodes||[])];
    while(pending.length){
      const index=pending.pop(),node=document.nodes?.[index];
      if(!node||visited.has(index))throw Error('Invalid GLB node hierarchy');
      visited.add(index);pending.push(...(node.children||[]));
      if(node.mesh===undefined)continue;
      const mesh=document.meshes?.[node.mesh];
      if(!mesh)throw Error('Invalid GLB mesh reference');
      meshInstances++;
      for(const primitive of mesh.primitives){
        const accessor=document.accessors?.[primitive.indices??primitive.attributes?.POSITION];
        if(!accessor||!Number.isInteger(accessor.count)||accessor.count<0)throw Error('Invalid GLB accessor reference');
        const mode=primitive.mode??4;
        drawCalls++;
        triangles+=mode===4?Math.floor(accessor.count/3):mode===5||mode===6?Math.max(0,accessor.count-2):0;
      }
    }
    const stats={bytes:source.byteLength,nodes:visited.size,meshes:document.meshes?.length||0,meshInstances,drawCalls,triangles};
    if(enforceBudget)checkCost(stats);
    return {document,binary,stats};
  }
  function modelStats(model){
    let drawCalls=0,triangles=0;
    model.traverse(part=>{
      if(!part.isMesh)return;
      const geometry=part.geometry,count=geometry.index?.count??geometry.attributes.position?.count??0;
      drawCalls+=Array.isArray(part.material)?Math.max(1,geometry.groups.length):1;
      triangles+=Math.floor(count/3);
    });
    return {drawCalls,triangles};
  }
  function disposeModel(model){
    const resources=new Set();
    model.traverse(part=>{
      if(part.geometry)resources.add(part.geometry);
      for(const material of Array.isArray(part.material)?part.material:part.material?[part.material]:[]){
        resources.add(material);
        for(const value of Object.values(material))if(value?.isTexture)resources.add(value);
      }
    });
    for(const resource of resources)resource.dispose?.();
  }
  return {LIMITS,checkSize,checkCost,inspectGlb,modelStats,disposeModel};
});
