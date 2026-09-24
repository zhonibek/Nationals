/* Actual production C++ controller and spline generator, compiled by build-wasm.ps1. */
(function(root){
  'use strict';
  class ProductionControl {
    constructor(instance,module){this.module=module;this.e=instance.exports;this.e._initialize?.();this.reset();}
    fork(){return ProductionControl.fromModule(this.module);}
    get buffer(){return new Float64Array(this.e.memory.buffer,this.e.control_buffer(),64);}
    reset(){this.e.control_reset();}
    configure(radius,speed){this.e.control_config(radius,speed,12/speed);}
    duration(a,b){return this.e.control_duration(...a,...b,0.45,0.8,3.5);}
    bezier(points,h0,h1){return this.e.control_bezier(...points.flat(),h0,h1);}
    bezierReference(t){if(!this.e.control_bezier_reference(t))throw Error('Bezier unavailable');return Array.from(this.buffer.slice(0,6));}
    reference(a,b,t,duration){this.e.control_reference(...a,...b,t,duration);return Array.from(this.buffer.slice(0,6));}
    spline(a,b,speed=0.45,accel=0.8,jerk=3.5,dt=0.01){
      const count=this.e.control_spline(a.x,a.y,a.theta,b.x,b.y,b.theta,speed,accel,jerk,dt),result=[];
      for(let i=0;i<count;i++){this.e.control_spline_sample(i);const s=this.buffer;result.push({x:s[0],y:s[1],heading:s[2],linear_vel:s[3],angular_vel:s[4],time:s[5]});}
      return result;
    }
    step(reference,feedback,dt){
      if(reference.length!==6||feedback.length!==10){this.reset();return {valid:false,volts:[0,0,0,0],targets:[0,0,0,0]};}
      this.buffer.set([...reference,...feedback,dt,1]);
      const valid=this.e.control_step()===1;
      return {valid,volts:Array.from(this.buffer.slice(32,36)),targets:Array.from(this.buffer.slice(36,40))};
    }
    static async load(url='control.wasm'){
      const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw Error(`C++ control download: ${response.status}`);
      const module=await WebAssembly.compile(await response.arrayBuffer());return ProductionControl.fromModule(module);
    }
    static fromModule(module){
      // No filesystem/clock/random imports are permitted in the control module.
      const wasi={proc_exit(code){throw Error(`C++ abort ${code}`);},fd_close(){return 8;},fd_seek(){return 8;},fd_write(){return 8;}};
      return new ProductionControl(new WebAssembly.Instance(module,{wasi_snapshot_preview1:wasi}),module);
    }
  }
  root.ProductionControl=ProductionControl;
  if(typeof module==='object')module.exports=ProductionControl;
})(globalThis);
