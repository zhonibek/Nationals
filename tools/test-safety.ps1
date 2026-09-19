param([Parameter(Mandatory=$true)][string]$Zig,[Parameter(Mandatory=$true)][string]$Cache)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$env:ZIG_GLOBAL_CACHE_DIR=$Cache
$env:ZIG_LOCAL_CACHE_DIR=$Cache
$testOutput=Join-Path $Cache 'safety-test.wasm'
& $Zig c++ (Join-Path $repo 'tests/safety.cpp') (Join-Path $repo 'tests/motion.cpp') (Join-Path $repo 'src/lemlib/safety.cpp') (Join-Path $repo 'src/subsystems/control/HolonomicMotion.cpp') -std=c++17 -O2 -fno-exceptions -target wasm32-wasi -mexec-model=reactor -I (Join-Path $repo 'tests/mocks') -I (Join-Path $repo 'include') '-Wl,--export=safety_test' '-Wl,--export=motion_test' -o $testOutput
if($LASTEXITCODE -ne 0){throw 'Safety test build failed'}
& node -e 'const fs=require("fs");const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));const e=new WebAssembly.Instance(m,{wasi_snapshot_preview1:{fd_write:()=>8,fd_close:()=>8,fd_seek:()=>8,proc_exit:code=>{throw Error("C++ abort "+code)}}}).exports;e._initialize();for(const test of ["safety_test","motion_test"]){const result=e[test]();if(result)throw Error(test+" assertion at line "+result);console.log("PASS: production "+test);}' $testOutput
if($LASTEXITCODE -ne 0){throw 'Safety regression failed'}
