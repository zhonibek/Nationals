param([Parameter(Mandatory=$true)][string]$Zig, [string]$Cache=(Join-Path $PSScriptRoot '../.cache/zig'))
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force $Cache | Out-Null
$env:ZIG_GLOBAL_CACHE_DIR=(Resolve-Path $Cache).Path
$env:ZIG_LOCAL_CACHE_DIR=$env:ZIG_GLOBAL_CACHE_DIR
& $Zig c++ (Join-Path $repo 'simulator/native/control.cpp') (Join-Path $repo 'src/subsystems/trajectory/QuinticSpline.cpp') (Join-Path $repo 'src/subsystems/pedro/BezierCurve.cpp') -std=c++17 -O2 -fno-exceptions -target wasm32-wasi -mexec-model=reactor -I (Join-Path $repo 'include') '-Wl,--export=control_buffer,--export=control_reset,--export=control_config,--export=control_step,--export=control_duration,--export=control_reference,--export=control_spline,--export=control_spline_sample,--export=control_bezier,--export=control_bezier_reference,--export=control_curve_eval' -o (Join-Path $repo 'simulator/control.wasm')
if($LASTEXITCODE -ne 0){throw 'WASM control build failed'}
$sources=@('include/subsystems/control/Cascade.hpp','include/subsystems/control/BezierReference.hpp','include/subsystems/trajectory/QuinticSpline.hpp','include/subsystems/ltv/State.hpp','include/lemlib/pose.hpp','include/subsystems/pedro/Point.hpp','include/subsystems/pedro/BezierCurve.hpp','src/subsystems/trajectory/QuinticSpline.cpp','src/subsystems/pedro/BezierCurve.cpp','simulator/native/control.cpp')
$hashes=[ordered]@{}
foreach($source in $sources){
    $bytes=[System.Text.Encoding]::UTF8.GetBytes((Get-Content -LiteralPath (Join-Path $repo $source) -Raw).Replace("`r`n","`n"))
    $sha=[System.Security.Cryptography.SHA256]::Create()
    $hashes[$source]=([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLower()
    $sha.Dispose()
}
@{compiler=(& $Zig version);sources=$hashes;wasm=(Get-FileHash (Join-Path $repo 'simulator/control.wasm') -Algorithm SHA256).Hash.ToLower()} | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $repo 'simulator/control-build.json') -Encoding utf8
Write-Output 'Built the production Cascade.hpp control core for the browser.'
