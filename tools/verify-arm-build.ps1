param(
    [Parameter(Mandatory=$true)][string]$Repository,
    [Parameter(Mandatory=$true)][string]$ToolchainBin,
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '..\.cache\arm-check')
)
$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path -LiteralPath $Repository).Path
New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
$buildPath = (Resolve-Path -LiteralPath $BuildDirectory).Path
$compiler = Join-Path $ToolchainBin 'arm-none-eabi-g++.exe'
$common = @('-std=gnu++20','-mcpu=cortex-a9','-mfpu=neon-fp16','-mfloat-abi=hard','-mthumb','-Os','-ffunction-sections','-fdata-sections','-funwind-tables','-D_POSIX_THREADS','-D_UNIX98_THREAD_MUTEX_ATTRIBUTES','-D_POSIX_TIMERS','-D_POSIX_MONOTONIC_CLOCK','-DEIGEN_DONT_VECTORIZE','-Wno-psabi','-Wall','-Wextra')
$log = Join-Path $buildPath 'compile.log'
'' | Set-Content -LiteralPath $log
$objects = @()
$failed = @()
foreach($file in Get-ChildItem (Join-Path $repoPath 'src') -Recurse -Filter '*.cpp') {
    $relative = $file.FullName.Substring(($repoPath+'\src\').Length)
    $includeDirectory = Join-Path (Join-Path $repoPath 'include') (Split-Path $relative -Parent)
    $objectFile = Join-Path $buildPath ($relative.Replace('\','_')+'.o')
    "FILE $relative" | Out-File $log -Append
    & $compiler -c $file.FullName @common -I (Join-Path $repoPath 'include') -iquote $includeDirectory -o $objectFile 2>&1 | Out-File $log -Append
    if($LASTEXITCODE -ne 0){$failed += $relative}
    $objects += $objectFile
}
if($failed.Count){throw ('Compilation failed: '+($failed -join ', '))}
$timestamp = Join-Path $buildPath 'timestamp.c'
'const int _PROS_COMPILE_TIMESTAMP_INT = 0; char const * const _PROS_COMPILE_TIMESTAMP = "AUDIT ONLY"; char const * const _PROS_COMPILE_DIRECTORY = "audit";' | Set-Content $timestamp
$timestampObject = Join-Path $buildPath 'timestamp.o'
& (Join-Path $ToolchainBin 'arm-none-eabi-gcc.exe') -c $timestamp -mcpu=cortex-a9 -mfpu=neon-fp16 -mfloat-abi=hard -mthumb -o $timestampObject
if($LASTEXITCODE -ne 0){throw 'Timestamp object failed'}
$objects += $timestampObject
$firmware = Join-Path $repoPath 'firmware'
$libraries = @(Get-ChildItem $firmware -Filter '*.a' | ForEach-Object FullName)
$elf = Join-Path $buildPath 'audit.elf'
& $compiler -mcpu=cortex-a9 -mfpu=neon-fp16 -mfloat-abi=hard -mthumb -Os -nostdlib '-Wl,--gc-sections' @objects '-Wl,--start-group' @libraries -lgcc -lstdc++ '-Wl,--end-group' -T (Join-Path $firmware 'v5.ld') -T (Join-Path $firmware 'v5-common.ld') '-Wl,--no-warn-rwx-segments,--sort-section=alignment,--sort-common' -o $elf 2>&1 | Tee-Object -FilePath (Join-Path $buildPath 'link.log')
if($LASTEXITCODE -ne 0){throw 'Link failed'}
& (Join-Path $ToolchainBin 'arm-none-eabi-size.exe') $elf
Write-Output "Compiled $($objects.Count-1) translation units and linked audit ELF. No upload was performed."
