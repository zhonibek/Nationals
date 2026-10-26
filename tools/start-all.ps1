# start-all.ps1 - Windows one-click launcher (used by start.bat).
# Runs the cross-platform tools/start.js: one RoboProof server serves the simulator, the portal and RoboProof.
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$node = (Get-Command node -ErrorAction SilentlyContinue).Source
if (-not $node) {
    $bundled = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    if (Test-Path -LiteralPath $bundled) { $node = $bundled }
}
if (-not $node) {
    Write-Host 'Node.js 20.3 or newer is required: https://nodejs.org' -ForegroundColor Red
    exit 1
}
& $node (Join-Path $repo 'tools/start.js') @args
