# Start-All.ps1 - 1-Click launcher for Nationals Physical AI Suite
$ErrorActionPreference = 'SilentlyContinue'
$repo = Split-Path $PSScriptRoot -Parent

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Nationals Physical AI Suite -- Master Launcher" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check or start Simulator (Port 8765)
$simPort = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
if (-not $simPort) {
    Write-Host "[1/3] Starting App 1: 3D Simulator on port 8765..." -ForegroundColor Green
    $bundled = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $bundled) { $py = $bundled } else { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
    if (-not $py) { $py = "python" }
    Start-Process -FilePath $py -ArgumentList "-m http.server 8765 --bind 127.0.0.1 --directory simulator" -WorkingDirectory $repo -WindowStyle Hidden
    Start-Sleep -Milliseconds 800
} else {
    Write-Host "[1/3] App 1: 3D Simulator is already running on port 8765" -ForegroundColor DarkGreen
}

# 2. Check or start RoboProof (Port 8766)
$roboPort = Get-NetTCPConnection -LocalPort 8766 -ErrorAction SilentlyContinue
if (-not $roboPort) {
    Write-Host "[2/3] Starting App 2: RoboProof Mission Control on port 8766..." -ForegroundColor Green
    $bundledNode = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    if (Test-Path -LiteralPath $bundledNode) { $node = $bundledNode } else { $node = (Get-Command node -ErrorAction SilentlyContinue).Source }
    if (-not $node) { $node = "node" }
    Start-Process -FilePath $node -ArgumentList "roboproof/server.js" -WorkingDirectory $repo -WindowStyle Hidden
    Start-Sleep -Milliseconds 800
} else {
    Write-Host "[2/3] App 2: RoboProof is already running on port 8766" -ForegroundColor DarkGreen
}

# 3. Open the Unified Master Portal in Browser
$portalUrl = "http://127.0.0.1:8765/portal.html"
Write-Host "[3/3] Opening Master Portal: $portalUrl" -ForegroundColor Cyan
Start-Process $portalUrl

Write-Host ""
Write-Host "SUCCESS! Both App 1 and App 2 are active." -ForegroundColor Green
Write-Host "Portal: $portalUrl" -ForegroundColor White
Write-Host "App 1:  http://127.0.0.1:8765/index.html" -ForegroundColor Gray
Write-Host "App 2:  http://127.0.0.1:8766/" -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor Cyan
