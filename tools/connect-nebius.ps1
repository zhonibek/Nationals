# tools/connect-nebius.ps1 - Connect Nationals Physical AI to Nebius Token Factory
param(
    [string]$ApiKey = "",
    [string]$BaseUrl = "https://api.tokenfactory.nebius.com/v1",
    [string]$Model = "nvidia/Nemotron-3_5-Lightning"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$cacheDir = Join-Path $repo ".cache\robotai-nemotron"
$keyFile = Join-Path $cacheDir "nebius-api-key.txt"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Nebius Token Factory - NVIDIA Nemotron Integration" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Test-Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
}

if (-not $ApiKey) {
    if (Test-Path $keyFile) {
        $existing = (Get-Content $keyFile -Raw).Trim()
        if ($existing) {
            Write-Host "Existing Nebius API key found in cache." -ForegroundColor Gray
            $ApiKey = $existing
        }
    }
}

if (-not $ApiKey) {
    Write-Host "Please enter your Nebius Token Factory API Key (Bearer token):" -ForegroundColor Cyan
    $ApiKey = Read-Host
    if (-not $ApiKey) {
        Write-Host "ERROR: Nebius API key is required." -ForegroundColor Red
        exit 1
    }
}

$ApiKey = $ApiKey.Trim()
Set-Content -Path $keyFile -Value $ApiKey -NoNewline
Write-Host "Saved API key to: $keyFile" -ForegroundColor Green

# Set environment variables for current session
$env:NEBIUS_API_KEY = $ApiKey
$env:ROBOTAI_NEMOTRON_API_KEY = $ApiKey
$env:ROBOTAI_NEMOTRON_BASE_URL = $BaseUrl
$env:ROBOTAI_NEMOTRON_MODEL = $Model

Write-Host "Base URL: $BaseUrl" -ForegroundColor White
Write-Host "Model:    $Model" -ForegroundColor White

Write-Host "`nTesting live connection to Nebius Token Factory..." -ForegroundColor Yellow

try {
    $headers = @{
        "Authorization" = "Bearer $ApiKey"
        "Content-Type" = "application/json"
    }
    $response = Invoke-RestMethod -Uri "$BaseUrl/models" -Headers $headers -Method Get -TimeoutSec 10
    if ($response.data) {
        Write-Host "SUCCESS! Connected to Nebius Token Factory." -ForegroundColor Green
        Write-Host "Available Models:" -ForegroundColor Cyan
        $nemotronModels = $response.data | Where-Object { $_.id -match 'nemotron' }
        if ($nemotronModels) {
            $nemotronModels | ForEach-Object { Write-Host " - $($_.id)" -ForegroundColor DarkGreen }
        } else {
            $response.data | Select-Object -First 5 | ForEach-Object { Write-Host " - $($_.id)" -ForegroundColor Gray }
        }
    }
} catch {
    Write-Host "WARNING: Could not verify connection to $BaseUrl ($($_.Exception.Message))" -ForegroundColor Yellow
    Write-Host "Key was saved and will be used by RoboProof server." -ForegroundColor Gray
}

Write-Host "`nReady! You can now start or restart RoboProof:" -ForegroundColor Green
Write-Host "  ./start.bat" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
