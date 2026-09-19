param([int]$Port=8765)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$bundled=Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if(Test-Path -LiteralPath $bundled){$python=$bundled}else{$python=(Get-Command python -ErrorAction Stop).Source}
Write-Host "Nationals-work3 simulator: http://127.0.0.1:$Port"
& $python -m http.server $Port --bind 127.0.0.1 --directory (Join-Path $repo 'simulator')
