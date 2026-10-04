$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $root 'roboproof/nemotron-runtime.json') -Raw | ConvertFrom-Json
$cache = Join-Path $root '.cache/robotai-nemotron'
New-Item -ItemType Directory -Path $cache -Force | Out-Null

function Get-VerifiedFile([string]$Uri, [string]$Destination, [string]$Sha256) {
    if (Test-Path -LiteralPath $Destination) {
        if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {
            throw "Existing file has a checksum mismatch: $Destination"
        }
        Write-Output "Already verified: $Destination"
        return
    }
    $partial = "$Destination.part"
    Write-Output "Downloading official artifact: $Uri"
    & curl.exe --fail --location --silent --show-error --retry 2 --retry-delay 2 --max-time 3600 --continue-at - --output $partial $Uri
    if ($LASTEXITCODE -ne 0) { throw "Download failed; partial file retained for resume: $partial" }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {
        throw "Downloaded file has a checksum mismatch; it will not be executed: $partial"
    }
    Move-Item -LiteralPath $partial -Destination $Destination
    Write-Output "SHA-256 verified: $Destination"
}

$archive = Join-Path $cache $manifest.runtime.filename
Get-VerifiedFile $manifest.runtime.url $archive $manifest.runtime.sha256
$runtime = Join-Path $cache $manifest.runtime.tag
if (-not (Test-Path -LiteralPath (Join-Path $runtime 'llama-server.exe'))) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archive)
    try {
        $prefix = [IO.Path]::GetFullPath($runtime) + [IO.Path]::DirectorySeparatorChar
        foreach ($entry in $zip.Entries) {
            $target = [IO.Path]::GetFullPath((Join-Path $runtime $entry.FullName))
            if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe archive entry' }
        }
    } finally { $zip.Dispose() }
    Expand-Archive -LiteralPath $archive -DestinationPath $runtime
}
$model = Join-Path $cache $manifest.model.filename
$modelUri = "https://huggingface.co/$($manifest.model.repository)/resolve/$($manifest.model.revision)/$($manifest.model.filename)"
Get-VerifiedFile $modelUri $model $manifest.model.sha256
if ((Get-Item -LiteralPath $model).Length -ne $manifest.model.bytes) { throw 'Unexpected model size' }
Write-Output 'Portable local Nemotron is ready. Run tools/start-nemotron.ps1, then node roboproof/server.js.'
