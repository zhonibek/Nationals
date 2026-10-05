param([int]$Port = 8080, [int]$Threads = 4, [int]$CacheMiB = 128)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be 1024..65535' }
if ($Threads -lt 1 -or $Threads -gt 64) { throw 'Threads must be 1..64' }
if ($CacheMiB -lt 0 -or $CacheMiB -gt 1024) { throw 'CacheMiB must be 0..1024; unlimited caching is disabled' }
$root = Split-Path -Parent $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $root 'roboproof/nemotron-runtime.json') -Raw | ConvertFrom-Json
$cache = Join-Path $root '.cache/robotai-nemotron'
$model = Join-Path $cache $manifest.model.filename
$archive = Join-Path $cache $manifest.runtime.filename
$runtime = Join-Path $cache $manifest.runtime.tag
$executable = Join-Path $runtime 'llama-server.exe'
if (-not (Test-Path -LiteralPath $model) -or -not (Test-Path -LiteralPath $executable)) { throw 'Run tools/setup-nemotron.ps1 first' }
if ((Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.model.sha256) { throw 'Model checksum mismatch' }
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.runtime.sha256) { throw 'Runtime archive checksum mismatch' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    foreach ($entry in $zip.Entries) {
        if (-not $entry.Name) { continue }
        $installed = Join-Path $runtime $entry.FullName
        $stream = $entry.Open()
        $hasher = [Security.Cryptography.SHA256]::Create()
        try { $expected = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
        finally { $hasher.Dispose(); $stream.Dispose() }
        if ((Get-FileHash -LiteralPath $installed -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw "Runtime file checksum mismatch: $($entry.FullName)" }
    }
} finally { $zip.Dispose() }
$keyFile = Join-Path $cache 'local-api-key.txt'
if (-not (Test-Path -LiteralPath $keyFile)) {
    $random = [Security.Cryptography.RandomNumberGenerator]::Create()
    $bytes = New-Object byte[] 32
    try { $random.GetBytes($bytes) } finally { $random.Dispose() }
    [IO.File]::WriteAllText($keyFile, [Convert]::ToBase64String($bytes))
}
$arguments = @('-m', "`"$model`"", '--alias', 'robotai-nemotron', '--host', '127.0.0.1', '--port', "$Port", '-c', '4096', '-t', "$Threads", '-tb', "$Threads", '-ngl', '0', '-b', '128', '-ub', '128', '--cache-ram', "$CacheMiB", '--parallel', '1', '--jinja', '--no-webui', '--no-agent', '--cors-origins', 'http://127.0.0.1:8766', '--api-key-file', "`"$keyFile`"")
$process = Start-Process -FilePath $executable -ArgumentList $arguments -WorkingDirectory $runtime -WindowStyle Hidden -RedirectStandardOutput (Join-Path $cache 'server.stdout.log') -RedirectStandardError (Join-Path $cache 'server.stderr.log') -PassThru
Write-Output "Nemotron local CPU server PID $($process.Id): http://127.0.0.1:$Port/v1"
Write-Output "Prompt cache capped at $CacheMiB MiB; this is not a total process-memory limit"
Write-Output 'Model loading can take time. Use Check connection in RoboProof. Logs: .cache/robotai-nemotron/server.stderr.log'
