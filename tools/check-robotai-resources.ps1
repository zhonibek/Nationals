param(
    [switch]$Json,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [int]$CacheMiB = 128
)

function Get-RobotAiNode {
    $result = [ordered]@{ status = 'missing'; version = $null; minimumMajor = 22; versionSource = 'file-metadata'; executableRun = $false }
    try {
        $command = Get-Command -Name node.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1
        if ($command.Source -match '^(\\\\|//)' -or -not [IO.Path]::IsPathRooted($command.Source)) {
            $result.status = 'unavailable'
            return [pscustomobject]$result
        }
        $item = Get-Item -LiteralPath $command.Source -ErrorAction Stop
        $result.status = 'unknown-version'
        $version = [string]$item.VersionInfo.ProductVersion
        if ($version -match '^v?([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,5})(?:\.[0-9]{1,5})?$') {
            $major = [int]$Matches[1]
            $result.version = '{0}.{1}.{2}' -f $major, [int]$Matches[2], [int]$Matches[3]
            if ($major -ge $result.minimumMajor) { $result.status = 'detected' }
            else { $result.status = 'outdated' }
        }
    } catch {
        if ($result.status -ne 'missing') { $result.status = 'unavailable' }
    }
    return [pscustomobject]$result
}

function Get-RobotAiPhysicalMemory {
    $result = [ordered]@{ status = 'unavailable'; source = 'local Win32_OperatingSystem physical memory'; totalUsableBytes = $null; availableBytes = $null }
    try {
        $memory = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        $totalKiB = [decimal]$memory.TotalVisibleMemorySize
        $availableKiB = [decimal]$memory.FreePhysicalMemory
        if ($null -eq $memory.TotalVisibleMemorySize -or $null -eq $memory.FreePhysicalMemory -or
            $totalKiB -le 0 -or $availableKiB -lt 0 -or $availableKiB -gt $totalKiB -or
            $totalKiB -ne [decimal]::Truncate($totalKiB) -or $availableKiB -ne [decimal]::Truncate($availableKiB) -or
            $totalKiB -gt ([decimal][long]::MaxValue / 1024)) {
            return [pscustomobject]$result
        }
        $result.totalUsableBytes = [long]($totalKiB * 1024)
        $result.availableBytes = [long]($availableKiB * 1024)
        $result.status = 'measured'
    } catch {}
    return [pscustomobject]$result
}

function Get-RobotAiArtifact([string]$LiteralPath) {
    $result = [ordered]@{ status = 'missing'; sizeBytes = $null }
    try {
        $item = Get-Item -LiteralPath $LiteralPath -Force -ErrorAction Stop
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            $result.status = 'invalid'
        } else {
            $result.sizeBytes = [long]$item.Length
            if ($result.sizeBytes -gt 0) { $result.status = 'present' }
            else { $result.status = 'empty' }
        }
    } catch {
        if ($_.CategoryInfo.Category -ne 'ObjectNotFound') { $result.status = 'unavailable' }
    }
    return [pscustomobject]$result
}

function Get-RobotAiResourceReport([string]$RepoRoot, [int]$CacheMiB = 128) {
    $issues = New-Object 'System.Collections.Generic.List[object]'
    $node = Get-RobotAiNode
    switch ($node.status) {
        'missing' { $issues.Add([ordered]@{ code = 'node-missing'; severity = 'error'; message = 'Node.js executable was not found on PATH. Node.js 22 or newer is required.' }) }
        'outdated' { $issues.Add([ordered]@{ code = 'node-outdated'; severity = 'error'; message = 'Detected Node.js file version is below the required major version 22.' }) }
        'unknown-version' { $issues.Add([ordered]@{ code = 'node-version-unknown'; severity = 'warning'; message = 'Node.js was found, but its version could not be verified without running it.' }) }
        'unavailable' { $issues.Add([ordered]@{ code = 'node-unavailable'; severity = 'warning'; message = 'Local Node.js metadata could not be inspected.' }) }
    }

    $memory = Get-RobotAiPhysicalMemory
    if ($memory.status -ne 'measured') {
        $issues.Add([ordered]@{ code = 'physical-memory-unavailable'; severity = 'warning'; message = 'Physical RAM information is unavailable; no pagefile or virtual-memory value was substituted.' })
    } elseif ($memory.availableBytes -lt 1GB) {
        $issues.Add([ordered]@{ code = 'low-available-physical-memory'; severity = 'warning'; message = 'Less than 1 GiB of physical RAM is currently available. Avoid concurrent local inference, training and heavy validation.' })
    }

    $artifacts = [ordered]@{
        manifestStatus = 'unavailable'
        model = [pscustomobject]@{ status = 'not-checked'; sizeBytes = $null }
        runtimeArchive = [pscustomobject]@{ status = 'not-checked'; sizeBytes = $null }
        runtimeExecutable = [pscustomobject]@{ status = 'not-checked'; sizeBytes = $null }
        expectedModelBytes = $null
        modelSizeMatchesManifest = $null
        integrityVerified = $false
    }
    try {
        $root = [IO.Path]::GetFullPath($RepoRoot)
        if ($RepoRoot -match '^(\\\\|//)' -or -not [IO.Path]::IsPathRooted($RepoRoot)) { throw 'Local absolute root required' }
        $manifestPath = Join-Path $root 'roboproof/nemotron-runtime.json'
        $manifestFile = Get-RobotAiArtifact $manifestPath
        $artifacts.manifestStatus = $manifestFile.status
        if ($manifestFile.status -eq 'present') {
            $artifacts.manifestStatus = 'invalid'
            if ($manifestFile.sizeBytes -gt 65536) { throw 'Manifest too large' }
            $manifest = Get-Content -LiteralPath $manifestPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            if ($manifest.model.filename -isnot [string] -or $manifest.model.filename -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.gguf$' -or
                $manifest.runtime.filename -isnot [string] -or $manifest.runtime.filename -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,159}\.zip$' -or
                $manifest.runtime.tag -isnot [string] -or $manifest.runtime.tag -cnotmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$' -or
                ($manifest.model.bytes -isnot [int] -and $manifest.model.bytes -isnot [long]) -or $manifest.model.bytes -le 0) {
                throw 'Invalid manifest fields'
            }
            $artifacts.manifestStatus = 'valid'
            $artifacts.expectedModelBytes = [long]$manifest.model.bytes
            $cache = Join-Path $root '.cache/robotai-nemotron'
            $artifacts.model = Get-RobotAiArtifact (Join-Path $cache $manifest.model.filename)
            $artifacts.runtimeArchive = Get-RobotAiArtifact (Join-Path $cache $manifest.runtime.filename)
            $artifacts.runtimeExecutable = Get-RobotAiArtifact (Join-Path (Join-Path $cache $manifest.runtime.tag) 'llama-server.exe')
            if ($artifacts.model.status -eq 'present') {
                $artifacts.modelSizeMatchesManifest = $artifacts.model.sizeBytes -eq $artifacts.expectedModelBytes
                if (-not $artifacts.modelSizeMatchesManifest) {
                    $issues.Add([ordered]@{ code = 'model-size-mismatch'; severity = 'error'; message = 'Model file size differs from the pinned manifest. Presence alone is not a usable-model or integrity check.' })
                }
            }
        }
    } catch {}
    if ($artifacts.manifestStatus -ne 'valid') {
        $issues.Add([ordered]@{ code = 'manifest-unavailable'; severity = 'error'; message = 'The local Nemotron manifest is missing, invalid or unreadable. Artifact paths were not inferred from external sources.' })
    } else {
        foreach ($artifactName in @('model', 'runtimeArchive', 'runtimeExecutable')) {
            if ($artifacts[$artifactName].status -ne 'present') {
                $issues.Add([ordered]@{ code = "$artifactName-unavailable"; severity = 'error'; message = "$artifactName is missing, empty, invalid or unreadable. Review the existing local setup instructions; nothing was installed or started." })
            }
        }
    }

    $estimate = [ordered]@{
        status = 'unavailable'
        weightsSource = $null
        weightsBytes = $null
        configuredPromptCacheMiB = $null
        promptCacheBytes = $null
        assumedRuntimeAndContextBytes = 1GB
        assumedHeadroomBytes = 1GB
        estimatedProcessBytes = $null
        estimatedAvailableBudgetBytes = $null
        availableMinusEstimatedBudgetBytes = $null
        assessment = 'unknown'
        guarantee = $false
    }
    if ($CacheMiB -lt 0 -or $CacheMiB -gt 1024) {
        $issues.Add([ordered]@{ code = 'invalid-cache-budget'; severity = 'error'; message = 'CacheMiB must be between 0 and 1024, matching the local startup script. Unlimited prompt caching is not assumed.' })
    } else {
        $estimate.configuredPromptCacheMiB = $CacheMiB
        $estimate.promptCacheBytes = [long]$CacheMiB * 1MB
        $weights = $artifacts.expectedModelBytes
        $estimate.weightsSource = 'manifest-declared-size'
        if ($null -ne $artifacts.model.sizeBytes -and $artifacts.model.sizeBytes -gt 0) {
            $weights = $artifacts.model.sizeBytes
            $estimate.weightsSource = 'observed-file-size'
        }
        if ($null -ne $weights -and $weights -le ([long]::MaxValue - $estimate.promptCacheBytes - 2GB)) {
            $estimate.status = 'estimate'
            $estimate.weightsBytes = $weights
            $estimate.estimatedProcessBytes = $weights + $estimate.promptCacheBytes + $estimate.assumedRuntimeAndContextBytes
            $estimate.estimatedAvailableBudgetBytes = $estimate.estimatedProcessBytes + $estimate.assumedHeadroomBytes
            if ($memory.status -eq 'measured') {
                $estimate.availableMinusEstimatedBudgetBytes = $memory.availableBytes - $estimate.estimatedAvailableBudgetBytes
                if ($estimate.availableMinusEstimatedBudgetBytes -lt 0) {
                    $estimate.assessment = 'below-estimate'
                    $issues.Add([ordered]@{ code = 'available-memory-below-estimate'; severity = 'warning'; message = 'Available physical RAM is below the estimated model, prompt-cache, runtime/context and headroom budget. Avoid simultaneous inference, training or heavy tests; no process was changed.' })
                } else { $estimate.assessment = 'at-or-above-estimate' }
            }
        } else { $estimate.weightsSource = $null }
    }

    $exitCode = 0
    $status = 'observed'
    if (@($issues | Where-Object { $_.severity -eq 'error' }).Count -gt 0) { $exitCode = 2; $status = 'blocked' }
    elseif ($issues.Count -gt 0) { $exitCode = 1; $status = 'warning' }
    return [pscustomobject][ordered]@{
        schemaVersion = 1
        readOnly = $true
        status = $status
        exitCode = $exitCode
        node = $node
        artifacts = [pscustomobject]$artifacts
        physicalMemory = $memory
        modelMemoryEstimate = [pscustomobject]$estimate
        issues = @($issues.ToArray())
        limitations = @(
            'Metadata only: no processes were launched, stopped or inspected; no network requests, inference, training, downloads, installs or execution-policy changes.'
            'No environment values, command lines, paths, API keys or log contents are included in this report.'
            'Node file metadata does not prove that Node can execute. Artifact presence and size do not verify checksums, runtime compatibility, model loading or inference.'
            'Model memory values are estimates, not hard limits or startup guarantees. Memory mapping, KV/context caches, runtime allocations and other applications can change actual RAM needs.'
            'Prompt-cache MiB matches CacheMiB in start-nemotron.ps1, not total process RAM. Usable and available physical RAM exclude pagefile capacity and may differ from installed RAM.'
            'Exit codes: 0 = observations without identified issues, 1 = warnings or unknown resource checks, 2 = missing or invalid prerequisites. None means inference readiness.'
        )
    }
}

function Write-RobotAiResourceReport($Report, [switch]$Json) {
    if ($Json) {
        $Report | ConvertTo-Json -Depth 8
        return
    }
    Write-Output "RobotAI local resource diagnostic: $($Report.status) (exit $($Report.exitCode))"
    Write-Output 'Read-only; metadata observations are not startup or inference verification.'
    $nodeVersion = 'unknown'
    if ($Report.node.version) { $nodeVersion = $Report.node.version }
    Write-Output "Node.js: $($Report.node.status); file version $nodeVersion; required major >= 22; executable not run."
    Write-Output "Manifest: $($Report.artifacts.manifestStatus); model: $($Report.artifacts.model.status); runtime archive: $($Report.artifacts.runtimeArchive.status); runtime executable: $($Report.artifacts.runtimeExecutable.status)."
    if ($null -ne $Report.artifacts.model.sizeBytes) {
        Write-Output ('Model artifact size: {0} bytes; checksum integrity NOT verified.' -f $Report.artifacts.model.sizeBytes)
    }
    if ($Report.physicalMemory.status -eq 'measured') {
        Write-Output ('Physical RAM: {0:N2} GiB usable; {1:N2} GiB available (not pagefile).' -f ($Report.physicalMemory.totalUsableBytes / 1GB), ($Report.physicalMemory.availableBytes / 1GB))
    } else { Write-Output 'Physical RAM: unavailable; no virtual-memory fallback.' }
    $estimate = $Report.modelMemoryEstimate
    if ($estimate.status -eq 'estimate') {
        Write-Output ('ESTIMATE: weights {0:N2} GiB + prompt cache {1} MiB + assumed runtime/context 1 GiB + headroom 1 GiB = {2:N2} GiB available-memory budget ({3}).' -f ($estimate.weightsBytes / 1GB), $estimate.configuredPromptCacheMiB, ($estimate.estimatedAvailableBudgetBytes / 1GB), $estimate.assessment)
    } else { Write-Output 'Model-memory estimate: unavailable.' }
    foreach ($issue in $Report.issues) { Write-Output "$($issue.severity.ToUpperInvariant()): $($issue.message)" }
    foreach ($limitation in $Report.limitations) { Write-Output $limitation }
}

if ($MyInvocation.InvocationName -ne '.') {
    $report = Get-RobotAiResourceReport -RepoRoot $RepoRoot -CacheMiB $CacheMiB
    Write-RobotAiResourceReport -Report $report -Json:$Json
    exit $report.exitCode
}
