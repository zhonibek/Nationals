'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');

const script = path.resolve(__dirname, '../tools/check-robotai-resources.ps1');
const powershell = process.platform === 'win32'
  ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32/WindowsPowerShell/v1.0/powershell.exe')
  : 'pwsh';
const privateMarker = 'fixture-private-secret-never-print';
const gib = 1024 ** 3;
const quote = value => `'${String(value).replaceAll("'", "''")}'`;

function fixture(context, options = {}) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), `robotai-resources-${privateMarker}-`));
  context.after(() => fs.rmSync(directory, {recursive: true, force: true}));
  const manifest = {
    model: {filename: 'fixture.gguf', bytes: 32},
    runtime: {filename: 'fixture.zip', tag: 'fixture-runtime'},
    privateValue: privateMarker,
  };
  const cache = path.join(directory, '.cache/robotai-nemotron');
  fs.mkdirSync(path.join(directory, 'roboproof'), {recursive: true});
  fs.mkdirSync(path.join(cache, manifest.runtime.tag), {recursive: true});
  const manifestPath = path.join(directory, 'roboproof/nemotron-runtime.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  fs.writeFileSync(path.join(cache, manifest.model.filename), Buffer.alloc(32, 1));
  fs.writeFileSync(path.join(cache, manifest.runtime.filename), Buffer.alloc(8, 2));
  fs.writeFileSync(path.join(cache, manifest.runtime.tag, 'llama-server.exe'), Buffer.alloc(8, 3));
  fs.writeFileSync(path.join(cache, 'local-api-key.txt'), privateMarker);
  fs.writeFileSync(path.join(cache, 'server.stderr.log'), privateMarker);
  return {directory, manifest, manifestPath, cache, ...options};
}

function snapshot(directory) {
  const result = {};
  for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) Object.assign(result, snapshot(filename));
    else {
      const stat = fs.statSync(filename);
      result[filename] = {mtimeMs: stat.mtimeMs, size: stat.size,
        sha256: crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex')};
    }
  }
  return result;
}

function runPowerShell(argumentsValue) {
  const result = spawnSync(powershell, ['-NoLogo', '-NoProfile', '-NonInteractive',
    ...(process.platform === 'win32' ? ['-ExecutionPolicy', 'Bypass'] : []), ...argumentsValue], {
    encoding: 'utf8', timeout: 30000, windowsHide: true,
    env: {...process.env, ROBOTAI_NEMOTRON_API_KEY: privateMarker, ROBOTAI_NEMOTRON_BASE_URL: `http://user:${privateMarker}@127.0.0.1/v1`},
    maxBuffer: 1024 * 1024,
  });
  assert.ifError(result.error);
  assert.equal(result.stderr.trim(), '', result.stderr);
  assert.equal(result.stdout.includes(privateMarker), false);
  return result;
}

function inspect(resources, options = {}) {
  const version = options.version === undefined ? '22.14.0' : options.version;
  const totalKiB = options.totalKiB === undefined ? 12 * gib / 1024 : options.totalKiB;
  const availableKiB = options.availableKiB === undefined ? 8 * gib / 1024 : options.availableKiB;
  const nodeProbe = options.nodeMissing
    ? `throw ${quote(privateMarker)}`
    : "[pscustomobject]@{ Source = 'C:\\fixture-node-metadata.exe' }";
  const memoryProbe = options.memoryUnavailable
    ? `throw ${quote(privateMarker)}`
    : `[pscustomobject]@{ TotalVisibleMemorySize = ${totalKiB}; FreePhysicalMemory = ${availableKiB} }`;
  const source = `
    $ErrorActionPreference = 'Stop'
    . ${quote(script)}
    function Get-Command { param($Name, $CommandType, $ErrorAction) ${nodeProbe} }
    function Get-Item {
      param($LiteralPath, [switch]$Force, $ErrorAction)
      if ($LiteralPath -eq 'C:\\fixture-node-metadata.exe') {
        return [pscustomobject]@{ VersionInfo = [pscustomobject]@{ ProductVersion = ${quote(version)} } }
      }
      Microsoft.PowerShell.Management\\Get-Item @PSBoundParameters
    }
    function Get-CimInstance {
      param($ClassName, $ErrorAction)
      if ($ClassName -ne 'Win32_OperatingSystem') { throw 'Unexpected non-physical memory probe' }
      ${memoryProbe}
    }
    function Get-Content {
      param($LiteralPath, [switch]$Raw, $ErrorAction)
      if ([IO.Path]::GetFullPath($LiteralPath) -ne [IO.Path]::GetFullPath(${quote(resources.manifestPath)})) { throw 'Only the pinned manifest may be read' }
      Microsoft.PowerShell.Management\\Get-Content @PSBoundParameters
    }
    function Start-Process { throw 'Diagnostic must not start processes' }
    function Stop-Process { throw 'Diagnostic must not stop processes' }
    function Get-Process { throw 'Diagnostic must not inspect processes' }
    function Invoke-WebRequest { throw 'Diagnostic must not use the network' }
    function Set-ExecutionPolicy { throw 'Diagnostic must not change execution policy' }
    $report = Get-RobotAiResourceReport -RepoRoot ${quote(options.repoRoot ?? resources.directory)} -CacheMiB ${options.cacheMiB ?? 128}
    Write-RobotAiResourceReport -Report $report ${options.human ? '' : '-Json'}
    exit $report.exitCode
  `;
  const result = runPowerShell(['-Command', source]);
  if (options.human) return result;
  const report = JSON.parse(result.stdout.replace(/^\uFEFF/, ''));
  assert.equal(result.status, report.exitCode);
  return report;
}

test('diagnostic source contains no process, network, secret-content or write operations', () => {
  const source = fs.readFileSync(script, 'utf8');
  assert.doesNotMatch(source, /\b(?:Start-Process|Stop-Process|Get-Process|Invoke-WebRequest|Invoke-RestMethod|Set-ExecutionPolicy|Set-Content|Add-Content|Out-File|New-Item|Remove-Item|Move-Item|Copy-Item|Get-FileHash|Invoke-Expression|Start-Job|Start-ThreadJob)\b/i);
  assert.doesNotMatch(source, /\b(?:CommandLine|Win32_Process|Invoke-CimMethod|DownloadFile|DownloadString|WriteAllText|WriteAllBytes|ProcessStartInfo)\b|\$env:|&/i);
  assert.doesNotMatch(source, /local-api-key\.txt|server\.(?:stdout|stderr)\.log|ROBOTAI_NEMOTRON_API_KEY/);
  assert.match(source, /Get-Command -Name node\.exe -CommandType Application/);
  assert.match(source, /Get-CimInstance -ClassName Win32_OperatingSystem/);
  assert.match(source, /ConvertTo-Json -Depth 8/);
});

test('PowerShell parses the diagnostic and its commands stay within a read-only allowlist', () => {
  const result = runPowerShell(['-Command', `
    $ErrorActionPreference = 'Stop'
    $tokens = $null
    $parseErrors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile(${quote(script)}, [ref]$tokens, [ref]$parseErrors)
    $commands = @($ast.FindAll({ param($entry) $entry -is [System.Management.Automation.Language.CommandAst] }, $true))
    [pscustomobject]@{
      errorCount = @($parseErrors).Count
      names = @($commands | ForEach-Object { $_.GetCommandName() } | Sort-Object -Unique)
      operators = @($commands | ForEach-Object { [string]$_.InvocationOperator } | Sort-Object -Unique)
    } | ConvertTo-Json -Depth 4
  `]);
  assert.equal(result.status, 0);
  const parsed = JSON.parse(result.stdout.replace(/^\uFEFF/, ''));
  assert.equal(parsed.errorCount, 0);
  assert.deepEqual(parsed.operators, ['Unknown']);
  const allowed = new Set(['Split-Path', 'Get-Command', 'Select-Object', 'Get-Item', 'Get-CimInstance',
    'New-Object', 'Join-Path', 'Get-Content', 'ConvertFrom-Json', 'Where-Object', 'ConvertTo-Json',
    'Write-Output', 'Get-RobotAiNode', 'Get-RobotAiPhysicalMemory', 'Get-RobotAiArtifact',
    'Get-RobotAiResourceReport', 'Write-RobotAiResourceReport']);
  for (const name of parsed.names) assert(allowed.has(name), `Unexpected diagnostic command: ${name}`);
});

test('healthy metadata produces explicit observations, physical bytes and non-guaranteed estimates without writes', context => {
  const resources = fixture(context);
  const before = snapshot(resources.directory);
  const report = inspect(resources);
  assert.deepEqual(snapshot(resources.directory), before);
  assert.equal(report.schemaVersion, 1);
  assert.equal(report.readOnly, true);
  assert.equal(report.status, 'observed');
  assert.equal(report.exitCode, 0);
  assert.deepEqual(report.issues, []);
  assert.equal(report.node.version, '22.14.0');
  assert.equal(report.node.executableRun, false);
  assert.equal(report.artifacts.manifestStatus, 'valid');
  for (const name of ['model', 'runtimeArchive', 'runtimeExecutable']) assert.equal(report.artifacts[name].status, 'present');
  assert.equal(report.artifacts.modelSizeMatchesManifest, true);
  assert.equal(report.artifacts.integrityVerified, false);
  assert.equal(report.physicalMemory.totalUsableBytes, 12 * gib);
  assert.equal(report.physicalMemory.availableBytes, 8 * gib);
  assert.equal(report.modelMemoryEstimate.status, 'estimate');
  assert.equal(report.modelMemoryEstimate.weightsSource, 'observed-file-size');
  assert.equal(report.modelMemoryEstimate.estimatedProcessBytes, 32 + 128 * 1024 ** 2 + gib);
  assert.equal(report.modelMemoryEstimate.estimatedAvailableBudgetBytes, 32 + 128 * 1024 ** 2 + 2 * gib);
  assert.equal(report.modelMemoryEstimate.assessment, 'at-or-above-estimate');
  assert.equal(report.modelMemoryEstimate.guarantee, false);
  assert.match(report.limitations.join(' '), /estimates, not hard limits or startup guarantees/);
  assert.equal(Object.hasOwn(report, 'ready'), false);
});

test('low available physical RAM warns despite ample total RAM and never implies a hard process cap', context => {
  const resources = fixture(context);
  const report = inspect(resources, {totalKiB: 6 * gib / 1024, availableKiB: gib / 2048});
  assert.equal(report.status, 'warning');
  assert.equal(report.exitCode, 1);
  assert.equal(report.physicalMemory.availableBytes, gib / 2);
  assert.equal(report.modelMemoryEstimate.assessment, 'below-estimate');
  assert(report.modelMemoryEstimate.availableMinusEstimatedBudgetBytes < 0);
  assert(report.issues.some(issue => issue.code === 'low-available-physical-memory'));
  assert(report.issues.some(issue => issue.code === 'available-memory-below-estimate'));
  const exhausted = inspect(resources, {availableKiB: 0});
  assert.equal(exhausted.physicalMemory.status, 'measured');
  assert.equal(exhausted.physicalMemory.availableBytes, 0);
  assert.equal(exhausted.modelMemoryEstimate.assessment, 'below-estimate');
  assert.equal(exhausted.exitCode, 1);
});

test('unavailable or invalid physical RAM remains unknown rather than becoming zero or pagefile capacity', context => {
  const resources = fixture(context);
  for (const options of [{memoryUnavailable: true}, {totalKiB: 0}, {availableKiB: -1},
    {availableKiB: 20 * gib / 1024}, {availableKiB: 1.5}, {availableKiB: '$null'}, {totalKiB: '9223372036854775807'}]) {
    const report = inspect(resources, options);
    assert.equal(report.physicalMemory.status, 'unavailable');
    assert.equal(report.physicalMemory.availableBytes, null);
    assert.equal(report.physicalMemory.totalUsableBytes, null);
    assert.equal(report.modelMemoryEstimate.assessment, 'unknown');
    assert.equal(report.modelMemoryEstimate.availableMinusEstimatedBudgetBytes, null);
    assert.equal(report.exitCode, 1);
  }
});

test('Node version metadata gates old or missing Node and sanitizes unparseable version strings', context => {
  const resources = fixture(context);
  const outdated = inspect(resources, {version: '20.19.0'});
  assert.equal(outdated.node.status, 'outdated');
  assert.equal(outdated.exitCode, 2);
  const missing = inspect(resources, {nodeMissing: true});
  assert.equal(missing.node.status, 'missing');
  assert.equal(missing.exitCode, 2);
  const unknown = inspect(resources, {version: privateMarker});
  assert.equal(unknown.node.status, 'unknown-version');
  assert.equal(unknown.node.version, null);
  assert.equal(unknown.exitCode, 1);
});

test('missing, empty or wrong-sized model files block and do not claim checksum verification', context => {
  const resources = fixture(context);
  const model = path.join(resources.cache, resources.manifest.model.filename);
  fs.unlinkSync(model);
  const missing = inspect(resources);
  assert.equal(missing.artifacts.model.status, 'missing');
  assert.equal(missing.exitCode, 2);
  assert.equal(missing.modelMemoryEstimate.weightsSource, 'manifest-declared-size');
  fs.writeFileSync(model, '');
  assert.equal(inspect(resources).artifacts.model.status, 'empty');
  fs.writeFileSync(model, 'short fixture');
  const mismatch = inspect(resources);
  assert.equal(mismatch.artifacts.modelSizeMatchesManifest, false);
  assert(mismatch.issues.some(issue => issue.code === 'model-size-mismatch'));
  assert.equal(mismatch.exitCode, 2);
  assert.equal(mismatch.artifacts.integrityVerified, false);
});

test('startup archive and executable are independently required', context => {
  const resources = fixture(context);
  fs.unlinkSync(path.join(resources.cache, resources.manifest.runtime.filename));
  fs.unlinkSync(path.join(resources.cache, resources.manifest.runtime.tag, 'llama-server.exe'));
  const report = inspect(resources);
  assert.equal(report.artifacts.model.status, 'present');
  assert.equal(report.artifacts.runtimeArchive.status, 'missing');
  assert.equal(report.artifacts.runtimeExecutable.status, 'missing');
  assert.equal(report.exitCode, 2);
});

test('invalid, missing and oversized manifests fail closed without reading arbitrary files or leaking errors', context => {
  const resources = fixture(context);
  const invalid = ['{', JSON.stringify({...resources.manifest, model: {filename: '../local-api-key.txt', bytes: 32}}),
    JSON.stringify({...resources.manifest, runtime: {filename: 'fixture.zip', tag: '..'}}),
    JSON.stringify({...resources.manifest, model: {filename: 'fixture.gguf', bytes: '32'}}),
    JSON.stringify({...resources.manifest, model: {filename: 'fixture.gguf', bytes: -1}}),
    `${' '.repeat(65537)}${privateMarker}`];
  for (const content of invalid) {
    fs.writeFileSync(resources.manifestPath, content);
    const report = inspect(resources);
    assert.equal(report.artifacts.manifestStatus, 'invalid');
    assert.equal(report.artifacts.model.status, 'not-checked');
    assert.equal(report.exitCode, 2);
  }
  fs.unlinkSync(resources.manifestPath);
  const missing = inspect(resources);
  assert.equal(missing.artifacts.manifestStatus, 'missing');
  assert.equal(missing.exitCode, 2);
});

test('UNC repository roots are rejected before any manifest or artifact inspection', context => {
  const resources = fixture(context);
  const report = inspect(resources, {repoRoot: `\\\\${privateMarker}\\share`});
  assert.equal(report.artifacts.manifestStatus, 'unavailable');
  assert.equal(report.artifacts.model.status, 'not-checked');
  assert.equal(report.exitCode, 2);
});

test('prompt-cache bounds match startup and never act as a total-memory limit', context => {
  const resources = fixture(context);
  const disabled = inspect(resources, {cacheMiB: 0});
  assert.equal(disabled.modelMemoryEstimate.promptCacheBytes, 0);
  assert.equal(disabled.modelMemoryEstimate.estimatedAvailableBudgetBytes, 32 + 2 * gib);
  const capped = inspect(resources, {cacheMiB: 1024});
  assert.equal(capped.modelMemoryEstimate.promptCacheBytes, gib);
  for (const cacheMiB of [-1, 1025]) {
    const invalid = inspect(resources, {cacheMiB});
    assert.equal(invalid.modelMemoryEstimate.status, 'unavailable');
    assert.equal(invalid.modelMemoryEstimate.promptCacheBytes, null);
    assert.equal(invalid.exitCode, 2);
    assert(invalid.issues.some(issue => issue.code === 'invalid-cache-budget'));
  }
});

test('human output exposes measured units, uncertainty and exit semantics without paths or secrets', context => {
  const output = inspect(fixture(context), {human: true}).stdout;
  assert.match(output, /RobotAI local resource diagnostic: observed \(exit 0\)/);
  assert.match(output, /Physical RAM:.*GiB usable;.*GiB available/);
  assert.match(output, /ESTIMATE:.*prompt cache 128 MiB.*headroom 1 GiB/);
  assert.match(output, /checksum integrity NOT verified/);
  assert.match(output, /None means inference readiness/);
  assert.match(output, /estimates, not hard limits or startup guarantees/);
});

test('the actual command-line entry produces JSON only and uses its documented exit code without modifying files', context => {
  const resources = fixture(context);
  const before = snapshot(resources.directory);
  const result = runPowerShell(['-File', script, '-RepoRoot', resources.directory, '-CacheMiB', '0', '-Json']);
  const report = JSON.parse(result.stdout.replace(/^\uFEFF/, ''));
  assert.equal(result.status, report.exitCode);
  assert.equal(report.schemaVersion, 1);
  assert.equal(report.readOnly, true);
  assert.equal(report.artifacts.manifestStatus, 'valid');
  assert.equal(report.artifacts.modelSizeMatchesManifest, true);
  assert.equal(report.modelMemoryEstimate.configuredPromptCacheMiB, 0);
  assert.equal(report.modelMemoryEstimate.guarantee, false);
  assert.deepEqual(snapshot(resources.directory), before);
});
