# Serial source collection only. Never retries a failed batch or scores labels.
# Omit -AllowNetwork for a dry run. Per-request and project costs remain guarded
# by run_operator_study; this launcher does not have its own API client.
param(
    [ValidateRange(0, 499)][int]$Start = 25,
    [ValidateRange(1, 500)][int]$End = 500,
    [ValidateRange(1, 25)][int]$BatchSize = 25,
    [ValidatePattern('^[0-9a-f]{64}$')]
    [string]$ExpectedMethod = 'e9122473d934e50a6800d909e6e90fe0db97dfd15f49b51d9a549655f7089836',
    [switch]$AllowNetwork
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($End -le $Start) { throw 'End must be greater than Start.' }
$taskWorkspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskPython = Join-Path $taskWorkspace '.venv\Scripts\python.exe'
$taskOldPath = $env:PYTHONPATH
$taskOldUtf8 = $env:PYTHONUTF8
Push-Location -LiteralPath $taskWorkspace
try {
    $env:PYTHONPATH = Join-Path $taskWorkspace 'src'
    $env:PYTHONUTF8 = '1'
    for ($taskOffset = $Start; $taskOffset -lt $End; $taskOffset += $BatchSize) {
        $taskCount = [Math]::Min($BatchSize, $End - $taskOffset)
        $taskArguments = @(
            '-X', 'utf8', '-m', 'growrag.experiments.run_operator_study',
            '--manifest', 'data/hotpotqa/operator_scale_sep30_v1/manifest.json',
            '--expected-manifest-sha256', 'd8596fd830c3cfe04255f18678a0a0454662aefa76b75ff8556f3cccb57c9225',
            '--expected-execution-sha256', $ExpectedMethod,
            '--phase', 'source', '--start', [string]$taskOffset,
            '--count', [string]$taskCount, '--arms', 'base', 'fresh', 'static',
            '--budget-cny', '2', '--api-config', 'qwenAPI.md'
        )
        if ($AllowNetwork) { $taskArguments += '--allow-network' }
        Write-Host "Source interval [$taskOffset, $($taskOffset + $taskCount)); network=$AllowNetwork"
        & $taskPython @taskArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Stopped at source offset $taskOffset. Inspect frozen reports; no automatic retry."
        }
    }
} finally {
    $env:PYTHONPATH = $taskOldPath
    $env:PYTHONUTF8 = $taskOldUtf8
    Pop-Location
}
