[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ExpectedHead,
    [string]$AuthorizationToken = "AUTHORIZE_E0_FORMAL_RUN"
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Formal tranche launcher requires $Python" }

$currentHead = (git rev-parse HEAD).Trim()
if ($currentHead -ne $ExpectedHead) {
    throw "Formal tranche launcher refused: HEAD $currentHead does not equal required $ExpectedHead"
}

$OutputRoot = "data/processed/m5_meter_specific_learner_gap/formal"
$CheckpointRoot = "data/processed/m5_meter_specific_learner_gap/formal_checkpoints"
$LogRoot = "data/processed/m5_meter_specific_learner_gap/formal_logs"
New-Item -ItemType Directory -Force -Path $OutputRoot, $CheckpointRoot, $LogRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutLog = Join-Path $LogRoot "tranche1-$stamp.stdout.log"
$stderrLog = Join-Path $LogRoot "tranche1-$stamp.stderr.log"
$null = Start-Transcript -Path $stdoutLog -Force

$activePython = Get-Process -Name python -ErrorAction SilentlyContinue
if ($activePython) {
    throw "Formal tranche launcher refused: a Python process is already active."
}

& $Python scripts/check_long_running_timeout_policy.py 2>> $stderrLog
$policyExitCode = [int]$LASTEXITCODE
if ($policyExitCode -ne 0) {
    throw "Long-running execution-policy scan failed with exit code $policyExitCode; tranche was not launched."
}

$arguments = @(
    "-u", "scripts/analyze_m5_meter_specific_learner_gap.py",
    "--formal",
    "--authorization-token", $AuthorizationToken,
    "--resume",
    "--max-new-draws-per-meter", "42",
    "--output-root", $OutputRoot,
    "--checkpoint-root", $CheckpointRoot,
    "--log-root", $LogRoot
)

Write-Host "[formal-preflight] started $(Get-Date -Format o)"
& $Python @arguments --formal-preflight 2>> $stderrLog
$preflightExitCode = [int]$LASTEXITCODE
if ($preflightExitCode -ne 0) {
    throw "Formal preflight failed with exit code $preflightExitCode; tranche was not launched. See $stdoutLog and $stderrLog"
}

Write-Host "[formal-tranche-1] started $(Get-Date -Format o)"
& $Python @arguments 2>> $stderrLog
$exitCode = [int]$LASTEXITCODE
Write-Host "[formal-tranche-1] exit code: $exitCode"
if ($exitCode -ne 0) { exit $exitCode }
Stop-Transcript | Out-Null
