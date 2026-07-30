[CmdletBinding()]
param(
    [string]$Python = ".venv\\Scripts\\python.exe",
    [string]$ValidationRoot = "data\\processed\\m5_e0_validation",
    [int]$BootstrapDraws = 1,
    [int]$LooBuildings = 3,
    [int]$SegmentDraws = 1,
    [switch]$EvidenceSuite,
    [int]$ValidationStopAfterUnits = 12
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-JsonAtomically([string]$Path, [object]$Value) {
    $temporary = "$Path.$PID.$([guid]::NewGuid().ToString('N')).tmp"
    try {
        $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $temporary -Encoding utf8
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}

function Get-ValidatedHeartbeatRecords([string]$RunRoot) {
    $heartbeatFiles = @(Get-ChildItem -LiteralPath (Join-Path $RunRoot "checkpoints") -Recurse -Filter "heartbeat.json" -ErrorAction SilentlyContinue)
    if ($heartbeatFiles.Count -eq 0) { throw "Heartbeat snapshot refused: no source heartbeat exists." }
    $records = @()
    foreach ($heartbeatFile in $heartbeatFiles) {
        try { $heartbeat = Get-Content -Raw -LiteralPath $heartbeatFile.FullName | ConvertFrom-Json }
        catch { throw "Heartbeat snapshot refused: invalid JSON at $($heartbeatFile.FullName)." }
        $phaseRoot = Split-Path -Parent $heartbeatFile.FullName
        $provenancePath = Join-Path $phaseRoot "provenance.json"
        if (-not (Test-Path -LiteralPath $provenancePath)) { throw "Heartbeat snapshot refused: provenance missing for $($heartbeatFile.FullName)." }
        try { $provenance = Get-Content -Raw -LiteralPath $provenancePath | ConvertFrom-Json }
        catch { throw "Heartbeat snapshot refused: invalid provenance at $provenancePath." }
        if (-not $heartbeat.phase -or -not $heartbeat.provenance_digest -or $heartbeat.provenance_digest -ne $provenance.digest -or $heartbeat.phase -ne $provenance.provenance.phase) {
            throw "Heartbeat snapshot refused: provenance mismatch at $($heartbeatFile.FullName)."
        }
        foreach ($field in "total", "completed", "computed", "reused", "pending", "status") {
            if ($null -eq $heartbeat.$field) { throw "Heartbeat snapshot refused: $field missing at $($heartbeatFile.FullName)." }
        }
        if (($heartbeat.completed + $heartbeat.pending) -ne $heartbeat.total) { throw "Heartbeat snapshot refused: invalid counters at $($heartbeatFile.FullName)." }
        $records += [ordered]@{
            source_heartbeat_path = $heartbeatFile.FullName
            source_heartbeat_sha256 = (Get-FileHash -LiteralPath $heartbeatFile.FullName -Algorithm SHA256).Hash
            source_provenance_path = $provenancePath
            source_provenance_sha256 = (Get-FileHash -LiteralPath $provenancePath -Algorithm SHA256).Hash
            heartbeat = $heartbeat
        }
    }
    return $records
}

function Assert-HeartbeatSnapshotAcceptance([object[]]$Records, [string]$Name, [int]$ExitCode, [object]$RuntimeSummary) {
    $heartbeatComputed = @($Records | ForEach-Object { [int]$_.heartbeat.computed } | Measure-Object -Sum).Sum
    $heartbeatReused = @($Records | ForEach-Object { [int]$_.heartbeat.reused } | Measure-Object -Sum).Sum
    if ($heartbeatComputed -ne $RuntimeSummary.computed_units -or $heartbeatReused -ne $RuntimeSummary.reused_units) {
        throw "[$Name] heartbeat counters do not match the runtime summary."
    }
    if ($Name -eq "run1-interruption") {
        if ($ExitCode -ne 75) { throw "Run 1 heartbeat snapshot requires exit 75." }
        $partial = @($Records | Where-Object { $_.heartbeat.status -ne "completed" })
        if ($partial.Count -eq 0) { throw "Run 1 heartbeat snapshot falsely reports every phase completed." }
        foreach ($record in $partial) {
            $heartbeat = $record.heartbeat
            if ($heartbeat.completed -le 0 -or $heartbeat.pending -le 0 -or -not (Test-Path -LiteralPath $heartbeat.last_completed_checkpoint)) {
                throw "Run 1 heartbeat snapshot lacks valid partial-progress evidence."
            }
            if (Test-Path -LiteralPath (Join-Path (Split-Path -Parent $record.source_heartbeat_path) "COMPLETE.json")) {
                throw "Run 1 heartbeat snapshot has an early completion marker for its interrupted phase."
            }
        }
        return
    }
    foreach ($record in $Records) {
        $heartbeat = $record.heartbeat
        if ($heartbeat.status -ne "completed" -or $heartbeat.completed -ne $heartbeat.total -or $heartbeat.pending -ne 0 -or $null -ne $heartbeat.current_unit -or -not (Test-Path -LiteralPath $heartbeat.phase_completion_marker)) {
            throw "[$Name] heartbeat snapshot does not prove completed phase state."
        }
    }
    if ($Name -eq "run2-resume" -and ($RuntimeSummary.computed_units -le 0 -or $RuntimeSummary.reused_units -le 0)) {
        throw "Run 2 heartbeat snapshot does not prove partial reuse and missing-unit computation."
    }
    if ($Name -eq "run3-reuse" -and ($RuntimeSummary.computed_units -ne 0 -or $RuntimeSummary.reused_units -le 0)) {
        throw "Run 3 heartbeat snapshot does not prove full reuse."
    }
}

function Save-HeartbeatSnapshot([string]$RunRoot, [string]$Name, [int]$ExitCode) {
    $runId = ($Name -split "-")[0]
    $snapshotPath = Join-Path $RunRoot "$runId.heartbeat.json"
    if (Test-Path -LiteralPath $snapshotPath) { throw "Heartbeat snapshot already exists: $snapshotPath" }
    $records = @(Get-ValidatedHeartbeatRecords $RunRoot)
    $runtimeSummaryPath = Join-Path $RunRoot "runtime_summary.json"
    if (-not (Test-Path -LiteralPath $runtimeSummaryPath)) { throw "Heartbeat snapshot refused: runtime summary is missing." }
    try { $runtimeSummary = Get-Content -Raw -LiteralPath $runtimeSummaryPath | ConvertFrom-Json }
    catch { throw "Heartbeat snapshot refused: runtime summary is invalid." }
    Assert-HeartbeatSnapshotAcceptance $records $Name $ExitCode $runtimeSummary
    Write-JsonAtomically $snapshotPath ([ordered]@{
        run_id = $runId; run_name = $Name; exit_code = $ExitCode
        snapshot_timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
        runtime_summary_path = $runtimeSummaryPath
        runtime_summary_sha256 = (Get-FileHash -LiteralPath $runtimeSummaryPath -Algorithm SHA256).Hash
        computed_units = $runtimeSummary.computed_units; reused_units = $runtimeSummary.reused_units
        heartbeats = $records
    })
    return $snapshotPath
}

function Write-CheckpointCensus([string]$RunRoot, [string]$Name, [int]$ExitCode, [datetime]$Started, [datetime]$Ended) {
    $checkpointRoot = Join-Path $RunRoot "checkpoints"
    $units = @(Get-ChildItem -LiteralPath $checkpointRoot -Recurse -Filter "*.json" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match '\\units\\' })
    $markers = @(Get-ChildItem -LiteralPath $checkpointRoot -Recurse -Filter "COMPLETE.json" -ErrorAction SilentlyContinue)
    $temporary = @(Get-ChildItem -LiteralPath $checkpointRoot -Recurse -Filter "*.tmp" -ErrorAction SilentlyContinue)
    Write-JsonAtomically (Join-Path $RunRoot "$Name.checkpoint_census.json") ([ordered]@{
        run = $Name; exit_code = $ExitCode; started_utc = $Started.ToUniversalTime().ToString("o"); ended_utc = $Ended.ToUniversalTime().ToString("o")
        unit_count = $units.Count; phase_completion_marker_count = $markers.Count; temporary_file_count = $temporary.Count
        units = @($units | ForEach-Object { [ordered]@{ path = $_.FullName; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash } })
    })
}

function Get-CheckpointHashes([string]$RunRoot) {
    @(Get-ChildItem -LiteralPath (Join-Path $RunRoot "checkpoints") -Recurse -Filter "*.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\units\\' } |
        Sort-Object FullName |
        ForEach-Object { "$($_.FullName.Substring($RunRoot.Length))=$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash)" })
}

function Get-CheckpointModificationTimes([string]$RunRoot) {
    @(Get-ChildItem -LiteralPath (Join-Path $RunRoot "checkpoints") -Recurse -Filter "*.json" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\units\\' } |
        Sort-Object FullName |
        ForEach-Object { "$($_.FullName.Substring($RunRoot.Length))=$($_.LastWriteTimeUtc.Ticks)" })
}

function Invoke-ValidationRun([string]$RunRoot, [string]$Name, [int]$ExpectedExitCode, [switch]$StopAfterUnits) {
    $stdout = Join-Path $RunRoot "$Name.stdout.log"
    $stderr = Join-Path $RunRoot "$Name.stderr.log"
    $arguments = @(
        "-u", "scripts/analyze_m5_meter_specific_learner_gap.py",
        "--validation-mode", "--phase", "all",
        "--bootstrap-draws", "$BootstrapDraws",
        "--loo-buildings", "$LooBuildings",
        "--segment-draws", "$SegmentDraws",
        "--checkpoint-root", "$RunRoot",
        "--output-root", "$RunRoot"
    )
    if ($StopAfterUnits) { $arguments += @("--validation-stop-after-units", "$ValidationStopAfterUnits") }
    $started = Get-Date
    Write-Host "[$Name] foreground run started: $($started.ToString('o'))"
    # A 75 exit is expected validation control flow. Keep native stderr from
    # becoming a terminating PowerShell error before we can inspect that code.
    $priorErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @arguments 2> $stderr | Tee-Object -FilePath $stdout
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    $ended = Get-Date
    Write-CheckpointCensus $RunRoot $Name $exitCode $started $ended
    $runtimeSummary = Join-Path $RunRoot "runtime_summary.json"
    if (Test-Path $runtimeSummary) { Copy-Item -LiteralPath $runtimeSummary -Destination (Join-Path $RunRoot "$Name.runtime_summary.json") -Force }
    if ($exitCode -ne $ExpectedExitCode) { throw "[$Name] expected exit $ExpectedExitCode but got $exitCode." }
    if ((Get-Item -LiteralPath $stderr).Length -ne 0) { throw "[$Name] stderr log is not empty." }
    if ($StopAfterUnits -and -not (Test-Path (Join-Path $RunRoot "EXPECTED_VALIDATION_INTERRUPTION.json"))) {
        throw "[$Name] expected interruption marker is missing."
    }
    $heartbeatSnapshot = Save-HeartbeatSnapshot $RunRoot $Name $exitCode
    Write-JsonAtomically (Join-Path $RunRoot "$Name.exit.json") ([ordered]@{
        run = $Name; exit_code = $exitCode; expected_exit_code = $ExpectedExitCode
        started_utc = $started.ToUniversalTime().ToString("o"); ended_utc = $ended.ToUniversalTime().ToString("o")
        stdout = $stdout; stderr = $stderr; heartbeat_root = (Join-Path $RunRoot "checkpoints")
        heartbeat_snapshot = $heartbeatSnapshot; runtime_summary = (Join-Path $RunRoot "$Name.runtime_summary.json")
    })
    Write-Host "[$Name] exit code: $exitCode"
}

if ($BootstrapDraws -lt 1 -or $LooBuildings -lt 1 -or $SegmentDraws -lt 1 -or $ValidationStopAfterUnits -lt 1) {
    throw "Every expensive validation phase requires an explicit positive work-unit limit."
}
& $Python scripts/check_long_running_timeout_policy.py
if ($LASTEXITCODE -ne 0) { throw "Repository timeout-policy compliance failed." }

New-Item -ItemType Directory -Force -Path $ValidationRoot | Out-Null
if (-not $EvidenceSuite) {
    Invoke-ValidationRun $ValidationRoot "validation-$(Get-Date -Format 'yyyyMMdd-HHmmss')" 0
    exit 0
}

$SuiteRoot = Join-Path $ValidationRoot "evidence-suite-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
New-Item -ItemType Directory -Force -Path $SuiteRoot | Out-Null
Write-Host "EvidenceSuite root: $SuiteRoot"
Write-Host "Foreground only: no timeout, job, detached process, or auto-kill is configured."

Invoke-ValidationRun $SuiteRoot "run1-interruption" 75 -StopAfterUnits
$beforeReuse = Get-CheckpointHashes $SuiteRoot
Invoke-ValidationRun $SuiteRoot "run2-resume" 0
$beforeRun3 = Get-CheckpointHashes $SuiteRoot
$beforeRun3ModificationTimes = Get-CheckpointModificationTimes $SuiteRoot
$run1HashesMissingAfterResume = @($beforeReuse | Where-Object { $_ -notin $beforeRun3 })
if ($run1HashesMissingAfterResume.Count -ne 0) {
    throw "Run 2 changed or removed completed Run 1 checkpoint SHA256 values."
}
$run2Summary = Get-Content -Raw (Join-Path $SuiteRoot "run2-resume.runtime_summary.json") | ConvertFrom-Json
$run2ComputedExpected = $beforeRun3.Count - $beforeReuse.Count
if ($run2Summary.computed_units -ne $run2ComputedExpected -or $run2Summary.reused_units -ne $beforeReuse.Count) {
    throw "Run 2 did not compute only missing units and reuse every completed Run 1 unit."
}
Invoke-ValidationRun $SuiteRoot "run3-reuse" 0
$afterRun3 = Get-CheckpointHashes $SuiteRoot
$afterRun3ModificationTimes = Get-CheckpointModificationTimes $SuiteRoot
if (Compare-Object $beforeRun3 $afterRun3) { throw "Run 3 changed checkpoint SHA256 values; reuse proof failed." }
if (Compare-Object $beforeRun3ModificationTimes $afterRun3ModificationTimes) { throw "Run 3 changed checkpoint modification times; reuse proof failed." }

$summary = Get-Content -Raw (Join-Path $SuiteRoot "runtime_summary.json") | ConvertFrom-Json
if ($summary.computed_units -ne 0 -or $summary.reused_units -ne $afterRun3.Count) { throw "Run 3 did not reuse every checkpoint." }
Write-JsonAtomically (Join-Path $SuiteRoot "evidence_suite_summary.json") ([ordered]@{
    execution_mode = "NON_SCIENTIFIC_VALIDATION"; root = $SuiteRoot; run1_expected_exit = 75; run2_exit = 0; run3_exit = 0
    run1_checkpoint_hash_count = $beforeReuse.Count; run3_checkpoint_hash_count = $afterRun3.Count
    run1_checkpoint_hashes_preserved_after_run2 = $true; run2_computed_units = $run2Summary.computed_units; run2_reused_units = $run2Summary.reused_units
    run3_checkpoint_modification_times_unchanged = $true
    all_stderr_logs_empty = $true
    run3_computed_units = $summary.computed_units; run3_reused_units = $summary.reused_units
    heartbeat = (Join-Path $SuiteRoot "checkpoints"); runtime_summary = (Join-Path $SuiteRoot "runtime_summary.json")
})
Write-Host "EvidenceSuite complete: $SuiteRoot"
