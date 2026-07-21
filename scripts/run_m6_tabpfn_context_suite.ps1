[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(10000, 100000)]
    [int]$ContextRows,

    [int]$QueryChunkSize = 4000,
    [int[]]$Seeds = @(42, 123, 999),
    [string[]]$MeterBudgets = @("50", "100", "200", "400", "all")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $RepoRoot "scripts\run_m6_tabpfn_context_cell.py"
$Processed = Join-Path $RepoRoot "data\processed"
$Logs = Join-Path $RepoRoot "logs\m6-tabpfn-context${ContextRows}"
$PidFile = Join-Path $Logs "active-cell.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$(Join-Path $RepoRoot 'scripts')"

function Test-CompletedResult {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    try {
        $Payload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json
        return $Payload.status -eq "completed"
    }
    catch {
        return $false
    }
}

function Assert-NoLiveDuplicate {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return
    }
    $Active = Get-Content -Raw -Encoding UTF8 -LiteralPath $PidFile | ConvertFrom-Json
    $Process = Get-Process -Id ([int]$Active.pid) -ErrorAction SilentlyContinue
    if ($null -ne $Process) {
        throw (
            "A TabPFN cell is already running (PID {0}, stem {1}). " +
            "No duplicate/retry was started."
        ) -f $Active.pid, $Active.stem
    }
    Remove-Item -LiteralPath $PidFile
}

function Invoke-CellWithoutTimeout {
    param(
        [Parameter(Mandatory = $true)][string]$Stem,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $LogPath = Join-Path $Logs "${Stem}.log"
    $ErrorPath = Join-Path $Logs "${Stem}.stderr.log"
    Write-Host "START (no timeout): $Stem" -ForegroundColor Cyan
    @{
        # The suite itself is launched as one detached hidden process. Its PID
        # remains stable while PowerShell waits on every Python cell without a
        # wall-time or output-silence timeout.
        pid = $PID
        stem = $Stem
        started_utc = [DateTime]::UtcNow.ToString("o")
        timeout_seconds = $null
        retry_on_silence = $false
    } | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $PidFile

    # Deliberately no detached child process or output-silence watchdog. The
    # call operator blocks without a time limit until Python exits on its own.
    & $Python @Arguments 1>> $LogPath 2>> $ErrorPath
    $ExitCode = $LASTEXITCODE
    Remove-Item -LiteralPath $PidFile -ErrorAction SilentlyContinue
    if ($ExitCode -ne 0) {
        throw "TabPFN cell failed with exit code $ExitCode. See $ErrorPath"
    }
    Write-Host "DONE: $Stem" -ForegroundColor Green
}

Push-Location $RepoRoot
try {
    Assert-NoLiveDuplicate
    foreach ($Budget in $MeterBudgets) {
        foreach ($Seed in $Seeds) {
            foreach ($Direction in @("a1", "a2")) {
                $Stem = "m6_tabpfn_b1_${Direction}_meters${Budget}_seed${Seed}_context${ContextRows}"
                $Result = Join-Path $Processed "${Stem}.json"
                if (Test-CompletedResult -Path $Result) {
                    Write-Host "SKIP completed: $Stem" -ForegroundColor DarkGray
                    continue
                }
                $Manifest = Join-Path $Processed "m6_site_transfer_b1_${Direction}_meters${Budget}_seed${Seed}_manifest.json"
                if (-not (Test-Path -LiteralPath $Manifest)) {
                    throw "Prepared B1 manifest is missing: $Manifest"
                }
                Invoke-CellWithoutTimeout -Stem $Stem -Arguments @(
                    $Runner
                    "--direction", $Direction
                    "--meter-budget", $Budget
                    "--selection-seed", [string]$Seed
                    "--context-rows", [string]$ContextRows
                    "--query-chunk-size", [string]$QueryChunkSize
                    "--manifest-in", $Manifest
                    "--out", $Result
                    "--predictions-out", (Join-Path $Processed "${Stem}_predictions.npz")
                )
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host "TabPFN context $ContextRows suite completed without a wall-time timeout." -ForegroundColor Green
