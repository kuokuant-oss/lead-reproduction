[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("A2", "A3", "A4", "A5", "B1", "B2", "Aggregate", "All")]
    [string[]]$Stage,

    [switch]$IncludeCalibration,
    [switch]$IncludeA4,
    [switch]$ManifestOnly,

    [int[]]$Seeds = @(42, 123, 999),
    [string[]]$MeterBudgets = @("50", "100", "200", "400", "all")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $RepoRoot "scripts\run_m6_site_transfer.py"
$OracleComparator = Join-Path $RepoRoot "scripts\compare_m6_site_oracle.py"
$Aggregator = Join-Path $RepoRoot "scripts\aggregate_m6_site_transfer.py"
$Processed = Join-Path $RepoRoot "data\processed"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$(Join-Path $RepoRoot 'src');$(Join-Path $RepoRoot 'scripts')"

$RunAll = $Stage -contains "All"
$CompletedCells = [System.Collections.Generic.List[string]]::new()
$OracleOutputs = [System.Collections.Generic.List[string]]::new()

function Test-Stage {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $RunAll -or ($Stage -contains $Name)
}

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    Write-Host ""
    Write-Host ("RUN: {0} {1}" -f $Python, ($Arguments -join " ")) -ForegroundColor Cyan
    & $Python @Arguments | Out-Host
    $PythonExitCode = $LASTEXITCODE
    if ($PythonExitCode -ne 0) {
        throw "Python command failed with exit code $PythonExitCode"
    }
}

function Invoke-M6Cell {
    param(
        [Parameter(Mandatory = $true)][string]$Stem,
        [Parameter(Mandatory = $true)][string[]]$CellArguments
    )
    $Manifest = Join-Path $Processed "${Stem}_manifest.json"
    $Result = Join-Path $Processed "${Stem}.json"
    $Predictions = Join-Path $Processed "${Stem}_predictions.npz"

    Invoke-Python -Arguments @(
        $Runner
        $CellArguments
        "--out", $Manifest
        "--prepare-only"
    )

    if ($ManifestOnly) {
        return
    }

    Invoke-Python -Arguments @(
        $Runner
        $CellArguments
        "--manifest-in", $Manifest
        "--out", $Result
        "--predictions-out", $Predictions
    )
    $script:CompletedCells.Add($Result)
}

function Invoke-A2 {
    Invoke-M6Cell -Stem "m6_site_transfer_a2_odd_to_even" -CellArguments @(
        "--experiment", "a2"
    )
    if ($IncludeCalibration) {
        Invoke-M6Cell -Stem "m6_site_transfer_a2_odd_to_even_sourcecal" -CellArguments @(
            "--experiment", "a2"
            "--source-calibration"
        )
    }
}

function Invoke-A3 {
    foreach ($Fold in 0..3) {
        Invoke-M6Cell -Stem "m6_site_transfer_a3_fold${Fold}" -CellArguments @(
            "--experiment", "a3"
            "--fold", [string]$Fold
        )
        if ($IncludeCalibration) {
            Invoke-M6Cell -Stem "m6_site_transfer_a3_fold${Fold}_sourcecal" -CellArguments @(
                "--experiment", "a3"
                "--fold", [string]$Fold
                "--source-calibration"
            )
        }
    }
}

function Invoke-A4 {
    foreach ($Site in 0..15) {
        Invoke-M6Cell -Stem "m6_site_transfer_a4_site${Site}" -CellArguments @(
            "--experiment", "a4"
            "--site-id", [string]$Site
        )
    }
}

function Invoke-A5 {
    foreach ($Site in 0..15) {
        Invoke-M6Cell -Stem "m6_site_transfer_a5_oracle_site${Site}" -CellArguments @(
            "--experiment", "a5"
            "--site-id", [string]$Site
        )
    }

    if ($ManifestOnly) {
        return
    }

    foreach ($Site in 0..15) {
        $Fold = $Site % 4
        $CrossPredictions = Join-Path $Processed "m6_site_transfer_a3_fold${Fold}_predictions.npz"
        $OraclePredictions = Join-Path $Processed "m6_site_transfer_a5_oracle_site${Site}_predictions.npz"
        $OracleOutput = Join-Path $Processed "m6_paired_oracle_site${Site}.json"
        if ((Test-Path -LiteralPath $CrossPredictions) -and (Test-Path -LiteralPath $OraclePredictions)) {
            Invoke-Python -Arguments @(
                $OracleComparator
                "--cross-predictions", $CrossPredictions
                "--oracle-predictions", $OraclePredictions
                "--out", $OracleOutput
            )
            $script:OracleOutputs.Add($OracleOutput)
        }
        else {
            Write-Warning "Skipping paired oracle site $Site because its A3 or A5 predictions are missing."
        }
    }
}

function Invoke-B1 {
    foreach ($Direction in @("a1", "a2")) {
        foreach ($Seed in $Seeds) {
            foreach ($Budget in $MeterBudgets) {
                $Stem = "m6_site_transfer_b1_${Direction}_meters${Budget}_seed${Seed}"
                Invoke-M6Cell -Stem $Stem -CellArguments @(
                    "--experiment", "b1"
                    "--direction", $Direction
                    "--meter-budget", $Budget
                    "--selection-seed", [string]$Seed
                )
            }
        }
    }
}

function Get-B2CommonPositiveBudget {
    $Available = [System.Collections.Generic.List[int64]]::new()
    foreach ($Split in @("a0", "a1", "a2")) {
        $Stem = "m6_site_transfer_b2_${Split}_pos1_seed42"
        $Manifest = Join-Path $Processed "${Stem}_manifest.json"
        Invoke-Python -Arguments @(
            $Runner
            "--experiment", "b2"
            "--base-split", $Split
            "--positive-budget", "1"
            "--selection-seed", "42"
            "--out", $Manifest
            "--prepare-only"
        )
        $Payload = Get-Content -Raw -Encoding UTF8 -LiteralPath $Manifest | ConvertFrom-Json
        $Available.Add([int64]$Payload.split.b2_available_source_anomalies)
    }
    $Minimum = ($Available | Measure-Object -Minimum).Minimum
    if ($null -eq $Minimum -or $Minimum -le 0) {
        throw "Could not determine a positive B2 common anomaly budget."
    }
    Write-Host "B2 common unique anomaly budget: $Minimum" -ForegroundColor Green
    return [int64]$Minimum
}

function Invoke-B2 {
    $CommonPositiveBudget = Get-B2CommonPositiveBudget
    foreach ($Split in @("a0", "a1", "a2")) {
        foreach ($Seed in $Seeds) {
            $Stem = "m6_site_transfer_b2_${Split}_pos${CommonPositiveBudget}_seed${Seed}"
            Invoke-M6Cell -Stem $Stem -CellArguments @(
                "--experiment", "b2"
                "--base-split", $Split
                "--positive-budget", [string]$CommonPositiveBudget
                "--selection-seed", [string]$Seed
            )
        }
    }
}

function Get-CompletedCellFiles {
    return @(
        Get-ChildItem -LiteralPath $Processed -Filter "m6_site_transfer_*.json" -File |
            Where-Object {
                $_.Name -notlike "*_manifest.json" -and
                $_.Name -ne "m6_site_transfer_plot_data.json"
            } |
            Select-Object -ExpandProperty FullName
    )
}

function Get-OracleFiles {
    return @(
        Get-ChildItem -LiteralPath $Processed -Filter "m6_paired_oracle_site*.json" -File |
            Select-Object -ExpandProperty FullName
    )
}

function Invoke-Aggregation {
    if ($ManifestOnly) {
        Write-Host "ManifestOnly is active; plot-data aggregation is skipped."
        return
    }
    $Cells = Get-CompletedCellFiles
    if ($Cells.Count -eq 0) {
        throw "No completed M6 cell JSON files were found for aggregation."
    }
    $Oracles = Get-OracleFiles
    $Output = Join-Path $Processed "m6_site_transfer_plot_data.json"
    $Arguments = [System.Collections.Generic.List[string]]::new()
    $Arguments.Add($Aggregator)
    $Arguments.Add("--inputs")
    foreach ($Cell in $Cells) {
        $Arguments.Add($Cell)
    }
    if ($Oracles.Count -gt 0) {
        $Arguments.Add("--oracle-inputs")
        foreach ($Oracle in $Oracles) {
            $Arguments.Add($Oracle)
        }
    }
    $Arguments.Add("--out")
    $Arguments.Add($Output)
    Invoke-Python -Arguments $Arguments.ToArray()
}

Push-Location $RepoRoot
try {
    if (Test-Stage "A2") {
        Invoke-A2
    }
    if (Test-Stage "A3") {
        Invoke-A3
    }
    if (Test-Stage "A5") {
        Invoke-A5
    }
    if (Test-Stage "B1") {
        Invoke-B1
    }
    if (Test-Stage "B2") {
        Invoke-B2
    }
    if (($Stage -contains "A4") -or ($RunAll -and $IncludeA4)) {
        Invoke-A4
    }
    if ((Test-Stage "Aggregate") -or $RunAll) {
        Invoke-Aggregation
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Requested M6 stages finished." -ForegroundColor Green
