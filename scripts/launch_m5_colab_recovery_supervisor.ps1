$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$supervisor = Join-Path $repoRoot "scripts\supervise_m5_tabpfn_recovery.py"
$processed = Join-Path $repoRoot "data\processed"
$stdout = Join-Path $processed "m5_tabpfn_recovery_supervisor.stdout.log"
$stderr = Join-Path $processed "m5_tabpfn_recovery_supervisor.stderr.log"

Set-Location -LiteralPath $repoRoot
& $python $supervisor --scope colab 1>> $stdout 2>> $stderr
exit $LASTEXITCODE
