$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$supervisor = Join-Path $repoRoot "scripts\supervise_m5_tabpfn_recovery.py"
$processed = Join-Path $repoRoot "data\processed"
$stdout = Join-Path $processed "m5_tabpfn_colab_head_recovery_supervisor.stdout.log"
$stderr = Join-Path $processed "m5_tabpfn_colab_head_recovery_supervisor.stderr.log"

Set-Location -LiteralPath $repoRoot
$env:TABPFN_COLAB_SHARD = "head"
$env:TABPFN_COLAB_HOME = "/home/tonykuo/.colab-hank"
$env:TABPFN_COLAB_AUTH = "oauth2"
$env:TABPFN_COLAB_ACCELERATOR = "L4"
& $python $supervisor --scope colab 1>> $stdout 2>> $stderr
exit $LASTEXITCODE
