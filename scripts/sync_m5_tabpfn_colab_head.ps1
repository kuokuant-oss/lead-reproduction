$repoRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "sync_m5_tabpfn_colab_tail.ps1") `
    -Session "lead-tabpfn-tail-2" `
    -ExpectedCheckpointCount 253 `
    -LocalDirectory (Join-Path $repoRoot "data\processed\m5_tabpfn_distributed_context100000\head-results") `
    -RemoteRoot "/content/lead_tabpfn_head"
exit $LASTEXITCODE
