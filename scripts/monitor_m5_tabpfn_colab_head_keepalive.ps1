$repoRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "monitor_m5_tabpfn_colab_keepalive.ps1") `
    -Session "lead-tabpfn-tail-2" `
    -LogPath (Join-Path $repoRoot "data\processed\m5_tabpfn_colab_head_keepalive.log")
exit $LASTEXITCODE
