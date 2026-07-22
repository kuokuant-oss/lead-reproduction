param(
    [int]$TargetCheckpointCount = 253,
    [int]$PollSeconds = 10,
    [string]$WorkDir = "C:\Users\tonykuo\projects\lead-reproduction\data\processed\m5_tabpfn_canonical_full_test_context100000.work",
    [string]$LogPath = "C:\Users\tonykuo\projects\lead-reproduction\data\processed\m5_tabpfn_local_head_monitor.log"
)

$ErrorActionPreference = "Stop"
$resolvedWorkDir = [System.IO.Path]::GetFullPath($WorkDir)
$allowedRoot = [System.IO.Path]::GetFullPath("C:\Users\tonykuo\projects\lead-reproduction\data\processed")
if (-not $resolvedWorkDir.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "WorkDir must remain under $allowedRoot"
}
if ($TargetCheckpointCount -lt 1) {
    throw "TargetCheckpointCount must be positive"
}

$chunksDir = Join-Path $resolvedWorkDir "chunks"
$stopPath = Join-Path $resolvedWorkDir "stop.json"
$temporaryStopPath = "$stopPath.tmp"

while ($true) {
    $count = @(Get-ChildItem -LiteralPath $chunksDir -Filter "chunk_*.npz" -ErrorAction SilentlyContinue).Count
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    Add-Content -LiteralPath $LogPath -Value "$timestamp checkpoints=$count target=$TargetCheckpointCount"
    if ($count -ge $TargetCheckpointCount) {
        $payload = @{
            reason = "local head boundary reached"
            timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
            checkpoint_count = $count
            target_checkpoint_count = $TargetCheckpointCount
        } | ConvertTo-Json
        Set-Content -LiteralPath $temporaryStopPath -Value $payload -Encoding UTF8
        Move-Item -LiteralPath $temporaryStopPath -Destination $stopPath -Force
        Add-Content -LiteralPath $LogPath -Value "$timestamp stop_requested=true"
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}
