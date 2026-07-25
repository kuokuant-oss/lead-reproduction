param(
    [Parameter(Mandatory = $true)][int]$Site,
    [Parameter(Mandatory = $true)][ValidateSet("head", "tail")][string]$Shard,
    [Parameter(Mandatory = $true)][int]$NEstimators,
    [Parameter(Mandatory = $true)][string]$Session,
    [Parameter(Mandatory = $true)][string]$ColabHome,
    [Parameter(Mandatory = $true)][int]$ExpectedCheckpointCount,
    [string]$ShardRootName = $null,
    [int]$Microbatch = 20000,
    [int]$MaxAttempts = 60,
    [int]$RetrySeconds = 60,
    [switch]$RelaxTokenScope
)

# Waits for A100 capacity, then brings one shard fully online in the order that
# actually works:
#
#   allocate -> keep-alive -> deploy -> sync + supervisor
#
# Keep-alive must be running *before* the upload, not after: a fresh session can
# be reclaimed during the minutes-long transfer, and a deploy that finishes
# against a reclaimed session leaves an assignment burning compute units with no
# worker on it. Doing this by hand each time is how that ordering got missed, so
# it lives in one script.

$ErrorActionPreference = "Continue"
$repo = "C:\Users\tonykuo\projects\lead-reproduction"
if (-not $ShardRootName) {
    $ShardRootName = "m5_tabpfn_site${Site}_context100000_n${NEstimators}"
}
$shardRoot = Join-Path $repo "data\processed\$ShardRootName"
$manifest = Get-Content -LiteralPath (Join-Path $shardRoot "$Shard\manifest.json") -Raw | ConvertFrom-Json
$remoteRoot = $manifest.remote_root
$results = Join-Path $shardRoot "$Shard-results"
New-Item -ItemType Directory -Path $results -Force | Out-Null

[string[]]$extraEnv = @()
if ($RelaxTokenScope) { $extraEnv = @("OAUTHLIB_RELAX_TOKEN_SCOPE=1") }

Write-Output "[queue] waiting for A100 for site$Site $Shard n=$NEstimators ($remoteRoot)"
$allocated = $false
for ($i = 1; $i -le $MaxAttempts; $i++) {
    & wsl.exe -d Ubuntu -- env "HOME=$ColabHome" @extraEnv /home/tonykuo/.local/bin/colab --auth oauth2 new -s $Session --gpu A100 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Output "[queue] allocated on attempt $i"
        $allocated = $true
        break
    }
    Start-Sleep -Seconds $RetrySeconds
}
if (-not $allocated) {
    Write-Output "[queue] FAILED to allocate A100 for site$Site $Shard after $MaxAttempts attempts"
    exit 1
}

$keepAliveArgs = @(
    "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
    "-File", "$repo\scripts\monitor_m5_tabpfn_colab_keepalive.ps1",
    "-Session", $Session, "-RemoteRoot", $remoteRoot, "-ColabHome", $ColabHome,
    "-LogPath", (Join-Path $results "keepalive.log"),
    # Retire the keep-alive with the shard, so a queue of shards does not leave a
    # growing pile of pollers behind it.
    "-CompletionDirectory", $results,
    "-ExpectedCheckpointCount", $ExpectedCheckpointCount
)
if ($RelaxTokenScope) { $keepAliveArgs += "-RelaxTokenScope" }
Start-Process powershell.exe -ArgumentList $keepAliveArgs -WindowStyle Hidden
Write-Output "[queue] keep-alive attached before upload"
Start-Sleep -Seconds 5

$deployArgs = @(
    "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
    "-File", "$repo\scripts\deploy_m5_tabpfn_site_shard.ps1",
    "-Site", $Site, "-Shard", $Shard, "-NEstimators", $NEstimators,
    "-Session", $Session, "-ColabHome", $ColabHome, "-Microbatch", $Microbatch,
    "-ShardRootName", $ShardRootName
)
if ($RelaxTokenScope) { $deployArgs += "-RelaxTokenScope" }
& powershell.exe @deployArgs 2>&1 | Select-Object -Last 3
Write-Output "[queue] deploy exit=$LASTEXITCODE"

foreach ($script in @("sync_m5_tabpfn_colab_tail.ps1", "supervise_m5_tabpfn_site_shard.ps1")) {
    if ($script -like "sync*") {
        $a = @(
            "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", "$repo\scripts\$script",
            "-Session", $Session, "-LocalDirectory", $results,
            "-RemoteRoot", $remoteRoot, "-ColabHome", $ColabHome,
            "-ExpectedCheckpointCount", $ExpectedCheckpointCount
        )
    }
    else {
        $a = @(
            "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", "$repo\scripts\$script",
            "-Site", $Site, "-Shard", $Shard, "-NEstimators", $NEstimators,
            "-Session", $Session, "-ColabHome", $ColabHome,
            "-ExpectedCheckpointCount", $ExpectedCheckpointCount,
            "-Microbatch", $Microbatch, "-ReapOrphans",
            "-ShardRootName", $ShardRootName
        )
    }
    if ($RelaxTokenScope) { $a += "-RelaxTokenScope" }
    Start-Process powershell.exe -ArgumentList $a -WindowStyle Hidden
}
Write-Output "[queue] site$Site $Shard n=$NEstimators online with sync + supervisor"
