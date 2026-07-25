param(
    [Parameter(Mandatory = $true)][int]$Site,
    [Parameter(Mandatory = $true)][ValidateSet("head", "tail")][string]$Shard,
    [Parameter(Mandatory = $true)][int]$NEstimators,
    [Parameter(Mandatory = $true)][string]$Session,
    [Parameter(Mandatory = $true)][string]$ColabHome,
    [Parameter(Mandatory = $true)][int]$ExpectedCheckpointCount,
    # The 137-feature line stores its shards under a different root name.
    [string]$ShardRootName = $null,
    [int]$Microbatch = 20000,
    [int]$PollSeconds = 90,
    [int]$StallCycles = 8,
    [string]$ColabCli = "/home/tonykuo/.local/bin/colab",
    [string]$ColabPython = "/home/tonykuo/.local/share/uv/tools/google-colab-cli/bin/python",
    [ValidateSet("adc", "oauth2")][string]$Auth = "oauth2",
    [switch]$RelaxTokenScope,
    # Release unnamed assignments too. They are only produced when the CLI drops
    # a lost session's local record while the server-side assignment survives,
    # and both accounts here are dedicated to this run, so nothing else can own
    # them. Off by default so it can never reap a bystander's runtime.
    [switch]$ReapOrphans,
    # Consecutive polls an assignment must stay unnamed before it is treated as
    # a true orphan rather than a session still registering.
    [int]$OrphanConfirmPolls = 3
)

# Recovery supervisor for one estimator-sweep shard (runbook section 7.1).
#
# The sync monitor and keep-alive only observe; neither rebuilds a shard whose
# Colab session Google has reclaimed. Without a supervisor a lost session sits
# there holding an assignment that still burns compute units while doing no
# work, and the run silently stops advancing. This process closes that gap: it
# detects a lost or stalled session, recreates it under the same name, and
# redeploys -- which re-uploads every durable local checkpoint so the worker
# resumes at the frontier rather than restarting the shard.
#
# Health is judged by durable local checkpoints advancing, never by heartbeat or
# session existence alone.

$ErrorActionPreference = "Continue"
$repo = "C:\Users\tonykuo\projects\lead-reproduction"
if (-not $ShardRootName) {
    $ShardRootName = "m5_tabpfn_site${Site}_context100000_n${NEstimators}"
}
$shardRoot = Join-Path $repo "data\processed\$ShardRootName"
$results = Join-Path $shardRoot "$Shard-results"
$chunkDir = Join-Path $results "chunks"
$logPath = Join-Path $results "supervisor.log"
New-Item -ItemType Directory -Path $chunkDir -Force | Out-Null

[string[]]$extraEnv = @()
if ($RelaxTokenScope) { $extraEnv = @("OAUTHLIB_RELAX_TOKEN_SCOPE=1") }

function Write-Log([string]$Message) {
    $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    Add-Content -LiteralPath $logPath -Value "$stamp $Message"
}

function Invoke-ColabRaw([string[]]$Arguments) {
    return @(& wsl.exe -d Ubuntu -- env "HOME=$ColabHome" @extraEnv $ColabCli --auth $Auth @Arguments 2>&1)
}

function Get-LocalCheckpointCount {
    return @(Get-ChildItem -LiteralPath $chunkDir -Filter "rows_*.npz" -File -ErrorAction SilentlyContinue).Count
}

function Release-Endpoint([string]$Endpoint) {
    # `colab stop` addresses a session by name, so it is useless once the CLI has
    # dropped the local name->token record -- which is exactly what it does when
    # a session returns 404/401. unassign() addresses the assignment by endpoint
    # ID, which stays visible in `sessions` even for unnamed orphans, so this is
    # the only reliable way to stop an assignment from burning compute units.
    if ($Endpoint -notmatch '^gpu-[a-z0-9-]+$') {
        Write-Log "endpoint_release_refused unsafe_endpoint=$Endpoint"
        return $false
    }
    $provider = if ($Auth -eq "adc") { "AuthProvider.ADC" } else { "AuthProvider.OAUTH2" }
    $code = "from colab_cli.auth import AuthProvider; from colab_cli.common import state; state.auth_provider=$provider; state.client.unassign('$Endpoint')"
    & wsl.exe -d Ubuntu -- env "HOME=$ColabHome" @extraEnv $ColabPython -c $code 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Log "released_exact_endpoint=$Endpoint"
        return $true
    }
    Write-Log "endpoint_release_failed=true endpoint=$Endpoint"
    return $false
}

function Get-SessionListing {
    return (Invoke-ColabRaw @("sessions")) -join "`n"
}

function Get-NamedEndpoint([string]$Text) {
    $m = [regex]::Match($Text, "^\[$([regex]::Escape($Session))\]\s+(gpu-[a-z0-9-]+)\s+\|", "Multiline")
    if ($m.Success) { return $m.Groups[1].Value }
    return $null
}

$script:UnnamedSeen = @{}

function Test-SessionAlive {
    $text = Get-SessionListing
    $current = @{}
    foreach ($m in [regex]::Matches($text, "^\[\?\]\s+(gpu-[a-z0-9-]+)\s+\|", "Multiline")) {
        $endpoint = $m.Groups[1].Value
        $current[$endpoint] = $true
        # A freshly allocated session briefly lists as unnamed before its name is
        # registered, so reaping on first sight can destroy a sibling shard's
        # session that is still coming up -- which happened once. Only act on an
        # endpoint that has stayed unnamed across several consecutive polls.
        $seen = 1 + [int]$script:UnnamedSeen[$endpoint]
        $script:UnnamedSeen[$endpoint] = $seen
        Write-Log "unnamed_assignment endpoint=$endpoint consecutive=$seen"
        if ($ReapOrphans -and $seen -ge $OrphanConfirmPolls) {
            Write-Log "orphan_confirmed=true endpoint=$endpoint polls=$seen"
            if (Release-Endpoint $endpoint) { $script:UnnamedSeen.Remove($endpoint) }
        }
    }
    foreach ($key in @($script:UnnamedSeen.Keys)) {
        if (-not $current.ContainsKey($key)) { $script:UnnamedSeen.Remove($key) }
    }
    return ($text -match [regex]::Escape("[$Session]"))
}

function Invoke-Recovery([string]$Reason) {
    Write-Log "recovery_start reason=$Reason"
    $baseline = Get-LocalCheckpointCount

    # Release this shard's own assignment by endpoint before rebuilding, so the
    # retry is never blocked by our own stale allocation and no assignment is
    # left burning compute units. Name-based stop is only a fallback.
    $listing = Get-SessionListing
    $endpoint = Get-NamedEndpoint $listing
    if ($endpoint) {
        Release-Endpoint $endpoint | Out-Null
    }
    Invoke-ColabRaw @("stop", "-s", $Session) | Out-Null

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        $created = Invoke-ColabRaw @("new", "-s", $Session, "--gpu", "A100")
        if ($LASTEXITCODE -eq 0) {
            Write-Log "session_recreated=true attempt=$attempt"
            break
        }
        $detail = ($created -join ' ')
        if ($detail -match 'TooManyAssignments') {
            Write-Log "session_create_blocked=too_many_assignments attempt=$attempt"
        }
        else {
            Write-Log "session_create_failed attempt=$attempt detail=$($detail.Substring(0, [Math]::Min(200, $detail.Length)))"
        }
        Start-Sleep -Seconds ([Math]::Min(300, 30 * $attempt))
    }

    $deployArgs = @(
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", "$repo\scripts\deploy_m5_tabpfn_site_shard.ps1",
        "-Site", $Site, "-Shard", $Shard, "-NEstimators", $NEstimators,
        "-Session", $Session, "-ColabHome", $ColabHome, "-Microbatch", $Microbatch,
        # Forward the shard root so a 137-batch recovery redeploys from its own
        # exported inputs. Without this, deploy falls back to the per-site default
        # (m5_tabpfn_site<Site>_context100000_n<N>) and would upload the wrong,
        # already-completed site shard onto this batch's session.
        "-ShardRootName", $ShardRootName
    )
    if ($RelaxTokenScope) { $deployArgs += "-RelaxTokenScope" }
    & powershell.exe @deployArgs *> (Join-Path $results "supervisor_deploy.log")
    Write-Log "redeploy_exit=$LASTEXITCODE baseline_chunks=$baseline"
}

Write-Log "supervisor_start site=$Site shard=$Shard n=$NEstimators session=$Session expected=$ExpectedCheckpointCount"
$lastCount = -1
$stalled = 0

while ($true) {
    $count = Get-LocalCheckpointCount
    $resultPath = Join-Path $results "result.json"
    if ((Test-Path -LiteralPath $resultPath) -and $count -ge $ExpectedCheckpointCount) {
        Write-Log "shard_complete=true chunks=$count"
        break
    }

    if ($count -gt $lastCount) {
        $stalled = 0
        Write-Log "progress chunks=$count expected=$ExpectedCheckpointCount"
    }
    else {
        $stalled++
    }
    $lastCount = $count

    if (-not (Test-SessionAlive)) {
        Invoke-Recovery "session_missing"
        $stalled = 0
    }
    elseif ($stalled -ge $StallCycles) {
        Write-Log "stall_detected cycles=$stalled chunks=$count"
        Invoke-Recovery "durable_progress_stalled"
        $stalled = 0
    }

    Start-Sleep -Seconds $PollSeconds
}
