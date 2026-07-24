param(
    [int]$PollSeconds = 120,
    [int]$OrphanConfirmPolls = 3,
    [string]$ColabCli = "/home/tonykuo/.local/bin/colab",
    [string]$ColabPython = "/home/tonykuo/.local/share/uv/tools/google-colab-cli/bin/python",
    [ValidateSet("adc", "oauth2")][string]$Auth = "oauth2",
    [switch]$WhatIf
)

# Final-stage reaper: with no further work queued, any Colab session that is not
# actively producing checkpoints is pure compute-unit burn.
#
# The sync monitor already stops a session when its shard completes, but that
# only covers the happy path. It does not cover a session whose deploy failed
# after allocation, a session whose sync monitor died, or an assignment left
# unnamed after the CLI dropped its local record. This process closes those
# gaps, then exits once every tracked shard is finished and nothing is left
# running.
#
# It only ever touches sessions in the lead-tabpfn-* namespace.

$ErrorActionPreference = "Continue"
$repo = "C:\Users\tonykuo\projects\lead-reproduction"
$proc = Join-Path $repo "data\processed"
$logPath = Join-Path $proc "m5_tabpfn_idle_session_reaper.log"

$accounts = @(
    @{ name = "hank"; home = "/home/tonykuo/.colab-hank"; relax = $false },
    @{ name = "tony"; home = "/home/tonykuo/.colab-tony"; relax = $true }
)

# Sessions expected to be doing real work, with the local evidence of completion.
$tracked = @(
    @{ session = "lead-tabpfn-s2-head-f137n8"; dir = "m5_tabpfn_site2_f137_context100000_n8\head-results"; expected = 32 },
    @{ session = "lead-tabpfn-s2-tail-f137n8"; dir = "m5_tabpfn_site2_f137_context100000_n8\tail-results"; expected = 32 },
    @{ session = "lead-tabpfn-s3-head-f137n8"; dir = "m5_tabpfn_site3_f137_context100000_n8\head-results"; expected = 30 },
    @{ session = "lead-tabpfn-s3-tail-f137n8"; dir = "m5_tabpfn_site3_f137_context100000_n8\tail-results"; expected = 30 }
)

function Write-Log([string]$Message) {
    $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    Add-Content -LiteralPath $logPath -Value "$stamp $Message"
}

function Get-Listing($account) {
    [string[]]$extra = @()
    if ($account.relax) { $extra = @("OAUTHLIB_RELAX_TOKEN_SCOPE=1") }
    return (& wsl.exe -d Ubuntu -- env "HOME=$($account.home)" @extra $ColabCli --auth $Auth sessions 2>&1) -join "`n"
}

function Stop-Named($account, [string]$Session) {
    if ($WhatIf) { Write-Log "would_stop session=$Session account=$($account.name)"; return }
    [string[]]$extra = @()
    if ($account.relax) { $extra = @("OAUTHLIB_RELAX_TOKEN_SCOPE=1") }
    & wsl.exe -d Ubuntu -- env "HOME=$($account.home)" @extra $ColabCli --auth $Auth stop -s $Session 2>&1 | Out-Null
    Write-Log "stopped_idle_session=$Session account=$($account.name) exit=$LASTEXITCODE"
}

function Release-Endpoint($account, [string]$Endpoint) {
    if ($Endpoint -notmatch '^gpu-[a-z0-9-]+$') { return }
    if ($WhatIf) { Write-Log "would_release endpoint=$Endpoint"; return }
    [string[]]$extra = @()
    if ($account.relax) { $extra = @("OAUTHLIB_RELAX_TOKEN_SCOPE=1") }
    $provider = if ($Auth -eq "adc") { "AuthProvider.ADC" } else { "AuthProvider.OAUTH2" }
    $code = "from colab_cli.auth import AuthProvider; from colab_cli.common import state; state.auth_provider=$provider; state.client.unassign('$Endpoint')"
    & wsl.exe -d Ubuntu -- env "HOME=$($account.home)" @extra $ColabPython -c $code 2>&1 | Out-Null
    Write-Log "released_endpoint=$Endpoint account=$($account.name) exit=$LASTEXITCODE"
}

function Test-ShardComplete($entry) {
    $dir = Join-Path $proc $entry.dir
    $result = Join-Path $dir "result.json"
    if (-not (Test-Path -LiteralPath $result)) { return $false }
    $count = @(Get-ChildItem -LiteralPath (Join-Path $dir "chunks") -Filter "rows_*.npz" -File -ErrorAction SilentlyContinue).Count
    return ($count -ge $entry.expected)
}

Write-Log "reaper_start tracked=$($tracked.Count) poll=${PollSeconds}s whatif=$WhatIf"
$unnamedSeen = @{}

while ($true) {
    $allComplete = $true
    foreach ($entry in $tracked) {
        if (-not (Test-ShardComplete $entry)) { $allComplete = $false }
    }

    foreach ($account in $accounts) {
        $text = Get-Listing $account
        foreach ($m in [regex]::Matches($text, "^\[([A-Za-z0-9_-]+)\]\s+(gpu-[a-z0-9-]+)\s+\|", "Multiline")) {
            $session = $m.Groups[1].Value
            if ($session -notlike "lead-tabpfn-*") { continue }
            $entry = $tracked | Where-Object { $_.session -eq $session } | Select-Object -First 1
            if ($null -eq $entry) {
                # Not something we expect to be working: nothing will ever stop it.
                Write-Log "untracked_session=$session account=$($account.name)"
                Stop-Named $account $session
            }
            elseif (Test-ShardComplete $entry) {
                Write-Log "shard_done_session_idle=$session"
                Stop-Named $account $session
            }
        }
        $current = @{}
        foreach ($m in [regex]::Matches($text, "^\[\?\]\s+(gpu-[a-z0-9-]+)\s+\|", "Multiline")) {
            $endpoint = $m.Groups[1].Value
            $current[$endpoint] = $true
            $seen = 1 + [int]$unnamedSeen[$endpoint]
            $unnamedSeen[$endpoint] = $seen
            if ($seen -ge $OrphanConfirmPolls) {
                Write-Log "orphan_confirmed=$endpoint polls=$seen"
                Release-Endpoint $account $endpoint
                $unnamedSeen.Remove($endpoint)
            }
        }
        foreach ($key in @($unnamedSeen.Keys)) {
            if (-not $current.ContainsKey($key)) { $unnamedSeen.Remove($key) }
        }
    }

    if ($allComplete) {
        Write-Log "all_tracked_shards_complete=true reaper_exiting"
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
