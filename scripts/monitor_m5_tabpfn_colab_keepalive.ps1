param(
    [string]$Session = "lead-tabpfn-tail",
    [int]$PollSeconds = 60,
    [string]$LogPath = "C:\Users\tonykuo\projects\lead-reproduction\data\processed\m5_tabpfn_colab_keepalive_monitor.log",
    [string]$RemoteRoot = "/content/lead_tabpfn_tail",
    [int]$TouchSeconds = 2700,
    [ValidateSet("adc", "oauth2")]
    [string]$Auth = "oauth2",
    [string]$ColabHome = "/home/tonykuo/.colab-hank",
    # The tonykuo account only receives the drive.file scope when this is set;
    # without it the CLI aborts before writing anything.
    [switch]$RelaxTokenScope,
    # Durable evidence that this shard is finished, so the monitor can retire
    # itself. Without it the loop runs forever: the sync monitor stops the Colab
    # session on completion, but this process keeps shelling into WSL every poll
    # for a session that no longer exists. Across a queue of shards those leaked
    # pollers accumulate, and one of them touching a session name that has since
    # been reused is worse than noise.
    [string]$CompletionDirectory = "",
    [int]$ExpectedCheckpointCount = 0,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$colabCli = "/home/tonykuo/.local/bin/colab"
$cliPython = "/home/tonykuo/.local/share/uv/tools/google-colab-cli/bin/python"
$sessionsPath = "$ColabHome/.config/colab-cli/sessions.json"
if ($Session -notmatch '^[A-Za-z0-9_-]+$') {
    throw "Unsafe session name: $Session"
}
if ($PollSeconds -lt 10) {
    throw "PollSeconds must be at least 10"
}
if ($TouchSeconds -lt 300) {
    throw "TouchSeconds must be at least 300"
}

# Explicitly typed: PowerShell unrolls a single-element array returned from an
# if-expression into a scalar, and splatting a string passes it one char at a
# time ("env: 'O': No such file or directory").
[string[]]$extraEnv = @()
if ($RelaxTokenScope) { $extraEnv = @("OAUTHLIB_RELAX_TOKEN_SCOPE=1") }
$relaxAssignment = if ($RelaxTokenScope) { "OAUTHLIB_RELAX_TOKEN_SCOPE=1 " } else { "" }
$lastTouch = 0

function Test-ShardComplete {
    if (-not $CompletionDirectory -or $ExpectedCheckpointCount -le 0) { return $false }
    if (-not (Test-Path -LiteralPath (Join-Path $CompletionDirectory "result.json"))) {
        return $false
    }
    $count = @(Get-ChildItem -LiteralPath (Join-Path $CompletionDirectory "chunks") `
            -Filter "rows_*.npz" -File -ErrorAction SilentlyContinue).Count
    return ($count -ge $ExpectedCheckpointCount)
}

while ($true) {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    if (Test-ShardComplete) {
        Add-Content -LiteralPath $LogPath -Value "$timestamp shard_complete=true keepalive_exiting"
        break
    }
    try {
        $pythonCode = "import json,pathlib; d=json.loads(pathlib.Path('$sessionsPath').read_text()); print(d.get('$Session',{}).get('endpoint',''))"
        $endpoint = (& wsl.exe -d Ubuntu -- env "HOME=$ColabHome" @extraEnv $cliPython -c $pythonCode 2>$null | Select-Object -First 1).Trim()
        if (-not $endpoint) {
            Add-Content -LiteralPath $LogPath -Value "$timestamp session_missing=true"
        }
        elseif ($endpoint -notmatch '^[a-z0-9-]+$') {
            throw "Unsafe endpoint returned by Colab state: $endpoint"
        }
        else {
            $pattern = "colab_cli.cli --auth=$Auth keep-alive $endpoint $Session"
            $existing = @(& wsl.exe -d Ubuntu -- pgrep -f -- $pattern 2>$null)
            if ($existing.Count -eq 0) {
                $launch = "nohup env HOME=$ColabHome $relaxAssignment$cliPython -m colab_cli.cli --auth=$Auth keep-alive $endpoint $Session >/dev/null 2>&1 &"
                & wsl.exe -d Ubuntu -- bash -lc $launch | Out-Null
                Start-Sleep -Seconds 2
                $existing = @(& wsl.exe -d Ubuntu -- pgrep -f -- $pattern 2>$null)
                if ($existing.Count -eq 0) {
                    throw "Failed to restore Colab keep-alive for $endpoint"
                }
                Add-Content -LiteralPath $LogPath -Value "$timestamp keep_alive_restarted=true endpoint=$endpoint"
            }
            else {
                Add-Content -LiteralPath $LogPath -Value "$timestamp keep_alive_ok=true endpoint=$endpoint"
            }

            if ($lastTouch -eq 0 -or ($timestamp - $lastTouch) -ge $TouchSeconds) {
                $touchOutput = & wsl.exe -d Ubuntu -- env "HOME=$ColabHome" @extraEnv $colabCli --auth $Auth ls -s $Session "$RemoteRoot/work" 2>&1
                if ($LASTEXITCODE -ne 0) {
                    throw "Work touch failed: $($touchOutput -join ' ')"
                }
                $lastTouch = $timestamp
                Add-Content -LiteralPath $LogPath -Value "$timestamp work_touch=true remote=$RemoteRoot/work interval_seconds=$TouchSeconds"
            }
        }
    }
    catch {
        $message = $_.Exception.Message.Replace([Environment]::NewLine, ' ')
        Add-Content -LiteralPath $LogPath -Value "$timestamp monitor_error=$message"
    }
    if ($Once) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
