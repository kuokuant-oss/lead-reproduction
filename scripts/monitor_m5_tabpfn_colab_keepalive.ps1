param(
    [string]$Session = "lead-tabpfn-tail",
    [int]$PollSeconds = 60,
    [string]$LogPath = "C:\Users\tonykuo\projects\lead-reproduction\data\processed\m5_tabpfn_colab_keepalive_monitor.log",
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$cliPython = "/home/tonykuo/.local/share/uv/tools/google-colab-cli/bin/python"
$sessionsPath = "/home/tonykuo/.config/colab-cli/sessions.json"
if ($Session -notmatch '^[A-Za-z0-9_-]+$') {
    throw "Unsafe session name: $Session"
}
if ($PollSeconds -lt 10) {
    throw "PollSeconds must be at least 10"
}

while ($true) {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    try {
        $pythonCode = "import json,pathlib; d=json.loads(pathlib.Path('$sessionsPath').read_text()); print(d.get('$Session',{}).get('endpoint',''))"
        $endpoint = (& wsl.exe -d Ubuntu -- $cliPython -c $pythonCode 2>$null | Select-Object -First 1).Trim()
        if (-not $endpoint) {
            Add-Content -LiteralPath $LogPath -Value "$timestamp session_missing=true"
        }
        elseif ($endpoint -notmatch '^[a-z0-9-]+$') {
            throw "Unsafe endpoint returned by Colab state: $endpoint"
        }
        else {
            $pattern = "colab_cli.cli --auth=adc keep-alive $endpoint $Session"
            $existing = @(& wsl.exe -d Ubuntu -- pgrep -f -- $pattern 2>$null)
            if ($existing.Count -eq 0) {
                $launch = "nohup $cliPython -m colab_cli.cli --auth=adc keep-alive $endpoint $Session >/dev/null 2>&1 &"
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
