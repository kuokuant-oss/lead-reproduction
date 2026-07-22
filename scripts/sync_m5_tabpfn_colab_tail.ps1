param(
    [string]$Session = "lead-tabpfn-tail",
    [int]$PollSeconds = 60,
    [int]$ExpectedCheckpointCount = 254,
    [string]$LocalDirectory = "C:\Users\tonykuo\projects\lead-reproduction\data\processed\m5_tabpfn_distributed_context100000\tail-results",
    [string]$ColabCli = "/home/tonykuo/.local/bin/colab"
)

$ErrorActionPreference = "Stop"
$remoteWork = "/content/lead_tabpfn_tail/work"
$remoteChunks = "$remoteWork/chunks"
$allowedRoot = [System.IO.Path]::GetFullPath("C:\Users\tonykuo\projects\lead-reproduction\data\processed")
$resolvedLocal = [System.IO.Path]::GetFullPath($LocalDirectory)
if (-not $resolvedLocal.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "LocalDirectory must remain under $allowedRoot"
}
if ($PollSeconds -lt 10) {
    throw "PollSeconds must be at least 10"
}

$chunksDirectory = Join-Path $resolvedLocal "chunks"
New-Item -ItemType Directory -Path $chunksDirectory -Force | Out-Null
$logPath = Join-Path $resolvedLocal "sync.log"

function Convert-ToWslPath([string]$WindowsPath) {
    $full = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($full -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Only absolute Windows drive paths are supported: $full"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace('\', '/')
    return "/mnt/$drive/$tail"
}

function Invoke-Colab([string[]]$Arguments) {
    $output = & wsl.exe -d Ubuntu -- $ColabCli --auth adc @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ($output -join [Environment]::NewLine)
    }
    return @($output)
}

function Download-Atomically([string]$RemotePath, [string]$LocalPath) {
    $temporary = "$LocalPath.partial"
    $wslTemporary = Convert-ToWslPath $temporary
    Invoke-Colab @("download", "-s", $Session, $RemotePath, $wslTemporary) | Out-Null
    Move-Item -LiteralPath $temporary -Destination $LocalPath -Force
}

while ($true) {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    try {
        $chunkListing = Invoke-Colab @("ls", "-s", $Session, $remoteChunks)
        $remoteChunkNames = @(
            $chunkListing |
                Select-String -AllMatches -Pattern 'rows_\d{8}_\d{8}\.npz' |
                ForEach-Object { $_.Matches.Value } |
                Sort-Object -Unique
        )
        foreach ($name in $remoteChunkNames) {
            $localPath = Join-Path $chunksDirectory $name
            if (-not (Test-Path -LiteralPath $localPath)) {
                Download-Atomically "$remoteChunks/$name" $localPath
                Add-Content -LiteralPath $logPath -Value "$timestamp downloaded=$name"
            }
        }

        $workListing = Invoke-Colab @("ls", "-s", $Session, $remoteWork)
        $stateNames = @("heartbeat.json", "progress.json", "result.json", "worker.log", "launcher.json")
        foreach ($name in $stateNames) {
            if (($workListing -join "`n") -match [regex]::Escape($name)) {
                Download-Atomically "$remoteWork/$name" (Join-Path $resolvedLocal $name)
            }
        }

        $localCount = @(Get-ChildItem -LiteralPath $chunksDirectory -Filter "rows_*.npz" -File).Count
        Add-Content -LiteralPath $logPath -Value "$timestamp checkpoints=$localCount expected=$ExpectedCheckpointCount"
        $resultPath = Join-Path $resolvedLocal "result.json"
        if ((Test-Path -LiteralPath $resultPath) -and $localCount -ge $ExpectedCheckpointCount) {
            Invoke-Colab @("stop", "-s", $Session) | Out-Null
            Add-Content -LiteralPath $logPath -Value "$timestamp completed=true session_stopped=true"
            exit 0
        }
    }
    catch {
        Add-Content -LiteralPath $logPath -Value "$timestamp sync_error=$($_.Exception.Message.Replace([Environment]::NewLine, ' '))"
    }
    Start-Sleep -Seconds $PollSeconds
}
