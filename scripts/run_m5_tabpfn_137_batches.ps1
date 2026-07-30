param(
    [int[]]$Batches = @(0, 1, 2, 3, 4, 5),
    [string]$ColabHome = "/home/tonykuo/.colab-tony",
    [switch]$RelaxTokenScope = $true,
    [int]$Microbatch = 20000,
    [int]$PollSeconds = 60,
    [int]$BatchTimeoutMinutes = 180
)

# LONG_RUNNING_RESEARCH_LAUNCH_BLOCKED: this legacy runner has a batch
# wall-clock timeout. It cannot be run until migrated to the repository
# checkpoint/resume execution contract. There is no override.
throw "Blocked: migrate this legacy long-running research path to atomic checkpoints, resume, provenance, and no-timeout execution first."

# Drives the remaining 137-feature batches through two A100s, one batch at a
# time: head and tail run in parallel, and the next batch only starts once both
# halves of the current one are durably complete.
#
# Sequential by batch is deliberate. The account holds two A100s, so running two
# batches at once would need four; more importantly a batch is the unit that
# gets merged and checked, and finishing them in order keeps partial progress
# meaningful if the run is interrupted.
#
# Per-shard bring-up is delegated to queue_m5_tabpfn_site_shard.ps1, which
# enforces the runbook ordering (allocate -> keep-alive -> deploy -> sync +
# supervisor). Completion is judged only by durable local checkpoints plus
# result.json, never by heartbeat or session existence.

$ErrorActionPreference = "Continue"
$repo = "C:\Users\tonykuo\projects\lead-reproduction"
$proc = Join-Path $repo "data\processed"
$plan = Get-Content -LiteralPath (Join-Path $proc "m5_tabpfn_137_remaining_batch_plan.json") -Raw | ConvertFrom-Json
$logPath = Join-Path $proc "m5_tabpfn_137_batch_runner.log"

function Write-Log([string]$Message) {
    $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $line = "$stamp $Message"
    Add-Content -LiteralPath $logPath -Value $line
    Write-Output $line
}

function Get-ShardState([int]$Batch, [string]$Shard, [int]$Expected) {
    $dir = Join-Path $proc "m5_tabpfn_f137_batch${Batch}_context100000_n8\$Shard-results"
    $chunks = @(Get-ChildItem -LiteralPath (Join-Path $dir "chunks") -Filter "rows_*.npz" -File -ErrorAction SilentlyContinue).Count
    $done = (Test-Path -LiteralPath (Join-Path $dir "result.json")) -and ($chunks -ge $Expected)
    return [pscustomobject]@{ Chunks = $chunks; Done = $done }
}

Write-Log "runner_start batches=$($Batches -join ',') microbatch=$Microbatch"

foreach ($batch in $Batches) {
    $entry = $plan.batches | Where-Object { $_.batch -eq $batch }
    if ($null -eq $entry) { Write-Log "batch_missing_in_plan=$batch"; continue }

    $shards = @(
        @{ shard = "head"; expected = [int]$entry.head_chunks },
        @{ shard = "tail"; expected = [int]$entry.tail_chunks }
    )

    $already = $true
    foreach ($s in $shards) {
        if (-not (Get-ShardState $batch $s.shard $s.expected).Done) { $already = $false }
    }
    if ($already) { Write-Log "batch_already_complete=$batch"; continue }

    Write-Log "batch_start=$batch rows=$($entry.rows) sites=$($entry.sites -join ',')"
    foreach ($s in $shards) {
        if ((Get-ShardState $batch $s.shard $s.expected).Done) {
            Write-Log "shard_already_complete batch=$batch shard=$($s.shard)"
            continue
        }
        $session = "lead-tabpfn-b$batch-$($s.shard)-f137n8"
        $queueLog = Join-Path $proc "m5_tabpfn_f137_batch${batch}_context100000_n8\$($s.shard)-results\queue.log"
        New-Item -ItemType Directory -Path (Split-Path $queueLog) -Force | Out-Null
        $inner = "& '$repo\scripts\queue_m5_tabpfn_site_shard.ps1' -Site $batch -Shard $($s.shard) -NEstimators 8 " +
        "-Session $session -ColabHome $ColabHome -ExpectedCheckpointCount $($s.expected) " +
        "-ShardRootName m5_tabpfn_f137_batch${batch}_context100000_n8 -Microbatch $Microbatch"
        if ($RelaxTokenScope) { $inner += " -RelaxTokenScope" }
        $inner += " *> '$queueLog'"
        Start-Process powershell.exe -ArgumentList @(
            "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", $inner
        ) -WindowStyle Hidden
        Write-Log "queued batch=$batch shard=$($s.shard) session=$session expected=$($s.expected)"
        Start-Sleep -Seconds 10
    }

    $deadline = (Get-Date).AddMinutes($BatchTimeoutMinutes)
    while ($true) {
        $states = $shards | ForEach-Object { Get-ShardState $batch $_.shard $_.expected }
        $line = ($shards | ForEach-Object { $st = Get-ShardState $batch $_.shard $_.expected; "$($_.shard)=$($st.Chunks)/$($_.expected)" }) -join ' '
        Write-Log "batch_progress=$batch $line"
        if (($states | Where-Object { -not $_.Done }).Count -eq 0) {
            Write-Log "batch_complete=$batch"
            break
        }
        if ((Get-Date) -gt $deadline) {
            Write-Log "batch_timeout=$batch after ${BatchTimeoutMinutes}m; moving on"
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-Log "runner_done"
