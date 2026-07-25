param(
    [string]$Plan = "m5_tabpfn_17_remaining_batch_plan.json",
    [string]$ShardRootTemplate = "m5_tabpfn_f17_batch{0}_context100000_n8",
    [string]$SessionTemplate = "lead-tabpfn-b{0}-{1}-f17n8",
    [int[]]$Batches = @(0, 1, 2, 3, 4, 5),
    [int]$Slots = 2,
    [string]$ColabHome = "/home/tonykuo/.colab-tony",
    [switch]$RelaxTokenScope = $true,
    [int]$NEstimators = 8,
    [int]$Microbatch = 20000,
    [int]$PollSeconds = 60,
    [int]$StallMinutes = 25,
    [string]$LogName = "m5_tabpfn_17_shard_pool.log"
)

# Drives a queue of head/tail shards through a fixed number of GPU slots, greedily:
# the moment any slot's shard is durably complete, the next shard in the queue
# starts on it.
#
# This replaces the batch-barrier runner (run_m5_tabpfn_137_batches.ps1), which
# waited for *both* halves of a batch before starting the next one. That barrier
# turned any single straggler into idle time on the other A100 and chained the
# delay into every later batch -- the 137-feature run abandoned it midway and was
# finished by hand. The scheduling that actually worked lives here instead of in
# someone's shell history.
#
# Per-shard bring-up is still delegated to queue_m5_tabpfn_site_shard.ps1, which
# enforces the runbook ordering (allocate -> keep-alive -> deploy -> sync +
# supervisor). Completion is judged only by durable local checkpoints plus
# result.json, never by heartbeat or session existence.
#
# Stalls are reported, never auto-repaired: a shard with a live supervisor must
# not be redeployed underneath it. The one exception is a shard whose queue
# script gave up waiting for an A100 -- nothing is running for it, so its slot is
# genuinely free and it goes back on the queue.

$ErrorActionPreference = "Continue"
$repo = "C:\Users\tonykuo\projects\lead-reproduction"
$proc = Join-Path $repo "data\processed"
$planData = Get-Content -LiteralPath (Join-Path $proc $Plan) -Raw | ConvertFrom-Json
$logPath = Join-Path $proc $LogName

# Hold the log open with FileShare.ReadWrite instead of reopening per line.
# Add-Content takes an exclusive handle, so anything tailing this file -- a
# `Get-Content -Wait`, a `tail -F` -- makes the *scheduler's* writes fail. The
# scheduler keeps running and keeps placing shards, but its log freezes, so the
# run looks dead from the outside precisely because someone was watching it.
# This was observed: the log stopped for 31 minutes while batch 1 was launched
# normally the whole time.
$logStream = [System.IO.StreamWriter]::new(
    [System.IO.FileStream]::new(
        $logPath,
        [System.IO.FileMode]::Append,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::ReadWrite
    )
)
$logStream.AutoFlush = $true

function Write-Log([string]$Message) {
    $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $line = "$stamp $Message"
    $logStream.WriteLine($line)
    Write-Output $line
}

function Get-ResultsDir($Item) {
    Join-Path $proc "$($Item.Root)\$($Item.Shard)-results"
}

function Get-ChunkCount($Item) {
    $dir = Join-Path (Get-ResultsDir $Item) "chunks"
    return @(Get-ChildItem -LiteralPath $dir -Filter "rows_*.npz" -File -ErrorAction SilentlyContinue).Count
}

function Test-ShardDone($Item) {
    $results = Get-ResultsDir $Item
    $hasResult = Test-Path -LiteralPath (Join-Path $results "result.json")
    return $hasResult -and ((Get-ChunkCount $Item) -ge $Item.Expected)
}

function Test-AllocationGaveUp($Item) {
    $queueLog = Join-Path (Get-ResultsDir $Item) "queue.log"
    if (-not (Test-Path -LiteralPath $queueLog)) { return $false }
    return (Select-String -LiteralPath $queueLog -Pattern "FAILED to allocate" -Quiet) -eq $true
}

function Start-Shard($Item) {
    $results = Get-ResultsDir $Item
    New-Item -ItemType Directory -Path $results -Force | Out-Null
    $queueLog = Join-Path $results "queue.log"
    # A fresh log per attempt, so the give-up probe cannot read a stale verdict.
    Remove-Item -LiteralPath $queueLog -Force -ErrorAction SilentlyContinue
    $inner = "& '$repo\scripts\queue_m5_tabpfn_site_shard.ps1' -Site $($Item.Batch) -Shard $($Item.Shard) " +
    "-NEstimators $NEstimators -Session $($Item.Session) -ColabHome $ColabHome " +
    "-ExpectedCheckpointCount $($Item.Expected) -ShardRootName $($Item.Root) -Microbatch $Microbatch"
    if ($RelaxTokenScope) { $inner += " -RelaxTokenScope" }
    $inner += " *> '$queueLog'"
    Start-Process powershell.exe -ArgumentList @(
        "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", $inner
    ) -WindowStyle Hidden
    Write-Log "launched batch=$($Item.Batch) shard=$($Item.Shard) session=$($Item.Session) expected=$($Item.Expected)"
}

# Queue order is batch 0 head, batch 0 tail, batch 1 head, ... Interleaving the
# halves this way means the two slots normally work the same batch at once, so a
# batch finishes as a unit and partial progress stays merge-meaningful.
$queue = [System.Collections.ArrayList]@()
foreach ($batch in $Batches) {
    $entry = $planData.batches | Where-Object { $_.batch -eq $batch }
    if ($null -eq $entry) { Write-Log "batch_missing_in_plan=$batch"; continue }
    foreach ($pair in @(@("head", [int]$entry.head_chunks), @("tail", [int]$entry.tail_chunks))) {
        [void]$queue.Add([pscustomobject]@{
                Batch    = $batch
                Shard    = $pair[0]
                Expected = $pair[1]
                Root     = ($ShardRootTemplate -f $batch)
                Session  = ($SessionTemplate -f $batch, $pair[0])
            })
    }
}

Write-Log "pool_start slots=$Slots shards=$($queue.Count) microbatch=$Microbatch home=$ColabHome"

$pending = [System.Collections.ArrayList]@()
foreach ($item in $queue) {
    if (Test-ShardDone $item) {
        Write-Log "already_complete batch=$($item.Batch) shard=$($item.Shard)"
    }
    else { [void]$pending.Add($item) }
}

$active = [System.Collections.ArrayList]@()
while ($pending.Count -gt 0 -or $active.Count -gt 0) {
    # Retire finished shards first, so their slots are reusable this same pass.
    foreach ($item in @($active)) {
        if (Test-ShardDone $item) {
            Write-Log "complete batch=$($item.Batch) shard=$($item.Shard)"
            $active.Remove($item)
            continue
        }
        $chunks = Get-ChunkCount $item
        if ($chunks -gt $item.Chunks) {
            $item.Chunks = $chunks
            $item.LastProgress = Get-Date
        }
        elseif (((Get-Date) - $item.LastProgress).TotalMinutes -ge $StallMinutes) {
            if (Test-AllocationGaveUp $item) {
                # Nothing is running for this shard, so requeueing cannot collide
                # with a live supervisor.
                Write-Log "requeue_no_gpu batch=$($item.Batch) shard=$($item.Shard)"
                $active.Remove($item)
                [void]$pending.Add($item)
            }
            else {
                Write-Log "STALL batch=$($item.Batch) shard=$($item.Shard) chunks=$chunks/$($item.Expected) for ${StallMinutes}m; check the remote, do not redeploy under a live supervisor"
                $item.LastProgress = Get-Date
            }
        }
    }

    while ($active.Count -lt $Slots -and $pending.Count -gt 0) {
        $item = $pending[0]
        $pending.RemoveAt(0)
        $item | Add-Member -NotePropertyName Chunks -NotePropertyValue (Get-ChunkCount $item) -Force
        $item | Add-Member -NotePropertyName LastProgress -NotePropertyValue (Get-Date) -Force
        Start-Shard $item
        [void]$active.Add($item)
        Start-Sleep -Seconds 10
    }

    if ($active.Count -gt 0) {
        $line = ($active | ForEach-Object { "b$($_.Batch)/$($_.Shard)=$(Get-ChunkCount $_)/$($_.Expected)" }) -join ' '
        Write-Log "progress queued=$($pending.Count) $line"
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Log "pool_done"
