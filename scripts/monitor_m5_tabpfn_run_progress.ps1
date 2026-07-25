param(
    [string]$Plan = "m5_tabpfn_17_remaining_batch_plan.json",
    [string]$ShardRootTemplate = "m5_tabpfn_f17_batch{0}_context100000_n8",
    [int[]]$Batches = @(0, 1, 2, 3, 4, 5),
    # Poll fast. The point of this monitor is to answer "did a chunk land just
    # now", so the sampling interval has to be far shorter than the chunk
    # interval, not comparable to it.
    [int]$PollSeconds = 10,
    # A heartbeat line on every poll would be 8,600 lines a day per shard, so the
    # dense record is emitted at this cadence instead. Every *chunk arrival* is
    # still logged the moment it happens, regardless of this.
    [int]$HeartbeatSeconds = 60,
    # Stall detection calibrates itself against the shard's own observed chunk
    # interval, because that interval is a property of the batch and the GPU, not
    # something worth hardcoding. The floor keeps a shard that has only just
    # started -- one sample, no median yet -- from alerting on normal warm-up.
    [int]$StallFloorSeconds = 300,
    [double]$StallFactor = 4.0,
    # A shard that never produces its *first* chunk is invisible to interval-based
    # stall detection -- there is no interval to compare against. That is exactly
    # the handoff failure: a slot frees, the next shard is launched, its deploy
    # dies, and because the other slot keeps producing the run-level stall check
    # never fires either. So a launched shard is also on the clock from launch.
    # Allocate + upload + install runs ~5 minutes and the first chunk ~1 more, so
    # this is generous while still catching a dead shard in minutes, not never.
    [int]$FirstChunkGraceMinutes = 12,
    # Nothing anywhere advancing for this long means the run as a whole is dead
    # (WSL gone, auth expired, network down) rather than one shard being unlucky.
    [int]$RunStallMinutes = 20,
    [string]$LogName = "m5_tabpfn_17_run_progress.log"
)

# Watches durable local chunk counts for every shard and says, in one place,
# whether the run is actually moving right now.
#
# Every other monitor here is scoped to a single shard and acts on it: the sync
# monitor downloads, the supervisor rebuilds a lost session, the pool scheduler
# fills GPU slots. None of them answers "is the whole run still producing, at
# what rate, and when will it finish". This one only observes and reports -- it
# never touches a session -- so it is safe alongside all of them.
#
# Progress is measured only by durable local checkpoints. Remote heartbeats and
# session listings both keep looking healthy while a worker produces nothing,
# which is exactly the failure this is meant to catch.

$ErrorActionPreference = "Continue"
$repo = "C:\Users\tonykuo\projects\lead-reproduction"
$proc = Join-Path $repo "data\processed"
$planData = Get-Content -LiteralPath (Join-Path $proc $Plan) -Raw | ConvertFrom-Json
$logPath = Join-Path $proc $LogName

# Hold the log open with FileShare.ReadWrite rather than reopening per line.
# Add-Content takes an exclusive handle, so anything tailing this file -- a
# `Get-Content -Wait`, a `tail -F` -- makes the *writer* fail, and the monitor
# silently stops recording exactly while someone is watching it. A log nobody can
# watch without breaking is worse than no log.
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
    $logStream.WriteLine("$stamp $Message")
    # Also to stdout, so a caller that redirects this process gets the same
    # stream without having to know the log path.
    Write-Output "$stamp $Message"
}

$shards = @()
foreach ($batch in $Batches) {
    $entry = $planData.batches | Where-Object { $_.batch -eq $batch }
    if ($null -eq $entry) { continue }
    foreach ($pair in @(@("head", [int]$entry.head_chunks), @("tail", [int]$entry.tail_chunks))) {
        $shards += [pscustomobject]@{
            Name      = "b$batch/$($pair[0])"
            Expected  = $pair[1]
            Results   = Join-Path $proc "$($ShardRootTemplate -f $batch)\$($pair[0])-results"
            Chunks    = 0
            LastChunk = $null
            Intervals = [System.Collections.ArrayList]@()
            Alerted   = $false
            Done      = $false
        }
    }
}
if ($shards.Count -eq 0) { throw "no shards derived from $Plan" }
$totalExpected = ($shards | Measure-Object -Property Expected -Sum).Sum

function Get-Chunks($Shard) {
    return @(Get-ChildItem -LiteralPath (Join-Path $Shard.Results "chunks") `
            -Filter "rows_*.npz" -File -ErrorAction SilentlyContinue).Count
}

function Get-MedianInterval($Shard) {
    if ($Shard.Intervals.Count -eq 0) { return $null }
    $sorted = @($Shard.Intervals | Sort-Object)
    return [double]$sorted[[int]([math]::Floor($sorted.Count / 2))]
}

# Chunks already on disk belong to a previous attempt; count them as the starting
# point so the rate figure describes this monitor's window, not the whole history.
foreach ($s in $shards) { $s.Chunks = Get-Chunks $s }
$start = Get-Date
$startChunks = ($shards | Measure-Object -Property Chunks -Sum).Sum
$lastRunProgress = $start
$lastTotal = $startChunks
$lastHeartbeat = [datetime]::MinValue
Write-Log "progress_monitor_start shards=$($shards.Count) expected_chunks=$totalExpected starting_chunks=$startChunks poll=${PollSeconds}s"

while ($true) {
    $now = Get-Date
    $total = 0
    $doneCount = 0

    foreach ($s in $shards) {
        $count = Get-Chunks $s
        $total += $count

        if ($count -gt $s.Chunks) {
            # Log every arrival the moment it is seen, with the gap since the
            # previous one -- this is the line that answers "is it still going".
            $gap = if ($null -ne $s.LastChunk) { [int]($now - $s.LastChunk).TotalSeconds } else { -1 }
            if ($gap -ge 0) {
                # A poll can pick up several chunks at once (the sync monitor
                # downloads in batches), so charge the elapsed time across them.
                $per = [int]($gap / ($count - $s.Chunks))
                for ($i = 0; $i -lt ($count - $s.Chunks); $i++) { [void]$s.Intervals.Add($per) }
                while ($s.Intervals.Count -gt 20) { $s.Intervals.RemoveAt(0) }
            }
            $median = Get-MedianInterval $s
            $medianText = if ($null -ne $median) { "$([int]$median)s" } else { "n/a" }
            $gapText = if ($gap -ge 0) { "+${gap}s" } else { "first" }
            Write-Log "chunk $($s.Name) $count/$($s.Expected) $gapText median=$medianText"
            $s.Chunks = $count
            $s.LastChunk = $now
            $s.Alerted = $false
        }

        $done = (Test-Path -LiteralPath (Join-Path $s.Results "result.json")) -and
            ($count -ge $s.Expected)
        if ($done) {
            if (-not $s.Done) { Write-Log "shard_done $($s.Name) chunks=$count/$($s.Expected)" }
            $s.Done = $true
            $doneCount++
        }
        elseif ($null -ne $s.LastChunk -and -not $s.Alerted) {
            $median = Get-MedianInterval $s
            $limit = $StallFloorSeconds
            if ($null -ne $median) {
                $limit = [math]::Max($StallFloorSeconds, [int]($StallFactor * $median))
            }
            $quiet = [int]($now - $s.LastChunk).TotalSeconds
            if ($quiet -ge $limit) {
                Write-Log "ALERT shard_stalled $($s.Name) chunks=$count/$($s.Expected) quiet=${quiet}s limit=${limit}s"
                $s.Alerted = $true
            }
        }
        elseif ($count -eq 0 -and -not $s.Alerted) {
            # queue.log is written by the scheduler immediately before it brings a
            # shard up, so its timestamp is this shard's launch time.
            $queueLog = Join-Path $s.Results "queue.log"
            if (Test-Path -LiteralPath $queueLog) {
                $since = ($now - (Get-Item -LiteralPath $queueLog).LastWriteTime).TotalMinutes
                if ($since -ge $FirstChunkGraceMinutes) {
                    Write-Log "ALERT no_first_chunk $($s.Name) launched_minutes_ago=$([math]::Round($since)) expected=$($s.Expected)"
                    $s.Alerted = $true
                }
            }
        }
    }

    if ($total -gt $lastTotal) {
        $lastRunProgress = $now
        $lastTotal = $total
    }
    elseif (($now - $lastRunProgress).TotalMinutes -ge $RunStallMinutes) {
        Write-Log "ALERT run_stalled no_new_chunks_for_minutes=$([math]::Round(($now - $lastRunProgress).TotalMinutes)) chunks=$total/$totalExpected"
        $lastRunProgress = $now
    }

    if (($now - $lastHeartbeat).TotalSeconds -ge $HeartbeatSeconds) {
        $lastHeartbeat = $now
        $elapsed = ($now - $start).TotalSeconds
        $gained = $total - $startChunks
        $rate = if ($elapsed -gt 0) { $gained / ($elapsed / 3600) } else { 0 }
        $etaText = if ($rate -gt 0) { "{0:N1}h" -f (($totalExpected - $total) / $rate) } else { "unknown" }
        $percent = if ($totalExpected -gt 0) { 100 * $total / $totalExpected } else { 0 }
        # Per-shard "quiet for Ns" is the field that makes this line answer, on
        # its own, whether anything is producing at this instant.
        $detail = @()
        foreach ($s in $shards) {
            if ($s.Done) { continue }
            if ($s.Chunks -eq 0 -and $null -eq $s.LastChunk) { continue }
            $quiet = if ($null -ne $s.LastChunk) { [int]($now - $s.LastChunk).TotalSeconds } else { -1 }
            $detail += "$($s.Name)=$($s.Chunks)/$($s.Expected)(quiet ${quiet}s)"
        }
        Write-Log ("progress {0}/{1} chunks ({2:N1}%) shards_done={3}/{4} rate={5:N1}ck/h eta={6} | {7}" -f `
                $total, $totalExpected, $percent, $doneCount, $shards.Count, $rate, $etaText, ($detail -join ' '))
    }

    if ($doneCount -eq $shards.Count) {
        Write-Log "all_shards_complete=true progress_monitor_exiting"
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
