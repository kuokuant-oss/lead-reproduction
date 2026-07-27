<#
Keep the context curve moving without being asked to advance it.

pool2 already chains one shard to the next on the pod. What it does not do is
notice that a context's two shards are both home and merge them, keep the
CPU-only tree arm fed, or come back after its own process dies. Those were the
seams a human had to sit across. This closes them.

Each pass, in this order:

  1. If pool2 is not running and any shard is still unscored, relaunch it.
     pool2 is resumable by design -- it rebuilds its queue from what is already
     pulled -- so a restart costs one poll interval, not a re-upload.
  2. Merge any context whose head and tail are both fully pulled.
  3. Keep the tree matched-N queue running.
  4. When every shard is pulled, say so loudly: the box bills per second, has no
     stop API, and is never reclaimed.

THE SINGLE MOST IMPORTANT RULE HERE IS THAT IT MUST NEVER START A SECOND POOL2.
Four processes appending to one vault file corrupted a 2.6 GiB matrix once
already, and only the post-upload SHA-256 caught it. Liveness is therefore
judged by inspecting real command lines through Win32_Process rather than by a
PID file that goes stale, or by `pgrep` under Git Bash -- which reports pipeline
subshells with the parent's command line and so cannot tell one pool from two.

    powershell -File scripts/m5_context_curve_supervisor.ps1 `
        -GputwHost pod-xxxx@ssh.gputw.ai
#>
param(
  [Parameter(Mandatory = $true)][string]$GputwHost,
  [int]$PollSeconds = 60
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$log = Join-Path $repo 'data\processed\m5_supervisor.log'
$bash = 'C:\Program Files\Git\bin\bash.exe'
$env:GPUTW_HOST = $GputwHost

# Nothing here may depend on a console. This process is started hidden, so its
# stdout goes to a console nobody drains; a writer that blocks on it wedges with
# ~0 s of CPU and no children, which is indistinguishable from "working, just
# slow". The tree queue died exactly that way once. Log to the file, retry if a
# reader momentarily holds it, and never let logging kill the loop.
function Say([string]$message) {
  $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'), $message
  foreach ($attempt in 1..5) {
    try { Add-Content -Path $log -Value $line -Encoding utf8 -ErrorAction Stop; return }
    catch { Start-Sleep -Milliseconds 200 }
  }
}

# Only one supervisor. A second one would double every action below, including
# the pool2 relaunch this whole file exists to keep singular.
$mine = $PID
# Matched on the -File invocation, not on the bare name: a console that merely
# greps for this script also carries its name, and treating that as a running
# supervisor would refuse to start the only one there is.
$others = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.ProcessId -ne $mine -and $_.CommandLine -like '*-File*m5_context_curve_supervisor.ps1*' }
if ($others) {
  Say "ABORT another supervisor is already running (pid $($others[0].ProcessId))"
  exit 1
}

# The image name is part of the test, not decoration. Every shell that greps for
# these scripts -- an operator's console, this file's own launcher -- carries the
# needle in its command line and would otherwise register as the job itself.
# Reading that as "pool2 is alive" is the failure that matters: it suppresses the
# restart this supervisor exists to perform, silently, for as long as the shell
# is open. Only bash.exe runs these.
# A merged artifact counts as done only if it opens. Existence is not enough: the
# merge writes the npz in place rather than atomically, so a crash -- or anything
# reading it mid-write -- leaves a truncated file that Test-Path happily calls
# finished, and the cell is then skipped forever. Opening the zip is the exact
# test for that failure, and it self-heals: a partial file fails, gets remerged,
# and passes. (A reader hitting the same window is what broke the first attempt
# to score this cell, so the window is real, not theoretical.)
Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
function Test-MergedOk([string]$path) {
  if (-not (Test-Path $path)) { return $false }
  try {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($path)
    $count = $zip.Entries.Count
    $zip.Dispose()
    return $count -gt 0
  } catch { return $false }
}

function Test-Live([string]$needle) {
  $procs = Get-CimInstance Win32_Process -Filter "Name='bash.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*$needle*" -and $_.ProcessId -ne $mine }
  return [bool]$procs
}

# Hidden processes get their output redirected to files, for the reason in Say.
function Start-Detached([string]$file, [string[]]$arguments, [string]$tag) {
  $out = Join-Path $repo "data\processed\m5_supervisor_${tag}.out"
  $err = Join-Path $repo "data\processed\m5_supervisor_${tag}.err"
  Start-Process -FilePath $file -ArgumentList $arguments -WorkingDirectory $repo `
    -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err | Out-Null
}

# head is 253 chunks, tail 254; together they cover all 10,137,155 holdout rows.
function Get-LocalChunks([int]$ctx, [int]$line, [string]$shard) {
  $dir = Join-Path $repo "data\processed\m5_tabpfn_f${line}_batch0_context${ctx}_n8\${shard}-results\chunks"
  if (-not (Test-Path $dir)) { return 0 }
  return @(Get-ChildItem -Path $dir -Filter 'rows_*.npz' -ErrorAction SilentlyContinue).Count
}
function Get-WantChunks([string]$shard) { if ($shard -eq 'head') { 253 } else { 254 } }

$cells = @()
foreach ($ctx in 5000, 10000, 20000, 50000) {
  foreach ($line in 17, 137) { $cells += , @{ ctx = $ctx; line = $line } }
}

Say "=== supervisor start (pid $mine, poll ${PollSeconds}s, host $GputwHost) ==="
$announcedSafe = $false

while ($true) {
  $shardsLeft = 0
  foreach ($cell in $cells) {
    foreach ($shard in 'head', 'tail') {
      if ((Get-LocalChunks $cell.ctx $cell.line $shard) -lt (Get-WantChunks $shard)) { $shardsLeft++ }
    }
  }

  # ---- 1. pool2 ----------------------------------------------------------
  $poolLive = Test-Live 'gputw_tabpfn_pool2.sh'
  if ($shardsLeft -gt 0 -and -not $poolLive) {
    Say "RESTART pool2 ($shardsLeft shard(s) unscored)"
    Start-Detached $bash @('scripts/gputw_tabpfn_pool2.sh') 'pool2'
  }

  # ---- 2. merge anything complete ---------------------------------------
  foreach ($cell in $cells) {
    $ctx = $cell.ctx; $line = $cell.line
    $out = Join-Path $repo "data\processed\m5_tabpfn_${line}_full_test_context${ctx}_n8_predictions.npz"
    if (Test-MergedOk $out) { continue }
    $head = Get-LocalChunks $ctx $line 'head'
    $tail = Get-LocalChunks $ctx $line 'tail'
    if ($head -lt 253 -or $tail -lt 254) { continue }
    if (Test-Live 'merge_m5_tabpfn_full_test') { continue }
    Say "MERGE f${line} c${ctx} (head $head, tail $tail)"
    # --roots is mandatory: the default path expects six batches, but the
    # context curve uses batch0 only -- head and tail already cover all
    # 10,137,155 rows -- so the default raises FileNotFoundError on batch1.
    $root = "data/processed/m5_tabpfn_f${line}_batch0_context${ctx}_n8"
    $merged = & uv run python scripts/merge_m5_tabpfn_full_test.py `
      --line $line --context-rows $ctx --roots $root 2>&1
    if ($LASTEXITCODE -eq 0) {
      Say "MERGED f${line} c${ctx}"
      ($merged | Select-Object -Last 12) | ForEach-Object { Say "    $_" }
    } else {
      Say "MERGE FAILED f${line} c${ctx} (exit $LASTEXITCODE)"
      ($merged | Select-Object -Last 20) | ForEach-Object { Say "    $_" }
    }
  }

  # ---- 3. tree arm -------------------------------------------------------
  # Free of the GPU and of the rented box, so it runs whenever nothing else is
  # already holding the frame in memory. The queue script skips finished cells,
  # which makes starting it idempotent.
  $treeLive = (Test-Live 'run_m5_tree_matched_queue.sh') -or
              (Test-Live 'run_m5_tree_ensemble_matched_context.py')
  if (-not $treeLive) {
    $queueDone = Test-Path (Join-Path $repo 'data\processed\m5_tree_matched_queue.DONE')
    if (-not $queueDone) {
      Say 'START tree matched-N queue'
      Start-Detached $bash @('scripts/run_m5_tree_matched_queue.sh') 'tree'
    }
  }

  # ---- 4. the money ------------------------------------------------------
  if ($shardsLeft -eq 0) {
    if (-not $announcedSafe) {
      Say '=== ALL SHARDS PULLED -- SAFE TO STOP THE INSTANCE (dashboard; it bills per second and is never reclaimed) ==='
      $announcedSafe = $true
    }
    $treeStillGoing = (Test-Live 'run_m5_tree_matched_queue.sh') -or
                      (Test-Live 'run_m5_tree_ensemble_matched_context.py')
    if (-not $treeStillGoing -and (Test-Path (Join-Path $repo 'data\processed\m5_tree_matched_queue.DONE'))) {
      Say '=== supervisor done: every shard scored, merged, and the tree arm finished ==='
      break
    }
  }

  Start-Sleep -Seconds $PollSeconds
}
