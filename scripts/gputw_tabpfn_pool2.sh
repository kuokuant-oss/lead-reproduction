#!/usr/bin/env bash
# Keep the rented 5090 busy until every context-curve shard is scored.
#
# Replaces gputw_tabpfn_pool.sh, which wedged in production. Two defects, both
# fixed here:
#
#   1. The uploader ran INSIDE the scheduler loop. A 2.6 GiB push therefore
#      blocked launching, polling and pulling for its whole duration. When every
#      shard already on the box had finished, the GPU sat at 0% for 25 minutes
#      while the loop was parked inside scp. Upload is now its own process; the
#      scheduler only ever reads a marker file.
#   2. It pushed with scp, which cannot resume. The gputw.ai SSH gateway drops
#      long transfers (observed at ~1.1 GiB and again at ~40 MiB), so each retry
#      restarted from zero and the 2.6 GiB matrices could never land. Uploads now
#      go through gputw_resumable_push.sh, which appends in 32 MiB blocks over
#      short-lived connections and resumes from the remote size.
#
# Completion is judged only by durable chunk count on the box, never by whether a
# session looks alive: a dead worker leaves a healthy-looking tmux session behind.
#
#   export GPUTW_HOST=pod-xxxx@ssh.gputw.ai
#   bash scripts/gputw_tabpfn_pool2.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${GPUTW_HOST:?set GPUTW_HOST}"
export GPUTW_HOST

SHARD=scripts/gputw_tabpfn_shard.sh
RPUSH=scripts/gputw_resumable_push.sh
LOG=data/processed/m5_tabpfn_pool2.log
STATE=data/processed/.pool2
# Two, measured. An earlier revision of this file defaulted to 1 and justified it
# at length from a single instantaneous nvidia-smi sample that read 100%. That
# sample was worthless: utilization oscillates between 30% and 100% as a worker
# alternates between attention and writing its checkpoint, and the sustained mean
# for one worker is ~69.5%. Sampling once lands wherever the cycle happens to be.
#
# With a second worker, over a 60 s window: GPU 69.5% -> 99%, and total output
# 5.14 -> 6.0 chunks/min while the co-tenant's own rate fell only 22% -- and the
# added chunks are 137-feature ones, which cost far more than the 17-feature
# chunks given up. Peak memory 6,659 MiB of 32,607, so the OOM the old comment
# feared for 50k/137 is not close. Never take a throughput decision from one
# nvidia-smi line; loop it.
SLOTS="${GPUTW_SLOTS:-2}"
POLL=30
mkdir -p "$STATE"

say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

src_dir() {
  if [[ "$2" == "137" ]]; then
    echo "data/processed/m5_tabpfn_137_distributed_context${1}_n8/${3}"
  else
    echo "data/processed/m5_tabpfn_f17_batch0_context${1}_n8/${3}"
  fi
}
dest_dir() { echo "data/processed/m5_tabpfn_f${2}_batch0_context${1}_n8/${3}-results"; }
vault_dir() { echo "/vault/lead-tabpfn/lead_tabpfn_c${1}_b0_${3}_f${2}_n8"; }
work_dir()  { echo "/workspace/lead_tabpfn_c${1}_b0_${3}_f${2}_n8"; }
session()   { echo "tabpfn_c${1}_b0_${3}_f${2}"; }
want_chunks() { [[ "$1" == "head" ]] && echo 253 || echo 254; }

# 5k last, per instruction. Within a context 17 before 137: it uploads eight
# times faster, so the GPU starts earning while the big matrix is still moving.
#
# Shards already scored and pulled are dropped from the queue entirely, because
# the scheduler judges completion by chunk count on the *pod* and /workspace is
# pod-local scratch. Resuming on a replacement pod therefore reads zero chunks
# for work that is finished and sitting on local disk, and would re-upload and
# re-score it at full price -- six shards, including two 2.6 GiB matrices, when
# this was written. The local pull is the durable record of a finished shard, so
# ask it rather than the pod.
JOBS=()
SKIPPED=()
for ctx in 10000 20000 50000 5000; do
  for line in 17 137; do
    for shard in head tail; do
      have="$(ls -1 "$(dest_dir "$ctx" "$line" "$shard")"/chunks/rows_*.npz 2>/dev/null | wc -l)"
      if (( have >= $(want_chunks "$shard") )); then
        SKIPPED+=("${ctx}:${line}:${shard} (${have} chunks local)")
      else
        JOBS+=("${ctx}:${line}:${shard}")
      fi
    done
  done
done

sshq() {
  ssh -p "${GPUTW_PORT:-2222}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$GPUTW_HOST" "$@" 2>/dev/null
}

# ---------------------------------------------------------------- uploader ---
# Serial on purpose: the uplink is the constraint and parallel streams just split
# it. Runs as its own process so a multi-GiB push never stalls the scheduler.
#
# Two passes, and the order matters. Hydration moves no bytes over the uplink --
# it is a local copy on the pod from the account's NFS mount -- so every shard
# the vault already holds is made runnable first. Only then does the first
# multi-GiB push begin. One pass in queue order would instead park the whole
# uploader inside a 32-minute 2.6 GiB push while shards that were already staged
# sat unhydrated and the card did nothing: the same idle-GPU failure the original
# pool had, arriving by a different route.
uploader() {
  local job ctx line shard src vault pending=()

  for job in "${JOBS[@]}"; do
    IFS=: read -r ctx line shard <<<"$job"
    src="$(src_dir "$ctx" "$line" "$shard")"
    if [[ -f "$STATE/${job}.uploaded" ]]; then
      # A marker means the bytes reached /vault, not that this pod can see them.
      # /workspace dies with the pod, so on a replacement pod a marked shard
      # still needs its working copy recreated; skipping outright left cmd_run
      # cd-ing into a directory that did not exist. hydrate re-stages from the
      # vault after checking every file is present at full length, so a vault
      # that did not survive -- or a marker written over a truncated push --
      # falls back to a full upload instead of scoring a short matrix.
      say "HYDRATE ${job}"
      if bash "$SHARD" hydrate "$src" "$ctx" "$line" 0 "$shard" >>"$LOG" 2>&1; then
        say "HYDRATED ${job}"
        continue
      fi
      say "VAULT LOST ${job} -- clearing marker and re-uploading"
      rm -f "$STATE/${job}.uploaded"
    fi
    pending+=("$job")
  done

  # Serial on purpose: the uplink is the constraint and parallel streams just
  # split it (1,237 KiB/s across two versus 1,384 KiB/s for one, measured).
  for job in "${pending[@]:-}"; do
    [[ -n "$job" ]] || continue
    IFS=: read -r ctx line shard <<<"$job"
    src="$(src_dir "$ctx" "$line" "$shard")"
    vault="$(vault_dir "$ctx" "$line" "$shard")"
    say "UPLOAD ${job}"
    if bash "$RPUSH" "$src/features.float32.npy" "$vault/features.float32.npy" >>"$LOG" 2>&1 &&
       bash "$SHARD" push "$src" "$ctx" "$line" 0 "$shard" >>"$LOG" 2>&1; then
      : > "$STATE/${job}.uploaded"
      say "UPLOADED ${job}"
    else
      say "UPLOAD FAILED ${job} -- retrying later"
    fi
  done
  say "=== uploader done ==="
}

# --------------------------------------------------------------- scheduler ---
scheduler() {
  local -A done_seen=() last_launch=() relaunches=()
  # A 137-feature shard memory-maps a 2.6 GiB matrix and loads the foundation
  # model before it can write its first 20,000-row checkpoint, which takes well
  # over a poll interval. cmd_run starts with `tmux kill-session`, so relaunching
  # a shard that is merely still starting up kills it and restarts the clock --
  # a loop that never makes progress and looks like a hung GPU. Observed live:
  # 10000:137:tail was launched six times in four minutes and never got going.
  # Nothing may relaunch a job until this grace period has elapsed.
  local GRACE=300
  while true; do
    local raw
    # R marks a shard this pod can actually run: the working copy exists and is
    # the same length as the vault original. A launch used to be gated on the
    # local .uploaded marker, which only records that bytes reached the vault at
    # some point, on some pod -- true and useless on a replacement pod whose
    # /workspace is empty, and true again for the window while the uploader is
    # still hydrating. Both launch a worker into a missing or half-copied
    # directory. Ask the pod what it has instead.
    raw="$(sshq '
      for d in /workspace/lead_tabpfn_c*_n8; do
        [ -d "$d" ] || continue
        n=$(basename "$d")
        echo "C $n $(ls -1 "$d"/work/chunks/rows_*.npz 2>/dev/null | wc -l)"
        fw=$(stat -c%s "$d/features.float32.npy" 2>/dev/null || echo 0)
        fv=$(stat -c%s "/vault/lead-tabpfn/$n/features.float32.npy" 2>/dev/null || echo 0)
        if [ "$fw" -gt 0 ] && [ "$fw" = "$fv" ] && [ -f "$d/model.portable.tabpfn_fit" ]; then
          echo "R $n 1"
        fi
      done
      tmux ls 2>/dev/null | sed "s/:.*//" | while read -r s; do echo "S $s"; done')" || {
      say "WARN remote unreadable; retrying"; sleep "$POLL"; continue; }

    local -A chunks=() alive=() ready=()
    while read -r kind name value; do
      case "$kind" in
        C) chunks[$name]="$value" ;;
        R) ready[$name]=1 ;;
        S) alive[$name]=1 ;;
      esac
    done <<<"$raw"

    local pending=0 running=0
    for job in "${JOBS[@]}"; do
      IFS=: read -r ctx line shard <<<"$job"
      local rt have want
      rt="$(basename "$(work_dir "$ctx" "$line" "$shard")")"
      have="${chunks[$rt]:-0}"; want="$(want_chunks "$shard")"
      if (( have >= want )); then
        if [[ -z "${done_seen[$job]:-}" ]]; then
          say "COMPLETE ${job} (${have}/${want}) -- pulling"
          bash "$SHARD" pull "$ctx" "$line" 0 "$shard" "$(dest_dir "$ctx" "$line" "$shard")" \
            >>"$LOG" 2>&1 && say "PULLED ${job}" || say "PULL FAILED ${job}"
          done_seen[$job]=1
        fi
        continue
      fi
      pending=$((pending + 1))
      [[ -n "${alive[$(session "$ctx" "$line" "$shard")]:-}" ]] && running=$((running + 1))
    done

    (( pending == 0 )) && { say "=== all shards complete ==="; return 0; }

    local now; now="$(date +%s)"
    for job in "${JOBS[@]}"; do
      (( running >= SLOTS )) && break
      IFS=: read -r ctx line shard <<<"$job"
      local rt have want
      rt="$(basename "$(work_dir "$ctx" "$line" "$shard")")"
      have="${chunks[$rt]:-0}"; want="$(want_chunks "$shard")"
      (( have >= want )) && continue
      [[ -n "${alive[$(session "$ctx" "$line" "$shard")]:-}" ]] && continue
      [[ -n "${ready[$rt]:-}" ]] || continue

      local since=$(( now - ${last_launch[$job]:-0} ))
      if (( ${last_launch[$job]:-0} > 0 && since < GRACE )); then
        continue          # still starting up; killing it now would restart the clock
      fi
      # Past the grace period with the session gone and no new chunk means the
      # worker died on startup rather than being slow. Say so: a silent relaunch
      # loop is what made the last failure look like an idle GPU for 25 minutes.
      if (( ${last_launch[$job]:-0} > 0 )); then
        relaunches[$job]=$(( ${relaunches[$job]:-0} + 1 ))
        say "WARN ${job} died after ${since}s with ${have}/${want} chunks (relaunch #${relaunches[$job]}); check ${rt}/worker.log"
        if (( ${relaunches[$job]} >= 3 )); then
          say "GIVE UP ${job} after 3 failed starts -- leaving the slot for other work"
          continue
        fi
      fi
      say "LAUNCH ${job}"
      last_launch[$job]="$now"
      bash "$SHARD" run "$ctx" "$line" 0 "$shard" >>"$LOG" 2>&1 && running=$((running + 1))
    done
    sleep "$POLL"
  done
}

for entry in "${SKIPPED[@]:-}"; do
  [[ -n "$entry" ]] && say "SKIP ${entry} -- already scored and pulled"
done
if (( ${#JOBS[@]} == 0 )); then
  say "=== nothing left to do: every shard is already scored locally ==="
  exit 0
fi
say "=== pool2 start: ${#JOBS[@]} jobs, ${SLOTS} slots ==="
uploader &
UP=$!
scheduler
kill "$UP" 2>/dev/null
say "=== pool2 finished ==="
