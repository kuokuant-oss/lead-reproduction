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
SLOTS=2
POLL=30
mkdir -p "$STATE"

# 5k last, per instruction. Within a context 17 before 137: it uploads eight
# times faster, so the GPU starts earning while the big matrix is still moving.
JOBS=()
for ctx in 10000 20000 50000 5000; do
  for line in 17 137; do
    for shard in head tail; do JOBS+=("${ctx}:${line}:${shard}"); done
  done
done

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

sshq() {
  ssh -p "${GPUTW_PORT:-2222}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "$GPUTW_HOST" "$@" 2>/dev/null
}

# ---------------------------------------------------------------- uploader ---
# Serial on purpose: the uplink is the constraint and parallel streams just split
# it. Runs as its own process so a multi-GiB push never stalls the scheduler.
uploader() {
  for job in "${JOBS[@]}"; do
    IFS=: read -r ctx line shard <<<"$job"
    [[ -f "$STATE/${job}.uploaded" ]] && continue
    local src vault
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
    raw="$(sshq '
      for d in /workspace/lead_tabpfn_c*_n8; do
        [ -d "$d" ] || continue
        echo "C $(basename "$d") $(ls -1 "$d"/work/chunks/rows_*.npz 2>/dev/null | wc -l)"
      done
      tmux ls 2>/dev/null | sed "s/:.*//" | while read -r s; do echo "S $s"; done')" || {
      say "WARN remote unreadable; retrying"; sleep "$POLL"; continue; }

    local -A chunks=() alive=()
    while read -r kind name value; do
      case "$kind" in C) chunks[$name]="$value" ;; S) alive[$name]=1 ;; esac
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
      [[ -f "$STATE/${job}.uploaded" ]] || continue

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

say "=== pool2 start: ${#JOBS[@]} jobs, ${SLOTS} slots ==="
uploader &
UP=$!
scheduler
kill "$UP" 2>/dev/null
say "=== pool2 finished ==="
