#!/usr/bin/env bash
# Keep two TabPFN workers busy on one gputw.ai box until every shard is scored.
#
# Why two: measured on the RTX 5090, a single 17-feature/10k worker leaves the
# GPU at 29% and 1.6 GiB of 32. Running two shards at once pushes it to 100% and
# yields 1.58x aggregate throughput (f17 kept 62% of its solo rate, f137 kept
# 96%). That turns ~10.3 GPU-hours of sequential work into ~6.5 -- about what a
# second rented box would buy, for nothing.
#
# Why not three: the card is already saturated at two. More slots would just
# split the same throughput and add failure surface.
#
# Upload runs one-at-a-time ahead of compute. At the measured 2.16 MiB/s uplink
# the whole queue is ~2.5 h of transfer against ~6.5 h of compute, so staying a
# shard ahead is enough and a second concurrent scp would only slow the one that
# matters.
#
#   export GPUTW_HOST=pod-xxxx@ssh.gputw.ai
#   nohup bash scripts/gputw_tabpfn_pool.sh > data/processed/pool.out 2>&1 &

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${GPUTW_HOST:?set GPUTW_HOST}"
export GPUTW_HOST

SHARD=scripts/gputw_tabpfn_shard.sh
LOG=data/processed/m5_tabpfn_pool.log
SLOTS=2
POLL=20

# 5k last, per instruction. Within a context 17 before 137: it uploads eight
# times faster, so the GPU starts earning while the big matrix is still moving.
JOBS=()
for ctx in 10000 20000 50000 5000; do
  for line in 17 137; do
    for shard in head tail; do
      JOBS+=("${ctx}:${line}:${shard}")
    done
  done
done

say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

src_dir() {
  local ctx="$1" line="$2" shard="$3"
  if [[ "$line" == "137" ]]; then
    echo "data/processed/m5_tabpfn_137_distributed_context${ctx}_n8/${shard}"
  else
    echo "data/processed/m5_tabpfn_f17_batch0_context${ctx}_n8/${shard}"
  fi
}

expected_chunks() {
  # head is 5,060,000 rows, tail 5,077,155; both at 20,000 per checkpoint.
  [[ "$1" == "head" ]] && echo 253 || echo 254
}

# One round trip per poll, not two per job. Sixteen jobs x two probes x a ~1.5 s
# SSH handshake would be 48 s of pure latency per cycle -- longer than the poll
# interval, so the pool would spend its life in handshakes instead of scheduling.
declare -A CHUNKS=()
declare -A ALIVE=()

refresh_state() {
  local raw
  raw="$(ssh -p "${GPUTW_PORT:-2222}" -o BatchMode=yes \
      -o StrictHostKeyChecking=accept-new "$GPUTW_HOST" '
    for d in /workspace/lead_tabpfn_c*_n8; do
      [ -d "$d" ] || continue
      n=$(ls -1 "$d"/work/chunks/rows_*.npz 2>/dev/null | wc -l)
      echo "C $(basename "$d") $n"
    done
    tmux ls 2>/dev/null | sed "s/:.*//" | while read -r s; do echo "S $s"; done
  ' 2>/dev/null)" || return 1
  CHUNKS=(); ALIVE=()
  while read -r kind name value; do
    case "$kind" in
      C) CHUNKS[$name]="$value" ;;
      S) ALIVE[$name]=1 ;;
    esac
  done <<<"$raw"
  return 0
}

root_name() { echo "lead_tabpfn_c${1}_b0_${3}_f${2}_n8"; }
session_name() { echo "tabpfn_c${1}_b0_${3}_f${2}"; }

session_alive() { [[ -n "${ALIVE[$(session_name "$1" "$2" "$3")]:-}" ]]; }

is_done() {
  local have want
  have="${CHUNKS[$(root_name "$1" "$2" "$3")]:-0}"
  want="$(expected_chunks "$3")"
  (( have >= want ))
}

say "=== pool start: ${#JOBS[@]} jobs, ${SLOTS} slots ==="
declare -A LAUNCHED=()
uploaded_upto=0

while true; do
  if ! refresh_state; then
    say "WARN could not read remote state; retrying"
    sleep "$POLL"
    continue
  fi
  running=0
  pending=0
  for job in "${JOBS[@]}"; do
    IFS=: read -r ctx line shard <<<"$job"
    if is_done "$ctx" "$line" "$shard"; then
      if [[ "${LAUNCHED[$job]:-}" != "done" ]]; then
        say "COMPLETE ${job}"
        LAUNCHED[$job]=done
      fi
      continue
    fi
    pending=$((pending + 1))
    if session_alive "$ctx" "$line" "$shard"; then
      running=$((running + 1))
    fi
  done

  (( pending == 0 )) && { say "=== all shards complete ==="; break; }

  # Fill free slots with the earliest job whose data is already on the box.
  for job in "${JOBS[@]}"; do
    (( running >= SLOTS )) && break
    IFS=: read -r ctx line shard <<<"$job"
    is_done "$ctx" "$line" "$shard" && continue
    session_alive "$ctx" "$line" "$shard" && continue
    [[ "${LAUNCHED[$job]:-}" == "uploaded" ]] || continue
    say "LAUNCH ${job}"
    bash "$SHARD" run "$ctx" "$line" 0 "$shard" >>"$LOG" 2>&1 && running=$((running + 1))
  done

  # Stay one shard ahead on upload. Serial on purpose: the uplink is the
  # constraint, and two parallel scps just split it.
  for job in "${JOBS[@]}"; do
    IFS=: read -r ctx line shard <<<"$job"
    is_done "$ctx" "$line" "$shard" && continue
    [[ -n "${LAUNCHED[$job]:-}" ]] && continue
    say "UPLOAD ${job}"
    if bash "$SHARD" push "$(src_dir "$ctx" "$line" "$shard")" "$ctx" "$line" 0 "$shard" \
        >>"$LOG" 2>&1; then
      LAUNCHED[$job]=uploaded
      say "UPLOADED ${job}"
    else
      say "UPLOAD FAILED ${job} -- will retry next pass"
    fi
    break
  done

  sleep "$POLL"
done
say "=== pool finished ==="
