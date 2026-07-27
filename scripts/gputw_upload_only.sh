#!/usr/bin/env bash
# Stage shards into /vault without scheduling any GPU work.
#
# Used when winding a session down: the pod bills until it is stopped, and
# uploads land on /vault, which is account-scoped NFS (192.168.7.2) rather than
# the pod-local /workspace. Bytes pushed during the wind-down therefore survive
# the pod and a later run finds them already there, so the remaining billed
# minutes buy something instead of nothing.
#
# Deliberately does NOT launch workers. Running the full pool here would start a
# fresh shard the moment a slot freed and extend exactly the GPU time being wound
# down. Interrupting this script is safe: gputw_resumable_push.sh resumes from the
# remote size on the next run.
#
#   export GPUTW_HOST=pod-xxxx@ssh.gputw.ai
#   bash scripts/gputw_upload_only.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${GPUTW_HOST:?set GPUTW_HOST}"
export GPUTW_HOST

LOG=data/processed/m5_tabpfn_upload_only.log
STATE=data/processed/.pool2
mkdir -p "$STATE"

# Cheapest-to-finish first: completing a cell whose other shard is already staged
# is worth more next session than starting a new one, and the 17-feature matrices
# move eight times faster than the 137-feature ones.
JOBS=(
  "20000:137:tail"
  "50000:17:head" "50000:17:tail" "5000:17:head" "5000:17:tail"
  "50000:137:head" "50000:137:tail" "5000:137:head" "5000:137:tail"
)

say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

src_dir() {
  if [[ "$2" == "137" ]]; then
    echo "data/processed/m5_tabpfn_137_distributed_context${1}_n8/${3}"
  else
    echo "data/processed/m5_tabpfn_f17_batch0_context${1}_n8/${3}"
  fi
}
vault_dir() { echo "/vault/lead-tabpfn/lead_tabpfn_c${1}_b0_${3}_f${2}_n8"; }

say "=== upload-only start: ${#JOBS[@]} shards, no workers will be launched ==="
for job in "${JOBS[@]}"; do
  IFS=: read -r ctx line shard <<<"$job"
  [[ -f "$STATE/${job}.uploaded" ]] && { say "SKIP ${job} (already staged)"; continue; }
  src="$(src_dir "$ctx" "$line" "$shard")"
  say "UPLOAD ${job}"
  if bash scripts/gputw_resumable_push.sh \
        "$src/features.float32.npy" \
        "$(vault_dir "$ctx" "$line" "$shard")/features.float32.npy" >>"$LOG" 2>&1 &&
     bash scripts/gputw_tabpfn_shard.sh push "$src" "$ctx" "$line" 0 "$shard" >>"$LOG" 2>&1; then
    : > "$STATE/${job}.uploaded"
    say "UPLOADED ${job}"
  else
    say "UPLOAD INCOMPLETE ${job} (resumes from remote size next run)"
  fi
done
say "=== upload-only finished ==="
