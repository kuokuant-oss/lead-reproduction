#!/usr/bin/env bash
# Export every context's head/tail shards, in the order they will be uploaded.
#
# Sequential, like the fits: each export materialises the 10.1M x 137 test
# matrix, and two at once does not fit in 31.6 GB. This is also why the tree arm
# waits -- it builds the same matrix.
#
# Order matches the run schedule: 10k, 20k, 50k, then 5k last, and within each
# context the 17-feature line first. 17 uploads eight times faster and computes
# three times faster, so putting it first gets the rented GPU working sooner and
# hides the 137 upload behind the 17 compute.
#
# Idempotent: a shard root that already holds both head and tail features is
# skipped, so this survives an interruption.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLAN=data/processed/m5_tabpfn_ctxcurve_single_batch_plan.json
LOG=data/processed/m5_tabpfn_context_exports.log

f137_root() { echo "data/processed/m5_tabpfn_137_distributed_context${1}_n8"; }
f17_root()  { echo "data/processed/m5_tabpfn_f17_batch0_context${1}_n8"; }

complete() {
  [[ -f "$1/head/features.float32.npy" && -f "$1/tail/features.float32.npy" ]]
}

echo "=== context exports started $(date -Is) ===" | tee -a "$LOG"
failed=0
for ctx in 10000 20000 50000 5000; do
  for line in 17 137; do
    if [[ "$line" == "137" ]]; then
      root="$(f137_root "$ctx")"
      cmd=(uv run python scripts/export_m5_tabpfn_137_shards.py
           --context-rows "$ctx" --n-estimators 8)
    else
      root="$(f17_root "$ctx")"
      # MSYS_NO_PATHCONV: Git Bash rewrites a POSIX absolute argument into a
      # Windows path, so --remote-prefix /workspace arrived as
      # "C:/Program Files/Git/workspace" and got baked into init_params.json.
      cmd=(env MSYS_NO_PATHCONV=1 uv run python
           scripts/export_m5_tabpfn_17_batch_shards.py
           --batches 0 --context-rows "$ctx" --plan "$PLAN"
           --remote-prefix /workspace --force)
    fi
    if complete "$root"; then
      echo "skip  f${line} ctx=${ctx} (already exported)" | tee -a "$LOG"
      continue
    fi
    echo "--- exporting f${line} ctx=${ctx} $(date -Is)" | tee -a "$LOG"
    if "${cmd[@]}" >>"$LOG" 2>&1; then
      # Ship the portable scaler next to the shard so the worker can apply it
      # at predict time if the matrix is ever sent unscaled.
      if [[ "$line" == "137" ]]; then
        src="data/processed/m5_tabpfn_137_full_test_context${ctx}_n8.work/scaler.npz"
      else
        src="data/processed/m5_tabpfn_canonical_full_test_context${ctx}_n8.work/scaler.npz"
      fi
      echo "ok    f${line} ctx=${ctx} -> ${root}" | tee -a "$LOG"
      du -sh "$root" 2>/dev/null | tee -a "$LOG"
      [[ -f "$src" ]] || echo "WARN  missing ${src}" | tee -a "$LOG"
    else
      echo "FAIL  f${line} ctx=${ctx} -- see $LOG" | tee -a "$LOG"
      failed=1
    fi
  done
done
echo "=== context exports finished $(date -Is), failed=${failed} ===" | tee -a "$LOG"
exit "$failed"
