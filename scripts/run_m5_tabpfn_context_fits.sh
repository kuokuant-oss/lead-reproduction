#!/usr/bin/env bash
# Fit TabPFN once per (context, feature width) for the training-size curve.
#
# Sequential on purpose. Each fit loads the 20.2M-row M3 frame and builds the
# 120 value-change columns over the training half; two of those at once does not
# fit in 31.6 GB. Running them in parallel trades an hour of wall clock for an
# OOM three fits deep.
#
# Idempotent: a context whose fit_manifest.json already exists is skipped, so
# this can be re-run after an interruption without redoing finished work.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONTEXTS=(5000 10000 20000 50000)
LOG=data/processed/m5_tabpfn_context_fits.log

work_dir() {
  local line="$1" ctx="$2"
  if [[ "$line" == "137" ]]; then
    echo "data/processed/m5_tabpfn_137_full_test_context${ctx}_n8.work"
  else
    echo "data/processed/m5_tabpfn_canonical_full_test_context${ctx}_n8.work"
  fi
}

echo "=== context fits started $(date -Is) ===" | tee -a "$LOG"
failed=0
for ctx in "${CONTEXTS[@]}"; do
  for line in 137 17; do
    dir="$(work_dir "$line" "$ctx")"
    if [[ -f "${dir}/fit_manifest.json" ]]; then
      echo "skip  f${line} ctx=${ctx} (already fitted)" | tee -a "$LOG"
      continue
    fi
    echo "--- fitting f${line} ctx=${ctx} $(date -Is)" | tee -a "$LOG"
    if uv run python "scripts/fit_m5_tabpfn_${line}_context100000.py" \
        --context-rows "$ctx" --n-estimators 8 >>"$LOG" 2>&1; then
      echo "ok    f${line} ctx=${ctx}" | tee -a "$LOG"
      # The portable scaler is what lets one unscaled matrix serve every
      # context, so it is produced next to the fit rather than later.
      uv run python scripts/export_m5_tabpfn_context_scaler.py \
          --work-dir "$dir" >>"$LOG" 2>&1 \
        && echo "ok    f${line} ctx=${ctx} scaler" | tee -a "$LOG" \
        || { echo "FAIL  f${line} ctx=${ctx} scaler" | tee -a "$LOG"; failed=1; }
    else
      echo "FAIL  f${line} ctx=${ctx} -- see $LOG" | tee -a "$LOG"
      failed=1
    fi
  done
done
echo "=== context fits finished $(date -Is), failed=${failed} ===" | tee -a "$LOG"
exit "$failed"
