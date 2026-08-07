#!/usr/bin/env bash
set -u

repo=${M5_BUILDING_V2_REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
audit_root=${M5_BUILDING_V2_AUDIT_ROOT:-"$repo/data/processed/m5_building_curve/sensitivity/building_candidate_pilot"}
out_root=${M5_BUILDING_V2_OUT_ROOT:-"$repo/data/processed/m5_building_curve/v2"}
sweep=building_seed_sweep_42-43-44-45-46
supervisor_root="$out_root/$sweep/overnight"
socket=m5-building-v2-overnight
session=m5-building-v2-supervisor
complete="$supervisor_root/COMPLETE.json"
failed="$supervisor_root/FAILED.json"
log="$supervisor_root/overnight.log"
mkdir -p "$(dirname "$log")"

while [[ ! -f "$complete" && ! -f "$failed" ]]; do
    if pgrep -f "scripts/run_m5_building_count_v2_tree_cell.py.*--mode formal" >/dev/null ||
       pgrep -f "scripts/run_m5_building_curve_tabpfn_cell.py.*--mode formal" >/dev/null; then
        printf '%s existing V2 formal cell owns execution; watchdog waiting\n'             "$(date --iso-8601=seconds)" >> "$log"
        sleep 60
        continue
    fi
    if ! tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
        tmux -L "$socket" new-session -d -s "$session"             "cd '$repo' && export PYTHONPATH='$repo/src:$repo/scripts' && export TABPFN_MODEL_CACHE_DIR='/home/kuant_kuo/.cache/tabpfn' && exec .venv/bin/python scripts/run_m5_building_count_v2_overnight.py --audit-root '$audit_root' --out-root '$out_root' --mode formal --retry-delay 120 --unit-retries 2 --finalize-retries 2 --push-retries 5 --git-push-timeout 120 --gpu-wait-checks 30 --publish-results >> '$log' 2>&1"
        printf '%s watchdog started V2 bounded-retry supervisor\n'             "$(date --iso-8601=seconds)" >> "$log"
    fi
    sleep 60
done

printf '%s watchdog stopped complete=%s failed=%s\n'     "$(date --iso-8601=seconds)"     "$([[ -f "$complete" ]] && printf yes || printf no)"     "$([[ -f "$failed" ]] && printf yes || printf no)" >> "$log"
