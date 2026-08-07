#!/usr/bin/env bash
set -u

repo=${M5_BUILDING_SENSITIVITY_REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
audit_root=${M5_BUILDING_SENSITIVITY_AUDIT_ROOT:-"$repo/data/processed/m5_building_curve/sensitivity/building_candidate_pilot"}
socket=m5-building-sensitivity-overnight
session=m5-building-sensitivity-supervisor
complete="$audit_root/overnight/COMPLETE.json"
failed="$audit_root/overnight/FAILED.json"
log="$audit_root/overnight/overnight.log"
mkdir -p "$(dirname "$log")"

while [[ ! -f "$complete" && ! -f "$failed" ]]; do
    if pgrep -f "scripts/run_m5_building_candidate_sensitivity_models.py.*--mode formal" >/dev/null; then
        printf '%s existing formal sweep owns execution; watchdog waiting\n' "$(date --iso-8601=seconds)" >> "$log"
        sleep 60
        continue
    fi
    if ! tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
        tmux -L "$socket" new-session -d -s "$session" \
            "cd '$repo' && export PYTHONPATH='$repo/src:$repo/scripts' && export TABPFN_MODEL_CACHE_DIR='/home/kuant_kuo/.cache/tabpfn' && exec .venv/bin/python scripts/run_m5_building_candidate_sensitivity_overnight.py --audit-root '$audit_root' --mode formal --retry-delay 120 --unit-retries 2 --finalize-retries 2 --push-retries 5 --gpu-wait-checks 30 --publish-results >> '$log' 2>&1"
        printf '%s watchdog started bounded-retry supervisor\n' "$(date --iso-8601=seconds)" >> "$log"
    fi
    sleep 60
done

printf '%s watchdog stopped complete=%s failed=%s\n' \
    "$(date --iso-8601=seconds)" "$([[ -f "$complete" ]] && printf yes || printf no)" "$([[ -f "$failed" ]] && printf yes || printf no)" >> "$log"
