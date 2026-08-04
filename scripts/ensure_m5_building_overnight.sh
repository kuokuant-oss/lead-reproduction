#!/usr/bin/env bash
set -u

repo=/home/kuant_kuo/projects/lead-reproduction-e3
session=m5-building-overnight
complete="$repo/data/processed/m5_building_curve/supervisor/COMPLETE.json"
log="$repo/data/processed/m5_building_curve/supervisor/overnight.log"
mkdir -p "$(dirname "$log")"

while [[ ! -f "$complete" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
        tmux new-session -d -s "$session" +            "cd '$repo' && exec .venv/bin/python scripts/run_m5_building_curve_overnight.py --retry-delay 120 >> '$log' 2>&1"
        printf '%s watchdog started tmux session %s\n' "$(date --iso-8601=seconds)" "$session" >> "$log"
    fi
    sleep 60
done
