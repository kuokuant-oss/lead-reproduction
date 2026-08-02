#!/usr/bin/env bash
# 唯讀單行狀態。只 stat 目錄與讀 log 尾端,不鎖任何正在寫入的檔案。
O="${1:-$HOME/m5-e6-gputw-probe-results}"
phase=start
[ -f "$O/preflight.json" ] && phase=preflight
[ -f "$O/sentinel_results.json" ] && phase=sentinel
[ -f "$O/single_worker_results.json" ] && phase=single
[ -f "$O/dual_worker_results.json" ] && phase=dual
workers=$(ps -eo args --no-headers 2>/dev/null | grep -c '[m]5_e6_gputw_single_worker' || true)
procs=$(ps -eo args --no-headers 2>/dev/null | grep -c '[m]5_e6_gputw' || true)
rounds=$(ls "$O"/worker_*_w*.json 2>/dev/null | wc -l | tr -d ' ')
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | head -1)
swap=$(free -m 2>/dev/null | awk '/^Swap:/{print $3}')
rss=$(ps -eo rss,comm --no-headers 2>/dev/null | awk '$2=="python"{s+=$1} END{printf "%.1f", s/1048576}')
pf=none
[ -f "$O/preflight.json" ] && { grep -q '"all_passed": true' "$O/preflight.json" && pf=ok || pf=FAILED; }
cv=none
[ -f "$O/compatibility_results.json" ] && cv=$(grep -o '"verdict": "[A-Z_]*"' "$O/compatibility_results.json" | head -1 | cut -d'"' -f4)
printf 'phase=%s workers=%s procs=%s worker_files=%s/6 pf=%s compat=%s gpu=%s rss=%sG swap=%s\n' \
  "$phase" "$workers" "$procs" "$rounds" "$pf" "$cv" "$gpu" "$rss" "$swap"
