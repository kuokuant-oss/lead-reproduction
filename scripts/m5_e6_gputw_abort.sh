#!/usr/bin/env bash
# 停止本任務在遠端的所有 benchmark process,不碰任何其他東西。
#
# 只依命令列比對本 benchmark 的腳本名稱,並排除自己的 PID —— 先前的教訓是
# 比對命令列的 kill 會連自己的 shell 一起殺掉。
set -u
O="${1:-$HOME/m5-e6-gputw-probe-results}"
SELF=$$
found=0
for pid in $(pgrep -f 'm5_e6_gputw_(single_worker|dual_worker|sentinel|preflight)' 2>/dev/null || true); do
  [ "$pid" = "$SELF" ] && continue
  printf 'terminating pid %s\n' "$pid"
  kill "$pid" 2>/dev/null || true
  found=$((found+1))
done
sleep 3
for pid in $(pgrep -f 'm5_e6_gputw_(single_worker|dual_worker|sentinel|preflight)' 2>/dev/null || true); do
  [ "$pid" = "$SELF" ] && continue
  printf 'force killing pid %s\n' "$pid"
  kill -9 "$pid" 2>/dev/null || true
done
remaining=$(pgrep -cf 'm5_e6_gputw_(single_worker|dual_worker|sentinel|preflight)' 2>/dev/null || echo 0)
printf 'terminated=%s remaining=%s\n' "$found" "$remaining"
printf 'GPU 上本任務的 process:\n'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
printf 'ABORTED。結果目錄保留於 %s。請自行關閉 instance 以停止計費。\n' "$O"
