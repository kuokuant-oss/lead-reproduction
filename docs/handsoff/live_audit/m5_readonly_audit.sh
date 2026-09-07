#!/usr/bin/env bash
# M5 Building-Count V5 fixed-50K —— 純唯讀稽核腳本
# 只讀取狀態，不啟動、不停止、不修改任何實驗檔案或 Colab session。
# 用法（在 Windows PowerShell 或 CMD 執行）：
#   wsl.exe -d Ubuntu -u kuant_kuo -- bash /mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/live_audit/m5_readonly_audit.sh

set -u
export PYTHONPATH=src:scripts
REPO=/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k
PY=/home/kuant_kuo/projects/lead-reproduction/.venv/bin/python
ROOT="$REPO/data/processed/m5_building_curve/v5_fixed_50k"
OUTDIR=/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/live_audit
mkdir -p "$OUTDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$OUTDIR/audit_$STAMP.txt"

{
echo "=== M5 V5 fixed-50K read-only audit ==="
echo "time: $(date -Is) ($(TZ=Asia/Taipei date '+%Y-%m-%d %H:%M:%S %Z'))"

echo; echo "--- [1] git HEAD / tracked clean ---"
cd "$REPO" || exit 1
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=no

echo; echo "--- [2] tmux sessions ---"
tmux ls 2>&1

echo; echo "--- [3] scheduler process ---"
pgrep -af 'run_m5_building_count_v5.py' 2>&1

echo; echo "--- [4] model cell processes ---"
pgrep -af 'run_m5_building_count_v5_tabpfn_cell.py|_tree_cell.py|tabpfn' 2>&1
echo "-- process tree --"
for p in $(pgrep -f 'run_m5_building_count_v5.py' 2>/dev/null); do ps -o pid,ppid,etime,pcpu,rss,cmd --forest -g "$(ps -o sid= -p "$p" | tr -d ' ')" 2>/dev/null; done

echo; echo "--- [5] supervisor status.json ---"
sed -n '1,200p' "$ROOT/supervisor/status.json" 2>&1

echo; echo "--- [6] supervisor heartbeat.json ---"
sed -n '1,120p' "$ROOT/supervisor/heartbeat.json" 2>&1

echo; echo "--- [7] active cell heartbeat (最新修改的 heartbeat.json) ---"
ACTIVE=$(find "$ROOT/model_runs" -name heartbeat.json -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -3 | awk '{print $2}')
for h in $ACTIVE; do
  echo "== $h ($(date -r "$h" -Is)) =="
  sed -n '1,160p' "$h"
done

echo; echo "--- [8] 最近 chunk 落盤時間 (推 sec/chunk) ---"
CELLDIR=$(echo "$ACTIVE" | head -1 | xargs -r dirname)
if [ -n "${CELLDIR:-}" ]; then
  echo "cell dir: $CELLDIR"
  find "$CELLDIR" -type f \( -name '*chunk*' -o -name '*.npz' \) -printf '%T@ %TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | sort -n | tail -25 | \
    awk '{t=$1; if(prev){printf "%s  delta=%.1fs  %s\n",$3,t-prev,$5} else {printf "%s  %s\n",$3,$5}; prev=t}'
fi

echo; echo "--- [9] FAILED.json census (formal root) ---"
find "$ROOT" -name FAILED.json -type f -print 2>&1
echo "-- 其他 root (K725 / phase3 / k200_3cells_v2) --"
find "$REPO/data/processed/m5_building_curve" -maxdepth 3 -name 'FAILED.json' -type f -printf '%p  (%TY-%Tm-%Td %TH:%TM)\n' 2>&1

echo; echo "--- [10] COMPLETE/FINALIZED markers of the two pending-finalization queues ---"
for r in v5_fixed_50k_phase3_colab v5_fixed_50k_k200_colab_3cells_v2 v5_fixed_50k_k725_colab; do
  d="$REPO/data/processed/m5_building_curve/$r/formal_queue"
  echo "== $r =="; ls -la "$d" 2>&1 | sed -n '1,40p'
done

echo; echo "--- [11] scheduler complete() 即時 84-unit census (read-only import) ---"
"$PY" - <<'PYEOF' 2>&1
import sys, json
sys.path[:0] = ["src", "scripts"]
try:
    import run_m5_building_count_v5 as R
except Exception as e:
    print("IMPORT_FAILED:", type(e).__name__, e); raise SystemExit(0)
cands = [n for n in dir(R) if "unit" in n.lower() or "queue" in n.lower() or "complete" in n.lower()]
print("module symbols of interest:", cands)
PYEOF

echo; echo "--- [12] COMPLETE.json census by budget ---"
find "$ROOT/model_runs" -name COMPLETE.json -type f 2>/dev/null | sed 's#.*/model_runs/##' | sort | tee /tmp/m5_complete_list.txt | wc -l
awk -F/ '{print $NF}' /tmp/m5_complete_list.txt | sed 's#/COMPLETE.json##' | sort | uniq -c
echo "-- 全部路徑 --"; cat /tmp/m5_complete_list.txt

echo; echo "--- [13] colab sessions (只查詢) ---"
HOME=/home/kuant_kuo/.colab-tony OAUTHLIB_RELAX_TOKEN_SCOPE=1 /home/kuant_kuo/.local/bin/colab sessions 2>&1 | sed -n '1,40p'

echo; echo "=== end of audit ==="
} > "$OUT" 2>&1

echo "audit written to: $OUT"
