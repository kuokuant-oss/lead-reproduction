#!/usr/bin/env bash
# GPUtw RTX PRO 6000 benchmark launcher。
#
# 連線資料只從環境變數讀,絕不寫進 repository、bundle、JSON 或報告:
#   GPUTW_HOST  GPUTW_USER  GPUTW_PORT  GPUTW_SSH_KEY
#
# 第一次連線會先顯示 server fingerprint 供人類確認,並把 host key 存進未追蹤的
# runtime 目錄,之後一律用 StrictHostKeyChecking=yes 比對。刻意不使用
# StrictHostKeyChecking=no —— 那等於放棄辨識對方是誰。
set -eu

: "${GPUTW_HOST:?請設定 GPUTW_HOST}"
: "${GPUTW_USER:?請設定 GPUTW_USER}"
: "${GPUTW_PORT:?請設定 GPUTW_PORT}"
: "${GPUTW_SSH_KEY:?請設定 GPUTW_SSH_KEY(本機檔案路徑)}"

RUNTIME="${GPUTW_RUNTIME_DIR:-$HOME/.gputw-probe-runtime}"
KNOWN_HOSTS="$RUNTIME/known_hosts"
BUNDLE="${GPUTW_BUNDLE:?請設定 GPUTW_BUNDLE(bundle 檔案路徑)}"
BUNDLE_SHA="${GPUTW_BUNDLE_SHA256:?請設定 GPUTW_BUNDLE_SHA256}"
REMOTE_STAGE="${GPUTW_REMOTE_STAGE:-\$HOME/m5-e6-gputw-probe}"
REMOTE_OUT="${GPUTW_REMOTE_OUT:-\$HOME/m5-e6-gputw-probe-results}"
LOCAL_OUT="${GPUTW_LOCAL_OUT:?請設定 GPUTW_LOCAL_OUT}"

mkdir -p "$RUNTIME"
chmod 700 "$RUNTIME"

if [ ! -f "$GPUTW_SSH_KEY" ]; then
  printf 'SSH key 不存在:%s\n' "$GPUTW_SSH_KEY" >&2
  exit 2
fi
PERM=$(stat -c '%a' "$GPUTW_SSH_KEY" 2>/dev/null || stat -f '%A' "$GPUTW_SSH_KEY")
case "$PERM" in
  600|400|0600|0400) ;;
  *) printf 'SSH key 權限為 %s,必須是 0600 或更嚴格\n' "$PERM" >&2; exit 2;;
esac

SSH_OPTS="-p $GPUTW_PORT -i $GPUTW_SSH_KEY -o IdentitiesOnly=yes -o BatchMode=yes"
SSH_OPTS="$SSH_OPTS -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes"
SSH_OPTS="$SSH_OPTS -o ServerAliveInterval=20 -o ServerAliveCountMax=3"
SSH_OPTS="$SSH_OPTS -o ConnectTimeout=30"

if [ ! -s "$KNOWN_HOSTS" ]; then
  printf '=== 第一次連線,以下是 %s:%s 的 host fingerprint ===\n' "$GPUTW_HOST" "$GPUTW_PORT"
  ssh-keyscan -p "$GPUTW_PORT" -H "$GPUTW_HOST" > "$KNOWN_HOSTS" 2>/dev/null
  chmod 600 "$KNOWN_HOSTS"
  ssh-keygen -lf "$KNOWN_HOSTS" || true
  printf '=== fingerprint 已存入未追蹤目錄 %s ===\n\n' "$KNOWN_HOSTS"
fi

run() { ssh $SSH_OPTS "$GPUTW_USER@$GPUTW_HOST" "$@"; }

printf '[1/7] 身分與 GPU 確認\n'
run "hostname; uname -m; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader"
GPU=$(run "nvidia-smi --query-gpu=name --format=csv,noheader" | head -1)
case "$GPU" in
  *"RTX PRO 6000"*) printf '  GPU = %s\n' "$GPU";;
  *) printf '  GPU 不是 RTX PRO 6000(%s),依規定停止\n' "$GPU" >&2; exit 3;;
esac

printf '[2/7] 建立乾淨 remote staging\n'
run "rm -rf $REMOTE_STAGE $REMOTE_OUT && mkdir -p $REMOTE_STAGE $REMOTE_OUT"

printf '[3/7] 上傳 bundle 並驗證 digest\n'
scp -P "$GPUTW_PORT" -i "$GPUTW_SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o UserKnownHostsFile="$KNOWN_HOSTS" -o StrictHostKeyChecking=yes \
    "$BUNDLE" "$GPUTW_USER@$GPUTW_HOST:$REMOTE_STAGE/"
REMOTE_SHA=$(run "sha256sum $REMOTE_STAGE/$(basename "$BUNDLE") | cut -d' ' -f1")
if [ "$REMOTE_SHA" != "$BUNDLE_SHA" ]; then
  printf '  bundle digest 不符\n    本機: %s\n    遠端: %s\n' "$BUNDLE_SHA" "$REMOTE_SHA" >&2
  exit 4
fi
printf '  digest 相符\n'

printf '[4/7] 解壓並逐檔驗證\n'
run "bash $REMOTE_STAGE/../m5_e6_gputw_remote_setup.sh" 2>/dev/null || \
  run "cd $REMOTE_STAGE && tar --use-compress-program=unzstd -xf $(basename "$BUNDLE") 2>/dev/null || tar -xzf $(basename "$BUNDLE")"
run "cd $REMOTE_STAGE && ls -1 | head -30"

printf '[5/7] preflight\n'
run "cd $REMOTE_STAGE && python m5_e6_gputw_preflight.py --bundle-root . --out $REMOTE_OUT"

printf '[6/7] compatibility sentinel\n'
run "cd $REMOTE_STAGE && python m5_e6_gputw_sentinel.py --bundle-root . --out $REMOTE_OUT"

printf '[7/7] 單 worker,然後雙 worker\n'
run "cd $REMOTE_STAGE && python m5_e6_gputw_single_worker.py --bundle-root . --out $REMOTE_OUT"
run "cd $REMOTE_STAGE && python m5_e6_gputw_dual_worker.py --bundle-root . --out $REMOTE_OUT --single-results $REMOTE_OUT/single_worker_results.json"

printf '收尾:確認沒有殘留 worker\n'
run "ps -eo args --no-headers | grep -c '[m]5_e6_gputw' || true"

printf '下載結果\n'
mkdir -p "$LOCAL_OUT"
scp -P "$GPUTW_PORT" -i "$GPUTW_SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o UserKnownHostsFile="$KNOWN_HOSTS" -o StrictHostKeyChecking=yes \
    "$GPUTW_USER@$GPUTW_HOST:$REMOTE_OUT/*.json" "$LOCAL_OUT/"

printf '\n完成。請記得自行關閉 GPUtw instance 以停止計費。\n'
