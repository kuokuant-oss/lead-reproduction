#!/usr/bin/env bash
# Drive one TabPFN shard on a gputw.ai instance over plain SSH.
#
# gputw.ai rents a dedicated, non-preemptible box reachable only by SSH. That
# removes almost everything the Colab runbooks needed: no keep-alive, no session
# reaper, no supervisor rebuilding a recycled VM, no 64 MB upload chunking, no
# orphan-assignment recovery. What remains is push, run, watch, pull.
#
# scp, not rsync: Windows Git Bash ships no rsync, and the control side of this
# run is a Windows box. Idempotence is recovered by comparing SHA-256 per file
# and skipping what already matches, so an interrupted push resumes at file
# granularity instead of restarting a multi-GiB transfer.
#
# Two things gputw.ai adds that Colab did not have:
#   1. Nothing reclaims an idle box. Per-second billing plus no preemption means
#      a forgotten instance bills until you stop it. `status` prints uptime for
#      exactly this reason.
#   2. Persistence across a stop is NOT documented. Treat /workspace as scratch:
#      never stop an instance before `pull` reports every chunk retrieved.
#
# Usage:
#   gputw_tabpfn_shard.sh push   <shard-dir> <context> <line> <batch> <head|tail>
#   gputw_tabpfn_shard.sh run    <context> <line> <batch> <head|tail> [extra args]
#   gputw_tabpfn_shard.sh status <context> <line> <batch> <head|tail>
#   gputw_tabpfn_shard.sh pull   <context> <line> <batch> <head|tail> <dest-dir>
#
# The host comes from $GPUTW_HOST and $GPUTW_PORT (default 2222), e.g.
#   export GPUTW_HOST=pod-xxxx@ssh.gputw.ai

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${REPO_ROOT}/.tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt"
WORKER="${REPO_ROOT}/scripts/run_m5_tabpfn_portable_shard.py"
PORT="${GPUTW_PORT:-2222}"
# One shared copy of the 203 MB foundation checkpoint. At 2 MiB/s uplink,
# re-sending it per shard would cost ~19 minutes of pure waste across twelve.
REMOTE_CKPT="/workspace/tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt"
VAULT_CKPT="/vault/lead-tabpfn/cache/tabpfn-v3-classifier-v3_default.ckpt"

die() { echo "ERROR: $*" >&2; exit 1; }
: "${GPUTW_HOST:?set GPUTW_HOST, e.g. pod-xxxx@ssh.gputw.ai}"

# Two boxes running heads and tails in parallel is the proven dual-shard shape.
# They usually share one key -- paste the same public key into both deploy forms
# -- but GPUTW_KEY covers the case where a box was given its own.
KEYARG=()
[[ -n "${GPUTW_KEY:-}" ]] && KEYARG=(-i "$GPUTW_KEY" -o IdentitiesOnly=yes)

sshx() {
  ssh -p "$PORT" "${KEYARG[@]}" -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new "$GPUTW_HOST" "$@"
}
scpx() {
  scp -P "$PORT" "${KEYARG[@]}" -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new -q "$@"
}

# Remote root carries context, batch, shard, feature width and estimator count.
# Every context scores the same rows with the same feature count, so two
# contexts differ only in the scores themselves -- a --resume that crossed that
# boundary would be finite, complete and silently the wrong experiment.
remote_root() {
  echo "/workspace/lead_tabpfn_c${1}_b${3}_${4}_f${2}_n8"
}

# Same leaf name under the account-scoped NFS mount, so a replacement pod finds
# its inputs by the identical path.
vault_root() {
  echo "/vault/lead-tabpfn/lead_tabpfn_c${1}_b${3}_${4}_f${2}_n8"
}

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

# scp a file only when the remote copy is missing or differs. Makes push
# re-runnable after a dropped connection without resending what landed.
push_if_changed() {
  local src="$1" dest="$2" label="$3"
  local want got
  want="$(sha_of "$src")"
  got="$(sshx "sha256sum '${dest}' 2>/dev/null | cut -d' ' -f1" || true)"
  if [[ "$want" == "$got" ]]; then
    echo "    skip $label (already matches)"
    return 0
  fi
  echo "    send $label ($(du -h "$src" | cut -f1))"
  scpx "$src" "${GPUTW_HOST}:${dest}"
  got="$(sshx "sha256sum '${dest}' | cut -d' ' -f1")"
  [[ "$want" == "$got" ]] || die "$label digest mismatch after upload"
  echo "    ok   $label"
}

cmd_push() {
  local src="$1" context="$2" line="$3" batch="$4" shard="$5"
  local root; root="$(remote_root "$context" "$line" "$batch" "$shard")"
  local vault; vault="$(vault_root "$context" "$line" "$batch" "$shard")"
  [[ -d "$src" ]] || die "shard dir not found: $src"
  for required in features.float32.npy metadata.npz model.portable.tabpfn_fit; do
    [[ -f "$src/$required" ]] || die "missing $required in $src"
  done

  # Uploads land in /vault, not /workspace. /vault is NFS bound to the *account*
  # (3.7 TB at 192.168.7.2), so it outlives the pod and every other pod the
  # account starts mounts the same tree. /workspace is pod-local NVMe and dies
  # with the pod. At a 2.16 MiB/s uplink, losing a pod with 18.7 GiB staged only
  # on /workspace would cost 2.5 hours of re-upload; from /vault a replacement
  # pod is a ~30 s local copy at the measured 86 MB/s.
  echo "==> ${GPUTW_HOST}  vault=${vault}  work=${root}"
  sshx "mkdir -p '${vault}' '${root}/work' /workspace/tabpfn-cache /vault/lead-tabpfn/cache"

  push_if_changed "$CHECKPOINT" "${VAULT_CKPT}" "foundation checkpoint"
  push_if_changed "$src/features.float32.npy" "${vault}/features.float32.npy" "features"
  push_if_changed "$src/metadata.npz" "${vault}/metadata.npz" "metadata"
  push_if_changed "$src/model.portable.tabpfn_fit" "${vault}/model.portable.tabpfn_fit" "portable fit"
  [[ -f "$src/scaler.npz" ]] &&
    push_if_changed "$src/scaler.npz" "${vault}/scaler.npz" "scaler"
  push_if_changed "$WORKER" "${vault}/run_m5_tabpfn_portable_shard.py" "worker"

  # Materialise the fast local working copy. Reading the matrix over NFS during
  # inference would work, but a local ext4 copy costs half a minute and removes
  # NFS from the hot path entirely.
  echo "    hydrating ${root} from vault"
  sshx "
    set -e
    cp -n '${VAULT_CKPT}' '${REMOTE_CKPT}' 2>/dev/null || true
    for f in features.float32.npy metadata.npz model.portable.tabpfn_fit scaler.npz run_m5_tabpfn_portable_shard.py; do
      [ -f '${vault}'/\$f ] || continue
      if [ ! -f '${root}'/\$f ] || [ \"\$(stat -c%s '${vault}'/\$f)\" != \"\$(stat -c%s '${root}'/\$f)\" ]; then
        cp '${vault}'/\$f '${root}'/\$f
      fi
    done
    ln -sfn '${REMOTE_CKPT}' '${root}/tabpfn-v3-classifier-v3_default.ckpt'
  "
  echo "push complete"
}

cmd_run() {
  local context="$1" line="$2" batch="$3" shard="$4"; shift 4
  local root; root="$(remote_root "$context" "$line" "$batch" "$shard")"
  local session="tabpfn_c${context}_b${batch}_${shard}_f${line}"
  local direction="reverse"
  [[ "$shard" == "head" ]] && direction="forward"

  # Gate on an explicit marker, never on the mere presence of scaler.npz. The
  # exported matrices are ALREADY standardised, and staging a scaler.npz beside
  # them is useful for other reasons -- if that file alone switched --scaler on,
  # the worker would standardise twice. The scores would still be finite, still
  # cover the right rows, and still merge cleanly: a silent corruption with no
  # downstream gate able to see it. The marker has to be written deliberately by
  # whoever exported an unscaled matrix.
  local scaler_arg=""
  if sshx "test -f '${root}/features.UNSCALED'"; then
    sshx "test -f '${root}/scaler.npz'" ||
      die "features.UNSCALED present but scaler.npz missing in ${root}"
    scaler_arg="--scaler ${root}/scaler.npz"
    echo "==> features marked UNSCALED: applying scaler at predict time"
  fi

  # python -u, not plain python: stdout redirected to a file is block-buffered,
  # so a healthy run looks like a dead one for many minutes. Learned the hard
  # way on the calibration run.
  sshx "tmux kill-session -t '${session}' 2>/dev/null; tmux new-session -d -s '${session}' \
    \"cd '${root}' && python -u run_m5_tabpfn_portable_shard.py \
      --features '${root}/features.float32.npy' \
      --metadata '${root}/metadata.npz' \
      --fit-state '${root}/model.portable.tabpfn_fit' \
      --work-dir '${root}/work' \
      --context-rows ${context} \
      --n-features ${line} \
      --n-estimators 8 \
      --query-microbatch-size 20000 \
      --checkpoint-rows 20000 \
      --direction ${direction} \
      ${scaler_arg} $* \
      --resume > '${root}/worker.log' 2>&1\""
  echo "launched tmux session ${session}; log at ${root}/worker.log"
}

cmd_status() {
  local context="$1" line="$2" batch="$3" shard="$4"
  local root; root="$(remote_root "$context" "$line" "$batch" "$shard")"
  sshx "
    echo '--- durable chunks ---'
    ls -1 '${root}/work/chunks'/rows_*.npz 2>/dev/null | wc -l
    echo '--- newest chunk age (s) ---'
    newest=\$(ls -t '${root}/work/chunks'/rows_*.npz 2>/dev/null | head -1)
    if [ -n \"\$newest\" ]; then echo \$(( \$(date +%s) - \$(stat -c %Y \"\$newest\") )); else echo 'none yet'; fi
    echo '--- gpu ---'
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    echo '--- uptime (bills until you stop the instance) ---'
    uptime -p
    echo '--- last 5 log lines ---'
    tail -5 '${root}/worker.log' 2>/dev/null || echo 'no worker.log'
  "
}

cmd_pull() {
  local context="$1" line="$2" batch="$3" shard="$4" dest="$5"
  local root; root="$(remote_root "$context" "$line" "$batch" "$shard")"
  mkdir -p "${dest}/chunks"
  # tar over ssh, not scp per file: a full shard is hundreds of small chunk
  # files and scp pays a round trip for each one.
  sshx "cd '${root}/work/chunks' && tar cf - rows_*.npz" | tar xf - -C "${dest}/chunks"
  local remote_count local_count
  remote_count="$(sshx "ls -1 '${root}/work/chunks'/rows_*.npz 2>/dev/null | wc -l" | tr -d '[:space:]')"
  local_count="$(find "${dest}/chunks" -name 'rows_*.npz' | wc -l | tr -d '[:space:]')"
  echo "remote chunks: ${remote_count}   local chunks: ${local_count}"
  [[ "$remote_count" == "$local_count" ]] || die "chunk count mismatch; DO NOT stop the instance"
  echo "all chunks retrieved; safe to stop this instance"
}

[[ $# -ge 1 ]] || die "no subcommand; see the header of this file"
sub="$1"; shift
case "$sub" in
  push)   [[ $# -eq 5 ]] || die "push needs 5 args";  cmd_push   "$@" ;;
  run)    [[ $# -ge 4 ]] || die "run needs >=4 args"; cmd_run    "$@" ;;
  status) [[ $# -eq 4 ]] || die "status needs 4 args"; cmd_status "$@" ;;
  pull)   [[ $# -eq 5 ]] || die "pull needs 5 args";  cmd_pull   "$@" ;;
  *) die "unknown subcommand: $sub" ;;
esac
