#!/usr/bin/env bash
# Resumable upload of one large file to a gputw.ai pod.
#
# Why this exists: scp cannot resume. The gputw.ai SSH gateway drops long-lived
# transfers ("Connection reset by peer" at ~1.1 GiB and again at ~40 MiB), so a
# 2.6 GiB matrix pushed with scp restarts from zero on every drop and can never
# finish. Retrying harder does not help -- the transfer has to be restartable.
#
# The file is sent in blocks, one short-lived ssh connection per block, appending
# to the remote copy. A dropped connection loses at most one block: the remote
# size is re-read each pass and the next block starts exactly there. `cat >>`
# only ever writes a byte-exact prefix of what was piped, so resuming from the
# observed size is safe even when a block is cut in half.
#
#   gputw_resumable_push.sh <local-file> <remote-path> [block-bytes]
set -uo pipefail

LOCAL="${1:?local file}"
REMOTE="${2:?remote path}"
BLOCK="${3:-33554432}"   # 32 MiB: small enough to survive the gateway, large
                         # enough that the ~1.5 s handshake stays under 5% overhead.
PORT="${GPUTW_PORT:-2222}"
: "${GPUTW_HOST:?set GPUTW_HOST}"

[[ -f "$LOCAL" ]] || { echo "ERROR: no such file: $LOCAL" >&2; exit 1; }
SIZE=$(stat -c%s "$LOCAL")

sshq() {
  ssh -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      "$GPUTW_HOST" "$@" 2>/dev/null
}

sshq "mkdir -p '$(dirname "$REMOTE")'"

stall=0
while true; do
  off=$(sshq "stat -c%s '$REMOTE' 2>/dev/null || echo 0" | tr -cd '0-9')
  off=${off:-0}
  if (( off >= SIZE )); then
    (( off > SIZE )) && { echo "ERROR: remote larger than local ($off > $SIZE); truncating"; \
                          sshq "truncate -s 0 '$REMOTE'"; continue; }
    break
  fi
  before=$off
  printf '\r  %s  %d/%d MiB (%d%%)' "$(basename "$REMOTE")" \
    $((off/1048576)) $((SIZE/1048576)) $((off*100/SIZE))
  tail -c +$((off + 1)) "$LOCAL" | head -c "$BLOCK" | sshq "cat >> '$REMOTE'"
  after=$(sshq "stat -c%s '$REMOTE' 2>/dev/null || echo 0" | tr -cd '0-9')
  if (( ${after:-0} <= before )); then
    stall=$((stall + 1))
    (( stall >= 8 )) && { echo; echo "ERROR: no progress after 8 attempts at offset $before" >&2; exit 1; }
    sleep $((stall * 3))
  else
    stall=0
  fi
done
echo
echo "  transferred, verifying digest..."
want=$(sha256sum "$LOCAL" | cut -d' ' -f1)
got=$(sshq "sha256sum '$REMOTE' | cut -d' ' -f1" | tr -cd '0-9a-f')
[[ "$want" == "$got" ]] || { echo "ERROR: digest mismatch on $REMOTE" >&2; exit 1; }
echo "  ok $REMOTE"
