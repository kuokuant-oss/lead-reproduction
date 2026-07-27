#!/usr/bin/env bash
# Run every tree matched-N cell the context curve needs, one at a time.
#
# The comparison the curve rests on is TabPFN(N) - Trees(N) at byte-identical
# rows, so every context TabPFN is scored at needs a tree cell beside it. This
# arm needs no GPU and no rented box, which makes it the right thing to run
# while the pod works: it costs nothing but wall clock.
#
# Strictly serial, and that is not laziness. Each cell rebuilds timestamp-merge
# features for both halves of a 20.2M-row frame and peaks around 7-15 GiB; two
# at once on a 32 GB box swaps, and a swapping cell is slower than two serial
# ones. --resume means an interrupted cell restarts at its last durable chunk
# rather than at the frame load.
#
# Order is by what it unblocks. 100k first: both TabPFN 100k lines are already
# published, so those two cells complete two whole comparisons the moment they
# land. Then the contexts TabPFN has finished, then the ones the pod is still
# producing.
#
#   bash scripts/run_m5_tree_matched_queue.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=data/processed/m5_tree_matched_queue.log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

CELLS=(
  "100000 17" "100000 137"   # TabPFN counterpart already published
  "10000 137" "20000 17"     # TabPFN counterpart already merged
  "20000 137" "50000 17" "50000 137" "5000 17" "5000 137"
)

say "=== tree matched-N queue: ${#CELLS[@]} cells ==="
for cell in "${CELLS[@]}"; do
  read -r ctx line <<<"$cell"
  out="data/processed/m5_tree_ensemble_f${line}_context${ctx}.json"
  if [[ -f "$out" ]]; then
    say "SKIP f${line} c${ctx} -- ${out} exists"
    continue
  fi
  say "RUN f${line} c${ctx}"
  if uv run python scripts/run_m5_tree_ensemble_matched_context.py \
       --context-rows "$ctx" --features "$line" --resume \
       >> "data/processed/m5_tree_f${line}_c${ctx}.log" 2>&1; then
    say "DONE f${line} c${ctx} -- $(grep -o 'pooled ROC-AUC.*' "data/processed/m5_tree_f${line}_c${ctx}.log" | tail -1)"
  else
    # Keep going. One cell failing (most likely on memory) should not cost the
    # other eight a night of unattended CPU that was free anyway.
    say "FAILED f${line} c${ctx} -- see data/processed/m5_tree_f${line}_c${ctx}.log"
  fi
done
say "=== tree matched-N queue finished ==="
# The completion marker is written here rather than by the caller. Chaining it on
# with `bash -c "... && touch ..."` meant the launcher had to quote a compound
# command through Start-Process, which it did not: bash received the single word
# `bash` and sat reading stdin forever -- no children, no output, ~0 s of CPU,
# and a supervisor that relaunched the same wedge every minute.
: > data/processed/m5_tree_matched_queue.DONE
