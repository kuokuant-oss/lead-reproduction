# M5 TabPFN context curve — gputw.ai run handoff (2026-07-27)

**Non-report handoff.** Plan and protocol live in
[docs/reference/m5-tabpfn-context-curve-runbook.md](../reference/m5-tabpfn-context-curve-runbook.md);
this file records what has actually executed, the transfer failures that cost GPU
time, and what the next session must not repeat.

The runbook's §13 "尚未執行任何 fit、export、租機或推論" is **stale** — it was written
before the run started. This file is authoritative for run state.

## 1. Where the run actually is

Rented box: `pod-895a3921-edf0-4ade-bddf-2f15a8beb0b9@ssh.gputw.ai:2222`,
RTX 5090 32 GB, account `kuantingkuo@ntu.edu.tw`. **Per-second billing, no
preemption — it bills until someone stops it.**

| context | line | shard | state |
|---|---|---|---|
| 10,000 | 17 | head+tail | **scored, pulled, merged** |
| 10,000 | 137 | head | scored, pulled (253 chunks) |
| 10,000 | 137 | tail | uploading (~52% of 2.65 GiB) |
| 20,000 | 17 | head | **scoring now** (~4,000 rows/s) |
| 20,000 | 17 | tail | exported, not uploaded |
| 20,000 / 50,000 / 5,000 | 17 + 137 | rest | exported, not uploaded |
| 100,000 | 17 + 137 | — | complete before this session |

Fits, portable scalers, batch plans and all 12 remaining shard exports are done
locally. Nothing is blocked on export.

Only merged artifact so far:
`data/processed/m5_tabpfn_17_full_test_context10000_n8_predictions.npz`.
Verified rather than assumed — 10,137,155 rows, 637,397 anomalies, all scores
finite in [2.77e-05, 0.999987], and `raw_index` / `anomaly` element-wise identical
to the published 100k line. Pearson correlation of scores against 100k is
**0.843**, i.e. shrinking the context genuinely moved the predictions.

`m5_tabpfn_137_full_test_context10000_n8_predictions.npz` can be merged as soon as
the 137 tail finishes; its head chunks are already local.

**Not started:** the tree matched-N arm
(`scripts/run_m5_tree_ensemble_matched_context.py`), which needs no GPU and no
money, and the formal Step 0 calibration JSON.

## 2. Measured numbers that replace the runbook's estimates

The runbook §9 numbers were extrapolation. These are measured on the 5090:

- **17 features @ 20k context: ~4,000 rows/s** (253 chunks of 20,000 rows in about
  20 minutes). The runbook guessed ~1,200 rows/s, so it was pessimistic by ~3.3x.
- **Uplink: ~1.5–2.0 MiB/s.** This is the binding constraint.
- One 17-feature worker at 20k context already pins the GPU at **100%**. The
  two-slot 1.58x aggregate gain recorded in `gputw_tabpfn_pool.sh` was measured at
  10k context, where a single worker only reached 29%. Do not assume the second
  slot pays for itself at larger contexts — measure before concluding.

**Consequence: this run is upload-bound, not compute-bound.** With ~19 GiB still to
send at ~1.75 MiB/s that is ~3.7 h of pure transfer against roughly 4–5 h of
remaining compute. Any serialization between the two shows up directly as an idle,
still-billing GPU.

## 3. The optimization that is being left on the table

The exported matrices are **already scaled**, so runbook §6.3 (upload one unscaled
matrix, ship a few-KB `scaler.npz` per context, apply at predict time) is not in
use. Every context therefore re-sends a full matrix: ~19 GiB instead of ~6 GiB.

`run_m5_tabpfn_portable_shard.py --scaler` and
`export_m5_tabpfn_context_scaler.py` are implemented and numerically verified to
2.4e-07, and `gputw_tabpfn_shard.sh` already gates on a deliberate
`features.UNSCALED` marker. Re-exporting unscaled would cost local CPU now and cut
roughly two-thirds of the remaining upload. **Undecided — needs a call.**

Note the marker discipline: `cmd_run` switches on `features.UNSCALED`, never on the
mere presence of `scaler.npz`. Standardising an already-standardised matrix would
produce finite scores over the right rows that merge cleanly — a silent corruption
no downstream gate can catch.

## 4. Three transfer failures that idled the GPU ~25 minutes

All three are permanent properties of this setup, not one-off glitches.

1. **gputw.ai's SSH gateway kills long transfers, and `scp` cannot resume.**
   `Connection reset by peer` was observed at ~1.1 GiB and again at ~40 MiB. Each
   retry restarted from byte zero, so the 2.6 GiB matrices could never land.
   Fixed by `scripts/gputw_resumable_push.sh`: 32 MiB blocks, one short-lived SSH
   connection each, appending, resuming from the observed remote size, SHA-256
   verified at the end. `rsync` is not available — Windows Git Bash ships none.
2. **The old pool uploaded inside its scheduler loop.**
   `gputw_tabpfn_pool.sh:144` called `push` synchronously, so a multi-GiB transfer
   blocked launching, polling *and* pulling. Once everything already on the box had
   finished, the card sat at 0% for the whole transfer.
   Fixed by `scripts/gputw_tabpfn_pool2.sh`, where the uploader is a separate
   process and the scheduler only reads marker files in `data/processed/.pool2/`.
3. **Operator error worth recording: `nohup cmd &` launched from the agent harness.**
   The task was reported "completed (exit 0)" the moment the outer shell returned,
   while the child kept running. Retrying on that false signal produced **four
   concurrent processes appending to the same vault file**, corrupting it. The
   post-upload SHA-256 check caught it; nothing else would have. Launch long work
   as a tracked background job with no inner `&`, or via `Start-Process`, and
   always check for a live process before retrying.

## 5. How to drive it

```bash
export GPUTW_HOST=pod-895a3921-edf0-4ade-bddf-2f15a8beb0b9@ssh.gputw.ai
bash scripts/gputw_tabpfn_pool2.sh          # detached; uploads + schedules + pulls
tail -f data/processed/m5_tabpfn_pool2.log
```

`data/processed/.pool2/<ctx>:<line>:<shard>.uploaded` marks a completed upload.
Pre-create a marker to make the pool skip a shard already on the box; delete one to
force a re-upload.

Merge a finished context:

```powershell
uv run python scripts/merge_m5_tabpfn_full_test.py --line 137 --context-rows 10000
```

Completion is judged **only** by durable chunk count on the box (253 head / 254
tail), never by whether a tmux session looks alive — a dead worker leaves a
healthy-looking session behind.

## 6. Before stopping the instance

`/workspace` is pod-local NVMe and dies with the pod. `/vault` is account-scoped
NFS (3.7 TB at `192.168.7.2`) and survives, which is why uploads land there and are
hydrated to `/workspace` for the hot path.

Run `gputw_tabpfn_shard.sh pull` for every shard and confirm it reports matching
remote/local chunk counts before stopping. It refuses to say "safe to stop" on a
mismatch. **The box bills continuously — stopping it is a required step, not
cleanup.**
