# M5 TabPFN context curve — resume handoff (2026-07-27)

**Read this first. It is authoritative for run state.** The protocol and the
reasoning behind the design live in
[docs/reference/m5-tabpfn-context-curve-runbook.md](../reference/m5-tabpfn-context-curve-runbook.md),
but that document's cost estimates were superseded by measurement (§3 here) and
its opening status line describes the pre-run state.

The rented GPU was **stopped on 2026-07-27 after 3h00m**. Nothing is running.
Everything scored so far is pulled to local disk and verified.

## 1. What the experiment asks

Does TabPFN beat tree ensembles *at the same amount of labelled data*, and does
any advantage grow as data shrinks? Both models sweep N, trained on byte-identical
rows, so `TabPFN(N) − Trees(N)` attributes to the model rather than to data volume.
Grid: N ∈ {5k, 10k, 20k, 50k, 100k} × {17, 137} features. 100k was already done
before this session.

**Do not compare a small-N TabPFN against the published 2.7M-row tree line.** That
measures data quantity, not model. This mistake was made once already.

## 2. Progress: 8 cells (4 new contexts × 2 feature lines)

Each cell needs two shards, head (253 chunks) and tail (254), covering all
10,137,155 holdout rows.

| cell | shards done | merged artifact |
|---|---|---|
| **10k / 17** | 2/2 | `m5_tabpfn_17_full_test_context10000_n8_predictions.npz` |
| **10k / 137** | 2/2 | `m5_tabpfn_137_full_test_context10000_n8_predictions.npz` |
| **20k / 17** | 2/2 | `m5_tabpfn_17_full_test_context20000_n8_predictions.npz` |
| 20k / 137 | 0/2 | head staged in `/vault`, never computed |
| 50k / 17 | 0/2 | exported locally only |
| 50k / 137 | 0/2 | exported locally only |
| 5k / 17 | 0/2 | exported locally only |
| 5k / 137 | 0/2 | exported locally only |

All three merged files pass the identity gates: 10,137,155 rows, 637,397
anomalies, scores finite, `raw_index`/`anomaly` element-wise identical to the
published 100k line. Every remote chunk had a matching local copy before shutdown.

**All 12 exported shards are ready to upload**; fits, scalers and batch plans are
complete. Nothing is blocked on export.

**The tree matched-N arm has never produced output.** Its `KeyError` is fixed
(§5) but the runner has not been re-run, so `TabPFN(N) − Trees(N)` cannot be
computed yet. This is the largest gap and it needs no GPU and no money.

## 3. Results so far, and the measured numbers that replace the estimates

Pooled over the full holdout:

| line | context | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| 17 | 10,000 | **0.9413** | **0.7617** |
| 17 | 20,000 | 0.9398 | 0.7484 |
| 17 | 100,000 | 0.9163 | 0.6944 |
| 137 | 10,000 | 0.9902 | 0.9208 |
| 137 | 100,000 | **0.9919** | **0.9314** |

**The two lines move in opposite directions.** At 17 features a smaller context is
better, monotonically across three points. At 137 features a larger context is
better. The 137 gap is small (ROC 0.0017, PR 0.0106) and rests on two points, so
it is not yet a trend; the 17-feature result is the sturdier one. 20k/137 and 50k
decide whether either holds. Per-site, the 17-feature 10k-vs-100k gain is a
redistribution rather than a lift: only 9 of 16 sites improve on ROC, and pooled
PR-AUC is dominated by Site 2 (+0.297) with Sites 6 and 10 behind it, while Sites
1, 7 and 9 get worse.

Measured on the RTX 5090, replacing runbook §9's extrapolation:

- **17 features @ 20k context: ~4,000 rows/s** — 253 chunks in ~20 min, about
  3.3x the projection.
- **Uplink: ~1.35 MB/s**, and this is a hard ceiling. Two parallel streams
  measured 1,237 KiB/s combined versus 1,384 KiB/s for one, so **parallelism does
  not help**; only sending fewer bytes does.
- One 17-feature worker at 20k context pins the GPU at 100%. The "two slots give
  1.58x" note in `gputw_tabpfn_pool.sh` was measured at 10k, where a single worker
  reached only 29%. Do not assume it generalises.

**The run is upload-bound, not compute-bound.**

## 4. Scripts

| script | role |
|---|---|
| `scripts/gputw_bootstrap.sh` | run first on a new pod: proves sm_120 kernels actually execute |
| `scripts/gputw_tabpfn_shard.sh` | `push` / `run` / `status` / `pull` for one shard |
| `scripts/gputw_resumable_push.sh` | 32 MiB block upload, resumes from remote size, SHA-256 verified |
| `scripts/gputw_tabpfn_pool2.sh` | full pool: uploads, launches, pulls. **Use this, not `gputw_tabpfn_pool.sh`** |
| `scripts/gputw_upload_only.sh` | stages shards into `/vault` with no GPU work; for winding down |
| `scripts/repair_m5_tabpfn_portable_fit_paths.py` | rewrites wrong `model_path` in exported fit states |
| `scripts/verify_m5_tabpfn_context_nesting.py` | Gate 1, already passed |
| `scripts/run_m5_tree_ensemble_matched_context.py` | the tree arm; fixed, not yet run |
| `scripts/merge_m5_tabpfn_full_test.py` | merge a context's shards |

`gputw_tabpfn_pool.sh` is kept only as the record of a design that failed; it
uploads inside its scheduler loop and will idle the GPU.

## 5. Traps — every one of these actually happened

1. **`scp` cannot resume and the gputw.ai gateway kills long transfers**
   (`Connection reset by peer` at ~1.1 GiB and again at ~40 MiB). Every retry
   restarted at zero, so 2.6 GiB matrices never landed. Always upload via
   `gputw_resumable_push.sh`. `rsync` is unavailable — Windows Git Bash ships none.
2. **Never upload inside the scheduler loop.** The original pool blocked launching,
   polling and pulling for the duration of each multi-GiB push; with everything
   already on the box finished, the card sat at 0% for 25 minutes.
3. **A wrong `model_path` in the portable fit does not fail loudly.** TabPFN
   concludes the weights are missing and tries to *download* them, surfacing as
   `TabPFNLicenseError` — a traceback entirely about licensing that never mentions
   a path. Twelve shards shipped with unusable paths: the 137 exporter emitted
   Colab roots without the context, and `--remote-prefix /workspace` through Git
   Bash was MSYS-rewritten to `C:/Program Files/Git/workspace`. Both exporters are
   fixed and verified against `remote_root()` in `gputw_tabpfn_shard.sh`, but
   **if a shard dies on startup, check `model_path` first**:

   ```bash
   python -c "import zipfile,json; print(json.loads(zipfile.ZipFile('<shard>/model.portable.tabpfn_fit').read('init_params.json'))['model_path'])"
   ```

   The 137 head shards only ever ran because a leftover `/content/lead_tabpfn_137_head`
   directory happened to survive on that pod. On a fresh pod the whole 137 line
   would have failed.
4. **`nohup cmd &` inside an agent tool call is a trap.** The harness reports the
   task "completed (exit 0)" the moment the outer shell returns while the child
   keeps running. Retrying on that false signal produced four processes appending
   to one vault file and corrupted it; only the post-upload SHA-256 caught it. Use
   a tracked background job with no inner `&`, or `Start-Process`, and check for a
   live process before retrying. Relatedly, Git Bash reports pipeline **subshells**
   with the parent's command line — several PIDs for one script is normal, so
   check parent PIDs before concluding there are duplicates.
5. **pool2 relaunched shards that were merely still starting.** `cmd_run` begins
   with `tmux kill-session`, so a 137 shard that needs >90 s to memory-map 2.6 GiB
   and load the model was killed on every poll: launched six times in four minutes,
   zero chunks, indistinguishable from an idle GPU. A 300 s grace period is now
   enforced. Judge liveness by **durable chunk count**, never by a session existing.
6. **`merge_m5_tabpfn_full_test.py` assumes six batches.** The context curve uses
   only batch0 (head+tail already cover all 10.1M rows), so the default path raises
   `FileNotFoundError` on `batch1`. Always pass `--roots` explicitly (§6).
7. **The feature builder destroys row identity.** `add_value_change_features` ends
   with `sort_values(...).reset_index(drop=True)`, dropping labels *and* reordering
   rows. The tree runner's `.loc[fit_index]` therefore raised `KeyError`. The
   KeyError was the lucky outcome — had labels been reused instead of dropped, the
   trees would have silently trained on different rows than TabPFN while every
   downstream check passed, destroying the only property the comparison rests on.
   Fixed by carrying `raw_index` through as a column, plus an assertion.

## 6. How to resume

```bash
# 1. Rent an RTX 5090 on gputw.ai (account kuantingkuo@ntu.edu.tw), Linux
#    PyTorch image. Then, BEFORE uploading anything:
export GPUTW_HOST=pod-<new-id>@ssh.gputw.ai
ssh -p 2222 "$GPUTW_HOST" 'bash -s' < scripts/gputw_bootstrap.sh   # proves sm_120 works

# 2. Check whether /vault survived (SEE THE WARNING BELOW)
ssh -p 2222 "$GPUTW_HOST" 'ls -la /vault/lead-tabpfn/'

# 3. Confirm the exported fit states carry correct paths
uv run python scripts/repair_m5_tabpfn_portable_fit_paths.py --dry-run

# 4. Run the pool (uploads, launches, pulls; safe to restart, resumes)
bash scripts/gputw_tabpfn_pool2.sh          # detached via Start-Process on Windows
tail -f data/processed/m5_tabpfn_pool2.log
```

`data/processed/.pool2/<ctx>:<line>:<shard>.uploaded` marks a finished upload.
Delete a marker to force re-upload; create one to make the pool skip a shard.
**A marker on an incomplete upload will launch a worker on a truncated matrix.**

Merging a finished context — note `--roots`:

```powershell
uv run python scripts/merge_m5_tabpfn_full_test.py --line 137 --context-rows 20000 `
  --roots data/processed/m5_tabpfn_f137_batch0_context20000_n8
```

The tree arm, which needs no GPU and can run at any time:

```powershell
uv run python scripts/run_m5_tree_ensemble_matched_context.py --context-rows 10000 --features 17
```

Before stopping any instance, `pull` every shard and confirm it reports matching
remote and local chunk counts. It refuses to say "safe to stop" on a mismatch.
**The box bills per second and is never reclaimed — stopping it is a required
step.** There is no API; stop it in the dashboard.

## 7. Open items

- **`/vault` persistence is still unverified.** Uploads land on
  `192.168.7.2:/vault/user-...`, account-scoped NFS that *should* outlive a pod,
  but stop/start was never actually tested — it has been an open runbook item from
  the start. Staged at shutdown: 10k all four shards, 20k 137-head, 20k 17
  head/tail, and 20k/137/tail at 1975 of 2653 MiB. Losing it costs only re-upload
  time; every scored result is already local.
- **Unscaled upload (runbook §6.3) was never adopted, and it is the single biggest
  lever.** Exports are pre-scaled, so each context re-sends a full matrix: ~19 GiB
  instead of ~6 GiB, because **the unscaled matrix is identical across contexts**
  and only a few-KB `scaler.npz` differs. The worker's `--scaler` path is
  implemented and verified to 2.4e-07, and `cmd_run` gates on a deliberate
  `features.UNSCALED` marker so an already-scaled matrix is never scaled twice.
  Only the exporters lack the flag. At 1.35 MB/s this is worth roughly 2.5 h of
  billed time.
- **TabPFN's ceiling on 32 GB was never probed.** The recorded
  `last_safe_budget: 100000` was measured on the 8 GB 4070. Where TabPFN stops is
  itself a reportable result.
- **5k was planned for the local RTX 4070** rather than the rented card; unchanged,
  and its shards are exported.
- No GitHub issue was opened for this work; recorded honestly per the
  change-checklist backfill policy rather than fabricated retroactively.
