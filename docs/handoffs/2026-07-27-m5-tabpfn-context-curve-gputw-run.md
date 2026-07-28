# M5 TabPFN context curve — resume handoff (2026-07-27 → 28)

**Read this first. It is authoritative for run state.** The protocol and the
reasoning behind the design live in
[docs/reference/m5-tabpfn-context-curve-runbook.md](../reference/m5-tabpfn-context-curve-runbook.md),
but that document's cost estimates were superseded by measurement (§3 here) and
its opening status line describes the pre-run state.

## 0. The grid is complete

**Finished 2026-07-28 14:00. Nothing is running, no rented GPU is up, and there
is nothing left to launch.** All twenty cells (5 contexts x 2 feature lines x
{TabPFN, trees}) are scored, merged and compared at matched N. Do not restart
`gputw_tabpfn_pool2.sh` or the supervisor; starting a pod would cost money for
nothing.

The last shard, 5k / 137 tail, was resumed from chunk 46 on the local RTX 4070
and ran to 254 with zero errors at a measured 0.61 chunk/min (4h35m). Its merge
passes the identity gates: 10,137,155 rows, 637,397 anomalies, `raw_index`
element-wise identical to the tree artifact.

| cell | TabPFN | trees | matched-N diff |
|---|---|---|---|
| 5k / 17 | merged | done | yes |
| 5k / 137 | merged | done | yes |
| 10k / 17 | merged | done | yes |
| 10k / 137 | merged | done | yes |
| 20k / 17 | merged | done | yes |
| 20k / 137 | merged | done | yes |
| 50k / 17 | merged | done | yes |
| 50k / 137 | merged | done | yes |
| 100k / 17 | published | done | yes |
| 100k / 137 | published | done | yes |

Disaggregated tables: `docs/reports/m5-matched-context-breakdown.md` (§3.0.1).
Not measured: TabPFN's own context-draw noise, currently proxied by the trees'
(§3.1). Measuring it needs GPU time to re-score new context draws.

If a 137-feature shard ever needs re-running locally, use the localised fit
state under `data/processed/m5_tabpfn_local_c5000_f137_{head,tail}/` and **not**
the shard's own `model.portable.tabpfn_fit`, which carries the pod's checkpoint
path and fails as `TabPFNLicenseError` (§5 trap 3). `--direction reverse` for a
tail shard is not optional: forward would rescore the head's rows.

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

| cell | shards | merged | tree counterpart |
|---|---|---|---|
| 5k / 17 | 2/2 | yes | yes |
| 5k / 137 | 2/2 | yes | yes |
| 10k / 17 | 2/2 | yes | yes |
| 10k / 137 | 2/2 | yes | yes |
| 20k / 17 | 2/2 | yes | yes |
| 20k / 137 | 2/2 | yes | yes |
| 50k / 17 | 2/2 | yes | yes |
| 50k / 137 | 2/2 | yes | yes |

Every merged file passes the identity gates: 10,137,155 rows, 637,397 anomalies,
scores finite, `raw_index`/`anomaly` element-wise identical to the published 100k
line. Chunk counts were verified per shard against the pod before it was stopped.

**The last cell ran locally.** The 137-feature 5k shards were scored on the local
4070 after the remote work finished. The head took 4 h 39 m at ~1 chunk/min, the
tail 4 h 35 m at 0.61 chunk/min. The portable fit carries the *pod's* checkpoint
path, so it must be localised first or it fails as `TabPFNLicenseError`
(§5 trap 3); this was done for both 5k/137 shards. The commands used:

```powershell
uv run python scripts/localize_m5_tabpfn_fit_state.py `
  --shard-dir data/processed/m5_tabpfn_137_distributed_context5000_n8/head `
  --out data/processed/m5_tabpfn_local_c5000_f137_head/model.portable.tabpfn_fit

uv run python scripts/run_m5_tabpfn_portable_shard.py `
  --features data/processed/m5_tabpfn_137_distributed_context5000_n8/head/features.float32.npy `
  --metadata data/processed/m5_tabpfn_137_distributed_context5000_n8/head/metadata.npz `
  --fit-state data/processed/m5_tabpfn_local_c5000_f137_head/model.portable.tabpfn_fit `
  --work-dir data/processed/m5_tabpfn_f137_batch0_context5000_n8/head-results `
  --context-rows 5000 --n-features 137 --n-estimators 8 `
  --query-microbatch-size 4096 --checkpoint-rows 20000 --direction forward --resume
```

`--work-dir` is the pull destination, so chunks land where the merge already
looks. `--direction reverse` for the tail. A 4096 microbatch keeps the 8 GB card
at 5.7 GB and 100% utilization; it is compute-bound, so raising it buys nothing.

## 3. Results and measured run figures

Pooled over the full holdout:

**17 features.** Complete, five points, TabPFN against trees on byte-identical rows:

| N | TabPFN ROC | Trees ROC | diff | TabPFN PR | Trees PR | diff |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | **0.9451** | 0.9609 | **-0.0158** | **0.7719** | 0.8208 | **-0.0490** |
| 10,000 | 0.9413 | 0.9634 | -0.0221 | 0.7617 | 0.8255 | -0.0638 |
| 20,000 | 0.9398 | 0.9659 | -0.0261 | 0.7484 | 0.8285 | -0.0801 |
| 50,000 | 0.9238 | 0.9665 | -0.0427 | 0.7159 | 0.8291 | -0.1132 |
| 100,000 | 0.9163 | 0.9650 | -0.0487 | 0.6944 | 0.8291 | -0.1348 |

**137 features.** Complete, five points:

| N | TabPFN ROC | Trees ROC | diff | TabPFN PR | Trees PR | diff |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.9878 | 0.9879 | -0.0001 | **0.9001** | 0.8965 | **+0.0036** |
| 10,000 | 0.9902 | 0.9901 | +0.0001 | 0.9208 | 0.9191 | +0.0017 |
| 20,000 | 0.9912 | 0.9912 | -0.0000 | 0.9263 | 0.9240 | +0.0023 |
| 50,000 | **0.9924** | 0.9918 | +0.0006 | **0.9324** | 0.9300 | +0.0024 |
| 100,000 | 0.9919 | 0.9920 | -0.0001 | 0.9314 | 0.9306 | +0.0008 |

Movement across the sweep, measured on the tables above:

- 17 features, 5k to 100k: TabPFN ROC -0.0288 / PR -0.0775, monotone at all five
  points. Trees stay between ROC 0.9609 and 0.9665.
- 137 features: TabPFN ROC rises +0.0022 from 10k to 50k, then -0.0005 by 100k.
  Its PR difference against trees is +0.0036 at 5k, +0.0017, +0.0023, +0.0024,
  +0.0008 as N rises.

Reference points measured on the same 10,137,155 rows:

- Ranking by `-meter_reading` alone, no model: **ROC 0.8998, PR 0.5143**. Single-
  feature |AUC-0.5| across the 17: meter_reading 0.419, dayofyear 0.155,
  month 0.154, all others below 0.11. Low readings are the anomalous direction.
- `SHIFTS` is symmetric, +-1..24 h plus +-48..168 h, so a 137-feature row carries
  up to seven days of its own meter's future as well as its past.

Noise floors, from redrawing the 100k context under four seeds and refitting the
trees (§3.1): ROC sd 0.0003 at 137 features, 0.0004 at 17. These are the *trees'*
draw noise. TabPFN's own has never been measured; nested prefixes also mean
adjacent contexts share rows, so the variance of a difference is smaller than
these independent-draw figures.

### 3.0 Per-site view of the same 17-feature tree numbers

Trees at 17 features, 5,000 rows versus 100,000 rows, PR-AUC per site:

| site | rows | anomaly rate | 5k | 100k | change |
|---:|---:|---:|---:|---:|---:|
| 4 | 370,460 | 0.05% | 0.5547 | 0.7320 | **+0.1773** |
| 1 | 289,853 | 13.50% | 0.3468 | 0.4738 | **+0.1270** |
| 9 | 1,367,482 | 3.85% | 0.8883 | 0.9357 | +0.0474 |
| 0 | 538,432 | 32.74% | 0.9973 | 0.9979 | +0.0006 |
| 11 | 43,626 | 0.45% | 0.0186 | 0.0185 | -0.0001 |
| 13 | 1,334,223 | 4.44% | 0.5859 | 0.5541 | -0.0318 |
| 6 | 345,117 | 8.30% | 0.6575 | 0.5421 | **-0.1154** |
| **pooled** | 10,137,155 | 6.29% | **0.8208** | **0.8291** | **+0.0083** |

The pooled change of +0.0083 spans per-site changes from +0.1773 to -0.1154.
Nine of sixteen sites improve on ROC. Site 0 is 538k rows at PR 0.998 at both
sizes; site 11 is at PR 0.018 at both.

### 3.0.1 The disaggregated grid

`docs/reports/m5-matched-context-breakdown.md` carries the per-site and per-meter
tables for both models at every context, plus the full-training-set tree for each
feature line. Regenerate with `scripts/report_m5_matched_context_breakdown.py`.
Numbers from it, for cross-reference:

- Per-meter PR difference at 137 features, 100k: chilledwater +0.0486,
  steam +0.0790, hotwater -0.0060, electricity -0.0065. Electricity is 6,035,071
  of the 10,137,155 rows and 356,679 of the 637,397 anomalies.
- The same at 5k: chilledwater +0.1090, steam -0.0406, hotwater -0.1022,
  electricity -0.0094.
- At 17 features the full-data tree scores PR 0.1436 at site 4 and 0.0188 at site
  11, against 0.5547 and 0.0186 for the same trees capped at 5,000 rows.
- Paired row bootstrap, 200 draws, sites with at most 3,000 anomalies: sd of the
  PR difference is 0.008-0.037 at the two 197-anomaly sites. It covers the row
  draw only, not the context draw.

### 3.1 Every number here comes from one draw

The context is a single frozen sample: seed 42, digest `e9ffe0cf`, nested
prefixes, reused by every TabPFN context and every tree cell. There is no
repetition and there are no error bars anywhere in the grid.

Both lines were redrawn at 100k to calibrate the noise -- trees, four seeds,
scored on one fixed 2M-row holdout subsample
(`scripts/audit_m5_matched_context_seed_stability.py --features {17,137}`):

| seed | 17f ROC | 17f PR | 137f ROC | 137f PR |
|---|---:|---:|---:|---:|
| 42 (frozen) | 0.9648 | 0.8287 | 0.9920 | 0.9304 |
| 1 | 0.9656 | 0.8326 | 0.9924 | 0.9319 |
| 7 | 0.9651 | 0.8311 | 0.9920 | 0.9312 |
| 2024 | 0.9657 | 0.8284 | 0.9918 | 0.9304 |
| | sd **0.0004** | sd **0.0020** | sd **0.0003** | sd **0.0007** |

The frozen draw sits below the redraw mean on both lines: 1.22 sd at 17 features,
0.25 sd at 137.

These floors are per line and per model family. Applying one line's sd to the
other, or the trees' to TabPFN, is an assumption the measurement does not cover.

Leakage was audited empirically rather than argued
(`scripts/audit_m5_matched_context_leakage.py`): reconstructed digest matches both
Gate 1 and the digest the tree run recorded, 100,000 unique rows exactly 50/50,
all fit buildings even and all holdout buildings odd with zero shared buildings,
zero shared row ids, and 385 of 2,000,000 sampled holdout rows byte-identical to
a fit row on 17 features plus label (0.019%, cross-building feature collisions).
Per-site on the 17-feature 10k-vs-100k step: 9 of 16 sites improve on ROC; the
pooled PR-AUC change is dominated by Site 2 (+0.297) with Sites 6 and 10 next,
while Sites 1, 7 and 9 fall.

Measured on the RTX 5090, replacing runbook §9's extrapolation:

- **17 features @ 20k context: ~4,000 rows/s** — 253 chunks in ~20 min, about
  3.3x the projection.
- **Uplink: ~1.35 MB/s on the first pod, ~2.1-2.5 MB/s on the second.** Measured
  on the second pod: 348 MB in 2m39s and 2m44s (2.19 and 2.12 MB/s), then
  2,653 MiB in 18m50s (2.46 MB/s). It varies by pod, so budget a resume from
  measurement. Two parallel streams measured 1,237 KiB/s combined against
  1,384 KiB/s for one.
- One 17-feature worker at 20k context pins the GPU at 100%. The "two slots give
  1.58x" note in `gputw_tabpfn_pool.sh` was measured at 10k, where a single worker
  reached only 29%.

## 4. Scripts

| script | role |
|---|---|
| `scripts/gputw_bootstrap.sh` | run first on a new pod: proves sm_120 kernels actually execute, and reports whether `/vault` survived |
| `scripts/gputw_tabpfn_shard.sh` | `push` / `hydrate` / `run` / `status` / `pull` for one shard |
| `scripts/gputw_resumable_push.sh` | 32 MiB block upload, resumes from remote size, SHA-256 verified |
| `scripts/gputw_tabpfn_pool2.sh` | full pool: uploads, launches, pulls. **Use this, not `gputw_tabpfn_pool.sh`** |
| `scripts/gputw_upload_only.sh` | stages shards into `/vault` with no GPU work; for winding down |
| `scripts/repair_m5_tabpfn_portable_fit_paths.py` | rewrites wrong `model_path` in exported fit states |
| `scripts/localize_m5_tabpfn_fit_state.py` | copy a portable fit with `model_path` pointed at the local checkpoint, for running a shard off the pod |
| `scripts/verify_m5_tabpfn_context_nesting.py` | Gate 1, already passed |
| `scripts/run_m5_tree_ensemble_matched_context.py` | the tree arm, one cell |
| `scripts/run_m5_tree_matched_queue.sh` | all nine tree cells, serial, resumable, skips finished ones |
| `scripts/m5_context_curve_supervisor.ps1` | unattended driver: restarts a dead pool, merges finished contexts, keeps the tree queue fed, announces when the pod is safe to stop |
| `scripts/merge_m5_tabpfn_full_test.py` | merge a context's shards |
| `scripts/report_m5_matched_context_breakdown.py` | per-site and per-meter tables for the whole grid, against the full-data tree |

Audits, all re-runnable, written because reading the sampler is not the same as
checking it (§3.1):

| script | answers |
|---|---|
| `scripts/audit_m5_matched_context_leakage.py` | do the fit rows touch the holdout? (no, four independent ways) |
| `scripts/audit_m5_matched_context_seed_stability.py` | does the result depend on which draw? (`--features 17\|137`) |
| `scripts/audit_m5_matched_context_representativeness.py` | are the drawn rows typical of their class? (yes) |
| `scripts/audit_m5_matched_context_composition.py` | what does 50/50 stratification do to site mix? (site 0 x2.85, site 3 x0.55) |

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
7. **A finished shard is only finished on the pod that ran it.** *(The one item
   here that did not happen: it was found reading the resume path before
   restarting, not by paying for it.)* The scheduler
   judges completion by chunk count under `/workspace`, which is pod-local
   scratch. On a replacement pod that count reads zero for work that is done and
   sitting on local disk, so the pool would have re-uploaded and re-scored six
   shards -- including two 2.6 GiB matrices -- at full price. The queue is now
   built from the *local* pull (`<line>/<shard>-results/chunks`), which is the
   durable record. Related: a `.uploaded` marker records that bytes reached the
   vault, not that this pod can see them, so it no longer authorises a launch.
   The pool asks the pod whether the working copy exists at the vault's length,
   and `hydrate` re-stages it if not.
8. **The two full-data M3 artifacts contradict themselves on row order.**
   `m3_17_feature_ensemble_predictions.npz` and `m3_figure_predictions_50_50.npz`
   store `validation_raw_index` ascending while their `anomaly` and score arrays
   are in the canonical scoring order. Their published AUCs are unaffected --
   score and label are mutually consistent, and scoring them positionally
   against the canonical labels reproduces 0.9663 and 0.9918 exactly -- but
   keying rows by that index instead collapses the 137-feature line to
   **ROC 0.4933**, which is noise wearing a plausible-looking number. Anything
   joining to those files must go through the canonical order, never their own
   index. `report_m5_matched_context_breakdown.py` gates on this.
9. **The feature builder destroys row identity.** `add_value_change_features` ends
   with `sort_values(...).reset_index(drop=True)`, dropping labels *and* reordering
   rows. The tree runner's `.loc[fit_index]` therefore raised `KeyError`. The
   KeyError was the lucky outcome — had labels been reused instead of dropped, the
   trees would have silently trained on different rows than TabPFN while every
   downstream check passed, destroying the only property the comparison rests on.
   Fixed by carrying `raw_index` through as a column, plus an assertion.

## 6. How to resume

Rent an RTX 5090 on gputw.ai (account kuantingkuo@ntu.edu.tw), Linux PyTorch
image, pasting `~/.ssh/id_ed25519.pub` into the deploy form. Then:

```bash
# 1. BEFORE uploading anything. Proves sm_120 kernels execute, and prints
#    whether /vault survived -- which decides whether resuming costs 30 s or 3 h.
export GPUTW_HOST=pod-<new-id>@ssh.gputw.ai
ssh -p 2222 "$GPUTW_HOST" 'bash -s' < scripts/gputw_bootstrap.sh

# 2. Run the pool: hydrates what the vault already holds, uploads the rest,
#    launches, pulls. Safe to restart; resumes.
bash scripts/gputw_tabpfn_pool2.sh          # detached via Start-Process on Windows
tail -f data/processed/m5_tabpfn_pool2.log
```

The local fit-state gate (`repair_m5_tabpfn_portable_fit_paths.py --dry-run`)
needs no pod and was run on 2026-07-27: all 16 archives ok, 0 need repair.

The pool builds its queue from what is already pulled to local disk, so the six
finished shards are dropped with a `SKIP` line and ten remain. It runs one worker
at a time (`GPUTW_SLOTS` overrides); a second slot was measured to help only at
10k context, and 50k/137 is the largest cell ever attempted here.

`data/processed/.pool2/<ctx>:<line>:<shard>.uploaded` marks bytes that reached
the vault. Delete one to force re-upload. It no longer authorises a launch: the
pool launches only when the pod holds a working copy matching the vault's
length, so a marker over a truncated push now fails into a re-upload instead of
scoring a short matrix.

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

- **`/vault` persistence: ANSWERED, it survives.** The second pod mounted
  `192.168.7.2:/vault/user-...` with all eight shards staged by the first pod
  intact, including the 20k/137 tail that had stopped mid-push at 1975 of 2653
  MiB — `gputw_resumable_push.sh` picked it up from exactly there. Hydrating a
  2.7 GiB shard from the vault to `/workspace` took **2-3 seconds** against the
  ~18 minutes it would have cost to re-upload. This closes an item open since the
  runbook was written. A replacement pod should always hydrate before uploading.
- **Unscaled upload (runbook §6.3): implemented, not adopted.** Exports are
  pre-scaled, so each context re-sends a full matrix, where the unscaled matrix is
  identical across contexts and only a few-KB `scaler.npz` differs. The worker's
  `--scaler` path is implemented and verified to 2.4e-07, and `cmd_run` gates on a
  deliberate `features.UNSCALED` marker so an already-scaled matrix is never
  scaled twice; only the exporters lack the flag.

  The earlier "single biggest lever" estimate priced upload against a
  *serialised* run. The uploader runs as its own process, so its ~3.5 h for the
  ten remaining shards (16.8 GiB at 1.35 MB/s, worst case of a lost vault)
  overlapped compute of roughly 10 h, scaling the measured 20 min/shard at
  17 features/20k and ~35 min/shard at 137/10k, dominated by 50k/137. Adopting it
  would require re-exporting six 2.6 GiB matrices and a shared-vault-path case in
  the pool. Revisit if the vault is lost *and* the queue is reordered to be
  upload-led.
- **TabPFN's ceiling on 32 GB is unprobed.** The largest cell attempted, 50k
  context at 137 features, used **7.9 GiB of 32.6** with two workers resident;
  the single-worker figure was 3.5 GiB. The recorded `last_safe_budget: 100000`
  was measured on the 8 GB 4070, not this card.
- **Slot count: two, measured.** Sustained sampling puts one worker at 69.5%;
  utilization swings 30-100% as a worker alternates between attention and
  checkpoint writes, so a single `nvidia-smi` sample does not size a slot count.
  A second worker took the card to 99% and total output from 5.14 to
  6.0 chunks/min.
- **5k / 137 finished on the local 4070, not the rented card.** The pod's
  remaining work ran out first. The other three 5k shards did run remotely. See
  §2 for the localisation step the portable fit needs.
- No GitHub issue was opened for this work; recorded honestly per the
  change-checklist backfill policy rather than fabricated retroactively.
