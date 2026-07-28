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

What is left is analysis, not compute. The disaggregated tables are in
`docs/reports/m5-matched-context-breakdown.md` (§3.0.1). The one measurement
still worth GPU time is TabPFN's own context-draw noise, which has never been
measured and is currently proxied by the trees' (§3.1).

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

**Running the last cell locally.** The 137-feature 5k shards are scored on the
local 4070 because the remote work was finished and keeping a pod alive for one
cheap cell is not worth the per-second bill. The head took 4 h 39 m at ~1
chunk/min, matching the estimate; the tail is paused partway (§0 has the exact
resume command). The portable fit carries the *pod's* checkpoint path, so it must
be localised first or it fails as `TabPFNLicenseError` (§5 trap 3) — this was
already done for both 5k/137 shards, so §0 only needs the run step:

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

## 3. Results so far, and the measured numbers that replace the estimates

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

**The headline answer is no.** At 17 features TabPFN loses at every N, by 40-120
sd of the draw noise in §3.1. At 137 features it ties on ROC (|diff| <= 0.0006 at
every N) and holds a consistent but tiny PR lead of +0.0008 to +0.0036, which is
1-5 sd. It never wins by an amount anyone would act on.

**With 5k merged, the PR lead on the 137 line is largest at the smallest N.**
+0.0036 at 5k, then +0.0017, +0.0023, +0.0024, +0.0008 as N rises. So the
hypothesis that TabPFN's advantage grows as labelled data shrinks is confirmed on
this line as well as refuted in magnitude: 5 sd of the 0.0007 PR floor, and ROC
stays flat at -0.0001. The ordering across the middle three points is not
monotone and is inside 2 sd, so the defensible claim is "largest at 5k", not "a
monotone trend".

**The hypothesised direction is confirmed and does not rescue the claim.** On the
17-feature line the deficit shrinks monotonically as data shrinks -- -0.0487 at
100k to -0.0158 at 5k -- and TabPFN's own best point on that line is its smallest
context. It is still 40 sd behind there. What the curve shows is not TabPFN
excelling at small N but TabPFN *degrading with context*: 0.9451 down to 0.9163
while the trees sit flat between 0.9609 and 0.9665.

**TabPFN at 137 features peaks at 50k, not at 100k.** An earlier draft of this
file called the 137 line monotone rising; with the 50k point merged it is not.
0.9924 at 50k falls to 0.9919 at 100k, a -0.0005 ROC / -0.0010 PR step against a
0.0003 / 0.0007 noise floor -- about 1.5 sd, so the defensible reading is that the
line stops improving after 50k, not that it declines.

**Read every number above against the one-feature baseline.** Ranking the holdout
by `-meter_reading` alone -- no model, no training -- scores **ROC 0.8998,
PR 0.5143** on the same 10,137,155 rows. Of the seventeen features it is the only
strong one (single-feature |AUC-0.5|: meter_reading 0.419, dayofyear 0.155,
month 0.154, everything else below 0.11), and the direction is that *low*
readings are anomalous. This is why 10,000 labelled rows is already enough for
both model families: most of the decision is a threshold on one raw column, and
5,000 positives calibrate it well. It also means ROC-AUC is a compressed axis
here -- it starts at 0.90 before anyone models anything -- while PR-AUC still
separates the methods (0.514 baseline, 0.762 TabPFN, 0.826 trees at 17f/10k).
Quote the baseline beside any model number, or a reader cannot tell how much of
it is the model.

A second thing to state rather than bury: `SHIFTS` is symmetric, +-1..24 h plus
+-48..168 h, so the 137-feature line gives each row up to **seven days of its own
meter's future**. That is a defensible framing for retrospective detection, but
it is not forecasting, and it is most of why 137 features reaches 0.99.

**The 17-feature line falls with context; the 137-feature line rises to 50k and
then stops.** Across 5k to 100k the 17 line moves ROC -0.0288 / PR -0.0775, and
does so monotonically at all five points. The 137 line climbs +0.0022 ROC from
10k to its 50k peak and gives back -0.0005 by 100k.

Both movements clear the noise floor, measured per line by redrawing the 100k
context under four seeds and refitting the trees (§3.1): **ROC sd 0.0003 at 137
features, 0.0004 at 17**. The 17-feature fall is ~70 sd; the 137-feature rise to
its peak is ~7 sd. Nested prefixes mean adjacent contexts share rows, so the
variance of a *difference* is smaller still than these independent-draw figures
imply.

This paragraph has been wrong twice, in opposite directions, and the sequence is
worth keeping as a caution. It first called the 137 rise settled on two points,
which it was not. Challenged, it then called the rise "not a finding" by applying
the 17-feature sd of 0.0004 -- an over-retraction, since the line's own sd is
0.0003. Merging the 50k point then showed the line is not monotone at all: it
peaks at 50k. Each revision was driven by data arriving, not by argument, which is
the only reason to trust the current one.

Caveat that the numbers cannot remove: the sd is the *tree's* draw noise, used as
a proxy for TabPFN's. TabPFN learns in-context, so it may well be more sensitive
to which rows the context holds than a fitted GBDT is, and 0.0003 is then an
underestimate for the TabPFN curve. Measuring it directly needs GPU time to
re-score new contexts, and has not been done.

### 3.0 The pooled metric hides the result

The single most misleading number in this experiment is the pooled one. Trees at
17 features, 5,000 rows versus 100,000 rows, PR-AUC per site:

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

Twentyfold more data moves the pooled figure by +0.0083, and that near-flatness
is the *net of +0.18 and -0.12 moving in opposite directions*. Sites 4 and 1
improve substantially; site 6 gets materially worse; site 11 is unsolved at both
sizes (PR 0.018). The pooled number is held up by a few large easy sites -- site
0 alone is 538k rows at PR 0.998.

**So "2,500 anomalies is enough" is a statement about the pooled metric, not
about the problem.** Any claim in this work about data volume mattering or not
mattering has to be made per site, or it is an artefact of aggregation. This is
the full version of the redistribution effect noted for the 17-feature line in
§3: only 9 of 16 sites improve, and the pooled gain is dominated by a handful.

### 3.0.1 The disaggregated grid

`docs/reports/m5-matched-context-breakdown.md` carries the full per-site and
per-meter tables for both models at every context, each read against the
full-training-set tree for its own feature line. Regenerate with
`scripts/report_m5_matched_context_breakdown.py`. Two things it shows that the
pooled table cannot:

- **The 137-feature "tie" is a cancellation, not a tie.** From 10k up, TabPFN
  beats the trees on chilledwater (+0.05 PR at 100k) and steam (+0.08), and
  loses on electricity (-0.007) -- which is 60% of the rows and 56% of the
  anomalies, so it alone pulls the pooled figure back to zero. Per-meter gaps run
  one to two orders of magnitude above the pooled +0.0008.
- **The 5k point is not a smaller version of the same picture.** At 5k the only
  meter TabPFN wins is chilledwater, and it wins it by +0.109; steam flips to
  -0.041 and hotwater to -0.102. So the pooled +0.0036 at 5k -- the grid's
  largest TabPFN lead -- is one meter carrying three, not a uniform edge. Do not
  describe the per-meter result without saying which N it is at.
- **More data is not uniformly better.** At 17 features the full-data tree scores
  PR 0.1436 at site 4 and 0.0188 at site 11, *below* the same trees capped at
  5,000 rows (0.5547) and below TabPFN at 5,000 (0.7416).
- **The thin sites survive their error bars.** A paired row bootstrap (200 draws)
  puts the sd of the PR difference at 0.008-0.037 for the 197-anomaly sites, so
  site 4's +0.1869 at 17f/5k and site 11's -0.3210 at 137f/10k are 7x and 13x
  their sampling noise. The caution that these sites are too thin to compare was
  written before the measurement and is wrong; what remains unmeasured for them
  is the *context* draw, not the row draw.

### 3.1 Every number here comes from one draw

The context is a single frozen sample: seed 42, digest `e9ffe0cf`, nested
prefixes, reused by every TabPFN context and every tree cell. There is no
repetition and there are no error bars anywhere in the grid. That is what makes
the effect sizes above the only defensible unit of judgement.

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

The frozen draw sits below the redraw mean on both lines (1.22 sd at 17
features, 0.25 sd at 137), which disposes of the worry that it was a favourable
sample. What the floors license:

- TabPFN minus trees at 17 features (ROC -0.0221 to -0.0487): 55-120 sd, solid.
- TabPFN minus trees at 137 features (|ROC| <= 0.0001 at 10k, 20k and 100k):
  under 0.35 sd at every context. A tie, now at a tighter tolerance.
- The 137 context rise (+0.0017 cumulative): 5.7 sd. Real, and negligible.

Applying a line's sd to the other line, or to TabPFN rather than the trees, is an
assumption -- and it has already produced one wrong call in this file.

Leakage was audited empirically rather than argued
(`scripts/audit_m5_matched_context_leakage.py`): reconstructed digest matches both
Gate 1 and the digest the tree run recorded, 100,000 unique rows exactly 50/50,
all fit buildings even and all holdout buildings odd with zero shared buildings,
zero shared row ids, and 385 of 2,000,000 sampled holdout rows byte-identical to
a fit row on 17 features plus label (0.019%, cross-building feature collisions). Per-site, the 17-feature 10k-vs-100k gain is a
redistribution rather than a lift: only 9 of 16 sites improve on ROC, and pooled
PR-AUC is dominated by Site 2 (+0.297) with Sites 6 and 10 behind it, while Sites
1, 7 and 9 get worse.

Measured on the RTX 5090, replacing runbook §9's extrapolation:

- **17 features @ 20k context: ~4,000 rows/s** — 253 chunks in ~20 min, about
  3.3x the projection.
- **Uplink: ~1.35 MB/s on the first pod, ~2.1-2.5 MB/s on the second.** Measured
  on the second pod: 348 MB in 2m39s and 2m44s (2.19 and 2.12 MB/s), then
  2,653 MiB in 18m50s (2.46 MB/s). So it is *not* the hard ceiling the first
  session's number suggested -- it varies by pod, and budgeting a resume should
  measure rather than assume. What does hold is that **parallelism does not
  help**: two streams measured 1,237 KiB/s combined versus 1,384 KiB/s for one.
  Only sending fewer bytes, or sending them off the critical path, does.
- One 17-feature worker at 20k context pins the GPU at 100%. The "two slots give
  1.58x" note in `gputw_tabpfn_pool.sh` was measured at 10k, where a single worker
  reached only 29%. Do not assume it generalises.

**The run is upload-bound, not compute-bound.**

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
- **Unscaled upload (runbook §6.3) is no longer worth doing, and the earlier
  "single biggest lever" estimate was wrong.** The mechanism is real: exports are
  pre-scaled, so each context re-sends a full matrix, where the unscaled matrix is
  identical across contexts and only a few-KB `scaler.npz` differs. The worker's
  `--scaler` path is implemented and verified to 2.4e-07, and `cmd_run` gates on a
  deliberate `features.UNSCALED` marker so an already-scaled matrix is never
  scaled twice; only the exporters lack the flag.

  What changed is the denominator. That estimate priced upload against a
  *serialised* run. The uploader now runs as its own process, so its ~3.5 h for
  the ten remaining shards (16.8 GiB at 1.35 MB/s, worst case of a lost vault)
  overlaps compute that is far longer: scaling the measured 20 min/shard at
  17 features/20k and ~35 min/shard at 137/10k gives roughly 10 h for what is
  left, dominated by 50k/137. The run is **compute-bound now, not upload-bound**,
  and deduplicating uploads would only shorten the wait before the *first* worker
  starts -- which hydrating from the vault already handles. It is not worth
  re-exporting six 2.6 GiB matrices and adding a shared-vault-path special case
  to a pool whose failure modes fill §5. Revisit only if the vault is lost *and*
  the queue is reordered to be upload-led.
- **TabPFN's ceiling on 32 GB is still unprobed, and now looks far away.** The
  largest cell attempted, 50k context at 137 features, used **7.9 GiB of 32.6**
  with two workers resident — the single-worker figure was 3.5 GiB. The recorded
  `last_safe_budget: 100000` came from the 8 GB 4070 and badly understates this
  card. Where TabPFN actually stops is still a reportable result nobody has
  measured.
- **Slot count: two, measured, not one.** An earlier revision defaulted to one
  slot on the strength of a single instantaneous `nvidia-smi` sample reading 100%.
  Sustained sampling puts one worker at 69.5% — utilization swings 30-100% as a
  worker alternates between attention and checkpoint writes, so a single sample is
  worthless. A second worker took the card to 99% and total output from 5.14 to
  6.0 chunks/min. Never size a slot count from one sample.
- **5k / 137 finished on the local 4070, not the rented card**, because the pod's
  remaining work ran out first and one cheap cell does not justify per-second
  billing. The other three 5k shards did run remotely. See §2 for the localisation
  step the portable fit needs.
- No GitHub issue was opened for this work; recorded honestly per the
  change-checklist backfill policy rather than fabricated retroactively.
