# TabPFN-3 500K single-context scaling

## Research question

Can one TabPFN-3 classifier on a single RTX 4070 Laptop GPU (8 GiB) use the
same complete context of 500,000 unique training rows for every prediction
batch with the 17 raw baseline features?

This additive experiment does not replace accepted M3/M5 metrics. A bounded
100K feasibility probe has completed, but the canonical 100K full-test run and
the later 200K--500K contexts have not been started.

## Single-context contract

A successful budget requires exactly one classifier, one fit call, one context,
one effective estimator, disabled sample subsampling, and completed non-empty
validation and test predictions. Fitted `n_train_samples_` must equal the
requested budget and `SUBSAMPLE_SAMPLES` must remain `None`.

TabPFN's internal memory-saving row chunking remains one context. External
multi-model sharding and averaging are forbidden. Query batching changes only
query rows per call, never training-context rows. An unverifiable installed API
stops as `blocked_unverified_context`.

## Data protocol

+ fit buildings: `building_id % 4 == 0`;
+ validation buildings: `building_id % 4 == 2`;
+ test buildings: `building_id % 2 == 1`;
+ features: `BASELINE_FEATURE_COLS` (raw 17 only);
+ contexts: deterministic balanced, unique, no-replacement nested prefixes;
+ score rows: fixed-seed natural-prevalence validation/test samples;
+ operating threshold: validation recall 0.90, applied unchanged to test.

Every budget archives its row-index SHA-256, first/last ten IDs, unique count,
and class counts.

## Isolation and watchdog

The parent controller never imports torch or TabPFN. Preflight and each budget
run in disposable subprocesses. Atomic summary/state files and an append-only
event log preserve completed work.

GPU monitoring prefers per-process `nvidia-smi` readings and falls back to
device-total WDDM readings, which include desktop and unrelated processes.
`psutil` records worker RSS and system RAM. Consecutive soft-limit samples write
a stop request checked between prediction batches. Hard limits, stale heartbeat,
or expired grace terminate and then kill the process tree. Model wall-time is
unlimited by default (`--budget-timeout-minutes 0`); a positive value remains
available only for an explicitly requested run.

A watchdog cannot guarantee interception of an instantaneous CUDA OOM. Query
OOM may halve only query batch size down to the configured minimum. Training
rows are never silently reduced.

On every exit path the worker drops its model, matrices, and score arrays,
runs garbage collection, and clears CUDA caches. The worker then exits, which
is the definitive RAM/VRAM release boundary. Under hard pressure the parent
recursively terminates and, after the grace period, kills the worker process
tree; completed budgets remain available for resume.

## Curve-ready output

Aggregate AUC values alone cannot reconstruct ROC or precision-recall curves.
Each completed budget therefore atomically saves a compressed scoring artifact
under `m5_tabpfn_single_context_scaling.predictions/`. It contains validation
and test labels, TabPFN probabilities, row IDs, building IDs, and site IDs, but
never the training matrix. Its path, SHA-256, size, and row counts are recorded
in the summary JSON.

The offline `plot_m5_tabpfn_single_context_curves.py` renderer creates four
analogs of the referenced M3 figures: pooled scaling ROC/PR comparisons and
largest-completed-budget ROC/PR panels by site. It does not load TabPFN or make
new predictions. A 4,000-row natural-prevalence sample can leave a rare site
without both classes; such a panel is explicitly marked `Not estimable`
instead of inventing a curve. To guarantee estimable site panels, enable
`--site-curve-rows-per-class`; those extra stratified queries run only for
`--site-curve-budget` (500K by default), remain separate from pooled metrics,
and are saved in the same artifact. Exact M3 feature-engineering semantics
cannot be claimed because this experiment intentionally uses raw 17 features
only.

## Current lightweight verification

+ TabPFN version observed: `8.0.8`.
+ Required constructor parameters are present, including estimator scaling,
  low-memory mode, memory-saving mode, and inference config overrides.
+ Current venv reports `torch 2.12.1+cu126`; CUDA is available on the RTX 4070
  Laptop GPU. The minimal CUDA allocation test passed and released its context.
+ Thirty focused unit/mock/fake-subprocess tests pass, including fitted
  state reload and full prediction-checkpoint reuse.
+ Fake smoke budgets 200 and 500 complete without initializing CUDA.
+ Fake-smoke artifacts successfully render all four offline figure types.

These checks do not establish any formal 100K--500K result.

## 100K feasibility and query-batch gate

The first 100K run was deliberately bounded to 4,000 validation plus 4,000
test queries under the older mod-4 feasibility split. It established that one
100,000-row, 17-feature context fits and predicts without internal context
subsampling, but it is not the canonical M3 comparison result:

+ fit: 0.84 seconds;
+ validation: 355.27 seconds, 11.26 rows/second;
+ test: 366.98 seconds, 10.90 rows/second;
+ Torch peak allocated/reserved: 3,431.7/3,826.0 MiB;
+ watchdog peak device GPU: 4,712 MiB;
+ watchdog peak worker process-tree RSS: 4,981.2 MiB.

A separate capacity gate rejected query microbatch 1,024 after it crossed the
GPU hard limit. Microbatch 512 completed at 23.67 validation rows/second and
23.19 test rows/second, with 5,986 MiB Torch reserved, 6,770 MiB peak device
GPU, and 4,947.3 MiB worker RSS. Therefore the canonical run uses query
microbatch 512. At the observed rate, 10,137,155 rows require approximately
121 hours; this is an estimate, not a completed result.

## Canonical full-test resumability

`run_m5_tabpfn_canonical_full_test.py` separates GPU query microbatches from
durable disk checkpoints:

+ context source: the exact M3 training half, `building_id % 2 == 0`, with a
  deterministic 100K balanced sample after reserving fixed validation rows;
+ prediction target: every row of the exact M3 test half,
  `building_id % 2 == 1` (10,137,155 rows). Target order comes from the M6 A0
  artifact, whose raw row IDs correctly map back to features and whose labels
  and ensemble scores are row-for-row identical to M3. The older M3 artifact's
  `validation_raw_index` is intentionally not used because it is known not to
  align with its saved labels/scores;
+ query microbatch: 512 rows; durable checkpoint: 20,000 rows;
+ immediately after fit, the official TabPFN
  `save_fitted_tabpfn_model` API atomically stores `model.tabpfn_fit`; the fitted
  `StandardScaler` and a context/model manifest are also atomically stored;
+ restart validates the manifest, calls `load_fitted_tabpfn_model`, and skips
  the saved validation result plus every already-complete prediction
  checkpoint. An interruption during one unfinished checkpoint loses at most
  that checkpoint; an interruption before the fitted model archive is
  committed requires refitting.

The parent watchdog has no wall-time limit. It continues to protect GPU and
RAM, and worker exit remains the fast whole-process memory-release boundary.
The contract also verifies the 17-feature by-site artifact, the 137-feature M3
artifact, and the A0 site artifact are row-aligned. The resulting TabPFN scores
can therefore be added to the two feature-engineering ROC/PR figures and the
two tree-ensemble-by-site ROC/PR figures named in the research request.

## Full no-fit preflight

The full-data preflight completed with `status = ready`; it did not call
`.fit()` or `predict_proba()`. All nine gates passed:

+ CUDA: `torch 2.12.1+cu126`, RTX 4070 Laptop GPU, 8,187.5 MiB;
+ TabPFN: `8.0.8`, local checkpoint SHA-256
  `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`;
+ context API: `SUBSAMPLE_SAMPLES=None` and source routing verified;
+ split rows: 5,035,406 fit / 5,043,539 validation / 10,137,155 test,
  with zero building overlap;
+ 500K context: 250,000 unique positives plus 250,000 unique negatives;
+ fixed scoring: 4,000 validation and 4,000 natural-prevalence test rows;
+ by-site scoring: 512 rows, 16 positives and 16 negatives per site;
+ WDDM monitoring: device-total scope; preflight GPU baseline 574 MiB;
+ limits: GPU soft/hard 7,041.68/7,532.96 MiB and no wall-time limit.

The preflight process tree exited normally and device memory returned to the
desktop baseline. This establishes readiness only, not model feasibility.

## Small real-GPU gate

A separate 200-row context gate completed on the real checkpoint and GPU. The
first attempt exposed a Windows heartbeat temporary-file race after fit and
both predictions had already succeeded. A write lock fixed the race; worker
RSS monitoring was also corrected to sum the Windows launcher process tree.
The repeated gate then completed end to end:

+ requested/effective context rows: 200/200;
+ effective estimators: 1; sample subsampling disabled;
+ fit / validation / test prediction: 0.82 / 6.47 / 0.62 seconds;
+ validation ROC-AUC / PR-AUC: 0.9939 / 0.9406;
+ test ROC-AUC / PR-AUC: 0.9579 / 0.6530;
+ Torch peak allocated/reserved: 232.6/252.0 MiB;
+ watchdog peak device GPU: 1,203 MiB;
+ watchdog peak worker process-tree RSS: 4,840.6 MiB;
+ scoring artifact: 4,614 bytes, SHA-256
  `b22e060de7012e16d9648c8dbec98e4a0eabebe737631fab8ce10c9ff07afd5`.

The worker exited, GPU memory returned to the desktop baseline, and all four
offline figure types rendered from the artifact without reloading TabPFN. The
gate used natural-prevalence rows only, so rare site panels being not estimable
is expected; the formal 500K budget enables the separate stratified site rows.

## Scaling results

| Context rows | Status | Validation ROC/PR | Test ROC/PR | Peak GPU |
| ---: | --- | --- | --- | ---: |
| 100,000 | Bounded feasibility only; canonical full test not run | -- | -- | 4,712 MiB (batch 256 probe) |
| 200,000 | Not run | -- | -- | -- |
| 300,000 | Not run | -- | -- | -- |
| 400,000 | Not run | -- | -- | -- |
| 500,000 | Not run | -- | -- | -- |

Last safe formal budget: **not established**. 500K success: **not
established**. `headline_500k_success` remains false.

## Commands

Machine-local CUDA Torch installation used for this run environment:

```powershell
uv pip install --python .venv\Scripts\python.exe --reinstall `
  "torch==2.12.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

Lightweight fake-model smoke:

```powershell
python scripts/run_m5_tabpfn_single_context_scaling.py `
  --budgets 200 500 --score-rows 100 --predict-batch-size 32 --smoke
```

Full-data/API preflight, with no model fit:

```powershell
python scripts/run_m5_tabpfn_single_context_scaling.py --preflight-only
```

Render curves after one or more budgets complete:

```powershell
python scripts/plot_m5_tabpfn_single_context_curves.py
```

Formal run, to be invoked only after explicit operator approval:

```powershell
python scripts/run_m5_tabpfn_single_context_scaling.py `
  --budgets 100000 200000 300000 400000 500000 `
  --score-rows 4000 --predict-batch-size 256 `
  --budget-timeout-minutes 0 `
  --max-budgets-this-run 1 `
  --site-curve-rows-per-class 16 --site-curve-budget 500000 `
  --gpu-soft-limit-fraction 0.86 --gpu-hard-limit-fraction 0.92 `
  --ram-soft-limit-fraction 0.85 --ram-hard-limit-fraction 0.92 `
  --resume
```

Canonical 100K context against the complete M3 test half, resumable without a
wall-time limit:

```powershell
python scripts/run_m5_tabpfn_canonical_full_test.py `
  --context-rows 100000 `
  --query-microbatch-size 512 --min-query-microbatch-size 256 `
  --checkpoint-rows 20000 `
  --gpu-soft-limit-fraction 0.86 --gpu-hard-limit-fraction 0.92 `
  --ram-soft-limit-fraction 0.85 --ram-hard-limit-fraction 0.92 `
  --resume
```

## Limitations

+ TabPFN behavior is version-specific; rerun preflight after upgrades.
+ Fit may be cheap while the first context forward during prediction OOMs.
+ WDDM device-total monitoring is conservative.
+ Formal results must come from generated artifacts, never smoke or fit-only
  completion.
+ The older bounded 100K mod-4 probe cannot be plotted as the canonical M3
  comparison and must never be reported as that result.
+ `--max-budgets-this-run 1` exits as `paused_after_budget_limit` after one
  successful budget; invoke the same command with `--resume` for the next one.
