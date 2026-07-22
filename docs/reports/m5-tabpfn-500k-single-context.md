# TabPFN-3 500K single-context scaling

## Research question

Can one TabPFN-3 classifier on a single RTX 4070 Laptop GPU (8 GiB) use the
same complete context of 500,000 unique training rows for every prediction
batch with the 17 raw baseline features?

This additive experiment does not replace accepted M3/M5 metrics. The formal
100K--500K run has not been started.

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
+ Twenty unit/mock/fake-subprocess tests pass.
+ Fake smoke budgets 200 and 500 complete without initializing CUDA.
+ Fake-smoke artifacts successfully render all four offline figure types.

These checks do not establish any formal 100K--500K result.

## Scaling results

| Context rows | Status | Validation ROC/PR | Test ROC/PR | Peak GPU |
| ---: | --- | --- | --- | ---: |
| 100,000 | Not run | -- | -- | -- |
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

## Limitations

+ TabPFN behavior is version-specific; rerun preflight after upgrades.
+ Fit may be cheap while the first context forward during prediction OOMs.
+ WDDM device-total monitoring is conservative.
+ Formal results must come from generated artifacts, never smoke or fit-only
  completion.
+ `--max-budgets-this-run 1` exits as `paused_after_budget_limit` after one
  successful budget; invoke the same command with `--resume` for the next one.
