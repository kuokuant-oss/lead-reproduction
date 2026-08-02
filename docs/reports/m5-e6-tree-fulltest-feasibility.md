# M5 E6 — fixed-tree full-test feasibility

## Measured throughput

Benchmarked on the laptop with **synthetic random float32 rows**, so no real
holdout row was scored and no holdout prediction was produced.

| Rows | Seconds | Rows/s | Peak RSS |
|---:|---:|---:|---:|
| 50,000 | 1.48 | 33,676 | 0.41 GB |
| 200,000 | 0.68 | 294,409 | 0.58 GB |
| 500,000 | 1.62 | 309,483 | 0.94 GB |

The 50,000-row figure is warm-up, not throughput. Steady state is
**about 309,000 rows/s** for the four-model ensemble.

Per model at 500,000 rows:

| Model | Seconds | Rows/s |
|---|---:|---:|
| lightgbm | 0.12 | 4,071,635 |
| xgboost | 0.06 | 8,959,835 |
| catboost | 0.18 | 2,784,915 |
| **hist_gradient_boosting** | **1.17** | **428,315** |

`hist_gradient_boosting` is 87% of the ensemble's inference time. If the tree
half ever needed to be faster, that is the only component worth touching — but
it does not need to be faster.

## Cost for the full holdout

| | |
|---|---:|
| one comparator over 10,137,155 rows | about 33 s |
| **24 comparators** | **about 13 minutes** |
| output, 24 x 10,137,155 float32 | 0.97 GB |
| peak RSS at a 500,000-row batch | under 1 GB |

Against the TabPFN half's 8.5 days, the tree half is about **0.1%** of the cost.

## Batch strategy

500,000-row batches keep RSS under 1 GB and sit in the flat part of the
throughput curve. 21 batches cover the holdout. Each batch writes atomically and
records its scaler identity, ensemble digest and row range, so a failure costs at
most one batch — about 1.6 seconds.

No special restart machinery is warranted at this cost. If a comparator fails,
rerun it.

## Gate before any holdout scoring

Unchanged from E5, and it is the reason the laptop is the execution environment
at all:

1. load the unit's `tree_ensemble.joblib` and `scaler.joblib`
2. verify the ensemble digest and the unit mapping
3. re-score E4's frozen 352-row query with the unit's exact scaler
4. require `max_abs_diff == 0` and **352/352 rows exact**
5. only after **24/24** pass may any holdout row be scored

No sampling, no tolerance, no refit. A single failure stops E6.

E5 ran this gate and got 24/24 bit-exact; on gpu-host the same ensembles matched
0/352 rows with a mean difference of 8.1e-03, which is comparable to the
TabPFN-minus-tree gap under test. That is why gpu-host tree output is prohibited
rather than merely discouraged.

## Feasibility verdict

The tree half is comfortably feasible on the laptop: about 13 minutes of
compute, under 1 GB of memory, and 1 GB of output. It imposes no constraint on
E6's schedule and needs no new engineering beyond the E5 gate.
