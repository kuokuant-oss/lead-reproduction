# M5.x partition-granularity handoff

**Non-report handoff notes.** The report narrative lives at [docs/reports/m5x-partition-granularity.md](../reports/m5x-partition-granularity.md); this file records run mechanics, timings, and known limits for follow-up.

## Run

- Smoke mode: `False`
- Value-change regime: `timestamp_merge(causal77)` (77 features: 17 baseline + 60 causal lag)
- Configs this run: `C1 C2 C3 C4` (single write, no merge helper)
- Single seed: `--seeds 42` (one model seed; `--seed 42` for eval/unit sampling)
- Budgets: `--fit-rows 10000 --eval-rows 4000 --calib-rows 4000 --max-units 400`
- Device: `cuda`, `tabpfn_ok=True` (local checkpoint `.tabpfn-cache/tabpfn-v3-classifier-v3_default.ckpt`)
- Executed command:
  `uv run python scripts/run_m5x_partition_granularity.py --seeds 42 --fit-rows 10000 --eval-rows 4000 --calib-rows 4000 --max-units 400 --out data/processed/m5x_partition_granularity.json`
- JSON: `data/processed/m5x_partition_granularity.json`
- Total elapsed: `15853.0s` (~4.4h), exit code 0.

## Per-stage / per-config timings

| Stage | Seconds |
|---|---:|
| C3 fallback cache build | 170.1 |
| C1 `(building_id, meter)` | 11817.9 (~3.28h) |
| C2 `(site_id, meter)` | 1479.8 (~24.7m) |
| C3 `(meter,)` | 173.1 |
| C4 `(primary_use, meter)` | 1627.5 (~27.1m) |

C1 dominates wall-clock (~75% of the run). Within C1 the heaviest model fits are CatBoost (`8026s`) and TabPFN (`68063s` summed fit_predict across trained units — GPU, serialized per unit).

## C1 performance limit and future optimization

C1 has `1,928` building×meter units in eval. With `max_units=400`, 400 are attempted and `303` are trainable (scorable train + non-empty calib); the remaining `1,625` units (`1,528` not selected + `97` non-scorable) fall back to the C3 meter-level cache. Coverage is only `15.25%` of eval rows; `84.75%` fall back.

Current cost driver: the seed loop processes each of the ~1,928 units **one at a time** (`seed_summary` iterates `unit_index_map`), even though every fallback unit only needs a cache lookup + row routing to a C3 meter model. The per-unit Python loop over 1,600+ fallback units is the bottleneck, not model fitting for those units.

Optimization (deferred until after multi-seed): vectorize the fallback path by **meter** — compute C3 predictions once per meter and scatter them to all fallback eval rows of that meter in one pass, instead of looping unit-by-unit. This does not change any computed value (fallback rows already read from the same C3 cache); it only removes redundant per-unit Python overhead. **Do not vectorize before the multi-seed run** — keep the current, verified single-write path for correctness first.

Do **not** re-introduce the two-pass merge helper. The clean script writes all requested configs once in `main()`; the earlier `merge_existing_results` / `recompute_top_level_fairness` path was removed (it hit a KeyError against old-schema JSON).

## Known limitations

- **Fixed-recall operating point double-counts fallback calib.** Fallback eval rows are scored by C3 meter models; the pooled `fixed_recall_0_90` threshold folds the corresponding C3 calib subsample in per routed row rather than deduplicating per C3 meter. This affects only `confusion@recall0.90` operating-point counts. It does **not** affect ROC-AUC or PR-AUC, which are threshold-free and carry every model comparison in the report.
- **Macro per-unit medians are mostly `1.0` for C1/C2** because scorable per-unit counts are small (`n_scorable` = 47 / 24 / 4 / 21 for C1/C2/C3/C4) and most scorable units are perfectly separable. Pooled ROC/PR is the headline metric; macro medians are only informative at C3.
- **Single seed.** All numbers are seed 42 only; no seed-to-seed band yet.

## Fairness asserts (computed this run)

| Assert | Result |
|---|---|
| `eval_idx_sha_all_equal` | True (four configs share one `eval_idx` sha256) |
| `calib_idx_sha_all_models_equal_within_scope` | True |
| `no_future_leak` | True (per trained unit, train timestamps precede test) |
| `no_future_leak_global` | True |
| `feature_count` | 77 |

## Headline numbers (seed 42, pooled)

Best tree PR-AUC vs TabPFN PR-AUC per config:

| Config | Coverage | Best tree PR-AUC | TabPFN PR-AUC | Gap |
|---|---:|---:|---:|---:|
| C1 `(building_id, meter)` | 0.1525 | 0.8168 (Ensemble) | 0.7369 | −0.0799 |
| C2 `(site_id, meter)` | 0.9975 | 0.8504 (CatBoost) | 0.8210 | −0.0294 |
| C3 `(meter,)` | 1.0000 | 0.8606 (Ensemble) | 0.6983 | −0.1623 |
| C4 `(primary_use, meter)` | 0.9978 | 0.8380 (Ensemble) | 0.7452 | −0.0928 |

Tree ensemble leads pooled PR-AUC at every granularity. TabPFN is closest at C2, worst at C3.

## Open questions

- Run the multi-seed band (`MODEL_SEEDS`) after the C1 fallback path is vectorized, so C1 wall-clock is tractable.
- C1's building-level signal is effectively untested (only 15% coverage). If building-level granularity matters, raise `max_units` well above 400 or restrict eval to units with scorable train history before comparing.
