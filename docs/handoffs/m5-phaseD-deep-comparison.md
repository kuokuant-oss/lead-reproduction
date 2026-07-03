# M5 Phase D deep comparison handoff

**Non-report handoff notes.** These numbers are for follow-up consumption; no `docs/reports/` narrative was updated.

## Run

- Smoke mode: `False`
- Requested axes this run: `stability_multiseed`
- JSON merge mode: `merge_preserve_unrequested_axes`
- Executed command: `uv run python scripts/run_m5_phaseD_deep_comparison.py --out C:\Users\tonykuo\projects\lead-reproduction\data\processed\m5_phaseD_deep_comparison.json --handoff C:\Users\tonykuo\projects\lead-reproduction\docs\handoffs\m5-phaseD-deep-comparison.md --fit-rows 10000 --score-rows 4000 --scarcity-sizes 20 50 100 150 300 500 1000 2000 --tune-trials 12 --seed 42`
- JSON: `C:/Users/tonykuo/projects/lead-reproduction/data/processed/m5_phaseD_deep_comparison.json`
- Full command for later real run: `uv run python scripts/run_m5_phaseD_deep_comparison.py --out data/processed/m5_phaseD_deep_comparison.json --handoff docs/handoffs/m5-phaseD-deep-comparison.md --fit-rows 10000 --score-rows 4000 --scarcity-sizes 20 50 100 150 300 500 1000 2000 --tune-trials 12 --seed 42 --axes default_vs_tuned sample_efficiency_fine dimensionality_at_small_n stability_multiseed`
- Value-change regime: `row_offset_meter_aware`

## Operating-Point Note

The test `threshold_0_5` and `fixed_recall_0_90` entries are post-hoc operating points. In particular, `fixed_recall_0_90` derives its threshold from the same split's labels, including test labels for the test summary. These entries are descriptive only and do not represent deployable performance. Model comparison and TabPFN-vs-tree claims should use threshold-free ROC-AUC / PR-AUC. For deployable operating points, choose thresholds on val and apply them once to test.

## Axis Headlines

### Axis 1: default_vs_tuned

Default trees:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.985839 | 0.859080 | 0.906083 |
| xgboost | completed | 0.981338 | 0.838969 | 0.895802 |
| catboost | completed | 0.978004 | 0.874993 | 0.881166 |
| hist_gradient_boosting | completed | 0.985067 | 0.849316 | 0.903388 |
| ensemble | completed | 0.986537 | 0.864730 | 0.902241 |

Tuned trees:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.983832 | 0.880182 | 0.914017 |
| xgboost | completed | 0.975509 | 0.871196 | 0.885947 |
| catboost | completed | 0.985876 | 0.874208 | 0.898999 |
| hist_gradient_boosting | completed | 0.973216 | 0.869045 | 0.891046 |
| ensemble | completed | 0.984344 | 0.867108 | 0.907967 |

TabPFN:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| tabpfn | completed | 0.983259 | 0.884399 | 0.907109 |

### Axis 2: sample_efficiency_fine

Support 20:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.074500 | 0.070750 | 0.068000 |
| xgboost | completed | 0.261644 | 0.239557 | 0.270101 |
| catboost | completed | 0.820090 | 0.743888 | 0.701554 |
| hist_gradient_boosting | completed | 0.074500 | 0.070750 | 0.068000 |
| ensemble | completed | 0.639322 | 0.595696 | 0.606189 |
| tabpfn | completed | 0.829324 | 0.773245 | 0.746472 |

Support 50:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.211130 | 0.198297 | 0.211744 |
| xgboost | completed | 0.358326 | 0.326783 | 0.355366 |
| catboost | completed | 0.786942 | 0.726805 | 0.751102 |
| hist_gradient_boosting | completed | 0.269947 | 0.277936 | 0.286139 |
| ensemble | completed | 0.552727 | 0.510103 | 0.573012 |
| tabpfn | completed | 0.808526 | 0.744482 | 0.769644 |

Support 100:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.695847 | 0.640239 | 0.670819 |
| xgboost | completed | 0.669947 | 0.572047 | 0.602547 |
| catboost | completed | 0.838814 | 0.764535 | 0.773485 |
| hist_gradient_boosting | completed | 0.659608 | 0.618436 | 0.674376 |
| ensemble | completed | 0.805376 | 0.740781 | 0.739646 |
| tabpfn | completed | 0.845072 | 0.754635 | 0.749428 |

Support 150:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.791319 | 0.730776 | 0.701549 |
| xgboost | completed | 0.737409 | 0.652730 | 0.669473 |
| catboost | completed | 0.862405 | 0.769917 | 0.776618 |
| hist_gradient_boosting | completed | 0.765320 | 0.719703 | 0.712828 |
| ensemble | completed | 0.843607 | 0.769550 | 0.757751 |
| tabpfn | completed | 0.904298 | 0.813724 | 0.821711 |

Support 300:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.839594 | 0.783440 | 0.778769 |
| xgboost | completed | 0.795263 | 0.778733 | 0.715601 |
| catboost | completed | 0.868607 | 0.807666 | 0.792656 |
| hist_gradient_boosting | completed | 0.813561 | 0.785555 | 0.762527 |
| ensemble | completed | 0.859860 | 0.818607 | 0.783001 |
| tabpfn | completed | 0.909067 | 0.829704 | 0.822686 |

Support 500:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.877308 | 0.792147 | 0.767814 |
| xgboost | completed | 0.833793 | 0.778056 | 0.742131 |
| catboost | completed | 0.868640 | 0.805313 | 0.778269 |
| hist_gradient_boosting | completed | 0.851493 | 0.766653 | 0.774985 |
| ensemble | completed | 0.881566 | 0.818521 | 0.783921 |
| tabpfn | completed | 0.914610 | 0.834277 | 0.829391 |

Support 1000:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.902920 | 0.806842 | 0.843864 |
| xgboost | completed | 0.908294 | 0.819851 | 0.855639 |
| catboost | completed | 0.905274 | 0.819902 | 0.808810 |
| hist_gradient_boosting | completed | 0.888104 | 0.804734 | 0.826895 |
| ensemble | completed | 0.918701 | 0.828663 | 0.837320 |
| tabpfn | completed | 0.926020 | 0.843846 | 0.850553 |

Support 2000:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.936145 | 0.823146 | 0.859249 |
| xgboost | completed | 0.934610 | 0.835361 | 0.854365 |
| catboost | completed | 0.911345 | 0.824729 | 0.810279 |
| hist_gradient_boosting | completed | 0.943799 | 0.826248 | 0.870220 |
| ensemble | completed | 0.931268 | 0.837579 | 0.832803 |
| tabpfn | completed | 0.930883 | 0.825895 | 0.828838 |

### Axis 3: dimensionality_at_small_n

raw_baseline_17 (17 features):
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.731594 | 0.618060 | 0.652225 |
| xgboost | completed | 0.707357 | 0.622807 | 0.659961 |
| catboost | completed | 0.775240 | 0.732718 | 0.736335 |
| hist_gradient_boosting | completed | 0.695014 | 0.620048 | 0.648094 |
| ensemble | completed | 0.781019 | 0.716322 | 0.731387 |
| tabpfn | completed | 0.801046 | 0.762183 | 0.782129 |

baseline_plus_first_33_value_change_50 (50 features):
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.836884 | 0.731889 | 0.729007 |
| xgboost | completed | 0.807624 | 0.724238 | 0.714079 |
| catboost | completed | 0.850146 | 0.792821 | 0.779427 |
| hist_gradient_boosting | completed | 0.834393 | 0.738158 | 0.736078 |
| ensemble | completed | 0.863387 | 0.797391 | 0.777629 |
| tabpfn | completed | 0.900676 | 0.810609 | 0.844328 |

full_137 (137 features):
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.877308 | 0.792147 | 0.767814 |
| xgboost | completed | 0.833793 | 0.778056 | 0.742131 |
| catboost | completed | 0.868640 | 0.805313 | 0.778269 |
| hist_gradient_boosting | completed | 0.851493 | 0.766653 | 0.774985 |
| ensemble | completed | 0.881566 | 0.818521 | 0.783921 |
| tabpfn | completed | 0.914610 | 0.834277 | 0.829391 |

### Axis 4: stability_multiseed

| model | train PR-AUC mean/std | val PR-AUC mean/std | test PR-AUC mean/std |
|---|---:|---:|---:|
| lightgbm | 0.977649/0.006639 | 0.892595/0.014762 | 0.896799/0.013215 |
| xgboost | 0.973974/0.007696 | 0.875535/0.013504 | 0.883586/0.013141 |
| catboost | 0.969585/0.008000 | 0.881758/0.017625 | 0.877603/0.018061 |
| hist_gradient_boosting | 0.978425/0.006747 | 0.881333/0.016193 | 0.891680/0.012003 |
| ensemble | 0.978213/0.007257 | 0.889774/0.014697 | 0.893219/0.014153 |
| tabpfn | 0.977989/0.006867 | 0.899034/0.014165 | 0.903402/0.008897 |

## Observations

- Axis 1 tuning used validation PR-AUC only; the held-out test half was scored after selection.
- Axis 2 crossover support: `{'support_size': 100, 'best_tree_test_pr_auc': 0.7734845569558051, 'tabpfn_test_pr_auc': 0.749428198237835}`.
- Axis 3 records the 50-feature rule as baseline plus the first 33 value-change columns.
- Axis 4 reports seed-grid variation and a separate same-input TabPFN rerun band when TabPFN is available.

## Open Questions

- Should tuned-tree search space be widened before any report-facing interpretation?
- Should later full runs raise `--tune-trials` beyond the current handoff budget?
- If TabPFN is skipped locally, rerun on the known local checkpoint/GPU path before comparing claims.
