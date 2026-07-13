# M5 Phase D deep comparison handoff

**Non-report handoff notes.** These numbers are for follow-up consumption; no `docs/reports/` narrative was updated.

## Run

- Smoke mode: `False`
- Requested axes this run: `default_vs_tuned sample_efficiency_fine dimensionality_at_small_n stability_multiseed`
- JSON merge mode: `merge_preserve_unrequested_axes`
- Executed command: `uv run python scripts/run_m5_phaseD_deep_comparison.py --out data\processed\m5_phaseD_deep_comparison.json --handoff docs\handoffs\m5-phaseD-deep-comparison.md --fit-rows 10000 --score-rows 4000 --scarcity-sizes 20 50 100 150 300 500 1000 2000 --tune-trials 12 --seed 42`
- JSON: `data/processed/m5_phaseD_deep_comparison.json`
- Full command for later real run: `uv run python scripts/run_m5_phaseD_deep_comparison.py --out data/processed/m5_phaseD_deep_comparison.json --handoff docs/handoffs/m5-phaseD-deep-comparison.md --fit-rows 10000 --score-rows 4000 --scarcity-sizes 20 50 100 150 300 500 1000 2000 --tune-trials 12 --seed 42 --axes default_vs_tuned sample_efficiency_fine dimensionality_at_small_n stability_multiseed`
- Value-change regime: `timestamp_merge`

## Operating-Point Note

The test `threshold_0_5` and `fixed_recall_0_90` entries are post-hoc operating points. In particular, `fixed_recall_0_90` derives its threshold from the same split's labels, including test labels for the test summary. These entries are descriptive only and do not represent deployable performance. Model comparison and TabPFN-vs-tree claims should use threshold-free ROC-AUC / PR-AUC. For deployable operating points, choose thresholds on val and apply them once to test.

## Axis Headlines

### Axis 1: default_vs_tuned

Default trees:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.974347 | 0.876254 | 0.908641 |
| xgboost | completed | 0.975347 | 0.862158 | 0.898601 |
| catboost | completed | 0.970743 | 0.868169 | 0.879127 |
| hist_gradient_boosting | completed | 0.973407 | 0.865964 | 0.888936 |
| ensemble | completed | 0.975980 | 0.872186 | 0.901102 |

Tuned trees:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.956753 | 0.883915 | 0.900632 |
| xgboost | completed | 0.972700 | 0.876733 | 0.910044 |
| catboost | completed | 0.966604 | 0.879703 | 0.889089 |
| hist_gradient_boosting | completed | 0.964666 | 0.882923 | 0.906778 |
| ensemble | completed | 0.969261 | 0.878913 | 0.910910 |

TabPFN:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| tabpfn | completed | 0.972691 | 0.889618 | 0.904514 |

### Axis 2: sample_efficiency_fine

Support 20:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.068750 | 0.063250 | 0.062250 |
| xgboost | completed | 0.188452 | 0.167468 | 0.165362 |
| catboost | completed | 0.848360 | 0.751155 | 0.756727 |
| hist_gradient_boosting | completed | 0.068750 | 0.063250 | 0.062250 |
| ensemble | completed | 0.831408 | 0.733856 | 0.738803 |
| tabpfn | completed | 0.828726 | 0.723929 | 0.745617 |

Support 50:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.478404 | 0.385975 | 0.399373 |
| xgboost | completed | 0.673738 | 0.542757 | 0.586210 |
| catboost | completed | 0.839247 | 0.736626 | 0.752483 |
| hist_gradient_boosting | completed | 0.558253 | 0.438871 | 0.496736 |
| ensemble | completed | 0.721592 | 0.582735 | 0.617610 |
| tabpfn | completed | 0.841518 | 0.742091 | 0.729338 |

Support 100:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.771357 | 0.586639 | 0.651645 |
| xgboost | completed | 0.791117 | 0.628827 | 0.692395 |
| catboost | completed | 0.810626 | 0.694461 | 0.722323 |
| hist_gradient_boosting | completed | 0.777287 | 0.627328 | 0.652429 |
| ensemble | completed | 0.796659 | 0.669785 | 0.704689 |
| tabpfn | completed | 0.837641 | 0.739263 | 0.756983 |

Support 150:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.807596 | 0.707429 | 0.707468 |
| xgboost | completed | 0.736838 | 0.669964 | 0.675539 |
| catboost | completed | 0.811530 | 0.719858 | 0.731562 |
| hist_gradient_boosting | completed | 0.800930 | 0.666019 | 0.694019 |
| ensemble | completed | 0.794212 | 0.723528 | 0.722822 |
| tabpfn | completed | 0.870821 | 0.804416 | 0.797858 |

Support 300:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.830603 | 0.737658 | 0.755108 |
| xgboost | completed | 0.804662 | 0.699477 | 0.757355 |
| catboost | completed | 0.823871 | 0.723724 | 0.717181 |
| hist_gradient_boosting | completed | 0.822961 | 0.716539 | 0.768125 |
| ensemble | completed | 0.835872 | 0.737696 | 0.744195 |
| tabpfn | completed | 0.903532 | 0.819054 | 0.819398 |

Support 500:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.855399 | 0.742254 | 0.800014 |
| xgboost | completed | 0.828571 | 0.733625 | 0.770805 |
| catboost | completed | 0.839463 | 0.738052 | 0.735048 |
| hist_gradient_boosting | completed | 0.842245 | 0.702540 | 0.765134 |
| ensemble | completed | 0.846056 | 0.752647 | 0.754961 |
| tabpfn | completed | 0.911039 | 0.853311 | 0.851953 |

Support 1000:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.894444 | 0.725124 | 0.801998 |
| xgboost | completed | 0.880346 | 0.755569 | 0.822801 |
| catboost | completed | 0.916213 | 0.784293 | 0.763787 |
| hist_gradient_boosting | completed | 0.888493 | 0.730558 | 0.818907 |
| ensemble | completed | 0.922470 | 0.809182 | 0.805292 |
| tabpfn | completed | 0.935077 | 0.858911 | 0.847271 |

Support 2000:
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.951111 | 0.824636 | 0.848693 |
| xgboost | completed | 0.939973 | 0.803185 | 0.824419 |
| catboost | completed | 0.936828 | 0.852854 | 0.853455 |
| hist_gradient_boosting | completed | 0.946329 | 0.791834 | 0.854519 |
| ensemble | completed | 0.947627 | 0.856438 | 0.865074 |
| tabpfn | completed | 0.947327 | 0.872030 | 0.868784 |

### Axis 3: dimensionality_at_small_n

raw_baseline_17 (17 features):
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.687743 | 0.524686 | 0.664897 |
| xgboost | completed | 0.686934 | 0.588712 | 0.662295 |
| catboost | completed | 0.723037 | 0.636272 | 0.707741 |
| hist_gradient_boosting | completed | 0.690034 | 0.542618 | 0.677570 |
| ensemble | completed | 0.721132 | 0.628843 | 0.707044 |
| tabpfn | completed | 0.796314 | 0.753637 | 0.758009 |

baseline_plus_first_33_value_change_50 (50 features):
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.817021 | 0.699268 | 0.779099 |
| xgboost | completed | 0.809975 | 0.706570 | 0.759632 |
| catboost | completed | 0.824454 | 0.728073 | 0.751901 |
| hist_gradient_boosting | completed | 0.821994 | 0.698685 | 0.771038 |
| ensemble | completed | 0.831318 | 0.737681 | 0.777163 |
| tabpfn | completed | 0.903062 | 0.850173 | 0.847876 |

full_137 (137 features):
| model | status | train PR-AUC | val PR-AUC | test PR-AUC |
|---|---:|---:|---:|---:|
| lightgbm | completed | 0.855399 | 0.742254 | 0.800014 |
| xgboost | completed | 0.828571 | 0.733625 | 0.770805 |
| catboost | completed | 0.839463 | 0.738052 | 0.735048 |
| hist_gradient_boosting | completed | 0.842245 | 0.702540 | 0.765134 |
| ensemble | completed | 0.846056 | 0.752647 | 0.754961 |
| tabpfn | completed | 0.911039 | 0.853311 | 0.851953 |

### Axis 4: stability_multiseed

| model | train PR-AUC mean/std | val PR-AUC mean/std | test PR-AUC mean/std |
|---|---:|---:|---:|
| lightgbm | 0.971963/0.005217 | 0.893948/0.013246 | 0.891275/0.013082 |
| xgboost | 0.971444/0.002524 | 0.878508/0.021906 | 0.881781/0.013802 |
| catboost | 0.965787/0.005929 | 0.887365/0.026663 | 0.874326/0.012920 |
| hist_gradient_boosting | 0.972226/0.004719 | 0.888351/0.016150 | 0.884867/0.013783 |
| ensemble | 0.972914/0.004660 | 0.892585/0.019995 | 0.887107/0.015497 |
| tabpfn | 0.972920/0.005728 | 0.903943/0.016467 | 0.897656/0.012128 |

## Observations

- Axis 1 tuning used validation PR-AUC only; the held-out test half was scored after selection.
- Axis 2 crossover support: `{'support_size': 20, 'best_tree_test_pr_auc': 0.7567274169293634, 'tabpfn_test_pr_auc': 0.7456166720080042}`.
- Axis 3 records the 50-feature rule as baseline plus the first 33 value-change columns.
- Axis 4 reports seed-grid variation and a separate same-input TabPFN rerun band when TabPFN is available.

## Open Questions

- Should tuned-tree search space be widened before any report-facing interpretation?
- Should later full runs raise `--tune-trials` beyond the current handoff budget?
- If TabPFN is skipped locally, rerun on the known local checkpoint/GPU path before comparing claims.
