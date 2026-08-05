# M5 building-count experiment

This report is atomically regenerated at every overnight publication gate. Training sources are restricted to building_id % 2 == 0; the complete canonical holdout contains only odd building IDs.

K=10/20/50/100 uses only 137 features. Average allocation is at most 500 rows per building, so total context is bounded by 5K/10K/25K/50K. Allocation within each incremental K block is proportional to each building's available rows, so individual buildings may contribute above or below 500. Rows are selected by a seed-42 stable hash of raw identity without consulting labels. Building and row sets are strict nested prefixes. Trees use building-disjoint 80/20 fit and early-stop roles to choose iteration counts, then final-refit on every selected row. TabPFN uses the same selected rows and has no task-specific epoch or weight-update loop, so early stopping is not applicable.

The direct TabPFN reference is the m5-matched-context-breakdown 50K / 137-feature / n_estimators=8 prediction artifact. Different context row/class distributions are allowed. Comparability is maintained by the same feature pipeline, scaler, checkpoint/config, canonical holdout identity and evaluator.

## Execution status

| stage | status |
|---|---|
| tree full K=725 f=17 | complete |
| tree full K=725 f=137 | complete |
| tree K=10 f=137 | pending |
| tabpfn K=10 f=137 | pending |
| tree K=20 f=137 | pending |
| tabpfn K=20 f=137 | pending |
| tree K=50 f=137 | pending |
| tabpfn K=50 f=137 | pending |
| tree K=100 f=137 | pending |
| tabpfn K=100 f=137 | pending |

## Overall PR-AUC / ROC-AUC

| source | building_budget | features | model | rows | anomalies | pr_auc | roc_auc |
|---|---|---|---|---|---|---|---|
| matched-context baseline | 0 | 137 | tabpfn_matched_50k | 10137155 | 637397 | 0.932394 | 0.992369 |
| building experiment | 725 | 17 | catboost | 10137155 | 637397 | 0.758745 | 0.958761 |
| building experiment | 725 | 17 | ensemble | 10137155 | 637397 | 0.816618 | 0.963737 |
| building experiment | 725 | 17 | hist_gradient_boosting | 10137155 | 637397 | 0.816588 | 0.960353 |
| building experiment | 725 | 17 | lightgbm | 10137155 | 637397 | 0.823255 | 0.963470 |
| building experiment | 725 | 17 | xgboost | 10137155 | 637397 | 0.799322 | 0.957699 |
| building experiment | 725 | 137 | catboost | 10137155 | 637397 | 0.921773 | 0.984871 |
| building experiment | 725 | 137 | ensemble | 10137155 | 637397 | 0.929914 | 0.992271 |
| building experiment | 725 | 137 | hist_gradient_boosting | 10137155 | 637397 | 0.925322 | 0.992198 |
| building experiment | 725 | 137 | lightgbm | 10137155 | 637397 | 0.923557 | 0.991714 |
| building experiment | 725 | 137 | xgboost | 10137155 | 637397 | 0.917151 | 0.989028 |

## Meter breakdown

| sampling_profile | building_budget | features | model | group_label | rows | anomalies | pr_auc | roc_auc |
|---|---|---|---|---|---|---|---|---|
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | electricity | 6035071 | 356679 | 0.985040 | 0.997838 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | chilledwater | 2115354 | 141139 | 0.829599 | 0.984698 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | steam | 1350609 | 48888 | 0.822684 | 0.985801 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | hotwater | 636121 | 90691 | 0.824857 | 0.953947 |
| representative | 725 | 17 | catboost | electricity | 6035071 | 356679 | 0.856823 | 0.971832 |
| representative | 725 | 17 | catboost | chilledwater | 2115354 | 141139 | 0.468219 | 0.934928 |
| representative | 725 | 17 | catboost | steam | 1350609 | 48888 | 0.415009 | 0.906972 |
| representative | 725 | 17 | catboost | hotwater | 636121 | 90691 | 0.512487 | 0.895664 |
| representative | 725 | 17 | ensemble | electricity | 6035071 | 356679 | 0.929352 | 0.976261 |
| representative | 725 | 17 | ensemble | chilledwater | 2115354 | 141139 | 0.534009 | 0.933547 |
| representative | 725 | 17 | ensemble | steam | 1350609 | 48888 | 0.494704 | 0.919054 |
| representative | 725 | 17 | ensemble | hotwater | 636121 | 90691 | 0.563184 | 0.927796 |
| representative | 725 | 17 | hist_gradient_boosting | electricity | 6035071 | 356679 | 0.919620 | 0.970577 |
| representative | 725 | 17 | hist_gradient_boosting | chilledwater | 2115354 | 141139 | 0.555697 | 0.932621 |
| representative | 725 | 17 | hist_gradient_boosting | steam | 1350609 | 48888 | 0.420892 | 0.919879 |
| representative | 725 | 17 | hist_gradient_boosting | hotwater | 636121 | 90691 | 0.545400 | 0.920419 |
| representative | 725 | 17 | lightgbm | electricity | 6035071 | 356679 | 0.927228 | 0.976569 |
| representative | 725 | 17 | lightgbm | chilledwater | 2115354 | 141139 | 0.543940 | 0.930920 |
| representative | 725 | 17 | lightgbm | steam | 1350609 | 48888 | 0.467633 | 0.916184 |
| representative | 725 | 17 | lightgbm | hotwater | 636121 | 90691 | 0.635022 | 0.927645 |
| representative | 725 | 17 | xgboost | electricity | 6035071 | 356679 | 0.921712 | 0.972566 |
| representative | 725 | 17 | xgboost | chilledwater | 2115354 | 141139 | 0.490769 | 0.928741 |
| representative | 725 | 17 | xgboost | steam | 1350609 | 48888 | 0.410249 | 0.912414 |
| representative | 725 | 17 | xgboost | hotwater | 636121 | 90691 | 0.532154 | 0.893594 |
| representative | 725 | 137 | catboost | electricity | 6035071 | 356679 | 0.992254 | 0.998972 |
| representative | 725 | 137 | catboost | chilledwater | 2115354 | 141139 | 0.764041 | 0.977616 |
| representative | 725 | 137 | catboost | steam | 1350609 | 48888 | 0.762702 | 0.954448 |
| representative | 725 | 137 | catboost | hotwater | 636121 | 90691 | 0.762358 | 0.922216 |
| representative | 725 | 137 | ensemble | electricity | 6035071 | 356679 | 0.993768 | 0.999327 |
| representative | 725 | 137 | ensemble | chilledwater | 2115354 | 141139 | 0.763622 | 0.980342 |
| representative | 725 | 137 | ensemble | steam | 1350609 | 48888 | 0.744169 | 0.966764 |
| representative | 725 | 137 | ensemble | hotwater | 636121 | 90691 | 0.815085 | 0.950563 |
| representative | 725 | 137 | hist_gradient_boosting | electricity | 6035071 | 356679 | 0.991800 | 0.999024 |
| representative | 725 | 137 | hist_gradient_boosting | chilledwater | 2115354 | 141139 | 0.727419 | 0.977756 |
| representative | 725 | 137 | hist_gradient_boosting | steam | 1350609 | 48888 | 0.717473 | 0.966932 |
| representative | 725 | 137 | hist_gradient_boosting | hotwater | 636121 | 90691 | 0.841508 | 0.958421 |
| representative | 725 | 137 | lightgbm | electricity | 6035071 | 356679 | 0.992103 | 0.999047 |
| representative | 725 | 137 | lightgbm | chilledwater | 2115354 | 141139 | 0.757214 | 0.979779 |
| representative | 725 | 137 | lightgbm | steam | 1350609 | 48888 | 0.705465 | 0.965881 |
| representative | 725 | 137 | lightgbm | hotwater | 636121 | 90691 | 0.804264 | 0.947007 |
| representative | 725 | 137 | xgboost | electricity | 6035071 | 356679 | 0.987858 | 0.998248 |
| representative | 725 | 137 | xgboost | chilledwater | 2115354 | 141139 | 0.756959 | 0.978769 |
| representative | 725 | 137 | xgboost | steam | 1350609 | 48888 | 0.700059 | 0.955090 |
| representative | 725 | 137 | xgboost | hotwater | 636121 | 90691 | 0.760747 | 0.938985 |

## Site breakdown

| sampling_profile | building_budget | features | model | group_label | rows | anomalies | pr_auc | roc_auc |
|---|---|---|---|---|---|---|---|---|
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 0 | 538432 | 176269 | 0.999846 | 0.999920 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 1 | 289853 | 39135 | 0.986904 | 0.996629 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 2 | 1263915 | 80897 | 0.905913 | 0.991966 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 3 | 1181463 | 2684 | 0.869856 | 0.998590 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 4 | 370460 | 197 | 0.710150 | 0.985135 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 5 | 386496 | 14435 | 0.973337 | 0.999060 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 6 | 345117 | 28654 | 0.815831 | 0.986176 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 7 | 200594 | 15886 | 0.715931 | 0.961793 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 8 | 284376 | 31083 | 0.874592 | 0.973599 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 9 | 1367482 | 52587 | 0.974235 | 0.994394 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 10 | 206430 | 15814 | 0.780662 | 0.959497 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 11 | 43626 | 197 | 0.571246 | 0.993574 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 12 | 158011 | 755 | 0.996508 | 0.999978 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 13 | 1334223 | 59283 | 0.761357 | 0.980529 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 14 | 1256775 | 101532 | 0.892405 | 0.980845 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | site 15 | 909902 | 17989 | 0.879272 | 0.996271 |
| representative | 725 | 17 | catboost | site 0 | 538432 | 176269 | 0.998009 | 0.998257 |
| representative | 725 | 17 | catboost | site 1 | 289853 | 39135 | 0.438881 | 0.783724 |
| representative | 725 | 17 | catboost | site 2 | 1263915 | 80897 | 0.572216 | 0.950797 |
| representative | 725 | 17 | catboost | site 3 | 1181463 | 2684 | 0.126274 | 0.954155 |
| representative | 725 | 17 | catboost | site 4 | 370460 | 197 | 0.104210 | 0.909759 |
| representative | 725 | 17 | catboost | site 5 | 386496 | 14435 | 0.952339 | 0.991926 |
| representative | 725 | 17 | catboost | site 6 | 345117 | 28654 | 0.476887 | 0.921258 |
| representative | 725 | 17 | catboost | site 7 | 200594 | 15886 | 0.854548 | 0.958707 |
| representative | 725 | 17 | catboost | site 8 | 284376 | 31083 | 0.867971 | 0.944891 |
| representative | 725 | 17 | catboost | site 9 | 1367482 | 52587 | 0.896088 | 0.983890 |
| representative | 725 | 17 | catboost | site 10 | 206430 | 15814 | 0.324065 | 0.911036 |
| representative | 725 | 17 | catboost | site 11 | 43626 | 197 | 0.022560 | 0.905415 |
| representative | 725 | 17 | catboost | site 12 | 158011 | 755 | 0.975075 | 0.990925 |
| representative | 725 | 17 | catboost | site 13 | 1334223 | 59283 | 0.469057 | 0.875525 |
| representative | 725 | 17 | catboost | site 14 | 1256775 | 101532 | 0.665949 | 0.925734 |
| representative | 725 | 17 | catboost | site 15 | 909902 | 17989 | 0.809541 | 0.984564 |
| representative | 725 | 17 | ensemble | site 0 | 538432 | 176269 | 0.997778 | 0.997931 |
| representative | 725 | 17 | ensemble | site 1 | 289853 | 39135 | 0.490675 | 0.828940 |
| representative | 725 | 17 | ensemble | site 2 | 1263915 | 80897 | 0.626716 | 0.955573 |
| representative | 725 | 17 | ensemble | site 3 | 1181463 | 2684 | 0.806153 | 0.950980 |
| representative | 725 | 17 | ensemble | site 4 | 370460 | 197 | 0.354762 | 0.848406 |
| representative | 725 | 17 | ensemble | site 5 | 386496 | 14435 | 0.959128 | 0.994097 |
| representative | 725 | 17 | ensemble | site 6 | 345117 | 28654 | 0.537532 | 0.926274 |
| representative | 725 | 17 | ensemble | site 7 | 200594 | 15886 | 0.853283 | 0.953011 |
| representative | 725 | 17 | ensemble | site 8 | 284376 | 31083 | 0.865774 | 0.946075 |
| representative | 725 | 17 | ensemble | site 9 | 1367482 | 52587 | 0.915457 | 0.989444 |
| representative | 725 | 17 | ensemble | site 10 | 206430 | 15814 | 0.575863 | 0.946642 |
| representative | 725 | 17 | ensemble | site 11 | 43626 | 197 | 0.024316 | 0.911709 |
| representative | 725 | 17 | ensemble | site 12 | 158011 | 755 | 0.975441 | 0.988577 |
| representative | 725 | 17 | ensemble | site 13 | 1334223 | 59283 | 0.561149 | 0.891652 |
| representative | 725 | 17 | ensemble | site 14 | 1256775 | 101532 | 0.683884 | 0.917699 |
| representative | 725 | 17 | ensemble | site 15 | 909902 | 17989 | 0.832176 | 0.986397 |
| representative | 725 | 17 | hist_gradient_boosting | site 0 | 538432 | 176269 | 0.997765 | 0.998010 |
| representative | 725 | 17 | hist_gradient_boosting | site 1 | 289853 | 39135 | 0.408291 | 0.792534 |
| representative | 725 | 17 | hist_gradient_boosting | site 2 | 1263915 | 80897 | 0.601689 | 0.948682 |
| representative | 725 | 17 | hist_gradient_boosting | site 3 | 1181463 | 2684 | 0.805262 | 0.951119 |
| representative | 725 | 17 | hist_gradient_boosting | site 4 | 370460 | 197 | 0.723907 | 0.858154 |
| representative | 725 | 17 | hist_gradient_boosting | site 5 | 386496 | 14435 | 0.938738 | 0.989364 |
| representative | 725 | 17 | hist_gradient_boosting | site 6 | 345117 | 28654 | 0.514634 | 0.919889 |
| representative | 725 | 17 | hist_gradient_boosting | site 7 | 200594 | 15886 | 0.849263 | 0.956596 |
| representative | 725 | 17 | hist_gradient_boosting | site 8 | 284376 | 31083 | 0.867154 | 0.939926 |
| representative | 725 | 17 | hist_gradient_boosting | site 9 | 1367482 | 52587 | 0.909867 | 0.988655 |
| representative | 725 | 17 | hist_gradient_boosting | site 10 | 206430 | 15814 | 0.567614 | 0.940014 |
| representative | 725 | 17 | hist_gradient_boosting | site 11 | 43626 | 197 | 0.048344 | 0.937126 |
| representative | 725 | 17 | hist_gradient_boosting | site 12 | 158011 | 755 | 0.974885 | 0.990938 |
| representative | 725 | 17 | hist_gradient_boosting | site 13 | 1334223 | 59283 | 0.554090 | 0.892202 |
| representative | 725 | 17 | hist_gradient_boosting | site 14 | 1256775 | 101532 | 0.682643 | 0.913757 |
| representative | 725 | 17 | hist_gradient_boosting | site 15 | 909902 | 17989 | 0.834945 | 0.985546 |
| representative | 725 | 17 | lightgbm | site 0 | 538432 | 176269 | 0.997621 | 0.997695 |
| representative | 725 | 17 | lightgbm | site 1 | 289853 | 39135 | 0.462548 | 0.796400 |
| representative | 725 | 17 | lightgbm | site 2 | 1263915 | 80897 | 0.680381 | 0.956785 |
| representative | 725 | 17 | lightgbm | site 3 | 1181463 | 2684 | 0.806161 | 0.942224 |
| representative | 725 | 17 | lightgbm | site 4 | 370460 | 197 | 0.341084 | 0.869674 |
| representative | 725 | 17 | lightgbm | site 5 | 386496 | 14435 | 0.962435 | 0.993202 |
| representative | 725 | 17 | lightgbm | site 6 | 345117 | 28654 | 0.568051 | 0.926574 |
| representative | 725 | 17 | lightgbm | site 7 | 200594 | 15886 | 0.865099 | 0.950358 |
| representative | 725 | 17 | lightgbm | site 8 | 284376 | 31083 | 0.864843 | 0.940682 |
| representative | 725 | 17 | lightgbm | site 9 | 1367482 | 52587 | 0.906328 | 0.986398 |
| representative | 725 | 17 | lightgbm | site 10 | 206430 | 15814 | 0.586549 | 0.948840 |
| representative | 725 | 17 | lightgbm | site 11 | 43626 | 197 | 0.058247 | 0.927916 |
| representative | 725 | 17 | lightgbm | site 12 | 158011 | 755 | 0.975351 | 0.987595 |
| representative | 725 | 17 | lightgbm | site 13 | 1334223 | 59283 | 0.554558 | 0.892406 |
| representative | 725 | 17 | lightgbm | site 14 | 1256775 | 101532 | 0.684032 | 0.912590 |
| representative | 725 | 17 | lightgbm | site 15 | 909902 | 17989 | 0.832635 | 0.982610 |
| representative | 725 | 17 | xgboost | site 0 | 538432 | 176269 | 0.996932 | 0.997142 |
| representative | 725 | 17 | xgboost | site 1 | 289853 | 39135 | 0.414892 | 0.794317 |
| representative | 725 | 17 | xgboost | site 2 | 1263915 | 80897 | 0.603308 | 0.944133 |
| representative | 725 | 17 | xgboost | site 3 | 1181463 | 2684 | 0.799748 | 0.930446 |
| representative | 725 | 17 | xgboost | site 4 | 370460 | 197 | 0.108084 | 0.793097 |
| representative | 725 | 17 | xgboost | site 5 | 386496 | 14435 | 0.946252 | 0.993132 |
| representative | 725 | 17 | xgboost | site 6 | 345117 | 28654 | 0.575710 | 0.930995 |
| representative | 725 | 17 | xgboost | site 7 | 200594 | 15886 | 0.818535 | 0.951743 |
| representative | 725 | 17 | xgboost | site 8 | 284376 | 31083 | 0.852259 | 0.939898 |
| representative | 725 | 17 | xgboost | site 9 | 1367482 | 52587 | 0.889188 | 0.963251 |
| representative | 725 | 17 | xgboost | site 10 | 206430 | 15814 | 0.561557 | 0.921692 |
| representative | 725 | 17 | xgboost | site 11 | 43626 | 197 | 0.050465 | 0.939932 |
| representative | 725 | 17 | xgboost | site 12 | 158011 | 755 | 0.976082 | 0.985975 |
| representative | 725 | 17 | xgboost | site 13 | 1334223 | 59283 | 0.555598 | 0.898157 |
| representative | 725 | 17 | xgboost | site 14 | 1256775 | 101532 | 0.667614 | 0.910673 |
| representative | 725 | 17 | xgboost | site 15 | 909902 | 17989 | 0.822609 | 0.985395 |
| representative | 725 | 137 | catboost | site 0 | 538432 | 176269 | 0.999878 | 0.999915 |
| representative | 725 | 137 | catboost | site 1 | 289853 | 39135 | 0.960112 | 0.978400 |
| representative | 725 | 137 | catboost | site 2 | 1263915 | 80897 | 0.817553 | 0.969019 |
| representative | 725 | 137 | catboost | site 3 | 1181463 | 2684 | 0.883877 | 0.996516 |
| representative | 725 | 137 | catboost | site 4 | 370460 | 197 | 0.772825 | 0.984388 |
| representative | 725 | 137 | catboost | site 5 | 386496 | 14435 | 0.985271 | 0.999430 |
| representative | 725 | 137 | catboost | site 6 | 345117 | 28654 | 0.840363 | 0.985777 |
| representative | 725 | 137 | catboost | site 7 | 200594 | 15886 | 0.887170 | 0.981438 |
| representative | 725 | 137 | catboost | site 8 | 284376 | 31083 | 0.960443 | 0.986520 |
| representative | 725 | 137 | catboost | site 9 | 1367482 | 52587 | 0.984132 | 0.998122 |
| representative | 725 | 137 | catboost | site 10 | 206430 | 15814 | 0.781606 | 0.968411 |
| representative | 725 | 137 | catboost | site 11 | 43626 | 197 | 0.268207 | 0.979261 |
| representative | 725 | 137 | catboost | site 12 | 158011 | 755 | 0.998569 | 0.999992 |
| representative | 725 | 137 | catboost | site 13 | 1334223 | 59283 | 0.675736 | 0.946706 |
| representative | 725 | 137 | catboost | site 14 | 1256775 | 101532 | 0.890748 | 0.970803 |
| representative | 725 | 137 | catboost | site 15 | 909902 | 17989 | 0.884438 | 0.992077 |
| representative | 725 | 137 | ensemble | site 0 | 538432 | 176269 | 0.999913 | 0.999946 |
| representative | 725 | 137 | ensemble | site 1 | 289853 | 39135 | 0.984153 | 0.996451 |
| representative | 725 | 137 | ensemble | site 2 | 1263915 | 80897 | 0.889365 | 0.991320 |
| representative | 725 | 137 | ensemble | site 3 | 1181463 | 2684 | 0.878247 | 0.998548 |
| representative | 725 | 137 | ensemble | site 4 | 370460 | 197 | 0.769100 | 0.981534 |
| representative | 725 | 137 | ensemble | site 5 | 386496 | 14435 | 0.993663 | 0.999729 |
| representative | 725 | 137 | ensemble | site 6 | 345117 | 28654 | 0.880333 | 0.987911 |
| representative | 725 | 137 | ensemble | site 7 | 200594 | 15886 | 0.892081 | 0.983330 |
| representative | 725 | 137 | ensemble | site 8 | 284376 | 31083 | 0.971710 | 0.991187 |
| representative | 725 | 137 | ensemble | site 9 | 1367482 | 52587 | 0.983272 | 0.998240 |
| representative | 725 | 137 | ensemble | site 10 | 206430 | 15814 | 0.830613 | 0.975939 |
| representative | 725 | 137 | ensemble | site 11 | 43626 | 197 | 0.502510 | 0.978115 |
| representative | 725 | 137 | ensemble | site 12 | 158011 | 755 | 0.998744 | 0.999985 |
| representative | 725 | 137 | ensemble | site 13 | 1334223 | 59283 | 0.648673 | 0.962595 |
| representative | 725 | 137 | ensemble | site 14 | 1256775 | 101532 | 0.875671 | 0.983071 |
| representative | 725 | 137 | ensemble | site 15 | 909902 | 17989 | 0.889700 | 0.996264 |
| representative | 725 | 137 | hist_gradient_boosting | site 0 | 538432 | 176269 | 0.999876 | 0.999929 |
| representative | 725 | 137 | hist_gradient_boosting | site 1 | 289853 | 39135 | 0.985010 | 0.996298 |
| representative | 725 | 137 | hist_gradient_boosting | site 2 | 1263915 | 80897 | 0.914577 | 0.992904 |
| representative | 725 | 137 | hist_gradient_boosting | site 3 | 1181463 | 2684 | 0.864421 | 0.998472 |
| representative | 725 | 137 | hist_gradient_boosting | site 4 | 370460 | 197 | 0.768903 | 0.977764 |
| representative | 725 | 137 | hist_gradient_boosting | site 5 | 386496 | 14435 | 0.995345 | 0.999753 |
| representative | 725 | 137 | hist_gradient_boosting | site 6 | 345117 | 28654 | 0.850884 | 0.986081 |
| representative | 725 | 137 | hist_gradient_boosting | site 7 | 200594 | 15886 | 0.886441 | 0.981537 |
| representative | 725 | 137 | hist_gradient_boosting | site 8 | 284376 | 31083 | 0.971346 | 0.991877 |
| representative | 725 | 137 | hist_gradient_boosting | site 9 | 1367482 | 52587 | 0.978103 | 0.997316 |
| representative | 725 | 137 | hist_gradient_boosting | site 10 | 206430 | 15814 | 0.803366 | 0.970875 |
| representative | 725 | 137 | hist_gradient_boosting | site 11 | 43626 | 197 | 0.567152 | 0.976148 |
| representative | 725 | 137 | hist_gradient_boosting | site 12 | 158011 | 755 | 0.998199 | 0.999964 |
| representative | 725 | 137 | hist_gradient_boosting | site 13 | 1334223 | 59283 | 0.634604 | 0.961834 |
| representative | 725 | 137 | hist_gradient_boosting | site 14 | 1256775 | 101532 | 0.851057 | 0.982896 |
| representative | 725 | 137 | hist_gradient_boosting | site 15 | 909902 | 17989 | 0.849514 | 0.995511 |
| representative | 725 | 137 | lightgbm | site 0 | 538432 | 176269 | 0.999842 | 0.999901 |
| representative | 725 | 137 | lightgbm | site 1 | 289853 | 39135 | 0.977235 | 0.993993 |
| representative | 725 | 137 | lightgbm | site 2 | 1263915 | 80897 | 0.884969 | 0.990504 |
| representative | 725 | 137 | lightgbm | site 3 | 1181463 | 2684 | 0.872192 | 0.998523 |
| representative | 725 | 137 | lightgbm | site 4 | 370460 | 197 | 0.759473 | 0.956753 |
| representative | 725 | 137 | lightgbm | site 5 | 386496 | 14435 | 0.994284 | 0.999676 |
| representative | 725 | 137 | lightgbm | site 6 | 345117 | 28654 | 0.862678 | 0.986578 |
| representative | 725 | 137 | lightgbm | site 7 | 200594 | 15886 | 0.866776 | 0.977925 |
| representative | 725 | 137 | lightgbm | site 8 | 284376 | 31083 | 0.973765 | 0.993234 |
| representative | 725 | 137 | lightgbm | site 9 | 1367482 | 52587 | 0.973096 | 0.996718 |
| representative | 725 | 137 | lightgbm | site 10 | 206430 | 15814 | 0.815484 | 0.973308 |
| representative | 725 | 137 | lightgbm | site 11 | 43626 | 197 | 0.293726 | 0.971833 |
| representative | 725 | 137 | lightgbm | site 12 | 158011 | 755 | 0.997875 | 0.999953 |
| representative | 725 | 137 | lightgbm | site 13 | 1334223 | 59283 | 0.665542 | 0.964390 |
| representative | 725 | 137 | lightgbm | site 14 | 1256775 | 101532 | 0.851342 | 0.981912 |
| representative | 725 | 137 | lightgbm | site 15 | 909902 | 17989 | 0.892309 | 0.996237 |
| representative | 725 | 137 | xgboost | site 0 | 538432 | 176269 | 0.999756 | 0.999882 |
| representative | 725 | 137 | xgboost | site 1 | 289853 | 39135 | 0.982315 | 0.995452 |
| representative | 725 | 137 | xgboost | site 2 | 1263915 | 80897 | 0.856392 | 0.989883 |
| representative | 725 | 137 | xgboost | site 3 | 1181463 | 2684 | 0.793752 | 0.997794 |
| representative | 725 | 137 | xgboost | site 4 | 370460 | 197 | 0.750211 | 0.962892 |
| representative | 725 | 137 | xgboost | site 5 | 386496 | 14435 | 0.990581 | 0.999495 |
| representative | 725 | 137 | xgboost | site 6 | 345117 | 28654 | 0.883162 | 0.989051 |
| representative | 725 | 137 | xgboost | site 7 | 200594 | 15886 | 0.874940 | 0.980577 |
| representative | 725 | 137 | xgboost | site 8 | 284376 | 31083 | 0.950721 | 0.985527 |
| representative | 725 | 137 | xgboost | site 9 | 1367482 | 52587 | 0.966406 | 0.994357 |
| representative | 725 | 137 | xgboost | site 10 | 206430 | 15814 | 0.783675 | 0.966241 |
| representative | 725 | 137 | xgboost | site 11 | 43626 | 197 | 0.480922 | 0.978993 |
| representative | 725 | 137 | xgboost | site 12 | 158011 | 755 | 0.997897 | 0.999957 |
| representative | 725 | 137 | xgboost | site 13 | 1334223 | 59283 | 0.658127 | 0.955245 |
| representative | 725 | 137 | xgboost | site 14 | 1256775 | 101532 | 0.815811 | 0.969191 |
| representative | 725 | 137 | xgboost | site 15 | 909902 | 17989 | 0.851786 | 0.994723 |

## Tree early-stopping audit

| K | features | model | best_iteration | ceiling | stop_reason | ES_PR_AUC | ES_ROC_AUC |
|---|---|---|---|---|---|---|---|
| 725 | 137 | lightgbm | 68 | 5000 | early_stopping | 0.895113 | 0.990047 |
| 725 | 137 | xgboost | 16 | 5000 | early_stopping | 0.903069 | 0.989273 |
| 725 | 137 | catboost | 301 | 5000 | early_stopping | 0.896370 | 0.988127 |
| 725 | 137 | hist_gradient_boosting | 83 | 1000 | early_stopping | 0.885952 | 0.989318 |
| 725 | 17 | lightgbm | 56 | 5000 | early_stopping | 0.766721 | 0.963477 |
| 725 | 17 | xgboost | 11 | 5000 | early_stopping | 0.753435 | 0.953549 |
| 725 | 17 | catboost | 85 | 5000 | early_stopping | 0.795568 | 0.958509 |
| 725 | 17 | hist_gradient_boosting | 72 | 1000 | early_stopping | 0.778459 | 0.963356 |

## K composition and selected-building audit

### K=10

- Selected rows: 5,000; anomalies: 407; anomaly rate: 0.081400.
- Site row composition: {"0": 539, "1": 316, "13": 631, "14": 947, "15": 799, "2": 508, "3": 630, "4": 315, "5": 315}.
- Meter row composition: {"0": 3107, "1": 991, "2": 580, "3": 322}.
- Anomalous building-meter pairs: 10/17 (0.588235).

| position | site_id | building_id | primary_use | role | selection_reason | row_allocation_reason | meters | allocated_row_quota | selected_rows | selected_anomalies | selected_anomaly_rate | available_rows | available_anomaly_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 314 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 14 | 0.044444 | 8758 | 0.044302 |
| 2 | 2 | 270 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0,1 | 508 | 508 | 2 | 0.003937 | 14135 | 0.000990 |
| 3 | 13 | 1174 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 631 | 631 | 31 | 0.049128 | 17568 | 0.038821 |
| 4 | 4 | 578 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 0 | 0.000000 | 8783 | 0.000000 |
| 5 | 3 | 476 | - | early_stop | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 2 | 0.006349 | 8758 | 0.012560 |
| 6 | 14 | 1302 | - | fit | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 947 | 947 | 232 | 0.244984 | 26351 | 0.244279 |
| 7 | 5 | 720 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 2 | 0.006349 | 8784 | 0.003757 |
| 8 | 15 | 1358 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 799 | 799 | 0 | 0.000000 | 22254 | 0.000000 |
| 9 | 1 | 148 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 316 | 316 | 8 | 0.025316 | 8784 | 0.018215 |
| 10 | 0 | 28 | - | early_stop | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0,1 | 539 | 539 | 116 | 0.215213 | 14991 | 0.225402 |

### K=20

- Selected rows: 10,000; anomalies: 601; anomaly rate: 0.060100.
- Site row composition: {"0": 869, "1": 316, "13": 1318, "14": 947, "15": 1091, "2": 1881, "3": 1316, "4": 315, "5": 315, "6": 635, "8": 311, "9": 686}.
- Meter row composition: {"0": 6450, "1": 1636, "2": 1267, "3": 647}.
- Anomalous building-meter pairs: 17/32 (0.531250).

| position | site_id | building_id | primary_use | role | selection_reason | row_allocation_reason | meters | allocated_row_quota | selected_rows | selected_anomalies | selected_anomaly_rate | available_rows | available_anomaly_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 314 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 14 | 0.044444 | 8758 | 0.044302 |
| 2 | 2 | 270 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0,1 | 508 | 508 | 2 | 0.003937 | 14135 | 0.000990 |
| 3 | 13 | 1174 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 631 | 631 | 31 | 0.049128 | 17568 | 0.038821 |
| 4 | 4 | 578 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 0 | 0.000000 | 8783 | 0.000000 |
| 5 | 3 | 476 | - | early_stop | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 2 | 0.006349 | 8758 | 0.012560 |
| 6 | 14 | 1302 | - | fit | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 947 | 947 | 232 | 0.244984 | 26351 | 0.244279 |
| 7 | 5 | 720 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 2 | 0.006349 | 8784 | 0.003757 |
| 8 | 15 | 1358 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 799 | 799 | 0 | 0.000000 | 22254 | 0.000000 |
| 9 | 1 | 148 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 316 | 316 | 8 | 0.025316 | 8784 | 0.018215 |
| 10 | 0 | 28 | - | early_stop | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0,1 | 539 | 539 | 116 | 0.215213 | 14991 | 0.225402 |
| 11 | 8 | 844 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 311 | 311 | 1 | 0.003215 | 7959 | 0.002387 |
| 12 | 3 | 312 | - | fit | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 8 | 0.023324 | 8782 | 0.011501 |
| 13 | 9 | 876 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 686 | 686 | 50 | 0.072886 | 17546 | 0.080417 |
| 14 | 2 | 234 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 1030 | 1030 | 0 | 0.000000 | 26346 | 0.000000 |
| 15 | 13 | 1198 | - | early_stop | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0,2 | 687 | 687 | 0 | 0.000000 | 17568 | 0.000342 |
| 16 | 0 | 40 | - | fit | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0 | 330 | 330 | 130 | 0.393939 | 8428 | 0.402824 |
| 17 | 6 | 748 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0,1 | 635 | 635 | 4 | 0.006299 | 16254 | 0.008921 |
| 18 | 3 | 562 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 1 | 0.002915 | 8782 | 0.003758 |
| 19 | 15 | 1416 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 292 | 292 | 0 | 0.000000 | 7472 | 0.000000 |
| 20 | 2 | 230 | - | early_stop | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 0 | 0.000000 | 8782 | 0.000000 |

### K=50

- Selected rows: 25,000; anomalies: 1,619; anomaly rate: 0.064760.
- Site row composition: {"0": 1191, "1": 316, "10": 321, "11": 965, "12": 322, "13": 2927, "14": 3148, "15": 2449, "2": 2846, "3": 2908, "4": 994, "5": 959, "6": 1599, "7": 630, "8": 619, "9": 2806}.
- Meter row composition: {"0": 15383, "1": 4651, "2": 3372, "3": 1594}.
- Anomalous building-meter pairs: 50/81 (0.617284).

| position | site_id | building_id | primary_use | role | selection_reason | row_allocation_reason | meters | allocated_row_quota | selected_rows | selected_anomalies | selected_anomaly_rate | available_rows | available_anomaly_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 314 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 14 | 0.044444 | 8758 | 0.044302 |
| 2 | 2 | 270 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0,1 | 508 | 508 | 2 | 0.003937 | 14135 | 0.000990 |
| 3 | 13 | 1174 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 631 | 631 | 31 | 0.049128 | 17568 | 0.038821 |
| 4 | 4 | 578 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 0 | 0.000000 | 8783 | 0.000000 |
| 5 | 3 | 476 | - | early_stop | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 2 | 0.006349 | 8758 | 0.012560 |
| 6 | 14 | 1302 | - | fit | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 947 | 947 | 232 | 0.244984 | 26351 | 0.244279 |
| 7 | 5 | 720 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 2 | 0.006349 | 8784 | 0.003757 |
| 8 | 15 | 1358 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 799 | 799 | 0 | 0.000000 | 22254 | 0.000000 |
| 9 | 1 | 148 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 316 | 316 | 8 | 0.025316 | 8784 | 0.018215 |
| 10 | 0 | 28 | - | early_stop | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0,1 | 539 | 539 | 116 | 0.215213 | 14991 | 0.225402 |
| 11 | 8 | 844 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 311 | 311 | 1 | 0.003215 | 7959 | 0.002387 |
| 12 | 3 | 312 | - | fit | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 8 | 0.023324 | 8782 | 0.011501 |
| 13 | 9 | 876 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 686 | 686 | 50 | 0.072886 | 17546 | 0.080417 |
| 14 | 2 | 234 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 1030 | 1030 | 0 | 0.000000 | 26346 | 0.000000 |
| 15 | 13 | 1198 | - | early_stop | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0,2 | 687 | 687 | 0 | 0.000000 | 17568 | 0.000342 |
| 16 | 0 | 40 | - | fit | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0 | 330 | 330 | 130 | 0.393939 | 8428 | 0.402824 |
| 17 | 6 | 748 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0,1 | 635 | 635 | 4 | 0.006299 | 16254 | 0.008921 |
| 18 | 3 | 562 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 1 | 0.002915 | 8782 | 0.003758 |
| 19 | 15 | 1416 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 292 | 292 | 0 | 0.000000 | 7472 | 0.000000 |
| 20 | 2 | 230 | - | early_stop | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 0 | 0.000000 | 8782 | 0.000000 |
| 21 | 9 | 886 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 965 | 965 | 187 | 0.193782 | 26329 | 0.182650 |
| 22 | 13 | 1128 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 10 | 0.031153 | 8778 | 0.030645 |
| 23 | 15 | 1380 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1 | 542 | 542 | 0 | 0.000000 | 14795 | 0.000811 |
| 24 | 3 | 404 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 2 | 0.006211 | 8782 | 0.008882 |
| 25 | 14 | 1300 | - | early_stop | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 913 | 913 | 62 | 0.067908 | 24916 | 0.065059 |
| 26 | 13 | 1076 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,2 | 644 | 644 | 2 | 0.003106 | 17567 | 0.002789 |
| 27 | 4 | 604 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 37 | 37 | 11 | 0.297297 | 1012 | 0.281621 |
| 28 | 12 | 1054 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8783 | 0.000000 |
| 29 | 9 | 878 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 834 | 834 | 23 | 0.027578 | 22764 | 0.027236 |
| 30 | 5 | 692 | - | early_stop | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8784 | 0.002732 |
| 31 | 3 | 296 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 309 | 309 | 2 | 0.006472 | 8438 | 0.003792 |
| 32 | 14 | 1264 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,1 | 644 | 644 | 202 | 0.313665 | 17568 | 0.348019 |
| 33 | 2 | 158 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8783 | 0.000000 |
| 34 | 7 | 792 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 1,2 | 630 | 630 | 4 | 0.006349 | 17192 | 0.008318 |
| 35 | 9 | 902 | - | early_stop | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 9 | 0.028037 | 8764 | 0.035828 |
| 36 | 0 | 52 | - | fit | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 120 | 0.372671 | 8784 | 0.384563 |
| 37 | 11 | 1032 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 965 | 965 | 0 | 0.000000 | 26346 | 0.000076 |
| 38 | 3 | 514 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 317 | 317 | 0 | 0.000000 | 8657 | 0.000000 |
| 39 | 5 | 670 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8784 | 0.002618 |
| 40 | 15 | 1326 | - | early_stop | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 816 | 816 | 0 | 0.000000 | 22276 | 0.000000 |
| 41 | 3 | 534 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 4 | 0.012422 | 8782 | 0.005124 |
| 42 | 8 | 856 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 308 | 308 | 34 | 0.110390 | 8423 | 0.128220 |
| 43 | 13 | 1122 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1 | 644 | 644 | 42 | 0.065217 | 17568 | 0.064435 |
| 44 | 14 | 1256 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 644 | 644 | 17 | 0.026398 | 17568 | 0.036259 |
| 45 | 3 | 504 | - | early_stop | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 1 | 0.003106 | 8782 | 0.002847 |
| 46 | 2 | 232 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0,3 | 643 | 643 | 3 | 0.004666 | 17564 | 0.001139 |
| 47 | 4 | 598 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8783 | 0.000000 |
| 48 | 6 | 774 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 964 | 964 | 142 | 0.147303 | 26325 | 0.152137 |
| 49 | 10 | 1010 | - | fit | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 5 | 0.015576 | 8766 | 0.017568 |
| 50 | 4 | 590 | - | early_stop | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 136 | 0.425000 | 8737 | 0.407806 |

### K=100

- Selected rows: 50,000; anomalies: 3,009; anomaly rate: 0.060180.
- Site row composition: {"0": 2423, "1": 1919, "10": 321, "11": 965, "12": 962, "13": 5811, "14": 6236, "15": 5410, "2": 5008, "3": 6039, "4": 1634, "5": 1920, "6": 1919, "7": 630, "8": 1520, "9": 7283}.
- Meter row composition: {"0": 30376, "1": 9816, "2": 6691, "3": 3117}.
- Anomalous building-meter pairs: 99/162 (0.611111).

| position | site_id | building_id | primary_use | role | selection_reason | row_allocation_reason | meters | allocated_row_quota | selected_rows | selected_anomalies | selected_anomaly_rate | available_rows | available_anomaly_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 314 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 14 | 0.044444 | 8758 | 0.044302 |
| 2 | 2 | 270 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0,1 | 508 | 508 | 2 | 0.003937 | 14135 | 0.000990 |
| 3 | 13 | 1174 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 631 | 631 | 31 | 0.049128 | 17568 | 0.038821 |
| 4 | 4 | 578 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 0 | 0.000000 | 8783 | 0.000000 |
| 5 | 3 | 476 | - | early_stop | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 2 | 0.006349 | 8758 | 0.012560 |
| 6 | 14 | 1302 | - | fit | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 947 | 947 | 232 | 0.244984 | 26351 | 0.244279 |
| 7 | 5 | 720 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 315 | 315 | 2 | 0.006349 | 8784 | 0.003757 |
| 8 | 15 | 1358 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 799 | 799 | 0 | 0.000000 | 22254 | 0.000000 |
| 9 | 1 | 148 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 316 | 316 | 8 | 0.025316 | 8784 | 0.018215 |
| 10 | 0 | 28 | - | early_stop | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0,1 | 539 | 539 | 116 | 0.215213 | 14991 | 0.225402 |
| 11 | 8 | 844 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 311 | 311 | 1 | 0.003215 | 7959 | 0.002387 |
| 12 | 3 | 312 | - | fit | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 8 | 0.023324 | 8782 | 0.011501 |
| 13 | 9 | 876 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 686 | 686 | 50 | 0.072886 | 17546 | 0.080417 |
| 14 | 2 | 234 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 1030 | 1030 | 0 | 0.000000 | 26346 | 0.000000 |
| 15 | 13 | 1198 | - | early_stop | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0,2 | 687 | 687 | 0 | 0.000000 | 17568 | 0.000342 |
| 16 | 0 | 40 | - | fit | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0 | 330 | 330 | 130 | 0.393939 | 8428 | 0.402824 |
| 17 | 6 | 748 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0,1 | 635 | 635 | 4 | 0.006299 | 16254 | 0.008921 |
| 18 | 3 | 562 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 1 | 0.002915 | 8782 | 0.003758 |
| 19 | 15 | 1416 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 292 | 292 | 0 | 0.000000 | 7472 | 0.000000 |
| 20 | 2 | 230 | - | early_stop | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 343 | 343 | 0 | 0.000000 | 8782 | 0.000000 |
| 21 | 9 | 886 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 965 | 965 | 187 | 0.193782 | 26329 | 0.182650 |
| 22 | 13 | 1128 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 10 | 0.031153 | 8778 | 0.030645 |
| 23 | 15 | 1380 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1 | 542 | 542 | 0 | 0.000000 | 14795 | 0.000811 |
| 24 | 3 | 404 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 2 | 0.006211 | 8782 | 0.008882 |
| 25 | 14 | 1300 | - | early_stop | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 913 | 913 | 62 | 0.067908 | 24916 | 0.065059 |
| 26 | 13 | 1076 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,2 | 644 | 644 | 2 | 0.003106 | 17567 | 0.002789 |
| 27 | 4 | 604 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 37 | 37 | 11 | 0.297297 | 1012 | 0.281621 |
| 28 | 12 | 1054 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8783 | 0.000000 |
| 29 | 9 | 878 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 834 | 834 | 23 | 0.027578 | 22764 | 0.027236 |
| 30 | 5 | 692 | - | early_stop | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8784 | 0.002732 |
| 31 | 3 | 296 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 309 | 309 | 2 | 0.006472 | 8438 | 0.003792 |
| 32 | 14 | 1264 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,1 | 644 | 644 | 202 | 0.313665 | 17568 | 0.348019 |
| 33 | 2 | 158 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8783 | 0.000000 |
| 34 | 7 | 792 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 1,2 | 630 | 630 | 4 | 0.006349 | 17192 | 0.008318 |
| 35 | 9 | 902 | - | early_stop | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 9 | 0.028037 | 8764 | 0.035828 |
| 36 | 0 | 52 | - | fit | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 120 | 0.372671 | 8784 | 0.384563 |
| 37 | 11 | 1032 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 965 | 965 | 0 | 0.000000 | 26346 | 0.000076 |
| 38 | 3 | 514 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 317 | 317 | 0 | 0.000000 | 8657 | 0.000000 |
| 39 | 5 | 670 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8784 | 0.002618 |
| 40 | 15 | 1326 | - | early_stop | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 816 | 816 | 0 | 0.000000 | 22276 | 0.000000 |
| 41 | 3 | 534 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 4 | 0.012422 | 8782 | 0.005124 |
| 42 | 8 | 856 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 308 | 308 | 34 | 0.110390 | 8423 | 0.128220 |
| 43 | 13 | 1122 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1 | 644 | 644 | 42 | 0.065217 | 17568 | 0.064435 |
| 44 | 14 | 1256 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,2 | 644 | 644 | 17 | 0.026398 | 17568 | 0.036259 |
| 45 | 3 | 504 | - | early_stop | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 1 | 0.003106 | 8782 | 0.002847 |
| 46 | 2 | 232 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0,3 | 643 | 643 | 3 | 0.004666 | 17564 | 0.001139 |
| 47 | 4 | 598 | - | fit | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0 | 322 | 322 | 0 | 0.000000 | 8783 | 0.000000 |
| 48 | 6 | 774 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 964 | 964 | 142 | 0.147303 | 26325 | 0.152137 |
| 49 | 10 | 1010 | - | fit | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 5 | 0.015576 | 8766 | 0.017568 |
| 50 | 4 | 590 | - | early_stop | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 136 | 0.425000 | 8737 | 0.407806 |
| 51 | 15 | 1334 | - | fit | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0,1 | 539 | 539 | 0 | 0.000000 | 14761 | 0.000068 |
| 52 | 1 | 118 | - | fit | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 4 | 0.012461 | 8784 | 0.012523 |
| 53 | 15 | 1366 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 813 | 813 | 0 | 0.000000 | 22271 | 0.000000 |
| 54 | 3 | 560 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 318 | 318 | 0 | 0.000000 | 8710 | 0.002755 |
| 55 | 13 | 1202 | - | early_stop | anomaly_bin:positive_low | proportional_to_available_rows_within_incremental_K_block | 0,1 | 641 | 641 | 6 | 0.009360 | 17567 | 0.009393 |
| 56 | 0 | 36 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 130 | 0.404984 | 8784 | 0.384563 |
| 57 | 9 | 898 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 960 | 960 | 17 | 0.017708 | 26313 | 0.023372 |
| 58 | 3 | 544 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 311 | 311 | 0 | 0.000000 | 8518 | 0.000000 |
| 59 | 1 | 114 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0,3 | 641 | 641 | 62 | 0.096724 | 17568 | 0.108265 |
| 60 | 12 | 1064 | - | early_stop | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 0 | 0.000000 | 8783 | 0.000455 |
| 61 | 13 | 1136 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0,1 | 641 | 641 | 1 | 0.001560 | 17568 | 0.000342 |
| 62 | 5 | 738 | - | fit | meter_presence:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 58 | 0.181250 | 8784 | 0.186931 |
| 63 | 2 | 156 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 2 | 0.006250 | 8783 | 0.004440 |
| 64 | 6 | 754 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 2 | 320 | 320 | 0 | 0.000000 | 8760 | 0.000000 |
| 65 | 15 | 1436 | - | early_stop | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0,2 | 545 | 545 | 0 | 0.000000 | 14940 | 0.000000 |
| 66 | 13 | 1146 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1 | 641 | 641 | 1 | 0.001560 | 17568 | 0.000342 |
| 67 | 3 | 492 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 312 | 312 | 3 | 0.009615 | 8542 | 0.011239 |
| 68 | 0 | 38 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 125 | 0.389408 | 8784 | 0.384563 |
| 69 | 2 | 222 | - | fit | meter_presence:1 | proportional_to_available_rows_within_incremental_K_block | 0,1 | 640 | 640 | 2 | 0.003125 | 17548 | 0.000513 |
| 70 | 14 | 1280 | - | early_stop | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0,1 | 641 | 641 | 150 | 0.234009 | 17568 | 0.233265 |
| 71 | 8 | 830 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 305 | 305 | 2 | 0.006557 | 8351 | 0.008622 |
| 72 | 3 | 542 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 0 | 0.000000 | 8782 | 0.000000 |
| 73 | 14 | 1258 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0,1,2,3 | 1166 | 1166 | 44 | 0.037736 | 31967 | 0.048488 |
| 74 | 5 | 660 | - | fit | meter_row_share:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 1 | 0.003115 | 8784 | 0.002618 |
| 75 | 3 | 470 | - | early_stop | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 2 | 0.006250 | 8758 | 0.008335 |
| 76 | 9 | 972 | - | fit | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 960 | 960 | 147 | 0.153125 | 26316 | 0.133341 |
| 77 | 4 | 620 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 0 | 0.000000 | 8783 | 0.000911 |
| 78 | 3 | 292 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 317 | 317 | 0 | 0.000000 | 8692 | 0.000000 |
| 79 | 9 | 990 | - | fit | anomaly_bin:positive_mid | proportional_to_available_rows_within_incremental_K_block | 0,1 | 641 | 641 | 17 | 0.026521 | 17557 | 0.028308 |
| 80 | 2 | 278 | - | early_stop | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 8 | 0.025000 | 8783 | 0.030400 |
| 81 | 13 | 1186 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0,2 | 641 | 641 | 1 | 0.001560 | 17568 | 0.001138 |
| 82 | 0 | 10 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 321 | 321 | 115 | 0.358255 | 8784 | 0.384563 |
| 83 | 2 | 184 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0,1,3 | 882 | 882 | 0 | 0.000000 | 24171 | 0.000000 |
| 84 | 3 | 376 | - | fit | size_bin:size_q4 | proportional_to_available_rows_within_incremental_K_block | 0 | 273 | 273 | 4 | 0.014652 | 7486 | 0.009752 |
| 85 | 9 | 928 | - | early_stop | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 955 | 955 | 56 | 0.058639 | 26169 | 0.046353 |
| 86 | 12 | 1050 | - | fit | size_bin:size_q4 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 0 | 0.000000 | 8784 | 0.000683 |
| 87 | 8 | 848 | - | fit | size_bin:size_q4 | proportional_to_available_rows_within_incremental_K_block | 0 | 305 | 305 | 89 | 0.291803 | 8351 | 0.272901 |
| 88 | 15 | 1408 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1 | 524 | 524 | 0 | 0.000000 | 14349 | 0.000000 |
| 89 | 13 | 1170 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 3 | 0.009375 | 8783 | 0.015826 |
| 90 | 8 | 828 | - | early_stop | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0 | 291 | 291 | 0 | 0.000000 | 7986 | 0.002379 |
| 91 | 14 | 1294 | - | fit | meter_presence:2 | proportional_to_available_rows_within_incremental_K_block | 0,1,2,3 | 1281 | 1281 | 99 | 0.077283 | 35108 | 0.088270 |
| 92 | 3 | 458 | - | fit | anomaly_bin:positive_high | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 0 | 0.000000 | 8782 | 0.002733 |
| 93 | 5 | 732 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 8 | 0.025000 | 8784 | 0.038593 |
| 94 | 15 | 1354 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 1,2 | 540 | 540 | 0 | 0.000000 | 14799 | 0.000000 |
| 95 | 3 | 468 | - | early_stop | meter_row_share:0 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 0 | 0.000000 | 8782 | 0.000000 |
| 96 | 0 | 48 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 269 | 269 | 76 | 0.282528 | 7371 | 0.267806 |
| 97 | 4 | 592 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 0 | 0.000000 | 8779 | 0.001595 |
| 98 | 9 | 996 | - | fit | zero_anomaly_share | proportional_to_available_rows_within_incremental_K_block | 0,1,2 | 961 | 961 | 39 | 0.040583 | 26329 | 0.039880 |
| 99 | 3 | 442 | - | fit | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0 | 320 | 320 | 0 | 0.000000 | 8782 | 0.002733 |
| 100 | 1 | 138 | - | early_stop | meter_presence:3 | proportional_to_available_rows_within_incremental_K_block | 0,3 | 641 | 641 | 118 | 0.184087 | 17564 | 0.170292 |

## Curve artifacts and reproducibility

- Detailed metrics: /home/kuant_kuo/projects/lead-reproduction-e3/data/processed/m5_building_curve/aggregate/metrics.csv.
- Plot-ready ROC/PR points for overall, every meter and every site: /home/kuant_kuo/projects/lead-reproduction-e3/data/processed/m5_building_curve/aggregate/curves.csv.
- Every model cell stores full predictions, provenance, heartbeats, model checkpoints, prediction chunks and an atomic COMPLETE.json.
- Report rows are accepted only when holdout raw-index and labels are byte-identical to the matched-context baseline.
