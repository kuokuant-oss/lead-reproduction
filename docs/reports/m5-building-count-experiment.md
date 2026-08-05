# M5 building-count experiment

This report is atomically regenerated at every overnight publication gate. Training sources are restricted to building_id % 2 == 0; the complete canonical holdout contains only odd building IDs.

K=10/20/50/100 uses only 137 features. Average allocation is at most 500 rows per building, so total context is bounded by 5K/10K/25K/50K. Allocation within each incremental K block is proportional to each building's available rows, so individual buildings may contribute above or below 500. Rows are selected by a seed-42 stable hash of raw identity without consulting labels. Building and row sets are strict nested prefixes. Trees use building-disjoint 80/20 fit and early-stop roles to choose iteration counts by PR-AUC, then final-refit using the exact M3 post-sort [negs1,pos,negs2,pos] sampling and float64 scaler path. TabPFN uses the same selected rows and has no task-specific epoch or weight-update loop, so early stopping is not applicable.

The direct TabPFN reference is the m5-matched-context-breakdown 50K / 137-feature / n_estimators=8 prediction artifact. Different context row/class distributions are allowed. Comparability is maintained by the same feature pipeline, scaler, checkpoint/config, canonical holdout identity and evaluator.

## Execution status

| stage | status |
|---|---|
| tree full K=725 f=17 | complete |
| tree full K=725 f=137 | complete |
| tree K=10 f=137 | complete |
| tabpfn K=10 f=137 | complete |
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
| building experiment | 725 | 17 | catboost | 10137155 | 637397 | 0.793279 | 0.955753 |
| building experiment | 725 | 17 | ensemble | 10137155 | 637397 | 0.821459 | 0.964359 |
| building experiment | 725 | 17 | hist_gradient_boosting | 10137155 | 637397 | 0.820174 | 0.960793 |
| building experiment | 725 | 17 | lightgbm | 10137155 | 637397 | 0.821083 | 0.964520 |
| building experiment | 725 | 17 | xgboost | 10137155 | 637397 | 0.796658 | 0.958488 |
| building experiment | 10 | 137 | catboost | 10137155 | 637397 | 0.640666 | 0.962066 |
| building experiment | 10 | 137 | ensemble | 10137155 | 637397 | 0.722452 | 0.968506 |
| building experiment | 10 | 137 | hist_gradient_boosting | 10137155 | 637397 | 0.644253 | 0.947803 |
| building experiment | 10 | 137 | lightgbm | 10137155 | 637397 | 0.603204 | 0.919940 |
| building experiment | 10 | 137 | tabpfn | 10137155 | 637397 | 0.710155 | 0.972253 |
| building experiment | 10 | 137 | xgboost | 10137155 | 637397 | 0.754752 | 0.971070 |
| building experiment | 725 | 137 | catboost | 10137155 | 637397 | 0.918181 | 0.985784 |
| building experiment | 725 | 137 | ensemble | 10137155 | 637397 | 0.927956 | 0.991717 |
| building experiment | 725 | 137 | hist_gradient_boosting | 10137155 | 637397 | 0.921726 | 0.991534 |
| building experiment | 725 | 137 | lightgbm | 10137155 | 637397 | 0.922766 | 0.991084 |
| building experiment | 725 | 137 | xgboost | 10137155 | 637397 | 0.923740 | 0.990130 |

## Meter breakdown

| sampling_profile | building_budget | features | model | group_label | rows | anomalies | pr_auc | roc_auc |
|---|---|---|---|---|---|---|---|---|
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | electricity | 6035071 | 356679 | 0.985040 | 0.997838 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | chilledwater | 2115354 | 141139 | 0.829599 | 0.984698 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | steam | 1350609 | 48888 | 0.822684 | 0.985801 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | hotwater | 636121 | 90691 | 0.824857 | 0.953947 |
| representative | 725 | 17 | catboost | electricity | 6035071 | 356679 | 0.914273 | 0.968324 |
| representative | 725 | 17 | catboost | chilledwater | 2115354 | 141139 | 0.499906 | 0.931966 |
| representative | 725 | 17 | catboost | steam | 1350609 | 48888 | 0.378693 | 0.887056 |
| representative | 725 | 17 | catboost | hotwater | 636121 | 90691 | 0.582166 | 0.921931 |
| representative | 725 | 17 | ensemble | electricity | 6035071 | 356679 | 0.929693 | 0.976557 |
| representative | 725 | 17 | ensemble | chilledwater | 2115354 | 141139 | 0.555149 | 0.935815 |
| representative | 725 | 17 | ensemble | steam | 1350609 | 48888 | 0.484859 | 0.920262 |
| representative | 725 | 17 | ensemble | hotwater | 636121 | 90691 | 0.594620 | 0.929954 |
| representative | 725 | 17 | hist_gradient_boosting | electricity | 6035071 | 356679 | 0.918205 | 0.970926 |
| representative | 725 | 17 | hist_gradient_boosting | chilledwater | 2115354 | 141139 | 0.595018 | 0.937551 |
| representative | 725 | 17 | hist_gradient_boosting | steam | 1350609 | 48888 | 0.521844 | 0.919535 |
| representative | 725 | 17 | hist_gradient_boosting | hotwater | 636121 | 90691 | 0.527937 | 0.918035 |
| representative | 725 | 17 | lightgbm | electricity | 6035071 | 356679 | 0.928131 | 0.975993 |
| representative | 725 | 17 | lightgbm | chilledwater | 2115354 | 141139 | 0.565065 | 0.935454 |
| representative | 725 | 17 | lightgbm | steam | 1350609 | 48888 | 0.428473 | 0.922515 |
| representative | 725 | 17 | lightgbm | hotwater | 636121 | 90691 | 0.565256 | 0.926902 |
| representative | 725 | 17 | xgboost | electricity | 6035071 | 356679 | 0.916323 | 0.971547 |
| representative | 725 | 17 | xgboost | chilledwater | 2115354 | 141139 | 0.460913 | 0.922625 |
| representative | 725 | 17 | xgboost | steam | 1350609 | 48888 | 0.419411 | 0.919662 |
| representative | 725 | 17 | xgboost | hotwater | 636121 | 90691 | 0.549154 | 0.912774 |
| representative | 10 | 137 | catboost | electricity | 6035071 | 356679 | 0.900553 | 0.982064 |
| representative | 10 | 137 | catboost | chilledwater | 2115354 | 141139 | 0.407916 | 0.938387 |
| representative | 10 | 137 | catboost | steam | 1350609 | 48888 | 0.332695 | 0.899253 |
| representative | 10 | 137 | catboost | hotwater | 636121 | 90691 | 0.572543 | 0.911823 |
| representative | 10 | 137 | ensemble | electricity | 6035071 | 356679 | 0.940382 | 0.987746 |
| representative | 10 | 137 | ensemble | chilledwater | 2115354 | 141139 | 0.461733 | 0.945806 |
| representative | 10 | 137 | ensemble | steam | 1350609 | 48888 | 0.320400 | 0.907757 |
| representative | 10 | 137 | ensemble | hotwater | 636121 | 90691 | 0.589285 | 0.920601 |
| representative | 10 | 137 | hist_gradient_boosting | electricity | 6035071 | 356679 | 0.890800 | 0.969285 |
| representative | 10 | 137 | hist_gradient_boosting | chilledwater | 2115354 | 141139 | 0.474532 | 0.938130 |
| representative | 10 | 137 | hist_gradient_boosting | steam | 1350609 | 48888 | 0.270886 | 0.893948 |
| representative | 10 | 137 | hist_gradient_boosting | hotwater | 636121 | 90691 | 0.541036 | 0.891012 |
| representative | 10 | 137 | lightgbm | electricity | 6035071 | 356679 | 0.852744 | 0.931103 |
| representative | 10 | 137 | lightgbm | chilledwater | 2115354 | 141139 | 0.449015 | 0.914216 |
| representative | 10 | 137 | lightgbm | steam | 1350609 | 48888 | 0.307762 | 0.863454 |
| representative | 10 | 137 | lightgbm | hotwater | 636121 | 90691 | 0.578829 | 0.902142 |
| representative | 10 | 137 | tabpfn | electricity | 6035071 | 356679 | 0.956596 | 0.991163 |
| representative | 10 | 137 | tabpfn | chilledwater | 2115354 | 141139 | 0.446448 | 0.951626 |
| representative | 10 | 137 | tabpfn | steam | 1350609 | 48888 | 0.395821 | 0.921316 |
| representative | 10 | 137 | tabpfn | hotwater | 636121 | 90691 | 0.720603 | 0.937102 |
| representative | 10 | 137 | xgboost | electricity | 6035071 | 356679 | 0.937669 | 0.988163 |
| representative | 10 | 137 | xgboost | chilledwater | 2115354 | 141139 | 0.513269 | 0.952360 |
| representative | 10 | 137 | xgboost | steam | 1350609 | 48888 | 0.326785 | 0.914563 |
| representative | 10 | 137 | xgboost | hotwater | 636121 | 90691 | 0.606344 | 0.921287 |
| representative | 725 | 137 | catboost | electricity | 6035071 | 356679 | 0.991389 | 0.998976 |
| representative | 725 | 137 | catboost | chilledwater | 2115354 | 141139 | 0.686212 | 0.974896 |
| representative | 725 | 137 | catboost | steam | 1350609 | 48888 | 0.732554 | 0.956646 |
| representative | 725 | 137 | catboost | hotwater | 636121 | 90691 | 0.776069 | 0.933124 |
| representative | 725 | 137 | ensemble | electricity | 6035071 | 356679 | 0.993072 | 0.999273 |
| representative | 725 | 137 | ensemble | chilledwater | 2115354 | 141139 | 0.767997 | 0.979424 |
| representative | 725 | 137 | ensemble | steam | 1350609 | 48888 | 0.744760 | 0.964122 |
| representative | 725 | 137 | ensemble | hotwater | 636121 | 90691 | 0.812718 | 0.948816 |
| representative | 725 | 137 | hist_gradient_boosting | electricity | 6035071 | 356679 | 0.991133 | 0.999004 |
| representative | 725 | 137 | hist_gradient_boosting | chilledwater | 2115354 | 141139 | 0.739761 | 0.978268 |
| representative | 725 | 137 | hist_gradient_boosting | steam | 1350609 | 48888 | 0.724034 | 0.961794 |
| representative | 725 | 137 | hist_gradient_boosting | hotwater | 636121 | 90691 | 0.800899 | 0.951111 |
| representative | 725 | 137 | lightgbm | electricity | 6035071 | 356679 | 0.991474 | 0.998940 |
| representative | 725 | 137 | lightgbm | chilledwater | 2115354 | 141139 | 0.761628 | 0.979502 |
| representative | 725 | 137 | lightgbm | steam | 1350609 | 48888 | 0.687226 | 0.959463 |
| representative | 725 | 137 | lightgbm | hotwater | 636121 | 90691 | 0.801744 | 0.945899 |
| representative | 725 | 137 | xgboost | electricity | 6035071 | 356679 | 0.991040 | 0.999035 |
| representative | 725 | 137 | xgboost | chilledwater | 2115354 | 141139 | 0.771136 | 0.978937 |
| representative | 725 | 137 | xgboost | steam | 1350609 | 48888 | 0.730435 | 0.963072 |
| representative | 725 | 137 | xgboost | hotwater | 636121 | 90691 | 0.787360 | 0.940245 |

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
| representative | 725 | 17 | catboost | site 0 | 538432 | 176269 | 0.997645 | 0.998022 |
| representative | 725 | 17 | catboost | site 1 | 289853 | 39135 | 0.383559 | 0.758608 |
| representative | 725 | 17 | catboost | site 2 | 1263915 | 80897 | 0.626710 | 0.957139 |
| representative | 725 | 17 | catboost | site 3 | 1181463 | 2684 | 0.807725 | 0.938436 |
| representative | 725 | 17 | catboost | site 4 | 370460 | 197 | 0.707548 | 0.916254 |
| representative | 725 | 17 | catboost | site 5 | 386496 | 14435 | 0.958644 | 0.992877 |
| representative | 725 | 17 | catboost | site 6 | 345117 | 28654 | 0.630915 | 0.920402 |
| representative | 725 | 17 | catboost | site 7 | 200594 | 15886 | 0.829344 | 0.958863 |
| representative | 725 | 17 | catboost | site 8 | 284376 | 31083 | 0.866162 | 0.951788 |
| representative | 725 | 17 | catboost | site 9 | 1367482 | 52587 | 0.876046 | 0.963827 |
| representative | 725 | 17 | catboost | site 10 | 206430 | 15814 | 0.426937 | 0.929467 |
| representative | 725 | 17 | catboost | site 11 | 43626 | 197 | 0.027327 | 0.894084 |
| representative | 725 | 17 | catboost | site 12 | 158011 | 755 | 0.974417 | 0.988732 |
| representative | 725 | 17 | catboost | site 13 | 1334223 | 59283 | 0.541994 | 0.882962 |
| representative | 725 | 17 | catboost | site 14 | 1256775 | 101532 | 0.636300 | 0.902755 |
| representative | 725 | 17 | catboost | site 15 | 909902 | 17989 | 0.792227 | 0.984902 |
| representative | 725 | 17 | ensemble | site 0 | 538432 | 176269 | 0.997876 | 0.997979 |
| representative | 725 | 17 | ensemble | site 1 | 289853 | 39135 | 0.472732 | 0.823691 |
| representative | 725 | 17 | ensemble | site 2 | 1263915 | 80897 | 0.626337 | 0.956057 |
| representative | 725 | 17 | ensemble | site 3 | 1181463 | 2684 | 0.810199 | 0.949839 |
| representative | 725 | 17 | ensemble | site 4 | 370460 | 197 | 0.726768 | 0.895085 |
| representative | 725 | 17 | ensemble | site 5 | 386496 | 14435 | 0.961038 | 0.995113 |
| representative | 725 | 17 | ensemble | site 6 | 345117 | 28654 | 0.618370 | 0.925376 |
| representative | 725 | 17 | ensemble | site 7 | 200594 | 15886 | 0.864359 | 0.957820 |
| representative | 725 | 17 | ensemble | site 8 | 284376 | 31083 | 0.865821 | 0.948244 |
| representative | 725 | 17 | ensemble | site 9 | 1367482 | 52587 | 0.921121 | 0.991286 |
| representative | 725 | 17 | ensemble | site 10 | 206430 | 15814 | 0.580938 | 0.947766 |
| representative | 725 | 17 | ensemble | site 11 | 43626 | 197 | 0.022780 | 0.904979 |
| representative | 725 | 17 | ensemble | site 12 | 158011 | 755 | 0.974970 | 0.989544 |
| representative | 725 | 17 | ensemble | site 13 | 1334223 | 59283 | 0.551758 | 0.897511 |
| representative | 725 | 17 | ensemble | site 14 | 1256775 | 101532 | 0.692590 | 0.916161 |
| representative | 725 | 17 | ensemble | site 15 | 909902 | 17989 | 0.816188 | 0.985639 |
| representative | 725 | 17 | hist_gradient_boosting | site 0 | 538432 | 176269 | 0.998196 | 0.998387 |
| representative | 725 | 17 | hist_gradient_boosting | site 1 | 289853 | 39135 | 0.444473 | 0.805704 |
| representative | 725 | 17 | hist_gradient_boosting | site 2 | 1263915 | 80897 | 0.564434 | 0.952156 |
| representative | 725 | 17 | hist_gradient_boosting | site 3 | 1181463 | 2684 | 0.812457 | 0.960399 |
| representative | 725 | 17 | hist_gradient_boosting | site 4 | 370460 | 197 | 0.527634 | 0.915523 |
| representative | 725 | 17 | hist_gradient_boosting | site 5 | 386496 | 14435 | 0.951593 | 0.992024 |
| representative | 725 | 17 | hist_gradient_boosting | site 6 | 345117 | 28654 | 0.663787 | 0.920455 |
| representative | 725 | 17 | hist_gradient_boosting | site 7 | 200594 | 15886 | 0.848938 | 0.953258 |
| representative | 725 | 17 | hist_gradient_boosting | site 8 | 284376 | 31083 | 0.870426 | 0.942145 |
| representative | 725 | 17 | hist_gradient_boosting | site 9 | 1367482 | 52587 | 0.933760 | 0.993655 |
| representative | 725 | 17 | hist_gradient_boosting | site 10 | 206430 | 15814 | 0.592102 | 0.947194 |
| representative | 725 | 17 | hist_gradient_boosting | site 11 | 43626 | 197 | 0.048983 | 0.948845 |
| representative | 725 | 17 | hist_gradient_boosting | site 12 | 158011 | 755 | 0.975484 | 0.992119 |
| representative | 725 | 17 | hist_gradient_boosting | site 13 | 1334223 | 59283 | 0.542955 | 0.881927 |
| representative | 725 | 17 | hist_gradient_boosting | site 14 | 1256775 | 101532 | 0.743488 | 0.927861 |
| representative | 725 | 17 | hist_gradient_boosting | site 15 | 909902 | 17989 | 0.820518 | 0.986222 |
| representative | 725 | 17 | lightgbm | site 0 | 538432 | 176269 | 0.997606 | 0.997642 |
| representative | 725 | 17 | lightgbm | site 1 | 289853 | 39135 | 0.452034 | 0.794757 |
| representative | 725 | 17 | lightgbm | site 2 | 1263915 | 80897 | 0.619207 | 0.953421 |
| representative | 725 | 17 | lightgbm | site 3 | 1181463 | 2684 | 0.813495 | 0.946526 |
| representative | 725 | 17 | lightgbm | site 4 | 370460 | 197 | 0.400691 | 0.848436 |
| representative | 725 | 17 | lightgbm | site 5 | 386496 | 14435 | 0.961254 | 0.992432 |
| representative | 725 | 17 | lightgbm | site 6 | 345117 | 28654 | 0.586161 | 0.927986 |
| representative | 725 | 17 | lightgbm | site 7 | 200594 | 15886 | 0.827519 | 0.950991 |
| representative | 725 | 17 | lightgbm | site 8 | 284376 | 31083 | 0.865332 | 0.944509 |
| representative | 725 | 17 | lightgbm | site 9 | 1367482 | 52587 | 0.905427 | 0.986358 |
| representative | 725 | 17 | lightgbm | site 10 | 206430 | 15814 | 0.675877 | 0.956351 |
| representative | 725 | 17 | lightgbm | site 11 | 43626 | 197 | 0.026089 | 0.907458 |
| representative | 725 | 17 | lightgbm | site 12 | 158011 | 755 | 0.974898 | 0.988268 |
| representative | 725 | 17 | lightgbm | site 13 | 1334223 | 59283 | 0.561294 | 0.897532 |
| representative | 725 | 17 | lightgbm | site 14 | 1256775 | 101532 | 0.692979 | 0.917875 |
| representative | 725 | 17 | lightgbm | site 15 | 909902 | 17989 | 0.826442 | 0.981621 |
| representative | 725 | 17 | xgboost | site 0 | 538432 | 176269 | 0.996886 | 0.996602 |
| representative | 725 | 17 | xgboost | site 1 | 289853 | 39135 | 0.350315 | 0.747793 |
| representative | 725 | 17 | xgboost | site 2 | 1263915 | 80897 | 0.610194 | 0.951726 |
| representative | 725 | 17 | xgboost | site 3 | 1181463 | 2684 | 0.797995 | 0.923854 |
| representative | 725 | 17 | xgboost | site 4 | 370460 | 197 | 0.112573 | 0.786483 |
| representative | 725 | 17 | xgboost | site 5 | 386496 | 14435 | 0.943309 | 0.992915 |
| representative | 725 | 17 | xgboost | site 6 | 345117 | 28654 | 0.541030 | 0.923529 |
| representative | 725 | 17 | xgboost | site 7 | 200594 | 15886 | 0.826260 | 0.958119 |
| representative | 725 | 17 | xgboost | site 8 | 284376 | 31083 | 0.843813 | 0.925127 |
| representative | 725 | 17 | xgboost | site 9 | 1367482 | 52587 | 0.874402 | 0.959932 |
| representative | 725 | 17 | xgboost | site 10 | 206430 | 15814 | 0.437890 | 0.925462 |
| representative | 725 | 17 | xgboost | site 11 | 43626 | 197 | 0.035783 | 0.907567 |
| representative | 725 | 17 | xgboost | site 12 | 158011 | 755 | 0.974466 | 0.986004 |
| representative | 725 | 17 | xgboost | site 13 | 1334223 | 59283 | 0.553583 | 0.909714 |
| representative | 725 | 17 | xgboost | site 14 | 1256775 | 101532 | 0.660048 | 0.907667 |
| representative | 725 | 17 | xgboost | site 15 | 909902 | 17989 | 0.784945 | 0.980862 |
| representative | 10 | 137 | catboost | site 0 | 538432 | 176269 | 0.993973 | 0.994792 |
| representative | 10 | 137 | catboost | site 1 | 289853 | 39135 | 0.574024 | 0.932253 |
| representative | 10 | 137 | catboost | site 2 | 1263915 | 80897 | 0.714949 | 0.977692 |
| representative | 10 | 137 | catboost | site 3 | 1181463 | 2684 | 0.207611 | 0.952644 |
| representative | 10 | 137 | catboost | site 4 | 370460 | 197 | 0.066388 | 0.924897 |
| representative | 10 | 137 | catboost | site 5 | 386496 | 14435 | 0.920509 | 0.988082 |
| representative | 10 | 137 | catboost | site 6 | 345117 | 28654 | 0.573352 | 0.957455 |
| representative | 10 | 137 | catboost | site 7 | 200594 | 15886 | 0.303412 | 0.895941 |
| representative | 10 | 137 | catboost | site 8 | 284376 | 31083 | 0.675416 | 0.910403 |
| representative | 10 | 137 | catboost | site 9 | 1367482 | 52587 | 0.658957 | 0.935626 |
| representative | 10 | 137 | catboost | site 10 | 206430 | 15814 | 0.338402 | 0.900476 |
| representative | 10 | 137 | catboost | site 11 | 43626 | 197 | 0.020739 | 0.868821 |
| representative | 10 | 137 | catboost | site 12 | 158011 | 755 | 0.882364 | 0.996227 |
| representative | 10 | 137 | catboost | site 13 | 1334223 | 59283 | 0.296738 | 0.920481 |
| representative | 10 | 137 | catboost | site 14 | 1256775 | 101532 | 0.678940 | 0.947351 |
| representative | 10 | 137 | catboost | site 15 | 909902 | 17989 | 0.658170 | 0.986298 |
| representative | 10 | 137 | ensemble | site 0 | 538432 | 176269 | 0.996942 | 0.997026 |
| representative | 10 | 137 | ensemble | site 1 | 289853 | 39135 | 0.491044 | 0.929513 |
| representative | 10 | 137 | ensemble | site 2 | 1263915 | 80897 | 0.756316 | 0.982008 |
| representative | 10 | 137 | ensemble | site 3 | 1181463 | 2684 | 0.865534 | 0.984794 |
| representative | 10 | 137 | ensemble | site 4 | 370460 | 197 | 0.762086 | 0.922216 |
| representative | 10 | 137 | ensemble | site 5 | 386496 | 14435 | 0.967236 | 0.995915 |
| representative | 10 | 137 | ensemble | site 6 | 345117 | 28654 | 0.579245 | 0.961672 |
| representative | 10 | 137 | ensemble | site 7 | 200594 | 15886 | 0.329610 | 0.896859 |
| representative | 10 | 137 | ensemble | site 8 | 284376 | 31083 | 0.870789 | 0.941631 |
| representative | 10 | 137 | ensemble | site 9 | 1367482 | 52587 | 0.780341 | 0.951365 |
| representative | 10 | 137 | ensemble | site 10 | 206430 | 15814 | 0.313873 | 0.911967 |
| representative | 10 | 137 | ensemble | site 11 | 43626 | 197 | 0.018133 | 0.873076 |
| representative | 10 | 137 | ensemble | site 12 | 158011 | 755 | 0.986295 | 0.999452 |
| representative | 10 | 137 | ensemble | site 13 | 1334223 | 59283 | 0.346855 | 0.929984 |
| representative | 10 | 137 | ensemble | site 14 | 1256775 | 101532 | 0.710653 | 0.947114 |
| representative | 10 | 137 | ensemble | site 15 | 909902 | 17989 | 0.721433 | 0.991534 |
| representative | 10 | 137 | hist_gradient_boosting | site 0 | 538432 | 176269 | 0.997482 | 0.997663 |
| representative | 10 | 137 | hist_gradient_boosting | site 1 | 289853 | 39135 | 0.324505 | 0.823541 |
| representative | 10 | 137 | hist_gradient_boosting | site 2 | 1263915 | 80897 | 0.651494 | 0.970767 |
| representative | 10 | 137 | hist_gradient_boosting | site 3 | 1181463 | 2684 | 0.872286 | 0.978853 |
| representative | 10 | 137 | hist_gradient_boosting | site 4 | 370460 | 197 | 0.762974 | 0.894854 |
| representative | 10 | 137 | hist_gradient_boosting | site 5 | 386496 | 14435 | 0.771387 | 0.963897 |
| representative | 10 | 137 | hist_gradient_boosting | site 6 | 345117 | 28654 | 0.465486 | 0.936070 |
| representative | 10 | 137 | hist_gradient_boosting | site 7 | 200594 | 15886 | 0.325583 | 0.887800 |
| representative | 10 | 137 | hist_gradient_boosting | site 8 | 284376 | 31083 | 0.865054 | 0.931071 |
| representative | 10 | 137 | hist_gradient_boosting | site 9 | 1367482 | 52587 | 0.712453 | 0.952792 |
| representative | 10 | 137 | hist_gradient_boosting | site 10 | 206430 | 15814 | 0.376154 | 0.923226 |
| representative | 10 | 137 | hist_gradient_boosting | site 11 | 43626 | 197 | 0.099946 | 0.929151 |
| representative | 10 | 137 | hist_gradient_boosting | site 12 | 158011 | 755 | 0.949628 | 0.999525 |
| representative | 10 | 137 | hist_gradient_boosting | site 13 | 1334223 | 59283 | 0.339134 | 0.924153 |
| representative | 10 | 137 | hist_gradient_boosting | site 14 | 1256775 | 101532 | 0.627985 | 0.893698 |
| representative | 10 | 137 | hist_gradient_boosting | site 15 | 909902 | 17989 | 0.711673 | 0.990027 |
| representative | 10 | 137 | lightgbm | site 0 | 538432 | 176269 | 0.996555 | 0.997311 |
| representative | 10 | 137 | lightgbm | site 1 | 289853 | 39135 | 0.171874 | 0.561157 |
| representative | 10 | 137 | lightgbm | site 2 | 1263915 | 80897 | 0.679356 | 0.966910 |
| representative | 10 | 137 | lightgbm | site 3 | 1181463 | 2684 | 0.767374 | 0.954469 |
| representative | 10 | 137 | lightgbm | site 4 | 370460 | 197 | 0.692494 | 0.892676 |
| representative | 10 | 137 | lightgbm | site 5 | 386496 | 14435 | 0.770192 | 0.896625 |
| representative | 10 | 137 | lightgbm | site 6 | 345117 | 28654 | 0.533294 | 0.941967 |
| representative | 10 | 137 | lightgbm | site 7 | 200594 | 15886 | 0.309128 | 0.889311 |
| representative | 10 | 137 | lightgbm | site 8 | 284376 | 31083 | 0.851704 | 0.921642 |
| representative | 10 | 137 | lightgbm | site 9 | 1367482 | 52587 | 0.699532 | 0.949002 |
| representative | 10 | 137 | lightgbm | site 10 | 206430 | 15814 | 0.387063 | 0.923955 |
| representative | 10 | 137 | lightgbm | site 11 | 43626 | 197 | 0.059934 | 0.913054 |
| representative | 10 | 137 | lightgbm | site 12 | 158011 | 755 | 0.774646 | 0.995239 |
| representative | 10 | 137 | lightgbm | site 13 | 1334223 | 59283 | 0.290218 | 0.884783 |
| representative | 10 | 137 | lightgbm | site 14 | 1256775 | 101532 | 0.588248 | 0.844296 |
| representative | 10 | 137 | lightgbm | site 15 | 909902 | 17989 | 0.659623 | 0.987402 |
| representative | 10 | 137 | tabpfn | site 0 | 538432 | 176269 | 0.996824 | 0.996436 |
| representative | 10 | 137 | tabpfn | site 1 | 289853 | 39135 | 0.873649 | 0.983759 |
| representative | 10 | 137 | tabpfn | site 2 | 1263915 | 80897 | 0.844897 | 0.983122 |
| representative | 10 | 137 | tabpfn | site 3 | 1181463 | 2684 | 0.799889 | 0.989868 |
| representative | 10 | 137 | tabpfn | site 4 | 370460 | 197 | 0.573982 | 0.889281 |
| representative | 10 | 137 | tabpfn | site 5 | 386496 | 14435 | 0.969446 | 0.996111 |
| representative | 10 | 137 | tabpfn | site 6 | 345117 | 28654 | 0.707594 | 0.975187 |
| representative | 10 | 137 | tabpfn | site 7 | 200594 | 15886 | 0.271864 | 0.891617 |
| representative | 10 | 137 | tabpfn | site 8 | 284376 | 31083 | 0.614011 | 0.918137 |
| representative | 10 | 137 | tabpfn | site 9 | 1367482 | 52587 | 0.863732 | 0.950203 |
| representative | 10 | 137 | tabpfn | site 10 | 206430 | 15814 | 0.381941 | 0.926987 |
| representative | 10 | 137 | tabpfn | site 11 | 43626 | 197 | 0.027282 | 0.916123 |
| representative | 10 | 137 | tabpfn | site 12 | 158011 | 755 | 0.914496 | 0.999414 |
| representative | 10 | 137 | tabpfn | site 13 | 1334223 | 59283 | 0.394035 | 0.932949 |
| representative | 10 | 137 | tabpfn | site 14 | 1256775 | 101532 | 0.807078 | 0.968429 |
| representative | 10 | 137 | tabpfn | site 15 | 909902 | 17989 | 0.840743 | 0.988223 |
| representative | 10 | 137 | xgboost | site 0 | 538432 | 176269 | 0.996787 | 0.996472 |
| representative | 10 | 137 | xgboost | site 1 | 289853 | 39135 | 0.550722 | 0.942450 |
| representative | 10 | 137 | xgboost | site 2 | 1263915 | 80897 | 0.776905 | 0.982682 |
| representative | 10 | 137 | xgboost | site 3 | 1181463 | 2684 | 0.437311 | 0.992115 |
| representative | 10 | 137 | xgboost | site 4 | 370460 | 197 | 0.732421 | 0.919836 |
| representative | 10 | 137 | xgboost | site 5 | 386496 | 14435 | 0.974028 | 0.997285 |
| representative | 10 | 137 | xgboost | site 6 | 345117 | 28654 | 0.603021 | 0.962787 |
| representative | 10 | 137 | xgboost | site 7 | 200594 | 15886 | 0.255502 | 0.887639 |
| representative | 10 | 137 | xgboost | site 8 | 284376 | 31083 | 0.675603 | 0.913052 |
| representative | 10 | 137 | xgboost | site 9 | 1367482 | 52587 | 0.834176 | 0.960082 |
| representative | 10 | 137 | xgboost | site 10 | 206430 | 15814 | 0.383032 | 0.926185 |
| representative | 10 | 137 | xgboost | site 11 | 43626 | 197 | 0.031248 | 0.922133 |
| representative | 10 | 137 | xgboost | site 12 | 158011 | 755 | 0.958232 | 0.999751 |
| representative | 10 | 137 | xgboost | site 13 | 1334223 | 59283 | 0.459468 | 0.939113 |
| representative | 10 | 137 | xgboost | site 14 | 1256775 | 101532 | 0.750715 | 0.950454 |
| representative | 10 | 137 | xgboost | site 15 | 909902 | 17989 | 0.773945 | 0.990321 |
| representative | 725 | 137 | catboost | site 0 | 538432 | 176269 | 0.999884 | 0.999936 |
| representative | 725 | 137 | catboost | site 1 | 289853 | 39135 | 0.970553 | 0.985625 |
| representative | 725 | 137 | catboost | site 2 | 1263915 | 80897 | 0.871372 | 0.988800 |
| representative | 725 | 137 | catboost | site 3 | 1181463 | 2684 | 0.867119 | 0.996834 |
| representative | 725 | 137 | catboost | site 4 | 370460 | 197 | 0.783801 | 0.954142 |
| representative | 725 | 137 | catboost | site 5 | 386496 | 14435 | 0.992843 | 0.999735 |
| representative | 725 | 137 | catboost | site 6 | 345117 | 28654 | 0.832326 | 0.984657 |
| representative | 725 | 137 | catboost | site 7 | 200594 | 15886 | 0.831199 | 0.975638 |
| representative | 725 | 137 | catboost | site 8 | 284376 | 31083 | 0.956497 | 0.985937 |
| representative | 725 | 137 | catboost | site 9 | 1367482 | 52587 | 0.978305 | 0.997424 |
| representative | 725 | 137 | catboost | site 10 | 206430 | 15814 | 0.558089 | 0.948111 |
| representative | 725 | 137 | catboost | site 11 | 43626 | 197 | 0.392721 | 0.973970 |
| representative | 725 | 137 | catboost | site 12 | 158011 | 755 | 0.995828 | 0.999964 |
| representative | 725 | 137 | catboost | site 13 | 1334223 | 59283 | 0.670544 | 0.949509 |
| representative | 725 | 137 | catboost | site 14 | 1256775 | 101532 | 0.873328 | 0.959872 |
| representative | 725 | 137 | catboost | site 15 | 909902 | 17989 | 0.841355 | 0.988273 |
| representative | 725 | 137 | ensemble | site 0 | 538432 | 176269 | 0.999901 | 0.999943 |
| representative | 725 | 137 | ensemble | site 1 | 289853 | 39135 | 0.986484 | 0.996610 |
| representative | 725 | 137 | ensemble | site 2 | 1263915 | 80897 | 0.890518 | 0.991087 |
| representative | 725 | 137 | ensemble | site 3 | 1181463 | 2684 | 0.875289 | 0.998579 |
| representative | 725 | 137 | ensemble | site 4 | 370460 | 197 | 0.780293 | 0.978702 |
| representative | 725 | 137 | ensemble | site 5 | 386496 | 14435 | 0.995431 | 0.999785 |
| representative | 725 | 137 | ensemble | site 6 | 345117 | 28654 | 0.890466 | 0.988778 |
| representative | 725 | 137 | ensemble | site 7 | 200594 | 15886 | 0.885694 | 0.982058 |
| representative | 725 | 137 | ensemble | site 8 | 284376 | 31083 | 0.970995 | 0.991698 |
| representative | 725 | 137 | ensemble | site 9 | 1367482 | 52587 | 0.982440 | 0.998129 |
| representative | 725 | 137 | ensemble | site 10 | 206430 | 15814 | 0.780635 | 0.970226 |
| representative | 725 | 137 | ensemble | site 11 | 43626 | 197 | 0.415379 | 0.975212 |
| representative | 725 | 137 | ensemble | site 12 | 158011 | 755 | 0.998337 | 0.999980 |
| representative | 725 | 137 | ensemble | site 13 | 1334223 | 59283 | 0.670012 | 0.962240 |
| representative | 725 | 137 | ensemble | site 14 | 1256775 | 101532 | 0.861050 | 0.980901 |
| representative | 725 | 137 | ensemble | site 15 | 909902 | 17989 | 0.867600 | 0.995648 |
| representative | 725 | 137 | hist_gradient_boosting | site 0 | 538432 | 176269 | 0.999884 | 0.999932 |
| representative | 725 | 137 | hist_gradient_boosting | site 1 | 289853 | 39135 | 0.984311 | 0.996126 |
| representative | 725 | 137 | hist_gradient_boosting | site 2 | 1263915 | 80897 | 0.884524 | 0.991255 |
| representative | 725 | 137 | hist_gradient_boosting | site 3 | 1181463 | 2684 | 0.848346 | 0.998396 |
| representative | 725 | 137 | hist_gradient_boosting | site 4 | 370460 | 197 | 0.762193 | 0.965644 |
| representative | 725 | 137 | hist_gradient_boosting | site 5 | 386496 | 14435 | 0.994966 | 0.999734 |
| representative | 725 | 137 | hist_gradient_boosting | site 6 | 345117 | 28654 | 0.893840 | 0.990072 |
| representative | 725 | 137 | hist_gradient_boosting | site 7 | 200594 | 15886 | 0.863789 | 0.979660 |
| representative | 725 | 137 | hist_gradient_boosting | site 8 | 284376 | 31083 | 0.972231 | 0.992475 |
| representative | 725 | 137 | hist_gradient_boosting | site 9 | 1367482 | 52587 | 0.976278 | 0.997322 |
| representative | 725 | 137 | hist_gradient_boosting | site 10 | 206430 | 15814 | 0.813401 | 0.971899 |
| representative | 725 | 137 | hist_gradient_boosting | site 11 | 43626 | 197 | 0.376184 | 0.971462 |
| representative | 725 | 137 | hist_gradient_boosting | site 12 | 158011 | 755 | 0.983013 | 0.999956 |
| representative | 725 | 137 | hist_gradient_boosting | site 13 | 1334223 | 59283 | 0.652265 | 0.961677 |
| representative | 725 | 137 | hist_gradient_boosting | site 14 | 1256775 | 101532 | 0.834028 | 0.980979 |
| representative | 725 | 137 | hist_gradient_boosting | site 15 | 909902 | 17989 | 0.856329 | 0.995412 |
| representative | 725 | 137 | lightgbm | site 0 | 538432 | 176269 | 0.999845 | 0.999906 |
| representative | 725 | 137 | lightgbm | site 1 | 289853 | 39135 | 0.978554 | 0.994862 |
| representative | 725 | 137 | lightgbm | site 2 | 1263915 | 80897 | 0.882954 | 0.990452 |
| representative | 725 | 137 | lightgbm | site 3 | 1181463 | 2684 | 0.866642 | 0.998229 |
| representative | 725 | 137 | lightgbm | site 4 | 370460 | 197 | 0.761182 | 0.965029 |
| representative | 725 | 137 | lightgbm | site 5 | 386496 | 14435 | 0.993698 | 0.999617 |
| representative | 725 | 137 | lightgbm | site 6 | 345117 | 28654 | 0.854350 | 0.986027 |
| representative | 725 | 137 | lightgbm | site 7 | 200594 | 15886 | 0.873670 | 0.978802 |
| representative | 725 | 137 | lightgbm | site 8 | 284376 | 31083 | 0.973343 | 0.992882 |
| representative | 725 | 137 | lightgbm | site 9 | 1367482 | 52587 | 0.971180 | 0.996659 |
| representative | 725 | 137 | lightgbm | site 10 | 206430 | 15814 | 0.786393 | 0.967864 |
| representative | 725 | 137 | lightgbm | site 11 | 43626 | 197 | 0.371944 | 0.976554 |
| representative | 725 | 137 | lightgbm | site 12 | 158011 | 755 | 0.997866 | 0.999937 |
| representative | 725 | 137 | lightgbm | site 13 | 1334223 | 59283 | 0.665710 | 0.960462 |
| representative | 725 | 137 | lightgbm | site 14 | 1256775 | 101532 | 0.848375 | 0.980614 |
| representative | 725 | 137 | lightgbm | site 15 | 909902 | 17989 | 0.858975 | 0.995646 |
| representative | 725 | 137 | xgboost | site 0 | 538432 | 176269 | 0.999839 | 0.999912 |
| representative | 725 | 137 | xgboost | site 1 | 289853 | 39135 | 0.987611 | 0.996590 |
| representative | 725 | 137 | xgboost | site 2 | 1263915 | 80897 | 0.859945 | 0.988575 |
| representative | 725 | 137 | xgboost | site 3 | 1181463 | 2684 | 0.852142 | 0.998572 |
| representative | 725 | 137 | xgboost | site 4 | 370460 | 197 | 0.772176 | 0.981299 |
| representative | 725 | 137 | xgboost | site 5 | 386496 | 14435 | 0.987530 | 0.999457 |
| representative | 725 | 137 | xgboost | site 6 | 345117 | 28654 | 0.851099 | 0.986455 |
| representative | 725 | 137 | xgboost | site 7 | 200594 | 15886 | 0.870589 | 0.980861 |
| representative | 725 | 137 | xgboost | site 8 | 284376 | 31083 | 0.946581 | 0.988220 |
| representative | 725 | 137 | xgboost | site 9 | 1367482 | 52587 | 0.977687 | 0.997105 |
| representative | 725 | 137 | xgboost | site 10 | 206430 | 15814 | 0.771120 | 0.965251 |
| representative | 725 | 137 | xgboost | site 11 | 43626 | 197 | 0.349826 | 0.976611 |
| representative | 725 | 137 | xgboost | site 12 | 158011 | 755 | 0.998960 | 0.999992 |
| representative | 725 | 137 | xgboost | site 13 | 1334223 | 59283 | 0.657088 | 0.959911 |
| representative | 725 | 137 | xgboost | site 14 | 1256775 | 101532 | 0.853701 | 0.972699 |
| representative | 725 | 137 | xgboost | site 15 | 909902 | 17989 | 0.885335 | 0.995095 |

## Tree early-stopping audit

| K | features | model | selection_metric | best_iteration | ceiling | stop_reason | ES_PR_AUC | ES_ROC_AUC |
|---|---|---|---|---|---|---|---|---|
| 725 | 137 | lightgbm | roc_auc | 61 | 5000 | early_stopping | 0.891059 | 0.989420 |
| 725 | 137 | xgboost | roc_auc | 30 | 5000 | early_stopping | 0.878814 | 0.988965 |
| 725 | 137 | catboost | roc_auc | 160 | 5000 | early_stopping | 0.883956 | 0.987368 |
| 725 | 137 | hist_gradient_boosting | roc_auc | 78 | 1000 | early_stopping | 0.885735 | 0.989251 |
| 725 | 17 | lightgbm | roc_auc | 62 | 5000 | early_stopping | 0.773070 | 0.968455 |
| 725 | 17 | xgboost | roc_auc | 9 | 5000 | early_stopping | 0.755698 | 0.957745 |
| 725 | 17 | catboost | roc_auc | 23 | 5000 | early_stopping | 0.780457 | 0.958921 |
| 725 | 17 | hist_gradient_boosting | roc_auc | 174 | 1000 | early_stopping | 0.756821 | 0.963975 |
| 10 | 137 | lightgbm | roc_auc | 8 | 5000 | early_stopping | 0.999573 | 0.999931 |
| 10 | 137 | xgboost | roc_auc | 28 | 5000 | early_stopping | 0.991589 | 0.998952 |
| 10 | 137 | catboost | roc_auc | 4 | 5000 | early_stopping | 1.000000 | 1.000000 |
| 10 | 137 | hist_gradient_boosting | roc_auc | 24 | 1000 | early_stopping | 1.000000 | 1.000000 |

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
