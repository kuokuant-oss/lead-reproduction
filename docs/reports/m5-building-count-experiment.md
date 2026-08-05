# M5 building-count experiment

This report is atomically regenerated at every overnight publication gate. Training sources are restricted to building_id % 2 == 0; the complete canonical holdout contains only odd building IDs.

K=10/20/50/100 uses only 137 features. Average allocation is at most 500 rows per building, so total context is bounded by 5K/10K/25K/50K. Allocation within each incremental K block is proportional to each building's available rows, so individual buildings may contribute above or below 500. Rows are selected by a seed-42 stable hash of raw identity without consulting labels. Building and row sets are strict nested prefixes. Trees use building-disjoint 80/20 fit and early-stop roles to choose iteration counts by PR-AUC, then final-refit using the exact M3 post-sort [negs1,pos,negs2,pos] sampling and float64 scaler path. TabPFN uses the same selected rows and has no task-specific epoch or weight-update loop, so early stopping is not applicable.

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
| building experiment | 725 | 17 | catboost | 10137155 | 637397 | 0.800766 | 0.957013 |
| building experiment | 725 | 17 | ensemble | 10137155 | 637397 | 0.815023 | 0.961747 |
| building experiment | 725 | 17 | hist_gradient_boosting | 10137155 | 637397 | 0.814399 | 0.960217 |
| building experiment | 725 | 17 | lightgbm | 10137155 | 637397 | 0.816312 | 0.962942 |
| building experiment | 725 | 17 | xgboost | 10137155 | 637397 | 0.780933 | 0.949001 |
| building experiment | 725 | 137 | catboost | 10137155 | 637397 | 0.918451 | 0.985740 |
| building experiment | 725 | 137 | ensemble | 10137155 | 637397 | 0.927335 | 0.991200 |
| building experiment | 725 | 137 | hist_gradient_boosting | 10137155 | 637397 | 0.921792 | 0.990755 |
| building experiment | 725 | 137 | lightgbm | 10137155 | 637397 | 0.923340 | 0.991000 |
| building experiment | 725 | 137 | xgboost | 10137155 | 637397 | 0.920309 | 0.988294 |

## Meter breakdown

| sampling_profile | building_budget | features | model | group_label | rows | anomalies | pr_auc | roc_auc |
|---|---|---|---|---|---|---|---|---|
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | electricity | 6035071 | 356679 | 0.985040 | 0.997838 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | chilledwater | 2115354 | 141139 | 0.829599 | 0.984698 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | steam | 1350609 | 48888 | 0.822684 | 0.985801 |
| matched_context_rows | 0 | 137 | tabpfn_matched_50k | hotwater | 636121 | 90691 | 0.824857 | 0.953947 |
| representative | 725 | 17 | catboost | electricity | 6035071 | 356679 | 0.915966 | 0.967785 |
| representative | 725 | 17 | catboost | chilledwater | 2115354 | 141139 | 0.508261 | 0.931266 |
| representative | 725 | 17 | catboost | steam | 1350609 | 48888 | 0.378768 | 0.908396 |
| representative | 725 | 17 | catboost | hotwater | 636121 | 90691 | 0.547746 | 0.922412 |
| representative | 725 | 17 | ensemble | electricity | 6035071 | 356679 | 0.925531 | 0.973047 |
| representative | 725 | 17 | ensemble | chilledwater | 2115354 | 141139 | 0.549716 | 0.932887 |
| representative | 725 | 17 | ensemble | steam | 1350609 | 48888 | 0.413798 | 0.921489 |
| representative | 725 | 17 | ensemble | hotwater | 636121 | 90691 | 0.584229 | 0.930988 |
| representative | 725 | 17 | hist_gradient_boosting | electricity | 6035071 | 356679 | 0.917910 | 0.971446 |
| representative | 725 | 17 | hist_gradient_boosting | chilledwater | 2115354 | 141139 | 0.562994 | 0.935345 |
| representative | 725 | 17 | hist_gradient_boosting | steam | 1350609 | 48888 | 0.419891 | 0.905906 |
| representative | 725 | 17 | hist_gradient_boosting | hotwater | 636121 | 90691 | 0.536088 | 0.922845 |
| representative | 725 | 17 | lightgbm | electricity | 6035071 | 356679 | 0.928092 | 0.975147 |
| representative | 725 | 17 | lightgbm | chilledwater | 2115354 | 141139 | 0.521970 | 0.930968 |
| representative | 725 | 17 | lightgbm | steam | 1350609 | 48888 | 0.419289 | 0.916004 |
| representative | 725 | 17 | lightgbm | hotwater | 636121 | 90691 | 0.579264 | 0.928990 |
| representative | 725 | 17 | xgboost | electricity | 6035071 | 356679 | 0.903228 | 0.959480 |
| representative | 725 | 17 | xgboost | chilledwater | 2115354 | 141139 | 0.434596 | 0.919861 |
| representative | 725 | 17 | xgboost | steam | 1350609 | 48888 | 0.317852 | 0.905686 |
| representative | 725 | 17 | xgboost | hotwater | 636121 | 90691 | 0.584676 | 0.913328 |
| representative | 725 | 137 | catboost | electricity | 6035071 | 356679 | 0.991444 | 0.998996 |
| representative | 725 | 137 | catboost | chilledwater | 2115354 | 141139 | 0.689245 | 0.975035 |
| representative | 725 | 137 | catboost | steam | 1350609 | 48888 | 0.736154 | 0.956130 |
| representative | 725 | 137 | catboost | hotwater | 636121 | 90691 | 0.774912 | 0.932285 |
| representative | 725 | 137 | ensemble | electricity | 6035071 | 356679 | 0.992678 | 0.999187 |
| representative | 725 | 137 | ensemble | chilledwater | 2115354 | 141139 | 0.771197 | 0.979818 |
| representative | 725 | 137 | ensemble | steam | 1350609 | 48888 | 0.725062 | 0.957077 |
| representative | 725 | 137 | ensemble | hotwater | 636121 | 90691 | 0.813012 | 0.948188 |
| representative | 725 | 137 | hist_gradient_boosting | electricity | 6035071 | 356679 | 0.990511 | 0.998809 |
| representative | 725 | 137 | hist_gradient_boosting | chilledwater | 2115354 | 141139 | 0.752583 | 0.978346 |
| representative | 725 | 137 | hist_gradient_boosting | steam | 1350609 | 48888 | 0.718162 | 0.951175 |
| representative | 725 | 137 | hist_gradient_boosting | hotwater | 636121 | 90691 | 0.811779 | 0.951402 |
| representative | 725 | 137 | lightgbm | electricity | 6035071 | 356679 | 0.991436 | 0.998931 |
| representative | 725 | 137 | lightgbm | chilledwater | 2115354 | 141139 | 0.762531 | 0.979522 |
| representative | 725 | 137 | lightgbm | steam | 1350609 | 48888 | 0.690482 | 0.957589 |
| representative | 725 | 137 | lightgbm | hotwater | 636121 | 90691 | 0.805876 | 0.946097 |
| representative | 725 | 137 | xgboost | electricity | 6035071 | 356679 | 0.989001 | 0.998383 |
| representative | 725 | 137 | xgboost | chilledwater | 2115354 | 141139 | 0.778501 | 0.980029 |
| representative | 725 | 137 | xgboost | steam | 1350609 | 48888 | 0.638042 | 0.948992 |
| representative | 725 | 137 | xgboost | hotwater | 636121 | 90691 | 0.760008 | 0.934574 |

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
| representative | 725 | 17 | catboost | site 0 | 538432 | 176269 | 0.997098 | 0.997132 |
| representative | 725 | 17 | catboost | site 1 | 289853 | 39135 | 0.451671 | 0.787918 |
| representative | 725 | 17 | catboost | site 2 | 1263915 | 80897 | 0.622543 | 0.953973 |
| representative | 725 | 17 | catboost | site 3 | 1181463 | 2684 | 0.805091 | 0.934379 |
| representative | 725 | 17 | catboost | site 4 | 370460 | 197 | 0.487887 | 0.864523 |
| representative | 725 | 17 | catboost | site 5 | 386496 | 14435 | 0.949730 | 0.991567 |
| representative | 725 | 17 | catboost | site 6 | 345117 | 28654 | 0.654866 | 0.935555 |
| representative | 725 | 17 | catboost | site 7 | 200594 | 15886 | 0.836169 | 0.960682 |
| representative | 725 | 17 | catboost | site 8 | 284376 | 31083 | 0.860608 | 0.947843 |
| representative | 725 | 17 | catboost | site 9 | 1367482 | 52587 | 0.876861 | 0.957521 |
| representative | 725 | 17 | catboost | site 10 | 206430 | 15814 | 0.634434 | 0.945455 |
| representative | 725 | 17 | catboost | site 11 | 43626 | 197 | 0.036926 | 0.902408 |
| representative | 725 | 17 | catboost | site 12 | 158011 | 755 | 0.974176 | 0.988632 |
| representative | 725 | 17 | catboost | site 13 | 1334223 | 59283 | 0.564508 | 0.905889 |
| representative | 725 | 17 | catboost | site 14 | 1256775 | 101532 | 0.639187 | 0.900331 |
| representative | 725 | 17 | catboost | site 15 | 909902 | 17989 | 0.808339 | 0.988770 |
| representative | 725 | 17 | ensemble | site 0 | 538432 | 176269 | 0.997290 | 0.997132 |
| representative | 725 | 17 | ensemble | site 1 | 289853 | 39135 | 0.457227 | 0.803435 |
| representative | 725 | 17 | ensemble | site 2 | 1263915 | 80897 | 0.628373 | 0.951814 |
| representative | 725 | 17 | ensemble | site 3 | 1181463 | 2684 | 0.808989 | 0.942122 |
| representative | 725 | 17 | ensemble | site 4 | 370460 | 197 | 0.726345 | 0.830866 |
| representative | 725 | 17 | ensemble | site 5 | 386496 | 14435 | 0.955154 | 0.991838 |
| representative | 725 | 17 | ensemble | site 6 | 345117 | 28654 | 0.602421 | 0.932108 |
| representative | 725 | 17 | ensemble | site 7 | 200594 | 15886 | 0.866056 | 0.958821 |
| representative | 725 | 17 | ensemble | site 8 | 284376 | 31083 | 0.861270 | 0.945850 |
| representative | 725 | 17 | ensemble | site 9 | 1367482 | 52587 | 0.894585 | 0.980563 |
| representative | 725 | 17 | ensemble | site 10 | 206430 | 15814 | 0.645004 | 0.951846 |
| representative | 725 | 17 | ensemble | site 11 | 43626 | 197 | 0.041406 | 0.927007 |
| representative | 725 | 17 | ensemble | site 12 | 158011 | 755 | 0.974212 | 0.988254 |
| representative | 725 | 17 | ensemble | site 13 | 1334223 | 59283 | 0.571104 | 0.906131 |
| representative | 725 | 17 | ensemble | site 14 | 1256775 | 101532 | 0.674621 | 0.909055 |
| representative | 725 | 17 | ensemble | site 15 | 909902 | 17989 | 0.823891 | 0.986638 |
| representative | 725 | 17 | hist_gradient_boosting | site 0 | 538432 | 176269 | 0.997723 | 0.997763 |
| representative | 725 | 17 | hist_gradient_boosting | site 1 | 289853 | 39135 | 0.363122 | 0.772295 |
| representative | 725 | 17 | hist_gradient_boosting | site 2 | 1263915 | 80897 | 0.599035 | 0.951671 |
| representative | 725 | 17 | hist_gradient_boosting | site 3 | 1181463 | 2684 | 0.815970 | 0.952374 |
| representative | 725 | 17 | hist_gradient_boosting | site 4 | 370460 | 197 | 0.467030 | 0.865992 |
| representative | 725 | 17 | hist_gradient_boosting | site 5 | 386496 | 14435 | 0.948485 | 0.989479 |
| representative | 725 | 17 | hist_gradient_boosting | site 6 | 345117 | 28654 | 0.580433 | 0.928249 |
| representative | 725 | 17 | hist_gradient_boosting | site 7 | 200594 | 15886 | 0.853864 | 0.955055 |
| representative | 725 | 17 | hist_gradient_boosting | site 8 | 284376 | 31083 | 0.861436 | 0.936445 |
| representative | 725 | 17 | hist_gradient_boosting | site 9 | 1367482 | 52587 | 0.900814 | 0.983903 |
| representative | 725 | 17 | hist_gradient_boosting | site 10 | 206430 | 15814 | 0.599839 | 0.951289 |
| representative | 725 | 17 | hist_gradient_boosting | site 11 | 43626 | 197 | 0.105769 | 0.963063 |
| representative | 725 | 17 | hist_gradient_boosting | site 12 | 158011 | 755 | 0.974781 | 0.989311 |
| representative | 725 | 17 | hist_gradient_boosting | site 13 | 1334223 | 59283 | 0.570987 | 0.885572 |
| representative | 725 | 17 | hist_gradient_boosting | site 14 | 1256775 | 101532 | 0.687942 | 0.917381 |
| representative | 725 | 17 | hist_gradient_boosting | site 15 | 909902 | 17989 | 0.854220 | 0.986811 |
| representative | 725 | 17 | lightgbm | site 0 | 538432 | 176269 | 0.997143 | 0.997085 |
| representative | 725 | 17 | lightgbm | site 1 | 289853 | 39135 | 0.463401 | 0.795001 |
| representative | 725 | 17 | lightgbm | site 2 | 1263915 | 80897 | 0.625040 | 0.948884 |
| representative | 725 | 17 | lightgbm | site 3 | 1181463 | 2684 | 0.809692 | 0.939400 |
| representative | 725 | 17 | lightgbm | site 4 | 370460 | 197 | 0.247781 | 0.855785 |
| representative | 725 | 17 | lightgbm | site 5 | 386496 | 14435 | 0.960334 | 0.990732 |
| representative | 725 | 17 | lightgbm | site 6 | 345117 | 28654 | 0.595749 | 0.928357 |
| representative | 725 | 17 | lightgbm | site 7 | 200594 | 15886 | 0.852375 | 0.953179 |
| representative | 725 | 17 | lightgbm | site 8 | 284376 | 31083 | 0.864631 | 0.943460 |
| representative | 725 | 17 | lightgbm | site 9 | 1367482 | 52587 | 0.886573 | 0.971718 |
| representative | 725 | 17 | lightgbm | site 10 | 206430 | 15814 | 0.645995 | 0.953163 |
| representative | 725 | 17 | lightgbm | site 11 | 43626 | 197 | 0.048589 | 0.939148 |
| representative | 725 | 17 | lightgbm | site 12 | 158011 | 755 | 0.974709 | 0.987349 |
| representative | 725 | 17 | lightgbm | site 13 | 1334223 | 59283 | 0.559338 | 0.900940 |
| representative | 725 | 17 | lightgbm | site 14 | 1256775 | 101532 | 0.685311 | 0.914975 |
| representative | 725 | 17 | lightgbm | site 15 | 909902 | 17989 | 0.837502 | 0.980079 |
| representative | 725 | 17 | xgboost | site 0 | 538432 | 176269 | 0.996035 | 0.995587 |
| representative | 725 | 17 | xgboost | site 1 | 289853 | 39135 | 0.321611 | 0.711421 |
| representative | 725 | 17 | xgboost | site 2 | 1263915 | 80897 | 0.624940 | 0.946177 |
| representative | 725 | 17 | xgboost | site 3 | 1181463 | 2684 | 0.793920 | 0.907399 |
| representative | 725 | 17 | xgboost | site 4 | 370460 | 197 | 0.100219 | 0.746138 |
| representative | 725 | 17 | xgboost | site 5 | 386496 | 14435 | 0.783283 | 0.981785 |
| representative | 725 | 17 | xgboost | site 6 | 345117 | 28654 | 0.623528 | 0.935592 |
| representative | 725 | 17 | xgboost | site 7 | 200594 | 15886 | 0.849246 | 0.953158 |
| representative | 725 | 17 | xgboost | site 8 | 284376 | 31083 | 0.830140 | 0.911257 |
| representative | 725 | 17 | xgboost | site 9 | 1367482 | 52587 | 0.856979 | 0.952513 |
| representative | 725 | 17 | xgboost | site 10 | 206430 | 15814 | 0.587967 | 0.935440 |
| representative | 725 | 17 | xgboost | site 11 | 43626 | 197 | 0.033692 | 0.924600 |
| representative | 725 | 17 | xgboost | site 12 | 158011 | 755 | 0.973966 | 0.988489 |
| representative | 725 | 17 | xgboost | site 13 | 1334223 | 59283 | 0.530303 | 0.905637 |
| representative | 725 | 17 | xgboost | site 14 | 1256775 | 101532 | 0.636074 | 0.884273 |
| representative | 725 | 17 | xgboost | site 15 | 909902 | 17989 | 0.784314 | 0.976651 |
| representative | 725 | 137 | catboost | site 0 | 538432 | 176269 | 0.999884 | 0.999936 |
| representative | 725 | 137 | catboost | site 1 | 289853 | 39135 | 0.968891 | 0.985326 |
| representative | 725 | 137 | catboost | site 2 | 1263915 | 80897 | 0.870669 | 0.988744 |
| representative | 725 | 137 | catboost | site 3 | 1181463 | 2684 | 0.866495 | 0.996603 |
| representative | 725 | 137 | catboost | site 4 | 370460 | 197 | 0.786702 | 0.957203 |
| representative | 725 | 137 | catboost | site 5 | 386496 | 14435 | 0.993040 | 0.999744 |
| representative | 725 | 137 | catboost | site 6 | 345117 | 28654 | 0.835615 | 0.984983 |
| representative | 725 | 137 | catboost | site 7 | 200594 | 15886 | 0.832472 | 0.975917 |
| representative | 725 | 137 | catboost | site 8 | 284376 | 31083 | 0.956853 | 0.986258 |
| representative | 725 | 137 | catboost | site 9 | 1367482 | 52587 | 0.978685 | 0.997465 |
| representative | 725 | 137 | catboost | site 10 | 206430 | 15814 | 0.563248 | 0.948844 |
| representative | 725 | 137 | catboost | site 11 | 43626 | 197 | 0.374109 | 0.970879 |
| representative | 725 | 137 | catboost | site 12 | 158011 | 755 | 0.995967 | 0.999967 |
| representative | 725 | 137 | catboost | site 13 | 1334223 | 59283 | 0.671083 | 0.949089 |
| representative | 725 | 137 | catboost | site 14 | 1256775 | 101532 | 0.873657 | 0.959862 |
| representative | 725 | 137 | catboost | site 15 | 909902 | 17989 | 0.841694 | 0.988192 |
| representative | 725 | 137 | ensemble | site 0 | 538432 | 176269 | 0.999885 | 0.999930 |
| representative | 725 | 137 | ensemble | site 1 | 289853 | 39135 | 0.986015 | 0.996654 |
| representative | 725 | 137 | ensemble | site 2 | 1263915 | 80897 | 0.894939 | 0.991451 |
| representative | 725 | 137 | ensemble | site 3 | 1181463 | 2684 | 0.872636 | 0.998484 |
| representative | 725 | 137 | ensemble | site 4 | 370460 | 197 | 0.779139 | 0.966586 |
| representative | 725 | 137 | ensemble | site 5 | 386496 | 14435 | 0.995334 | 0.999772 |
| representative | 725 | 137 | ensemble | site 6 | 345117 | 28654 | 0.893086 | 0.988880 |
| representative | 725 | 137 | ensemble | site 7 | 200594 | 15886 | 0.885028 | 0.980856 |
| representative | 725 | 137 | ensemble | site 8 | 284376 | 31083 | 0.970804 | 0.990848 |
| representative | 725 | 137 | ensemble | site 9 | 1367482 | 52587 | 0.980092 | 0.997857 |
| representative | 725 | 137 | ensemble | site 10 | 206430 | 15814 | 0.794756 | 0.971271 |
| representative | 725 | 137 | ensemble | site 11 | 43626 | 197 | 0.431054 | 0.975396 |
| representative | 725 | 137 | ensemble | site 12 | 158011 | 755 | 0.998276 | 0.999979 |
| representative | 725 | 137 | ensemble | site 13 | 1334223 | 59283 | 0.669905 | 0.957579 |
| representative | 725 | 137 | ensemble | site 14 | 1256775 | 101532 | 0.853918 | 0.980317 |
| representative | 725 | 137 | ensemble | site 15 | 909902 | 17989 | 0.864592 | 0.995590 |
| representative | 725 | 137 | hist_gradient_boosting | site 0 | 538432 | 176269 | 0.999856 | 0.999913 |
| representative | 725 | 137 | hist_gradient_boosting | site 1 | 289853 | 39135 | 0.982017 | 0.995689 |
| representative | 725 | 137 | hist_gradient_boosting | site 2 | 1263915 | 80897 | 0.898785 | 0.991824 |
| representative | 725 | 137 | hist_gradient_boosting | site 3 | 1181463 | 2684 | 0.859025 | 0.998196 |
| representative | 725 | 137 | hist_gradient_boosting | site 4 | 370460 | 197 | 0.754942 | 0.957196 |
| representative | 725 | 137 | hist_gradient_boosting | site 5 | 386496 | 14435 | 0.993892 | 0.999627 |
| representative | 725 | 137 | hist_gradient_boosting | site 6 | 345117 | 28654 | 0.906007 | 0.990754 |
| representative | 725 | 137 | hist_gradient_boosting | site 7 | 200594 | 15886 | 0.865864 | 0.980075 |
| representative | 725 | 137 | hist_gradient_boosting | site 8 | 284376 | 31083 | 0.971331 | 0.991824 |
| representative | 725 | 137 | hist_gradient_boosting | site 9 | 1367482 | 52587 | 0.972526 | 0.996542 |
| representative | 725 | 137 | hist_gradient_boosting | site 10 | 206430 | 15814 | 0.813378 | 0.970839 |
| representative | 725 | 137 | hist_gradient_boosting | site 11 | 43626 | 197 | 0.350965 | 0.972878 |
| representative | 725 | 137 | hist_gradient_boosting | site 12 | 158011 | 755 | 0.997871 | 0.999967 |
| representative | 725 | 137 | hist_gradient_boosting | site 13 | 1334223 | 59283 | 0.657598 | 0.954029 |
| representative | 725 | 137 | hist_gradient_boosting | site 14 | 1256775 | 101532 | 0.828830 | 0.980606 |
| representative | 725 | 137 | hist_gradient_boosting | site 15 | 909902 | 17989 | 0.845180 | 0.995166 |
| representative | 725 | 137 | lightgbm | site 0 | 538432 | 176269 | 0.999846 | 0.999907 |
| representative | 725 | 137 | lightgbm | site 1 | 289853 | 39135 | 0.978633 | 0.994865 |
| representative | 725 | 137 | lightgbm | site 2 | 1263915 | 80897 | 0.885893 | 0.990555 |
| representative | 725 | 137 | lightgbm | site 3 | 1181463 | 2684 | 0.865889 | 0.998226 |
| representative | 725 | 137 | lightgbm | site 4 | 370460 | 197 | 0.759451 | 0.954159 |
| representative | 725 | 137 | lightgbm | site 5 | 386496 | 14435 | 0.993721 | 0.999618 |
| representative | 725 | 137 | lightgbm | site 6 | 345117 | 28654 | 0.858796 | 0.986366 |
| representative | 725 | 137 | lightgbm | site 7 | 200594 | 15886 | 0.873083 | 0.978627 |
| representative | 725 | 137 | lightgbm | site 8 | 284376 | 31083 | 0.973338 | 0.992854 |
| representative | 725 | 137 | lightgbm | site 9 | 1367482 | 52587 | 0.971249 | 0.996649 |
| representative | 725 | 137 | lightgbm | site 10 | 206430 | 15814 | 0.790922 | 0.968447 |
| representative | 725 | 137 | lightgbm | site 11 | 43626 | 197 | 0.430129 | 0.977224 |
| representative | 725 | 137 | lightgbm | site 12 | 158011 | 755 | 0.997830 | 0.999941 |
| representative | 725 | 137 | lightgbm | site 13 | 1334223 | 59283 | 0.663670 | 0.959114 |
| representative | 725 | 137 | lightgbm | site 14 | 1256775 | 101532 | 0.846782 | 0.980563 |
| representative | 725 | 137 | lightgbm | site 15 | 909902 | 17989 | 0.859422 | 0.995607 |
| representative | 725 | 137 | xgboost | site 0 | 538432 | 176269 | 0.999760 | 0.999870 |
| representative | 725 | 137 | xgboost | site 1 | 289853 | 39135 | 0.985986 | 0.996491 |
| representative | 725 | 137 | xgboost | site 2 | 1263915 | 80897 | 0.850658 | 0.988940 |
| representative | 725 | 137 | xgboost | site 3 | 1181463 | 2684 | 0.847890 | 0.998143 |
| representative | 725 | 137 | xgboost | site 4 | 370460 | 197 | 0.774181 | 0.963897 |
| representative | 725 | 137 | xgboost | site 5 | 386496 | 14435 | 0.989672 | 0.999487 |
| representative | 725 | 137 | xgboost | site 6 | 345117 | 28654 | 0.831395 | 0.984922 |
| representative | 725 | 137 | xgboost | site 7 | 200594 | 15886 | 0.881477 | 0.980013 |
| representative | 725 | 137 | xgboost | site 8 | 284376 | 31083 | 0.953590 | 0.985265 |
| representative | 725 | 137 | xgboost | site 9 | 1367482 | 52587 | 0.969083 | 0.995271 |
| representative | 725 | 137 | xgboost | site 10 | 206430 | 15814 | 0.801981 | 0.965791 |
| representative | 725 | 137 | xgboost | site 11 | 43626 | 197 | 0.295605 | 0.976262 |
| representative | 725 | 137 | xgboost | site 12 | 158011 | 755 | 0.998658 | 0.999988 |
| representative | 725 | 137 | xgboost | site 13 | 1334223 | 59283 | 0.668370 | 0.952054 |
| representative | 725 | 137 | xgboost | site 14 | 1256775 | 101532 | 0.824080 | 0.966800 |
| representative | 725 | 137 | xgboost | site 15 | 909902 | 17989 | 0.874822 | 0.995147 |

## Tree early-stopping audit

| K | features | model | selection_metric | best_iteration | ceiling | stop_reason | ES_PR_AUC | ES_ROC_AUC |
|---|---|---|---|---|---|---|---|---|
| 725 | 137 | lightgbm | pr_auc | 60 | 5000 | early_stopping | 0.891333 | 0.989403 |
| 725 | 137 | xgboost | pr_auc | 16 | 5000 | early_stopping | 0.885224 | 0.988411 |
| 725 | 137 | catboost | pr_auc | 167 | 5000 | early_stopping | 0.889853 | 0.987230 |
| 725 | 137 | hist_gradient_boosting | pr_auc | 58 | 1000 | early_stopping | 0.882755 | 0.988763 |
| 725 | 17 | lightgbm | pr_auc | 39 | 5000 | early_stopping | 0.777521 | 0.967343 |
| 725 | 17 | xgboost | pr_auc | 4 | 5000 | early_stopping | 0.757258 | 0.943534 |
| 725 | 17 | catboost | pr_auc | 12 | 5000 | early_stopping | 0.788360 | 0.953934 |
| 725 | 17 | hist_gradient_boosting | pr_auc | 62 | 1000 | early_stopping | 0.765727 | 0.962127 |

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
