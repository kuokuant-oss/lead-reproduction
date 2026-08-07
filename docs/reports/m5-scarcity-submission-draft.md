# M5 labeled-data scarcity experiments (submission draft)

This package contains two distinct experiments. They share the odd-building holdout, 137-feature representation, models, and reporting metric, but they implement different scarcity interventions. Each experiment is therefore presented completely before any cross-experiment synthesis.

## Experiment A — matched-context row scarcity

### Experiment A objective

Measure sensitivity to the number of labeled context rows, N, while source-building coverage remains nearly complete. Training contexts are nested with a 1:1 class ratio; evaluation uses the natural holdout distribution.

### Experiment A protocol

![Experiment A protocol](tables/m5-scarcity/m5_exp_a_matched_context_setup.png)

### Experiment A aggregate performance

![Experiment A macro PR-AUC](assets/m5-scarcity/m5_exp_a_matched_context_macro_pr_auc.png)

![Experiment A numeric results](tables/m5-scarcity/m5_exp_a_matched_context_macro_pr_auc.png)

### Experiment A meter-level performance

![Experiment A meter trend panels](assets/m5-scarcity/m5_exp_a_matched_context_all_meters_pr_auc.png)

![Experiment A grouped meter comparison](assets/m5-scarcity/m5_exp_a_matched_context_meter_grouped_pr_auc.png)

![Experiment A budget-color meter blocks](assets/m5-scarcity/m5_exp_a_matched_context_meter_budget_blocks_pr_auc.png)

![Experiment A meter detail table](tables/m5-scarcity/m5_exp_a_matched_context_meter_pr_auc.png)

## Experiment B — representative source-building scarcity

### Experiment B objective

Measure sensitivity to the number of distinct labeled source buildings, K. The ladder is nested and representative; total context size is approximately 500 rows per building and the selected rows retain their natural class mix.

### Experiment B protocol

![Experiment B protocol](tables/m5-scarcity/m5_exp_b_building_count_setup.png)

### Experiment B aggregate performance

![Experiment B macro PR-AUC](assets/m5-scarcity/m5_exp_b_building_count_macro_pr_auc.png)

![Experiment B numeric results](tables/m5-scarcity/m5_exp_b_building_count_macro_pr_auc.png)

### Experiment B meter-level performance

![Experiment B meter trend panels](assets/m5-scarcity/m5_exp_b_building_count_all_meters_pr_auc.png)

![Experiment B grouped meter comparison](assets/m5-scarcity/m5_exp_b_building_count_meter_grouped_pr_auc.png)

![Experiment B budget-color meter blocks](assets/m5-scarcity/m5_exp_b_building_count_meter_budget_blocks_pr_auc.png)

![Experiment B meter detail table](tables/m5-scarcity/m5_exp_b_building_count_meter_pr_auc.png)

Paired result figures contain K=10, K=20, K=50, and K=100.

## Cross-experiment synthesis

These experiments answer complementary questions; their cells are not matched treatments. Experiment A varies row count under near-full building coverage and forced 1:1 training balance. Experiment B varies building diversity while row count grows with K and class balance remains natural. The following figures compare curve shape and coverage only; they must not be interpreted as a point-to-point contest between N and K budgets.

![Cross-experiment training coverage](assets/m5-scarcity/m5_cross_experiment_design_source_buildings.png)

![Cross-experiment macro curves](assets/m5-scarcity/m5_cross_experiment_macro_pr_auc.png)

## Reporting conventions

- Primary measure: equal-weight macro PR-AUC across electricity, chilled water, steam, and hot water.
- Meter-level figures always retain all four meter types.
- Seed 42 represents one sampled draw.
- Protocol and numeric tables are provided as compilable LaTeX source and LaTeX-rendered PDF/PNG.
