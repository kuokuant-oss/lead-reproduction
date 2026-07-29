# M5 137-feature context construction: data interpretation and paper direction

Last reviewed: 2026-07-29

This note records the current interpretation of the M5 evidence with the 137-feature line as the primary object. It is intended to sit between the result reports and the paper plan: it states the empirical structure currently visible in the predictions, the role of the raw-reading plots, and the paper direction that the next analyses can test.

Related reports:

- [M5 matched-context breakdown](../reports/m5-matched-context-breakdown.md)
- [M5 context-construction screening](../reports/m5-context-construction-screening.md)
- [M5 context-construction paper plan](../plans/m5-context-construction-paper-plan.md)

Related visual scripts:

- Existing plots: `scripts/plot_m5_context_grid_by_meter.py`
- New diagnostic plots: `scripts/plot_m5_137_anomaly_diagnostics.py`

## 1. Current empirical state

The 137-feature line changes the meaning of context scaling. From 5k to 100k rows, TabPFN PR-AUC increases for every meter:

- electricity: `0.9743 -> 0.9864`
- chilledwater: `0.7623 -> 0.8251`
- steam: `0.6413 -> 0.8320`
- hotwater: `0.6725 -> 0.8171`

The tree ensemble also improves:

- electricity: `0.9837 -> 0.9929`
- chilledwater: `0.6533 -> 0.7765`
- steam: `0.6819 -> 0.7530`
- hotwater: `0.7747 -> 0.8231`

The largest TabPFN gains occur for steam and hotwater. This indicates that the 120 lagged difference/ratio features give the learner reusable temporal structure that raw baseline features do not provide. Extra context becomes useful once the representation exposes temporal deviation, periodicity, and local regime changes.

The improvement is heterogeneous. Site-level curves still contain declines, such as TabPFN Moose PR-AUC falling from `0.8458` to `0.7419` across the 137-feature context curve. The relevant object is therefore a representation-dependent context response surface, with meter, site, and anomaly morphology as interacting axes.

## 2. What the 137-feature context curve means

The 137-feature results support a two-level reading.

At the global level, larger support improves anomaly ordering. At the local level, context composition still changes the score geometry of particular domain regimes. High pooled AUC does not remove this effect; it can conceal it because electricity and chilledwater dominate the holdout and have near-ceiling scores.

The fixed `0.5` rule gives a second view of the same process. For example, in the 137-feature hotwater panels, PR-AUC improves while the number of anomalies below `0.5` can rise from 5k to 100k. The score ordering improves while the location of the score cloud relative to the fixed decision line moves. The missed/false-positive counts are therefore direct fixed-threshold operating results and also evidence that context changes the effective alert queue.

The paper should read these two outputs together:

- PR-AUC and pairwise AUC describe ordering quality;
- fixed-`0.5` counts describe the resulting alert queue;
- score-density and anomaly-only plots show where the movement occurs.

## 3. Composition effects remain in the 137-feature regime

At 20k rows, TabPFN F4 has pooled PR-AUC:

- pooled reference: `0.9884`
- meter balanced: `0.9891`
- hotwater-heavy: `0.9877`
- hotwater-excluded: `0.9748`

The pooled change hides a much larger per-meter response. Under hotwater exclusion:

- electricity: `0.9920 -> 0.9915`
- chilledwater: `0.9989 -> 0.9993`
- steam: `0.9556 -> 0.9227`
- hotwater: `0.9618 -> 0.7751`

For the same intervention, hotwater normal scores move by `+0.3295` on average while hotwater anomaly scores move by `-0.0162`. The F4 intervention therefore acts mainly through the normal side of the cross-domain score scale in this probe. It changes which normal rows serve as background anchors for other domain anomalies.

Trees F4 has a much smaller pooled hotwater-exclusion change (`0.9885 -> 0.9868`) and a different affected-cell pattern. The comparison is useful as a topology contrast: the two learners receive the same rows and features, yet context composition produces different domain-level movements.

## 4. What the current raw-reading plots reveal

The existing grid plots show the full holdout, with raw `meter_reading` on the x-axis, predicted score on the y-axis, dark normal points, light anomaly points, and the fixed `0.5` line. They reveal two important local structures in the 137-feature line.

### Hotwater near-zero anomalies

In the `0–1` raw-reading region, hotwater anomalies can remain below `0.5` for both TabPFN and Trees. The score distributions differ, while the missed region is shared. This is evidence for a shared regime-level blind spot in the current feature representation or support, with learner-specific score spread layered on top.

The 137 features compute value changes and ratios around the current reading. Near-zero readings can produce a regime where the same raw level occurs in both normal and anomalous rows, and the temporal features must carry the class information. The current plot does not establish whether the failure comes from ambiguous labels, overlapping temporal patterns, or ratio geometry. It does establish that aggregate hotwater improvement leaves a concentrated low-reading failure region.

### Steam high-reading anomalies

Steam rows around `100k–300k` can remain below `0.5`. When these are true anomaly points, they define a high-load anomaly blind region: raw magnitude is extreme, yet the current 137-feature score does not enter the alert queue. When the points are normal rows, the same area shows that high steam load is a normal operating regime. The anomaly-only and regime-rate diagnostics separate these two readings by showing the score distribution conditional on the label.

This observation matters because it separates magnitude from anomaly mechanism. The model uses raw level together with temporal differences, ratios, time, weather, and building attributes. A high reading carries weak anomaly evidence when the surrounding temporal regime is coherent; a low reading carries weak anomaly evidence when it resembles a common low-flow state.

The first diagnostic run quantifies the two regimes at 100k context:

- hotwater `0–1`: 86,220 anomalies; TabPFN misses 17,745, Trees misses 15,272, and both miss 12,186;
- steam `100k–300k`: 70 anomalies; TabPFN misses 42, Trees miss 69, and both miss 42.

The hotwater band is therefore a large shared-miss tail rather than a uniform failure of every near-zero anomaly. The steam band is a much smaller, sharper failure region: at 100k, TabPFN detects 28/70 and Trees detects 1/70. At smaller contexts, the steam detection rate is close to zero for both learners.

## 5. Current paper story

The strongest current paper direction is:

> **Temporal representation unlocks context scaling while context composition continues to control cross-domain alert ranking.**

The full claim has four linked parts:

1. Raw 17-feature support scaling can produce adverse or weak responses for TabPFN.
2. The 137-feature temporal representation turns context scaling into a broadly positive response, especially for steam and hotwater.
3. Fixed composition interventions still reassign the score roles of domain-specific anomalies and normals, even when pooled AUC is near the ceiling.
4. The reassignment topology differs between TabPFN and tree ensembles and leaves localized blind regions, including hotwater near-zero anomalies and possible steam high-load anomalies.

This frames context construction as part of the effective predictor. The context determines which examples become reusable temporal prototypes, which examples anchor the normal score scale, and which domain regimes receive sufficient support in the global ranking.

## 6. Evidence hierarchy for the paper

The main evidence should be organized around geometry rather than a model leaderboard:

1. Matched 5k–100k curves for 17 and 137 features establish the representation-dependent scaling reversal.
2. Pairwise meter and meter×site AUC decompositions show whether changes occur within a domain or across domains.
3. Fixed-`0.5` counts show the alert-queue consequence under the repository operating rule.
4. Anomaly-only raw-reading/regime plots localize shared and learner-specific blind regions.
5. Label-role and frozen-scaler interventions identify whether composition acts through anomaly exemplars, normal anchors, or preprocessing geometry.

The raw-reading scatter is a diagnostic projection. The 137-feature learner has 120 lagged change/ratio features, so a single x-axis cannot explain the score. It remains valuable for localizing failures and generating morphology hypotheses; pairwise rank plots and conditional regime summaries carry the main causal argument.

The new diagnostic figures are stored separately from the canonical context grids:

- [Electricity anomaly-only grid — TabPFN](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_only_electricity_tabpfn_raw_reading_context_grid.png)
- [Electricity anomaly-only grid — Trees](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_only_electricity_trees_raw_reading_context_grid.png)
- [Chilledwater anomaly-only grid — TabPFN](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_only_chilledwater_tabpfn_raw_reading_context_grid.png)
- [Chilledwater anomaly-only grid — Trees](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_only_chilledwater_trees_raw_reading_context_grid.png)
- [Hotwater anomaly-only grid — TabPFN](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_only_hotwater_tabpfn_raw_reading_context_grid.png)
- [Hotwater anomaly-only grid — Trees](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_only_hotwater_trees_raw_reading_context_grid.png)
- [Steam anomaly-only grid — TabPFN](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_only_steam_tabpfn_raw_reading_context_grid.png)
- [Steam anomaly-only grid — Trees](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_only_steam_trees_raw_reading_context_grid.png)
- [137-feature detection rate by raw-reading regime](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_anomaly_detection_by_raw_regime.png)

The anomaly-only grids make the vertical movement of true anomalies readable. The regime figure compresses the same evidence into conditional detection rates and makes the steam `100k–300k` region immediately visible. The original full-row grids remain the canonical all-row visualizations.

The normal-side companion figures use the same layout and reverse the class reading:

- low-score rows are normal rows correctly excluded from the alert queue;
- high-score rows are normal false positives;
- both classes retain the meter's original color, with alpha separating the two outcomes.

Outputs:

- [Normal-only grids and false-positive regime figure](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/m5_137_normal_false_positive_by_raw_regime.png)
- [All diagnostic outputs](../reports/assets/m5-context-construction-screening/m5_137_anomaly_diagnostics/)

The two sides should be read together. Anomaly-only figures locate missed anomaly regimes; normal-only figures locate false-positive regimes and show where a context raises normal scores into the alert queue. This paired view distinguishes an anomaly regime that remains invisible from a score scale that also makes normal rows look anomalous.

## 7. Next paper development directions

### 7.1 Localize the two observed blind regions

For every meter, model, feature line, and context size, report anomaly-only statistics by raw-reading regime:

- anomaly count;
- detection rate at `0.5`;
- median score;
- 10th/90th score percentiles;
- shared-miss rate for TabPFN and Trees.

The key cells are hotwater `0–1` and steam `100k–300k`. Connect those cells to duration, magnitude, local trend, time-of-day, site, and temporal-feature values.

### 7.2 Separate temporal representation families

Use targeted contrasts:

- baseline plus past-only changes;
- baseline plus future-only changes;
- difference-only;
- ratio-only;
- temporal changes without explicit meter identity;
- meter identity without temporal changes.

The question is which feature family moves the two blind regions and which family changes the context-composition response.

### 7.3 Separate support examples from scaler geometry

Run the existing composition contexts with a frozen pooled scaler. Compare:

- composition-specific rows + composition-specific scaler;
- composition-specific rows + frozen pooled scaler;
- pooled rows + composition-specific scaler;
- pooled rows + frozen pooled scaler.

Read the result through anomaly-regime detection rates, class-conditional score movement, and pairwise AUC. Pooled PR alone will miss the relevant effect.

### 7.4 Expand the source-meter response surface

Hotwater is an effective first sentinel. The final response surface should use all four source meters at `0`, natural share, `0.5`, and `1.0`, with target-meter PR, fixed-threshold detection, and pairwise cross-domain AUC. The output should be a source→target topology for F4, with F0 as the representation contrast.

## 8. Scope

The current evidence concerns the GEPIII-derived M5 holdout, building-disjoint evaluation, 50/50 label-balanced training contexts, the current TabPFN-3 configuration, the current tree ensemble, and the frozen context draw. The 137-feature line includes future-shift information and should be presented as an offline detection representation. The 500k report provides feasibility infrastructure; it provides no completed 500k performance result.
