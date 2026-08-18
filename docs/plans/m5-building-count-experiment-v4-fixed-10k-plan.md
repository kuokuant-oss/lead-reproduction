# M5 building-count experiment V4: fixed 10K balanced contexts

- Status: implementation, context preparation, and bounded validation in progress
- Date: 2026-08-18
- Experiment version: `m5_building_count_v4_fixed_10k`
- Execution policy: `docs/policies/long-running-research-execution.md`

## 1. Research question

V4 isolates source-building diversity while holding the training-context size and
global class ratio fixed. It asks whether increasing the number of source
buildings changes Tree Ensemble and TabPFN performance when every model sees
exactly 10,000 unique training rows containing 5,000 anomalies and 5,000
normals.

V4 is a new experiment. It does not reuse V2/V3 building identities, ladders,
row identities, or row-allocation logic.

## 2. Frozen factors

| Factor | V4 setting |
| --- | --- |
| Source-building budgets | K = 50, 100, 200, 300, 400 |
| Context size | 10,000 unique rows at every K |
| Class ratio | exactly 5,000 anomalies and 5,000 normals |
| Building draws | 5; `building_draw_seed` = 0, 1, 2, 3, 4 |
| Row draws | 2 per building ladder; `row_draw_seed` = 0, 1 |
| Model seed | 42 for both model families |
| Features | preserved 137-feature pipeline |
| Models | frozen Tree Ensemble contract and TabPFN with 8 estimators |
| Holdout | canonical odd-building holdout at natural prevalence |
| Primary metrics | per-meter ROC-AUC and PR-AUC |

This produces 5 building ladders x 2 row draws x 5 K values = 50 frozen
contexts. Both model families consume each context, producing 100 formal model
cells.

## 3. Building selection

The candidate population is the sorted set of every even-numbered building in
the M3 training frame. For each `building_draw_seed`, NumPy's documented PCG64
pseudo-random generator creates one permutation of that complete candidate
population. The K sets are the first 50, 100, 200, 300, and 400 entries of the
same permutation. Therefore, for each building draw,

`B50` is an ordered prefix of `B100`, which is a prefix of `B200`, `B300`, and
`B400`.

No label, meter, site, row count, anomaly rate, diversity score, minimum
coverage rule, acceptance threshold, retry, repair, or redraw affects building
selection. A selected building is allowed to contribute zero sampled rows to a
particular 10K context.

## 4. Row selection

For every `(building_draw_seed, row_draw_seed, K)` cell, the eligible pool is
all unique training rows belonging to the exact K-building prefix. Rows are
partitioned only by the binary anomaly label. Exactly 5,000 rows are sampled
uniformly without replacement from each class, using independent PCG64 streams
whose complete seed material is recorded in the manifest. The selected class
vectors are deterministically interleaved anomaly/normal to freeze row order.

There is no per-building, per-meter, or per-site quota; no preference for any
previous experiment's rows; no oversampling; and no synthetic or duplicated
row. Meter/site/building composition is recorded only after sampling as an
audit outcome. K contexts share the nested building support but row sets are
independent fixed-size draws and are not claimed to be nested.

If either class has fewer than 5,000 eligible unique rows, preparation fails.
It must not replace a building, redraw a ladder, lower the target, or duplicate
rows.

## 5. Pairing and estimand

Tree Ensemble and TabPFN receive the identical ordered raw-row vector, labels,
137-feature matrix, and canonical holdout for a context. The primary replicate
structure is hierarchical:

1. retain every raw model cell;
2. within each building ladder and K, summarize the two row draws;
3. across K, compare matched prefixes within the same building ladder;
4. summarize the five building-ladder effects without treating the ten
   contexts as ten independent building draws;
5. report row-draw variability separately from between-building-ladder
   variability.

## 6. Provenance and gates

Preparation writes immutable building-ladder manifests, context artifacts,
composition audits, raw-input digests, and a strict `training_context_gate.json`.
The gate covers all 25 building-ladder cells and all 50 row contexts and checks:

- candidate identity and PCG64 seed/algorithm identity;
- exact ordered nesting of K building prefixes;
- exact K unique building IDs and even-building eligibility;
- 10,000 unique rows, exactly 5,000 per class;
- row membership in the selected K-building support;
- artifact and ordered-row SHA-256 identities;
- absence of sampling repair or redraw.

Bounded model validation is isolated under a visibly non-scientific directory.
It exercises both model families at the smallest and largest K with capped
context/holdout rows, while the preparation gate validates every formal
context. Validation outputs are never scientific results. Formal execution is
blocked until the context gate, focused tests, bounded model validation, an
explicit authorization flag, the TabPFN checkpoint, and a clean committed
implementation all pass.

## 7. Execution order and recovery

The formal queue is strictly K-major:

1. finish all 10 contexts and both model families at K=50;
2. only then begin K=100;
3. then K=200;
4. then K=300;
5. finally K=400.

Within K, order is building seed 0 through 4 and row seed 0 then 1; within a
context, Tree runs before TabPFN. Every model cell uses atomic prediction
chunks, provenance checks, heartbeat/status files, a completion marker, and
deterministic resume. A failed or interrupted unit blocks later K values and is
resumed from validated checkpoints; there is no automatic retry loop or
automatic restart of a healthy process.

## 8. Reporting contract

The formal aggregate must preserve `building_draw_seed`, `row_draw_seed`, K,
model family, meter, ROC-AUC, and PR-AUC. The main comparison reports per-meter
means and uncertainty using the hierarchical replicate structure above. Macro
metrics, if shown, are equal-weight means over electricity, chilled water,
steam, and hot water; they do not replace the meter-level table.
