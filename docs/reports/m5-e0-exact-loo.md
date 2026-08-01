# M5 E0 exact leave-one-building analysis

## Scope

Exact leave-one-building (LOO) influence for the formal E0 endpoints. Pinned
HEAD `d8e59da2c40cb5102367d6a73299e807680f6ca6`, execution mode `FORMAL_E0`,
phase provenance digest
`7fb67090f7fcd5c1c6ce3efe68ab26d694eaa4e396cbe7b0360ee8d9b6955362`.

No fit, no refit, no TabPFN inference, no GPU. Every quantity is recomputed from
the frozen prediction artifacts by omitting one building's rows and re-scoring
the matched-row tree comparator against TabPFN on the identical retained rows.

## Universe and coverage

Every building in every meter was omitted exactly once — the universe is
complete, not sampled.

| Meter | Buildings |
| --- | ---: |
| electricity | 709 |
| chilledwater | 252 |
| steam | 162 |
| hotwater | 73 |
| **Total** | **1,196** |

Each `meter × omitted-building` is its own atomic checkpoint under
`checkpoints/leave_one_building/units/`, so the phase is interruption-safe and
resumable; 5,980 endpoint records in total. `COMPLETE.json` was issued only
after a full census reported 0 missing units. `.tmp` count 0; worker `stderr`
0 bytes.

Each unit preserves the building ID, its row count, and its positive count
alongside the metrics and the learner gap.

## Execution provenance

The committed analyzer exposes `leave_one_building_unit()` and a checkpointed
LOO phase, but `--formal` permits only identity, base_metrics and bootstrap, and
the checkpointed LOO path is reachable only under `--validation-mode`, which
stamps `NON_SCIENTIFIC_VALIDATION` provenance and truncates each meter to the
first `--loo-buildings` entries. There is therefore no committed path to a
*formal, complete* LOO. External orchestration supplied only the loop and the
checkpointing; the committed `leave_one_building_unit()` performed every
computation, on the full base, exactly as the monolithic `leave_one_building()`
does.

| Item | Value |
| --- | --- |
| units computed on laptop | 16 |
| units computed on gpu-host | 1,180 (4 parallel workers × 295) |
| assignment | round-robin over a fixed 1,196-unit manifest |
| manifest SHA-256 | `b42655f7d1585371db6bc9a7f344f33c0d2f35919e9f162f572191d15e131226` |
| transfer | 1,180 files, 276,572 B, SHA-256 `6554f43b…` verified identical after download |
| import | 1,180 written, 0 reused, 0 rejected |

Round-robin rather than contiguous ranges: electricity units cost ~24.5 s each
against ~7.3 s for chilledwater, so contiguous slices would have left one worker
with only cheap meters. Measured per-meter cost confirmed the ratio carried over
from the bootstrap phase (chilledwater/electricity 0.297 measured vs 0.277
assumed).

## Influence results — score endpoints

`100k_learner_gap`, TabPFN minus matched tree. A **sign flip** means one omitted
building changes the direction of the reported effect; that is the concrete
"dominated by a single building" test.

| Meter | Metric | Full | LOO min | LOO max | max abs Δ | Sign flips |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| electricity | PR-AUC | −0.006502 | −0.006876 | −0.005845 | 0.000657 | **0** |
| electricity | ROC-AUC | −0.001129 | −0.001152 | −0.000954 | 0.000175 | **0** |
| chilledwater | PR-AUC | +0.048618 | +0.037634 | +0.056080 | 0.010983 | **0** |
| chilledwater | ROC-AUC | +0.004285 | +0.003513 | +0.004862 | 0.000772 | **0** |
| steam | PR-AUC | +0.079004 | +0.058778 | +0.096896 | 0.020226 | **0** |
| steam | ROC-AUC | +0.018556 | +0.003649 | +0.019888 | 0.014907 | **0** |
| hotwater | PR-AUC | −0.005953 | −0.017951 | +0.015032 | 0.020985 | **2** |
| hotwater | ROC-AUC | −0.000509 | −0.004248 | +0.002456 | 0.003738 | **7** |

## Influence results — positive-rank endpoint

Reported separately and never pooled with the score endpoints.

| Meter | Full | LOO min | LOO max | max abs Δ | Sign flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| electricity | −0.001062 | −0.001273 | −0.000833 | 0.000229 | **0** |
| chilledwater | +0.003999 | +0.002803 | +0.005083 | 0.001196 | **0** |
| steam | +0.017884 | +0.003456 | +0.018881 | 0.014428 | **0** |
| hotwater | −0.000437 | −0.004785 | +0.007032 | 0.007469 | **6** |

## Influence results — 5k → 100k gap change

| Meter | Metric | Full | LOO min | LOO max | Sign flips |
| --- | --- | ---: | ---: | ---: | ---: |
| electricity | PR-AUC | +0.002898 | +0.000036 | +0.003389 | 0 |
| electricity | ROC-AUC | +0.000049 | −0.000087 | +0.000201 | 2 |
| chilledwater | PR-AUC | −0.060333 | −0.070455 | −0.046066 | 0 |
| chilledwater | ROC-AUC | −0.004115 | −0.004744 | −0.002917 | 0 |
| steam | PR-AUC | +0.119672 | +0.099117 | +0.140017 | 0 |
| steam | ROC-AUC | +0.005223 | +0.003935 | +0.005831 | 0 |
| hotwater | PR-AUC | +0.096185 | +0.041139 | +0.110810 | 0 |
| hotwater | ROC-AUC | +0.013384 | −0.000966 | +0.017448 | 1 |

## What the LOO evidence establishes

- **No single building drives electricity, chilledwater, or steam** on the
  primary score endpoints: 0 sign flips across 709, 252 and 162 omissions
  respectively. Electricity's range is exceptionally tight
  (PR-AUC spans 0.001 across 709 omissions).
- **Steam's direction is robust but its ROC-AUC and rank magnitudes are not.**
  Omitting one building moves ROC-AUC from +0.0186 to +0.0037 and the rank gap
  from +0.0179 to +0.0035 — roughly 80% of each magnitude. The gap never turns
  negative under any omission, so this is a magnitude sensitivity, not a
  direction reversal. It is stated here because the direction-only reading
  would overstate the precision of the ROC-AUC and rank estimates.
- **Hotwater is single-building sensitive.** Sign flips occur on PR-AUC (2),
  ROC-AUC (7) and rank (6) out of only 73 buildings. Combined with a bootstrap
  interval that spans zero, no direction survives omission.
- Score movement and rank movement are reported separately throughout; they are
  not combined into any single influence statistic.

Artifacts: `data/processed/m5_meter_specific_learner_gap/formal/
formal_loo_influence_summary.{json,csv}`.

## Not done

No interpretation beyond influence, no partial-LOO early reading (the phase was
finalized only after all 1,196 units existed), no fitting, no inference, and no
C1 or downstream unlock is claimed here.
