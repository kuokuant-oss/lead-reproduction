# M5 E3 — TabPFN 8.0.8 variance pilot

**What this measures:** how much a TabPFN readout moves when the *only* thing
that changes is repeated inference on one fitted state. It does not measure
whether any cell's effect is real.

| | |
|---|---|
| Base commit | `e9cb59b9bbf7977f4bda2dfbb9779fb0659d9168` |
| Protocol artifact | `e3_protocol.json`, SHA-256 `679f8211…1419b14` |
| Realised cell order | `00 → 01 → 10 → 11` (PCG64, seed 42, order frozen in the protocol) |
| Cells | 4 · one fit each · 20,000 context rows · seed 42 · F4_137 |
| Repeats | 8 per cell, 32 total; none escalated |
| Query | 352 rows, SHA-256 `06874156…7718575`, identical across all four cells |
| Result root | `data/processed/m5_e3_variance_pilot/` |

## Design, in one paragraph

Each cell is one factorial combination of hotwater-positive and
hotwater-negative rows in the context. Each cell gets exactly **one** fit; the
repeats are repeated *inference* on that one fitted state, in the same process.
Only two steam endpoints gate continuation. Precision is a two-sided 95%
Student-t half-width on the repeat-level endpoint mean,
`t(0.975, n-1) · sd(ddof=1) / √n` — normal approximation, percentile bootstrap,
and averaging row probabilities before scoring are all forbidden. The bounded
target is 0.015; the continuous margin target is `0.02 × reference_IQR`, where
the reference IQR comes from the **fixed tree comparator** and was computed
before any TabPFN fit, so it cannot have been tuned to the result.

## The gate

Both endpoints passed in every cell at the first batch of 8. The cap of 40 was
never approached.

| Cell | Context composition | AUC half-width (≤ 0.015) | Margin half-width | Margin target | Verdict |
|---|---|---:|---:|---:|---|
| 00 | hw_pos ✗, hw_neg ✗ | 0.013179 | 0.005507 | 0.016183 | pass |
| 01 | hw_pos ✗, hw_neg ✓ | 0.000000 | 0.000936 | 0.001000 | pass |
| 10 | hw_pos ✓, hw_neg ✗ | 0.006094 | 0.005606 | 0.008743 | pass |
| 11 | hw_pos ✓, hw_neg ✓ | 0.002035 | 0.005547 | 0.012071 | pass |

Cell 00 has the least margin to spare (0.013179 against 0.015). Cell 01 is the
tightest in relative terms: its margin half-width of 0.000936 sits against a
target of 0.001000, because its reference IQR of 0.0500 is an order of magnitude
narrower than cell 00's 0.8091. A cell can therefore be *both* the most stable
in absolute terms and the closest to failing, and only the ratio shows it.

### Gating endpoints, repeat-level

| Cell | steam AUC mean | sd | range | steam margin mean | sd | range |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 0.826172 | 0.015764 | 0.039062 | 0.352844 | 0.006587 | 0.019742 |
| 01 | 1.000000 | 0.000000 | 0.000000 | 0.922186 | 0.001119 | 0.003102 |
| 10 | 0.473145 | 0.007289 | 0.025391 | 0.049777 | 0.006706 | 0.024187 |
| 11 | 0.970947 | 0.002434 | 0.007812 | 0.705480 | 0.006634 | 0.018109 |

Cell 01's AUC is 1.000000 in all 8 repeats — a saturated endpoint, not a
precise one. Its half-width of exactly zero means the metric has no headroom
left to vary in, so it carries no information about stability. The margin
endpoint, which is not saturated, is what actually certifies that cell.

## Repeated inference is not bitwise reproducible

Every cell produced **8 distinct score-vector digests from 8 repeats** — 32 of
32 across the pilot, zero collisions. Repeated inference on a single fitted
TabPFN state does not return identical floating-point output.

The endpoint-level consequence is small. The largest observed spread on a gating
endpoint is cell 00's AUC range of 0.039 across 8 repeats; the tightest non-
degenerate one is cell 01's margin range of 0.003. This is exactly the quantity
the pilot existed to measure, and it is the number that any future comparison
between cells has to clear before it can be called an effect.

## Non-gating readouts

Reported for completeness; none of them gates anything, and the chilledwater
cross-meter comparison is prefixed `RESOLUTION_LIMITED_DIAGNOSTIC_` in the
artifacts so it cannot be quoted as a finding by accident.

The within-meter chilledwater readouts are extremely stable across repeats
(half-widths of 1e-4 to 2e-3 in every cell), as are the rank readouts. Score and
rank are never pooled.

## Fresh-process reload diagnostic

The protocol fixes this to cell 11, two runs, each in its own process. Both
reloaded the persisted state via `load_fitted_tabpfn_model`, and both passed
every hard-failure check: version match (8.0.8), state digest match
(`45666736…`), query row identity, finite output.

| Run | Load | Score | steam AUC | steam margin | Endpoints inside the same-process range |
|---|---:|---:|---:|---:|---|
| 0 | 0.84 s | 9.08 s | 0.966797 | 0.690466 | 9 of 11 |
| 1 | 1.25 s | 9.23 s | 0.968750 | 0.708909 | 11 of 11 |

For reference, the same-process repeats of cell 11 gave a steam AUC mean of
0.970947 (range 0.007812) and a steam margin mean of 0.705480 (range 0.018109).

Run 0 fell outside the same-process min–max on two endpoints, by 0.90 SD
(steam margin) and 0.55 SD (chilledwater global rank). With n=8 the observed
min–max is a narrow envelope, so landing just outside it is unremarkable — but
the protocol says numerical differences are **recorded, not judged**, and no
pass/fail threshold was invented for them here.

These two runs are stored in `cell_11/fresh/`, flagged
`excluded_from_same_process_statistics: true` and `scientific_estimate: false`.
They enter no mean, SD, or confidence interval, and they did not influence how
many repeats any cell ran.

## Provenance

- Four distinct fit-state digests (`75f0537c`, `14422d3d`, `3438ecb6`,
  `45666736`) — four genuinely independent fits, not one state reused.
- Four distinct process UUIDs, one per cell.
- One query digest shared by all four cells: composition is the only thing that
  differs between them.
- Every gate statistic in this report was **recomputed on import** from the raw
  per-repeat records and matched the runner's own values to within 1e-12. The
  recomputation takes its targets from the frozen protocol, not from the
  recorded result.

## What this pilot does not establish

The differences between cells are large — steam AUC runs from 0.473 in cell 10
to 1.000 in cell 01, against repeat-level half-widths of 0.006 and 0.000. That
is a 40-to-1 ratio, and it is tempting to read it as the mechanism result.

It is not, and E3 does not license that reading. This pilot has one fit per
cell, so between-cell differences confound the composition change with
fit-to-fit variation, which is **not measured here at all**. What E3 establishes
is only that the inference-level noise floor is small enough that a future
design could resolve such differences — not that these particular differences
survive a design that separates the two sources.
