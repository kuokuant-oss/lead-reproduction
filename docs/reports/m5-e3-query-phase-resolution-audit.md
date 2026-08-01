# M5 E3 — query-phase resolution audit

**Artifact:** `data/processed/m5_e3_variance_pilot/e3_query_phase_audit.json`
**Stage:** pre-fit, read-only. No model was scored and the frozen 192-row query
was not touched.
**Rules:** the C1 query-resolution rules, reused unchanged.

This audit ran *before* any TabPFN fit, so its conclusions could not have been
chosen to suit a result.

## What was asked and what the artifacts can support

Two secondary readouts were requested on top of the E3 core. Each was checked
against the frozen artifacts, and each was resolved before execution.

| Requested readout | Status | Basis |
|---|---|---|
| chilledwater positive vs chilledwater negative, within meter | **RESOLVED** | 64 vs 64 rows, 4,096 valid pairs, AUC resolution 0.000244 |
| chilledwater positive vs hotwater negative | **RESOLUTION_LIMITED_DIAGNOSTIC** | 1,024 valid pairs but only 16 hotwater-negative rows; AUC resolution 0.000977 against a 0.0043 effect |
| onset / middle / recovery phase contrasts | **UNRESOLVED_NOT_EXECUTED** | no frozen artifact assigns a phase to a row |

## Within-meter contrast: adequate

The chilledwater-positive vs chilledwater-negative comparison passes every
predeclared threshold:

| Check | Threshold | Observed |
|---|---|---|
| valid pairs | ≥ 100 | 4,096 |
| top-building share, positives | ≤ 0.50 | 0.094 (34 buildings over 64 rows) |
| top-building share, negatives | ≤ 0.50 | 0.078 (45 buildings over 64 rows) |
| AUC resolution | — | 0.000244 |

Concentration is not a threat here: no single building supplies more than about
9% of either side. All 64 positive rows fall inside a segment, spread over 48
segments, with a top-segment share of 0.0625.

## Cross-meter contrast: resolution-limited, and labelled as such

The chilledwater-positive vs hotwater-negative comparison is arithmetically
computable — 1,024 valid pairs — but only 16 hotwater-negative rows exist. The
pairwise AUC can therefore only move in steps of 1/1024 = 0.000977, roughly
**4.4x coarser than the 0.0043 effect** it would need to resolve.

It is retained as a diagnostic and excluded from the scientific gate. In the
result artifacts its key is prefixed `RESOLUTION_LIMITED_DIAGNOSTIC_` so it
cannot be quoted as a finding by accident.

A continuous score margin was considered as a substitute and rejected: a margin
answers a different question than pairwise ordering, so it does not restore the
resolution that the 16-row denominator removed.

## Phase contrast: not executable without inventing a rule

The onset/middle/recovery contrasts cannot be computed from what is frozen.

`m5_137_anomaly_segment_phases.parquet` (27,957 rows) is a per-`(segment_id,
phase)` **aggregate**. It carries `rows`, `reading_mean`, `reading_slope`, and
movement columns — but **no timestamps and no row index**. Neither the
row-movement artifact nor the segment artifact carries a phase column.

So there is no path from a query row to a phase. Producing one would mean
inventing a within-segment cutpoint rule, which is forbidden: segments must not
be redefined and cutpoints must not be adjusted.

Status is **UNRESOLVED_NOT_EXECUTED** — recorded as unexecuted rather than
attempted-and-failed, because attempting it would itself have been the
violation. Consequence, as predeclared: the E3 core variance pilot proceeds
unchanged; the query is not modified, no rows are added, and the frozen 192-row
query is not read.

## Decision

The core E3 variance pilot was cleared to run on the within-meter contrast. The
cross-meter contrast is carried as a labelled diagnostic, and the phase contrast
is carried forward as unresolved. Neither limitation is a defect in E3 — both
are properties of the frozen inputs, and both were declared before the first
fit.
