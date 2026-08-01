# M5 C1 artifact census

## Scope

Enumeration of every input the chilledwater C1 localization is permitted to use.
Read-only. C1 is CPU-only aggregation over existing artifacts: no fit, no refit,
no TabPFN inference, no tree refit, and no scoring of the frozen 192-row query.

Execution base: E0 completion commit `5e44479`, C1 branch `m5-c1-localization`
in a separate clean worktree. The main working tree's pre-existing unrelated
modifications were left untouched and are not part of this work.

Protocol frozen before any C1 result was read:
`data/processed/m5_chilledwater_c1/c1_protocol.json`,
SHA-256 `fb6699d8ccf7fefda1213261fb1d64db36c71dcb6a6b96f879959c11a6b5ac1b`.
The protocol writer refuses to overwrite a frozen protocol with different
content.

## Result

**17 artifacts required, 17 present, 0 missing.** Full record with per-file
SHA-256 in `data/processed/m5_chilledwater_c1/c1_artifact_census.json`.

### Predictions (natural-prevalence full holdout)

Ten frozen NPZ artifacts, five contexts × two learners, identical to the set
verified in E0:

| Learner | Contexts |
| --- | --- |
| TabPFN | 5k, 10k, 20k, 50k, 100k |
| matched-row tree | 5k, 10k, 20k, 50k, 100k |

### Row-level movement artifacts

| Artifact | Rows | Columns |
| --- | ---: | ---: |
| `m5_137_row_score_rank_movement_tabpfn.parquet` | 10,137,155 | 26 |
| `m5_137_row_score_rank_movement_trees.parquet` | 10,137,155 | 26 |

Coverage: 724 buildings; 637,397 anomaly rows and 9,499,758 normal rows;
per meter — electricity 6,035,071, chilledwater 2,115,354, steam 1,350,609,
hotwater 636,121.

These carry per-row scores at all five contexts plus score, global-rank and
within-meter-rank deltas, together with `building_id`, `meter`, `anomaly`,
`meter_reading` and `reading_regime`. **This is why C1 needs no inference: every
quantity it reports already exists at row level.**

The two learner tables are asserted to be row-aligned on `raw_index` rather than
merged on it; a mismatch is a hard failure.

### Frozen segment artifacts (E1 definition, not re-derived)

| Artifact | Rows | Role |
| --- | ---: | --- |
| `m5_137_anomaly_segments.parquet` | 13,334 | segment identity plus duration, slope, 24h/168h deviation, diff and ratio morphology |
| `m5_137_anomaly_segment_phases.parquet` | 27,957 | onset / middle / recovery phases |

Chilledwater coverage: 2,604 segments across 208 buildings.

### Screening query

`m5_context_stories/queries/screening/queries.npz` — the original 352-row
screening query. 352 rows, 142 buildings, 4 sites, arrays `raw_index`,
`anomaly`, `meter`, `site_id`, `building_id`.

| Stratum | Rows |
| --- | ---: |
| electricity positive / negative | 64 / 64 |
| chilledwater positive / negative | 64 / 64 |
| steam positive / negative | 32 / 32 |
| hotwater positive / negative | 16 / 16 |

### Frozen 192-row independent query

`m5_hotwater_label_factorial/independent_query/queries.npz` is recorded as
present and **was not read, not scored, and not used**. Its status is unchanged.

## Environment

C1 ran on the main repository virtual environment, not the pinned FORMAL_E0
execution clone. The pinned clone's `uv.lock` does not contain `pyarrow`, so it
cannot read the Parquet artifacts at all; the same limitation was recorded in
the E0 segment phase. Numeric stack: pandas 3.0.3, numpy 2.4.6,
scikit-learn 1.8.0, pyarrow 23.0.1. The pinned formal environment was not
modified.

## Sufficiency

No required artifact is missing, so C1 proceeded. Had one been absent, the
census exits non-zero and C1 stops: a missing artifact is never grounds to run a
fit or an inference.
