# M5 — frozen 192-row query and final holdout identity audit

**Scope:** metadata, identity, coverage, and execution-scale only. No model was
scored, and no scientific outcome from either artifact was read or summarised.

This audit was run during E4 protocol preparation. Its purpose is to establish
*what* the remaining evidence stages would operate on, not to anticipate what
they would find.

## Evidence order

```
E4 formal Path A
  → E5 frozen 192-row independent replication
    → E6 complete other-half natural-prevalence full-test confirmation
```

The 192-row query is a replication check, **not** a substitute for the final
holdout result.

Two identity facts govern how these stages may be described, and both are
established below rather than assumed:

- the frozen 192-row query is still an **unscored independent query**;
- the 10,137,155 holdout rows have **already been scored** by prior
  context-curve runs, so E6 is a factorial confirmation on already-scored rows
  using new fitted states, not a first look at untouched data.

## 1. Frozen 192-row independent query

| | |
|---|---|
| Path | `data/processed/m5_hotwater_label_factorial/independent_query/` |
| `queries.npz` SHA-256 | `d780f0f8a96c47f49ffe061a72906728f1301056555350cabd979348aa41a2a0` |
| `manifest.json` SHA-256 | `1eca40088db317dc71ec71aa46060da3cc1078ba32b114551dab3d5ec69dc92d` |
| `raw_index` array SHA-256 | `2fc4a638a2a0880f2b4d7feac87875c941d155f5fe5172b75b13d041b654fa16` |
| Rows | 192 (192 unique) |
| Sampling seed | 20260730 |
| Strata | 3 × 64 rows: `hw01_negative`, `hw01_positive`, `steam_positive` |
| Concentration caps | ≤ 2 rows per building, ≤ 2 rows per segment |
| Meter counts | meter 2: 64, meter 3: 128 |
| Label counts | anomaly 0: 64, anomaly 1: 128 |
| Schema | `raw_index, anomaly, meter, building_id, meter_reading, segment_id, stratum` |

The manifest records `sampling_declared_before_prediction_read: true` and
`fit_rule: "no model fit or context change"`, and excludes the 352 screening
rows and their 142 buildings by construction.

**Scored by any current stage: no.** All 48 factorial prediction files under
`predictions/{tabpfn,trees}/…` carry 352 rows and match the screening query's
row set. No prediction artifact anywhere in the tree covers the 192-row set.

**Disjoint from the screening query: yes.** Intersection of the two `raw_index`
sets is **0 rows**.

## 2. The "complete other half" — identity resolved

The final holdout is defined by the split rule carried in every context and
query manifest:

```
fit_rule     : building_id % 2 == 0
holdout_rule : building_id % 2 == 1
```

| | |
|---|---|
| Rows | **10,137,155** (10,137,155 unique) |
| Buildings | 724 |
| Sites | 16 |
| Building-id parity | `{1}` — every row is an odd building, with no exceptions |
| Anomalies | 637,397 (natural prevalence 6.29%) |
| Sorted `raw_index` SHA-256 | `f0867d3e86ae2b017ea6fee2d1b9f6dead2ee241948346a467ea06305e220e76` |
| Schema | `raw_index, anomaly, tabpfn, site_id, building_id` |

**This answers the question the audit was asked to settle.** The "complete other
half" is **not** a separate untouched artifact waiting to be built — it *is* the
10,137,155-row row set that the existing full-test artifacts already cover. Ten
separate prediction files (17- and 137-feature, context sizes 5k/10k/20k/50k,
plus the ungated `_n8` variants) all carry the **identical** sorted `raw_index`
digest above.

So the row set has been scored many times. What has **not** been scored is this
row set **under the factorial design**: every existing full-test artifact comes
from the pooled-reference context-curve line, not from a hotwater
positive/negative-support factorial cell.

**Required language for E6.** E6 is a natural-prevalence *factorial*
confirmation: new factorial fitted states scored on holdout rows that prior
context-curve runs have already scored. It must **not** be described as a first
contact with an untouched holdout, and it must not be presented as an
untouched-data replication — those rows already carry context-curve
predictions. What is new in E6 is the states, not the rows.

## 3. Existing factorial predictions (352-row query)

48 prediction files exist and already span the exact E4 grid:

| Axis | Values | Count |
|---|---|---|
| Model | `tabpfn`, `trees` | 24 each |
| Context seed | 42, 123, 999 | — |
| Cell | 4 factorial cells | — |
| Scaler arm | `cell_specific`, `frozen_reference` | 24 each |

The 24 **tree** files are the matched-row fixed comparator E4 requires, and they
already exist for all 3 seeds × 4 cells × 2 arms. No tree refit is needed or
permitted.

The 24 **tabpfn** files predate the 8.0.8 pin and the repeated-inference design;
they carry one score vector each and no repeat structure. E4 replaces them with
24 fresh fits under TabPFN 8.0.8, each with 8 same-process repeats.

## 4. States reusable by E5 and E6

E4 persists one fitted state per (cell × context seed × scaler arm) — 24 states.
Both later stages are re-scoring problems, not refitting problems:

- **E5** would reload each of the 24 states and score the 192-row query. At E3's
  measured 8.2 s per 352-row inference, 192 rows is smaller than one E3 repeat;
  the whole of E5 is minutes of GPU time.
- **E6** would reload the same 24 states and score 10,137,155 rows.

E6 is the only stage whose cost is not trivial, and it is four orders of
magnitude larger than anything E3 or E4 has run.

## 5. Execution-scale references for E6

Do not extrapolate E6 from the 352-row figure — that measurement is dominated by
fixed overhead. Size it from the artifacts the earlier full-test line already
produced:

| Artifact | What it provides |
|---|---|
| `m5_tabpfn_137_remaining_batch_plan.json` | the realised batch decomposition: per-batch rows, buildings, sites, anomaly counts, prevalence, and canonical position ranges |
| `m5_tabpfn_137_shard_verification.json` | head/tail feature-matrix shapes (5,060,000 × 137 and 5,077,155 × 137), digest verification, and per-site row/anomaly identity |
| `m5_tabpfn_137_distributed_context*_n8/{head,tail}/` | the realised shard outputs for each context size |
| `m5_tabpfn_137_batch_runner.log`, `m5_tabpfn_17_run_progress.log` | realised wall-clock progress for the sharded runs |

The structural facts a plan must respect: the holdout was split **head/tail**
into roughly 5.06M and 5.08M rows, then further decomposed into batches by
building ranges, with per-batch prevalence varying widely (batch 0 at 0.156,
batch 1 at 0.074). Prevalence heterogeneity across batches means partial
coverage is not a representative sample of the holdout.

Multiplying by 24 states is the part that needs a decision before E6 is planned:
the earlier line scored the holdout once per context size, whereas E6 as
specified would score it once per fitted state.

## 6. What this audit did not do

No scores were read from either artifact. No metric, ranking, or outcome from
the 192-row query or the 10,137,155-row holdout appears in this report. The
`tabpfn` score column present in the full-test artifacts was not opened; only
`raw_index`, `building_id`, `site_id`, and `anomaly` were used, and `anomaly` only
to state prevalence and confirm the split.

Neither artifact was scored in this round.
