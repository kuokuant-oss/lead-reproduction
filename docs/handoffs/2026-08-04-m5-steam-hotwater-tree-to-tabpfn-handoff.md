# M5 Steam–Hotwater Tree series → TabPFN handoff — 2026-08-04

## 1. Purpose and current state

This handoff transfers the **exact training-context identities** from the
dedicated Steam/Hotwater Tree series to a future TabPFN implementation.  It
does not authorize a different sampler, a different meter scope, a reordering
of context rows, or reuse of fitted Tree preprocessing/models.

Completed Tree contexts:

| Context | Fits | Odd-Steam PR-AUC | Odd-Steam ROC-AUC |
|---|---:|---:|---:|
| 50K `steam_only` | 4 | 0.754802969 | 0.960782077 |
| 50K `steam_hw_normal` | 4 | 0.751545071 | 0.960817911 |
| 50K `steam_hw_anomaly` | 4 | 0.761709404 | 0.957282209 |
| 50K `steam_hw_all` | 4 | 0.804667436 | 0.958680409 |
| 100K `steam_only` | 4 | 0.737911677 | 0.959030229 |
| 100K `steam_hw_all` | 4 | 0.793433745 | 0.965544022 |
| 200K `steam_hw_all` | 4 | 0.793507471 | 0.971584974 |

The 100K runner/identity test commits are `d593554` and
`c6abb656aad80f167b84378a12ae9984f04ea341`.  The 200K all-Hotwater formal
commit recorded in its final marker is `c0b72def2ebac6c22d665bab5edfe6795b6c14d3`.

## 2. Direct context authorities

Use these paths; do not reconstruct a new random sample.

| Desired TabPFN context | Authoritative JSON field |
|---|---|
| 50K Steam only | `data/processed/m5_eh_50k_steam_hotwater_preflight/preflight.json` → `manifests.steam_only.raw_index` |
| 50K Steam + HW normal | same → `manifests.steam_hw_normal.raw_index` |
| 50K Steam + HW anomaly | same → `manifests.steam_hw_anomaly.raw_index` |
| 50K Steam + all HW | same → `manifests.steam_hw_all.raw_index` |
| 100K Steam only | `data/processed/m5_ek_steam_budget_preflight/preflight.json` → `items.steam_100k.raw_index` |
| 100K Steam + all HW | same → `items.steam_hw_100k.raw_index` |
| 200K Steam + all HW | `data/processed/m5_ej_200k_steam_hotwater_preflight/preflight.json` → `raw_index` |

Each field is an ordered `int64` context vector.  Its stored sequence is
scientific input: do not sort, deduplicate-and-sort, concatenate class blocks,
or join/reorder it by timestamp.

The full condition census, including every Steam / Hotwater × label count, is
in [`m5-steam-hotwater-training-data-overview.md`](../reports/m5-steam-hotwater-training-data-overview.md).

## 3. Mandatory TabPFN admission gates

Before fitting, a TabPFN runner must hard-fail unless all applicable gates pass.

1. M3 input SHA-256 values equal the four values in the DOV §Common source and
   split gates.
2. `raw_index` is read directly from the selected authoritative JSON field as
   `int64`, preserving its stored order.
3. `len(raw_index)` equals the context budget, `len(unique(raw_index))` equals
   that same budget, and the little-endian `int64` SHA-256 equals the digest
   below.
4. Every selected row has an even `building_id`; no selected row is in the
   fixed 4,000-row validation exclusion.
5. Labels are exactly balanced: N/2 anomaly and N/2 normal.  No class weight,
   sample weight, over/under-sampling, duplication, SMOTE, or replacement is
   permitted after loading the vector.
6. The meter/label membership exactly matches the selected condition.  In
   particular, `steam_hw_normal` must have zero Hotwater anomalies;
   `steam_hw_anomaly` must have zero Hotwater normals; `steam_only` must have
   zero Hotwater rows; and none may include meter 0 or 1.
7. Build the frozen F4/137 feature matrix with `timestamp_merge`, then select
   with `frame.loc[raw_index]`; assert that the returned index is byte-for-byte
   identical to `raw_index` in order.
8. Fit any TabPFN-specific scaler/typing only on the admitted context.  Never
   import a Tree scaler, Tree model, Tree score, or odd label while preparing
   the context.

### Exact raw-index digests

| Context | Rows | SHA-256 |
|---|---:|---|
| 50K `steam_only` | 50,000 | `4defe9e9498b302f308fcd902f140f5750195ff664df8f9945c8b5928abca65e` |
| 50K `steam_hw_normal` | 50,000 | `a0b50958f395668b3be25a17dc6720c98dca98ef73b1d3d49817d84933f54da5` |
| 50K `steam_hw_anomaly` | 50,000 | `9f4db0387c3d90220fdc22c85f4f536ad912169bfc2da4e3009eed8692c0b4c7` |
| 50K `steam_hw_all` | 50,000 | `acc441899b7aa14bc18833a8a9f4bf1014107d9b7500fd828c359493900fdfe3` |
| 100K `steam_only` | 100,000 | `0d897266ee502c30ff3ae3c042545345252fa0f05e3717a9985da8c0b76e3012` |
| 100K `steam_hw_all` | 100,000 | `edf2fc89ae3b276780543c8659d641f0720131f92dfd1ba79a13e4260c9d2d01` |
| 200K `steam_hw_all` | 200,000 | `5979c3478f7350fbb7366e2383bb66b8fdc91227d9f2523ec488401d490638b0` |

### Fixed validation reconstruction gate

The manifests intentionally store the validation digest rather than a second
copy of the vector.  Reconstruct it exactly from the verified `train.csv`:

```python
raw = np.arange(len(train), dtype="int64")
even = train["building_id"].to_numpy() % 2 == 0
validation_raw_index = np.random.RandomState(20_042).choice(
    raw[even], size=4_000, replace=False
)
assert sha256_le_int64(np.sort(validation_raw_index)) == (
    "4f2002bfad4feba4ac3cf235ad724496bcd9845947650ce64367f44e0baa99f9"
)
assert not np.intersect1d(raw_index, validation_raw_index).size
```

`sha256_le_int64` hashes `np.ascontiguousarray(values.astype("<i8")).tobytes()`.

## 4. Feature, holdout, and evaluation constraints

- Use only meters 2 and 3 for training.  The future experiment may evaluate
  only the canonical odd Steam rows if it is intended to compare with this
  series.
- The holdout canonical positional identity is
  `data/processed/legacy/m5_tabpfn_137_full_test_n8_predictions.npz` →
  `raw_index`.  Steam is `meter == 2` indexed in that stored order; it has
  exactly 1,350,609 rows and 48,888 anomalies.
- Never use `m3_figure_predictions_50_50.npz.validation_raw_index` as a join
  key.  It is an A001 ordering trap.  Historical score comparison is positional
  after checking its anomaly vector equals the A002 anomaly vector.
- Report PR-AUC only with `sklearn.metrics.average_precision_score` and ROC-AUC
  with `sklearn.metrics.roc_auc_score`.

## 5. Explicit non-authorizations

This handoff does **not** authorize a 20K Steam+Hotwater Tree comparison: no
such meter-only Tree context has been frozen or run.  The old E0/E1 20K Tree
contains all four meters and is not a substitute.  It also does not authorize
changing the label balance, adding Electricity/Chilledwater, resampling a
context, changing the F4 regime, or treating these Tree results as a TabPFN
result.

Before any long TabPFN execution, follow
[`docs/policies/long-running-research-execution.md`](../policies/long-running-research-execution.md): implement atomic per-unit checkpoints, dry-run the gates above, commit the implementation, then obtain separate formal-run authorization.
