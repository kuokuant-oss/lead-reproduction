# M5 Steam–Hotwater balanced-context data overview

## Scope

This is a data overview (DOV), not a new scientific result.  It records the
exact balanced, even-building training contexts used by the dedicated Steam /
Hotwater Tree experiments.  Every context below is evaluated only on the
canonical odd-building Steam holdout: 1,350,609 rows, 48,888 anomalies.

`steam` means `meter == 2`; `hotwater` means `meter == 3`.  Electricity
(`meter == 0`) and Chilledwater (`meter == 1`) are excluded from every context
in this document.

## Common source and split gates

All frozen contexts use the M3 files below.  Their SHA-256 values must match
before another learner consumes a context.

| Source | SHA-256 |
|---|---|
| `data/raw/m3/train.csv` | `2d75e0c4cfa93818647cf5272ef8d48f9cf8e9d3479dde357faae381d1abbcb3` |
| `data/raw/m3/bad_meter_readings.csv` | `e9846b746bc584f3f57f30f54d2077beb294e39407146ff6c9ad024f806bab93` |
| `data/raw/m3/building_metadata.csv` | `357bf585047359e9dfbaef2935453429f8ec19e5e80c08a8eb066789f28c4070` |
| `data/raw/m3/weather_train.csv` | `81022191f16dacc21494c15dac7975611cb39922fc7332e419a857cbb00cc125` |

The training side is defined by `building_id % 2 == 0`.  A fixed 4,000-row
validation exclusion is sampled from that side with
`numpy.random.RandomState(20042).choice(..., replace=False)`.  Its sorted
little-endian `int64` digest is
`4f2002bfad4feba4ac3cf235ad724496bcd9845947650ce64367f44e0baa99f9`.

After this exclusion, the candidate census is:

| Eligible meters | Rows | Anomaly | Normal | Maximum no-replacement 50:50 context |
|---|---:|---:|---:|---:|
| Steam only | 1,357,567 | 76,943 | 1,280,624 | 153,886 |
| Steam + Hotwater | 1,985,240 | 132,996 | 1,852,244 | 265,992 |
| Hotwater contribution | 627,673 | 56,053 | 571,620 | — |

Consequently a 200K Steam-only 50:50 context is impossible without duplicating
anomalies: it is short by 23,057 Steam anomaly rows.  It was correctly not run.
This is a feasibility fact, not a model result.

## The four 50K factorial conditions

The authoritative artifact is
`data/processed/m5_eh_50k_steam_hotwater_preflight/preflight.json` under
`manifests`.  Each context has exactly 50,000 unique raw rows: 25,000 anomaly
and 25,000 normal.  The selection is condition-local E0/E1
`nested_balanced_indices` semantics, seed 42, without replacement.

| Condition | Inclusion rule | Steam anomaly | Steam normal | HW anomaly | HW normal | Raw-index digest |
|---|---|---:|---:|---:|---:|---|
| `steam_only` | Steam | 25,000 | 25,000 | 0 | 0 | `4defe9e9498b302f308fcd902f140f5750195ff664df8f9945c8b5928abca65e` |
| `steam_hw_normal` | Steam, plus Hotwater normal only | 25,000 | 17,309 | 0 | 7,691 | `a0b50958f395668b3be25a17dc6720c98dca98ef73b1d3d49817d84933f54da5` |
| `steam_hw_anomaly` | Steam, plus Hotwater anomaly only | 14,433 | 25,000 | 10,567 | 0 | `9f4db0387c3d90220fdc22c85f4f536ad912169bfc2da4e3009eed8692c0b4c7` |
| `steam_hw_all` | Steam and all Hotwater types | 14,433 | 17,195 | 10,567 | 7,805 | `acc441899b7aa14bc18833a8a9f4bf1014107d9b7500fd828c359493900fdfe3` |

The four conditions are the only source for claims about Hotwater normal,
Hotwater anomaly, or the factorial interaction at 50K.  They are not neutral
controls and must not be silently replaced by a different random sample.

## Completed all-Hotwater budget points

These rows refer specifically to the `Steam + all Hotwater` inclusion rule.
Every training context is exactly 50:50 by label, uses unique even-building raw
rows, retains frozen raw-index order, uses F4/137 `timestamp_merge` features,
and the fixed four-Tree equal-weight ensemble.

| Budget | Artifact and raw-index location | Steam anomaly | Steam normal | HW anomaly | HW normal | PR-AUC | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 20K | Not run for this meter-only Tree line | — | — | — | — | — | — |
| 50K | `m5_eh_50k_steam_hotwater_preflight/preflight.json` → `manifests.steam_hw_all.raw_index` | 14,433 | 17,195 | 10,567 | 7,805 | 0.804667436 | 0.958680409 |
| 100K | `m5_ek_steam_budget_preflight/preflight.json` → `items.steam_hw_100k.raw_index` | 28,924 | 34,542 | 21,076 | 15,458 | 0.793433745 | 0.965544022 |
| 200K | `m5_ej_200k_steam_hotwater_preflight/preflight.json` → `raw_index` | 57,854 | 69,099 | 42,146 | 30,901 | 0.793507471 | 0.971584974 |

For reference, the fixed Historical Full Tree has odd-Steam PR-AUC
0.770821669 and ROC-AUC 0.967257910.  The dedicated meter-only 50K experiment
is currently the highest PR-AUC among the completed all-Hotwater budgets.

## Feature and model boundary

The Tree contexts used the frozen F4/137 layout, timestamp-merge value-change
regime, a context-fitted `StandardScaler`, and equal probabilities from
LightGBM, XGBoost, CatBoost, and HistGradientBoosting.  The raw-index contexts
are learner-neutral; a TabPFN experiment may consume the same ordered rows,
but it must not reuse a fitted Tree `StandardScaler` or Tree predictions.
Its TabPFN preprocessing/model contract must be declared separately.

For the complete handoff and executable identity gates, see
[`docs/handoffs/2026-08-04-m5-steam-hotwater-tree-to-tabpfn-handoff.md`](../handoffs/2026-08-04-m5-steam-hotwater-tree-to-tabpfn-handoff.md).
