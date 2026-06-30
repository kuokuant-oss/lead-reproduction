# BDG2 EDA Report

**Date**: 2026-06-30
**Issue**: [#40](https://github.com/kuokuant-oss/lead-reproduction/issues/40)
**Plan**: [docs/plans/bdg2-eda-plan.md](../plans/bdg2-eda-plan.md)

## Scope And Guardrails

This is a read-only, pre-modeling EDA slice. It reads BDG2 from `data/raw/bdg2`
and reads GEPIII comparison data from frozen GEPIII sources (`load_m3_frame` and
`data/raw/m3/building_metadata.csv`). It does not build a model, create scores,
fabricate labels, report supervised BDG2 metrics, or make readiness/transfer
claims.

The report uses neutral data-quality terms: zero-reading share, negative-reading
share, flatline share, missingness, coverage, and distribution distance.

## Headline Findings

+ BDG2 has 1636 buildings: 187
  BDG2-only and 1449 GEPIII-overlap.
+ BDG2-only chilledwater is sparse at the building level:
  26 BDG2-only
  buildings have chilledwater columns, but only
  3 meet the
  sufficient-observation rule (`missing_rate <= 0.50`). This reproduces the
  Phase E Step 4 pooled stop point from the data side.
+ The chilledwater underpowering root cause is mostly building availability plus
  per-building observation missingness: most BDG2-only buildings do not have a
  chilledwater meter, and the available BDG2-only chilledwater set still leaves
  23
  high-missing buildings.
+ OOD quantification gives comparable scalars: square_feet KS
  `0.2176`, meter_reading KS
  `0.4549`, and primary_use
  categorical PSI
  `1.415`.

## Per-Meter Structure

| Meter | Buildings | BDG2-only buildings | Raw null | Cleaned null | Raw zero | Raw negative | Raw flatline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| electricity | 1578 | 151 | 0.04739 | 0.08929 | 0.04189 | 0 | 0.1539 |
| chilledwater | 555 | 26 | 0.06948 | 0.07766 | 0.1654 | 0 | 0.1967 |
| steam | 370 | 3 | 0.1126 | 0.1218 | 0.1284 | 0 | 0.177 |
| hotwater | 185 | 19 | 0.06201 | 0.07425 | 0.3294 | 0 | 0.4551 |
| gas | 177 | 52 | 0.03337 | 0.0477 | 0.3776 | 0 | 0.4609 |
| water | 146 | 25 | 0.05783 | 0.06938 | 0.481 | 0 | 0.5284 |
| irrigation | 37 | 20 | 0.107 | 0.1189 | 0.7662 | 0 | 0.8319 |
| solar | 5 | 0 | 0.2013 | 0.2172 | 0.2719 | 0 | 0.331 |

## Missingness Decomposition

This table separates building-level meter availability from observation-level
missingness. `Absent buildings` means metadata buildings without a column in the
wide meter file.

| Meter | Absent buildings | Median timestamp coverage | Raw observation missingness | Cleaned observation missingness |
| --- | --- | --- | --- | --- |
| electricity | 58 | 0.9985 | 0.04739 | 0.08929 |
| chilledwater | 1081 | 0.999 | 0.06948 | 0.07766 |
| steam | 1266 | 0.999 | 0.1126 | 0.1218 |
| hotwater | 1451 | 0.9998 | 0.06201 | 0.07425 |
| gas | 1459 | 1 | 0.03337 | 0.0477 |
| water | 1490 | 0.9968 | 0.05783 | 0.06938 |
| irrigation | 1599 | 0.9444 | 0.107 | 0.1189 |
| solar | 1631 | 0.9987 | 0.2013 | 0.2172 |

## Cleaned-Vs-Raw Delta

| Meter | Null-rate delta | Raw present -> cleaned missing | Raw missing -> cleaned present | Changed observed cells |
| --- | --- | --- | --- | --- |
| electricity | 0.04189 | 0.04189 | 0 | 0.0004208 |
| chilledwater | 0.00818 | 0.00818 | 0 | 0.005495 |
| steam | 0.009229 | 0.009229 | 0 | 0.00359 |
| hotwater | 0.01224 | 0.01224 | 0 | 0.001415 |
| gas | 0.01433 | 0.01433 | 0 | 0.0005059 |
| water | 0.01155 | 0.01155 | 0 | 0.00391 |
| irrigation | 0.01195 | 0.01195 | 0 | 0.001761 |
| solar | 0.01595 | 0.01595 | 0 | 0 |

## BDG2-Only Sufficient-Observation Counts

| Meter | BDG2-only with meter | Sufficient obs | High missing | Median missing rate |
| --- | --- | --- | --- | --- |
| electricity | 151 | 99 | 52 | 0 |
| chilledwater | 26 | 3 | 23 | 0.5024 |
| steam | 3 | 3 | 0 | 0 |
| hotwater | 19 | 2 | 17 | 0.5018 |
| gas | 52 | 50 | 2 | 0 |
| water | 25 | 23 | 2 | 0.01146 |
| irrigation | 20 | 20 | 0 | 0.05424 |
| solar | 0 | 0 | 0 | n/a |

## Metadata And Meter Coverage

| Meter | All buildings marked yes | BDG2-only | GEPIII-overlap |
| --- | --- | --- | --- |
| electricity | 1578 | 151 | 1427 |
| chilledwater | 555 | 26 | 529 |
| steam | 370 | 3 | 367 |
| hotwater | 185 | 19 | 166 |
| gas | 177 | 52 | 125 |
| water | 146 | 25 | 121 |
| irrigation | 37 | 20 | 17 |
| solar | 5 | 0 | 5 |

Primary-use unseen/unmapped rate for BDG2-only vs GEPIII is
`0.1123`.
Unseen or unmapped normalized categories:
`(missing/unmapped)`.

Square-feet medians:

+ BDG2-only: `2.786e+04`.
+ GEPIII-overlap: `5.767e+04`.
+ GEPIII: `5.767e+04`.

## BDG2-Only Vs GEPIII-Overlap Main Line

+ BDG2-only buildings are concentrated in a smaller set of sites, especially
  Lamb, Panther, Rat, and Swan in the local archive.
+ Meter availability differs sharply by meter. Electricity is broadest; solar
  and irrigation remain narrow. Chilledwater has enough overlap buildings for a
  bridge baseline but not enough BDG2-only sufficient-observation buildings for
  the prior Step 4 frame.
+ Cleaned files increase null rates for every meter in this archive, consistent
  with the Stage 0 inventory. That is a data-quality delta, not a label.

## BDG2 Vs GEPIII OOD Quantification

| Feature | KS | PSI | Basis |
| --- | --- | --- | --- |
| square_feet | 0.2176 | 0.2235 | BDG2-only vs GEPIII metadata |
| meter_reading | 0.4549 | 1.102 | sampled raw BDG2-only cells vs GEPIII `load_m3_frame` cells |
| primary_use coverage | n/a | 1.415 | categorical PSI; unseen/unmapped rate 0.1123 |

Figures:

+ ![Square feet ECDF](../assets/bdg2-eda/square-feet-ecdf.png)
+ ![Meter reading histogram](../assets/bdg2-eda/meter-reading-hist.png)

Figure sizes:

| Figure | Bytes |
| --- | --- |
| docs/assets/bdg2-eda/square-feet-ecdf.png | 31412 |
| docs/assets/bdg2-eda/meter-reading-hist.png | 25387 |

## Temporal Profiles

The provenance JSON includes hour/month mean profiles for representative
electricity and chilledwater raw readings. These are descriptive profiles only;
they are not model features, scores, or readiness evidence.

## Provenance

+ Machine-readable summary: `data/processed/bdg2_eda.json` (gitignored shard).
+ Script: `scripts/run_bdg2_eda.py`.
+ BDG2 source: `data/raw/bdg2`.
+ GEPIII comparison source: `load_m3_frame(verbose=False)` and
  `data/raw/m3/building_metadata.csv`.
