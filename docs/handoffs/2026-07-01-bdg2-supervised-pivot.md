# 2026-07-01 BDG2 supervised pivot handoff

## What changed

The BDG2 FDD roadmap pivots from unlabeled audit-yield as the primary M6 frame
to supervised evaluation on the GEPIII-overlap subset.

The important distinction is narrow but decisive:

+ BDG2 still has no native per-row anomaly label in its own archive.
+ `data/raw/m3/bad_meter_readings.csv` is a real rank-1 manual GEPIII/Kaggle
  annotation file already used by M2/M3.
+ Those labels can bridge to BDG2 overlap buildings for 2016 meters 0-3 through
  `building_id_kaggle`, meter code, and timestamp.

## Current P0 state

This handoff belongs to the docs-only pivot slice. It adds Proposed ADR 0025 and
ADR 0026, replaces the audit-yield plan with the supervised M6 plan, and updates
the roadmap/README/unknowns/CONTEXT framing. No code, loader, script, test, or
metric output is part of this slice.

## Next slice

M6.1 should implement the label bridge and integrity gate only. It should not
report supervised transfer accuracy. Required output is a coverage/provenance
JSON with eligible rows, hit rates, null-label rates, positive counts, and
excluded BDG2-only/2017/other-meter counts.

## Do not do yet

+ Do not change M3 defaults or golden metrics.
+ Do not export a new public helper unless the slice includes additive API tests.
+ Do not apply the GEPIII-only `0.2931` correction in the BDG2 path.
+ Do not include BDG2-only, 2017, water, gas, solar, or irrigation rows in
  supervised denominators.
+ Do not reactivate LEAD1.0 dual-label work without a new issue and ADR.
