# M5 Phase D handoff: BDG2 ingestion skeleton

## Scope

This slice closes issue #33 under the Phase D umbrella issue #31. It adds a
bounded BDG2 ingestion skeleton and smoke test only. It does not download the
full BDG2 corpus, add heavy BDG2 dependencies, run a transfer benchmark, or
change the frozen `lead.__all__` public API.

## What Changed

+ Added `src/lead/bdg2.py` with `load_bdg2_frame`.
+ Added `tests/fixtures/bdg2_fox_smoke.csv` and
  `tests/fixtures/bdg2_fox_smoke_labels.csv` as a small Fox/ASU-style single-site
  fixture.
+ Added `tests/test_bdg2_loader.py` to cover unlabeled loading, optional label
  merge, building-level split compatibility, site-held-out mask plumbing, and
  the explicit no-download gate.

## Loader Contract

`load_bdg2_frame` expects a local bounded CSV with `timestamp`, `building_id`,
`meter`, and `meter_reading`. `site_id` can come from the CSV or from the
`site_id=` argument. Labels are optional; if provided, they are merged by
`(building_id, meter, timestamp)` and normalized to `anomaly`. With no labels,
the returned frame has no `anomaly` column, preserving the Phase D requirement
that BDG2 may be unlabeled or transfer-only.

`allow_download=True` is intentionally a `NotImplementedError`; the full BDG2
pull remains gated behind an explicit future approval.

## Next Slice

Slice 2 should add an approved bounded real Fox/ASU data acquisition path or
documented local source mapping, then define the first transfer-evaluation
contract for held-out or unlabeled sites. Keep offline and causal feature
regimes explicit; any real-time FDD claim still requires `PAST_SHIFTS`-only
features per ADR 0007 and ADR 0011.
