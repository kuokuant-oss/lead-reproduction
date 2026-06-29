# Isolate GEPIII-only assumptions for BDG2

## Status

Accepted (2026-06-29)

## Context

Stage 0 and ADR 0017 established that BDG2 is not just a larger GEPIII frame.
It has string building ids, eight meter types, raw and cleaned meter variants,
site-level weather keyed by `(site_id, timestamp)`, six timezones, two calendar
years, and no native per-row anomaly labels.

The existing M3/GEPIII path also contains several assumptions that were valid
for the 2016 GEPIII/Kaggle reproduction but are unsafe for BDG2:

+ holiday flags were generated with `holidays.country_holidays("US",
  years=[2016])`;
+ site 0 / meter 0 was multiplied by `0.2931`, a GEPIII/Kaggle unit correction;
+ row-offset value-change grouped only by `building_id`;
+ post-processing used a leap-year end boundary `dayofyear > 366.9583`;
+ post-processing named only GEPIII meter ids `{0, 1, 2, 3}`;
+ site-held-out masks were duplicated inline in scripts.

## Decision

Keep GEPIII behavior unchanged by default, but make every assumption explicit
and provide BDG2-safe branches:

+ Holiday years are now derived from the frame's timestamp years. When a
  `timezone` column exists, holiday country is selected from the measured BDG2
  timezone family: US timezones use US holidays, `Europe/London` uses GB
  holidays, and `Europe/Dublin` uses IE holidays. Unknown or missing timezones
  fall back to US and must be re-reviewed before headline BDG2 claims.
+ The `0.2931` correction is isolated in a named GEPIII/Kaggle-only helper and
  is not called from `load_bdg2_frame`.
+ `add_value_change_features` keeps `row_offset` as the M3 default and adds
  `row_offset_meter_aware` for BDG2-style multi-meter rows. `timestamp_merge`
  already keys by `meter` when present.
+ Post-processing end-of-year masks are computed from each row's timestamp year,
  so 2016 uses a leap-year boundary and 2017 uses a common-year boundary.
+ Post-processing meter names are split into GEPIII numeric ids and BDG2 meter
  strings. GEPIII remains the default for the M3.5 script.
+ `leave_site_out_mask` is exported and the M3.5 / M5 Phase D site-held-out
  scripts route through it.

## Rationale

The M3 numeric line is a reproduction target, so changing defaults would be
unnecessary risk. Adding explicit BDG2 branches keeps the current GEPIII path
stable while preventing Phase E from inheriting hidden 2016-only, four-meter,
integer-id assumptions.

The holiday mapping is intentionally conservative. It follows the measured
timezone families in `docs/reports/bdg2-data-reality.md`, but it is still a
feature-engineering assumption rather than a verified operational calendar for
every site. Future Phase E evaluation must document whether holiday features
are used and whether the country mapping is adequate.

## Consequences

+ `lead.__all__` gains one additive export: `leave_site_out_mask`.
+ Existing M3 row-offset value-change behavior remains the default.
+ Existing M3 GEPIII post-processing still defaults to numeric GEPIII meter ids.
+ BDG2 callers have explicit meter-aware value-change and string-meter
  post-processing paths.
+ Phase E still needs a separate evaluation-paradigm ADR before any headline
  BDG2 metric, especially if raw/cleaned pseudo-labels are used.
