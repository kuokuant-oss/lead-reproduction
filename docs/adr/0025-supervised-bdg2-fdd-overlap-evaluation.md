# Supervised BDG2 FDD evaluation on the GEPIII-overlap subset

## Status

Proposed (2026-07-01)

## Context

ADR 0017 measured the local BDG2 archive and found no native per-row anomaly
label in BDG2 itself. That remains true.

The pivot is that the repo already carries a separate label source:
`data/raw/m3/bad_meter_readings.csv`. It is the rank-1 GEPIII/Kaggle solution
team's manual bad-reading annotation file, row-aligned to Kaggle `train.csv`.
M2 and M3 already use it as supervised ground truth for GEPIII reproduction.

BDG2 metadata also preserves `building_id_kaggle` for the 1,449 buildings that
overlap GEPIII. Therefore those rank-1 GEPIII annotations can be keyed by
`(building_id, meter, timestamp)` and bridged onto BDG2 rows for:

+ GEPIII-overlap buildings only;
+ year 2016 only;
+ meters `electricity`, `chilledwater`, `steam`, and `hotwater` only, mapped to
  GEPIII meter codes `0`, `1`, `2`, and `3`.

They do not label BDG2-only buildings, 2017 data, or BDG2 meters outside the
GEPIII competition meter set such as water, gas, solar, or irrigation.

The label source is not part of the BDG2 official release. It comes from the
BUDS-lab GEPIII solution-analysis repo under `solutions/rank-1/input/`. The
correct framing is "rank-1 manual GEPIII annotations bridged onto BDG2's
overlap subset", not "BDG2 has native labels" or "BDG2 official labels".

## Decision

Use supervised FDD evaluation as the primary M6 paradigm for the BDG2
GEPIII-overlap, 2016, meters-0-3 subset.

For that labeled subset, M6 may report supervised ROC-AUC, PR-AUC, precision,
recall, and F1 using the bridged `bad_meter_readings` annotations. The label
scope must travel with every metric table, JSON key, figure, and prose claim.

BDG2-only buildings, 2017 rows, and non-GEPIII meters remain unlabeled. They may
be counted, excluded from supervised metrics, and later handled through an
explicitly-secondary pseudo-label or audit/review path. They must not be scored
as if they had bridged ground truth.

Raw BDG2 is the primary supervised scoring surface per ADR 0023. Cleaned BDG2
may be reported as a companion sensitivity, because cleaned data can remove the
very readings a fault-detection evaluation should examine.

GBDT remains the primary production-candidate scanner from the M5 model track.
TabPFN remains in scope for M6 comparison and label-scarce/offline evaluation,
but its license and about `6.3 ms/row` latency caveats remain attached.

## Superseded Decisions

This ADR supersedes ADR 0019 and ADR 0020 as the primary BDG2 FDD evaluation
paradigm.

ADR 0021 is superseded as moot for the supervised-overlap path because its
powered-gate decision belonged to the old unlabeled review branch.

ADR 0022 remains useful but is rescoped: electricity is the first labeled
supervised-evaluation meter, not merely the first unlabeled transfer/audit meter.

ADR 0023 remains useful and is rescoped: raw-first is required so supervised
evaluation does not score only after BDG2 cleaning removed candidate positives.

## Metric Contract

For the labeled overlap subset:

+ use a keyed label bridge defined by ADR 0026;
+ report metrics only for rows with non-null bridged labels;
+ stratify by meter, and keep the eligible labeled subset explicit;
+ use existing repo metric helpers such as `classification_metrics` instead of
  re-implementing metric math.

For unlabeled remainder rows:

+ report counts and exclusions before supervised metrics;
+ do not fabricate labels;
+ do not include rows in supervised denominators;
+ reserve raw-vs-cleaned pseudo-labels or review tooling for a later
  explicitly-secondary M6.4 slice.

For cross-dataset wording:

+ do not describe GEPIII-overlap metrics as pure BDG2-only transfer evidence;
+ do describe them as BDG2 raw/cleaned scoring under the GEPIII-overlap label
  bridge;
+ keep unknown #27 visible as a measured source-vs-target regime shift in M6.2.

## Consequences

+ M6 is replanned as supervised BDG2 FDD on the GEPIII-overlap subset first.
+ The prior unlabeled review plan is retired in favor of
  `docs/plans/bdg2-supervised-fdd-plan.md`.
+ The old unlabeled review tooling is demoted to the unlabeled
  remainder, not deleted from history.
+ M6.1 must build and prove the label bridge before any M6.2 supervised metric.
+ No code or metric is authorized by this ADR alone.
