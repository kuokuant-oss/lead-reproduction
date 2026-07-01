# BDG2 supervised FDD plan

**Stage**: M6 pivot plan after recognizing the GEPIII overlap label bridge
**Status**: Draft for review; P0 documentation pivot
**Decision records**:
[ADR 0025](../adr/0025-supervised-bdg2-fdd-overlap-evaluation.md),
[ADR 0026](../adr/0026-bdg2-label-bridge-integrity.md)

## Scope

M6 evaluates FDD on BDG2 where real supervised labels can be bridged:
GEPIII-overlap buildings, year 2016, meters `electricity`, `chilledwater`,
`steam`, and `hotwater`.

The label source is `data/raw/m3/bad_meter_readings.csv`, the rank-1 manual
GEPIII/Kaggle bad-reading annotation file used by M2 and M3. It is bridged onto
BDG2 through `building_id_kaggle`, meter code, and timestamp. It is not a native
BDG2 label release.

BDG2-only buildings, 2017 rows, and non-GEPIII meters remain unlabeled. They are
counted and excluded from supervised metrics until a later secondary
pseudo-label or audit slice.

## Guardrails

+ Preserve the M3 numeric line: M3.2 `0.9920`, M3.4 `0.9928`, seeds, default
  `row_offset`, and StandardScaler path stay fixed.
+ Keep `lead.__all__` frozen unless a later implementation makes an additive
  export with tests and ADR coverage.
+ Keep raw BDG2 as the primary scoring surface; cleaned BDG2 is a companion.
+ Do not apply the GEPIII-only `0.2931` correction inside the BDG2 path.
+ Use existing metric helpers, especially `classification_metrics`.
+ Scope every supervised metric to GEPIII-overlap, 2016, meters 0-3, bridged
  rank-1 GEPIII annotations.

## M6 Ladder

### M6.1 Label Bridge And Integrity

Build the keyed label bridge and prove it before metrics.

Done when:

+ a labeled overlap frame builds for eligible rows;
+ ADR 0026 integrity checks pass;
+ coverage JSON records row counts, label hit rate, null-label rate, and
  excluded rows;
+ no supervised accuracy metric is reported yet.

### M6.2 Supervised Transfer Accuracy

Score the GEPIII-trained M3.4 ensemble on BDG2 raw overlap rows and report
ROC-AUC, PR-AUC, precision, recall, and F1 by meter. Report cleaned as a
companion sensitivity.

Done when:

+ raw primary and cleaned companion supervised tables exist;
+ unknown #27 is measured as a source-vs-target / raw-vs-cleaned delta rather
  than only a caveat;
+ ADR 0025 can move from Proposed to Accepted if review approves the evidence.

### M6.3 GBDT Vs TabPFN On Labeled BDG2 Overlap

Compare GBDT and TabPFN on the same labeled BDG2 overlap frame, with accuracy
and latency side by side.

Done when:

+ the comparison mirrors the M5 Phase D report discipline;
+ TabPFN license and about `6.3 ms/row` latency caveats remain attached;
+ the verdict is based on labeled BDG2 overlap evidence, not preassigned roles.

### M6.4 Unlabeled Remainder

Handle BDG2-only, 2017, and other-meter rows as an explicitly-secondary
pseudo-label or audit screen. Raw-vs-cleaned can support this branch, but it is
not ground truth.

Done when:

+ all unlabeled outputs are labeled as pseudo-label, audit, or review evidence;
+ no unlabeled remainder row appears in supervised denominators.

### M6.5 Close-Out

Update README, plans, ADR status, reports, handoff, validation evidence, issue
state, and provenance placement.

## Slice Tracker

| Slice | Issue | Status | ADR |
| --- | --- | --- | --- |
| P0 docs-only pivot record | Not opened | In progress | ADR 0025/0026 Proposed |
| M6.1 label bridge + integrity | Not opened | Queued after P0 review | ADR 0026 |
| M6.2 supervised transfer accuracy | Not opened | Queued after M6.1 | ADR 0025 |
| M6.3 GBDT vs TabPFN supervised | Not opened | Queued after M6.2 | TBD |
| M6.4 unlabeled remainder | Not opened | Queued after M6.3 | TBD |
| M6.5 close-out | Not opened | Queued after M6.4 | TBD |

## Parked Contingency

LEAD1.0 dual-label electricity work is parked. Do not download it, wire it, or
reference it in active M6 metrics unless a later approved issue and ADR
reactivate it.
