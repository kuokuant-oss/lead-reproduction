# Phase E to M6 FDD roadmap

## Purpose And Status

Authoritative roadmap for the Phase E to M6 pivot.

Current status: Phase E Part A is complete. The old unlabeled review M6 ladder
has been superseded by the supervised BDG2 overlap pivot recorded in
[ADR 0025](../adr/0025-supervised-bdg2-fdd-overlap-evaluation.md) and
[ADR 0026](../adr/0026-bdg2-label-bridge-integrity.md).

This P0 pivot is documentation-only. It does not implement a label bridge, run
scoring, report metrics, or change `src/lead`.

## Pivot Fact

BDG2 has no native per-row anomaly labels in the local archive. However,
`data/raw/m3/bad_meter_readings.csv` is the rank-1 manual GEPIII/Kaggle
bad-reading annotation file already used by M2 and M3. Because BDG2 preserves
`building_id_kaggle` for GEPIII-overlap buildings, those labels can be bridged
by `(kaggle_building_id, meter_code, timestamp)` onto BDG2 overlap rows for
2016 meters `electricity`, `chilledwater`, `steam`, and `hotwater`.

That makes supervised metrics legitimate only for the bridged subset:
GEPIII-overlap buildings, 2016, meters 0-3. BDG2-only buildings, 2017 rows, and
other meters remain unlabeled.

## Fixed Constraints

+ M3 numeric line is frozen: `load_m3_frame` defaults, M3.2/M3.4 golden values,
  the +/- `0.0005` gate, downsampling semantics, and StandardScaler fit path.
+ `lead.__all__` is frozen unless a later implementation slice makes an
  additive export with ADR and `test_public_api.py` coverage.
+ Raw BDG2 is the primary scoring surface; cleaned is a companion sensitivity.
+ The GEPIII-only `0.2931` unit correction stays out of the BDG2 path.
+ Every supervised BDG2 metric must state: GEPIII-overlap, 2016, meters 0-3,
  bridged rank-1 GEPIII annotations.
+ BDG2-only, 2017, water, gas, solar, and irrigation rows are counted and
  excluded from supervised denominators.
+ One slice = one issue = one commit = stop for review; run the full
  change-checklist per slice.

## Superseded Part A Decisions

The following Part A decisions remain historical context but no longer define
the primary M6 evaluation paradigm:

+ [ADR 0019](../adr/0019-bdg2-evaluation-paradigm.md): superseded by ADR 0025.
+ [ADR 0020](../adr/0020-bdg2-fdd-audit-yield-evaluation.md): superseded by ADR
  0025; retained only for the unlabeled remainder.
+ [ADR 0021](../adr/0021-powered-gate-as-transfer-confidence.md): superseded as
  moot for the supervised-overlap path.

The following decisions continue with amended scope:

+ [ADR 0022](../adr/0022-electricity-entry-meter-for-bdg2-fdd.md): electricity
  remains the first BDG2 FDD meter, now as the first labeled supervised-eval
  meter.
+ [ADR 0023](../adr/0023-raw-first-bdg2-transfer-scoring.md): raw-first remains
  required for supervised evaluation correctness.
+ [ADR 0024](../adr/0024-value-change-regime-convergence.md): value-change
  semantics remain the multi-meter guardrail.

## M6 Phase Ladder

### M6.1 Label Bridge And Integrity

Build the keyed bridge from GEPIII labels to BDG2 overlap rows. Prove the
bridge before metrics.

Definition of Done:

+ labeled overlap frame builds for eligible rows;
+ ADR 0026 guards pass, including label-file schema/length/index checks,
  unique label keys, timestamp-grid sampling, and null-label-rate checks;
+ coverage provenance records eligible rows, hit rates, positive counts, and
  excluded BDG2-only/2017/other-meter rows;
+ no supervised accuracy metrics are reported.

### M6.2 Supervised Transfer Accuracy

Score the GEPIII-trained M3.4 ensemble on BDG2 raw overlap rows and report
ROC-AUC, PR-AUC, precision, recall, and F1 by meter. Cleaned BDG2 is reported
as a companion sensitivity.

Definition of Done:

+ raw primary metrics and cleaned companion metrics exist;
+ unknown #27 is measured as a raw-vs-source / BDG2-vs-Kaggle regime delta;
+ ADR 0025 can be accepted if review approves the evidence.

### M6.3 GBDT Vs TabPFN Supervised Comparison

Compare GBDT and TabPFN on the labeled BDG2 overlap frame.

Definition of Done:

+ accuracy and latency are reported side by side;
+ TabPFN research/internal-use license and about `6.3 ms/row` latency caveats
  remain attached;
+ the verdict mirrors M5 Phase D discipline and is based on labeled BDG2 overlap
  evidence.

### M6.4 Unlabeled Remainder

Handle BDG2-only buildings, 2017 rows, and other meters through an
explicitly-secondary pseudo-label or audit screen.

Definition of Done:

+ raw-vs-cleaned or audit outputs are labeled as pseudo-label/review evidence;
+ no unlabeled remainder rows enter supervised metrics.

### M6.5 Close-Out

Close M6 with README, plan, ADR, handoff, provenance, validation, issue, and CI
updates.

## Unknown #27

Unknown #27 remains active but changes role. It is no longer only a caveat that
travels with unlabeled score transfer. In M6.2 it becomes a measured
source-vs-target / raw-vs-cleaned regime-shift delta on the labeled overlap
subset.

The GEPIII/Kaggle source kept UTC weather timestamps and unit-conversion errors
left as-is, while BDG2 raw/cleaned uses local-time weather and corrected units.
The GEPIII-only `0.2931` correction remains outside the BDG2 path to avoid
double conversion.

## Slice Tracker

| Slice | Issue | Status | ADR |
| --- | --- | --- | --- |
| P0 docs-only supervised pivot | Not opened | In progress | ADR 0025/0026 Proposed |
| M6.1 label bridge + integrity | Not opened | Queued after P0 review | ADR 0026 |
| M6.2 supervised transfer accuracy | Not opened | Queued after M6.1 | ADR 0025 |
| M6.3 GBDT vs TabPFN supervised | Not opened | Queued after M6.2 | TBD |
| M6.4 unlabeled remainder | Not opened | Queued after M6.3 | ADR 0020 historical context |
| M6.5 milestone close-out | Not opened | Queued after M6.4 | TBD |
