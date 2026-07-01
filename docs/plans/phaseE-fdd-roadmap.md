# Phase E to M6 FDD roadmap

## Purpose And Status

Authoritative roadmap for clearing Phase E baggage (Part A) and implementing
FDD on BDG2 (M6). Living document; each slice updates its row.

Current status: A1 is done ([#42](https://github.com/kuokuant-oss/lead-reproduction/issues/42),
[ADR 0021](../adr/0021-powered-gate-as-transfer-confidence.md)) and A2 is done
([#44](https://github.com/kuokuant-oss/lead-reproduction/issues/44),
[ADR 0022](../adr/0022-electricity-entry-meter-for-bdg2-fdd.md)). A4 is done
([#45](https://github.com/kuokuant-oss/lead-reproduction/issues/45),
[ADR 0023](../adr/0023-raw-first-bdg2-transfer-scoring.md)). A5 is done
([#48](https://github.com/kuokuant-oss/lead-reproduction/issues/48),
[ADR 0024](../adr/0024-value-change-regime-convergence.md)). A3 is done
([#49](https://github.com/kuokuant-oss/lead-reproduction/issues/49)). Part A is
complete. The queued M6 comparison redesign and M6 remain queued and must run as
separate issue/commit/review slices.

Part A progress: A1, A2, A4, A5, and A3 are done. The next queued item is the
M6 comparison redesign, which is not started and must run before any M6
implementation.

This document is the single source of truth for the Phase E to M6 arc. It
archives the fixed constraints, Part A cleanup sequence, BDG2-paper-derived
folds, unknown #27, and M6 implementation ladder so the through-line
survives context loss.

## Fixed Constraints

These govern every slice:

+ GEPIII-trained-detector to BDG2 transfer paradigm is FIXED (ADR 0019). Not
  removed, not reframed to BDG2-native detection.
+ TabPFN is a RESEARCH FOCUS. Never deleted, deprioritized, or narrowed; its ADR
  0020 audit roles (re-rank, few-shot calibration, active-learning audit-set
  selection, disagreement diagnostics) carry forward.
+ M3 numeric line FROZEN: `load_m3_frame` defaults, M3.2/M3.4 golden values, the
  +/- `0.0005` gate, `downsample_indices` shape, and StandardScaler fit path.
+ `lead.__all__` FROZEN unless additive plus ADR plus `test_public_api.py`.
+ Honest contract: no BDG2 supervised AUC/PR-AUC/precision/recall/F1 as ground
  truth; no `confirmed` status; no absolute cross-dataset top-K; BDG2-only vs
  GEPIII-overlap always separated; TabPFN license and about `6.3 ms/row`
  latency caveats intact.
+ Process: one slice = one issue = one commit = stop for review; full
  change-checklist per slice; throwaway spikes may live in `.scratch/`.

## Part A Slices

### A1. Demote Powered Gate To Reporting Confidence

Scope: demote the powered-headline gate from a blocking entry gate to an
after-the-fact reporting-confidence dimension. The prior 5-building / 17,544-row
threshold remains measured as `multi_building_transfer_stability`, but
single-building BDG2-only sufficient-observation evidence can proceed to the
within-context packet path.

BDG2 paper fold: none specific to A1. This slice resolves the ADR 0019 powered
entry clause vs ADR 0020 within-context packet conflict.

Status: DONE.

Issue: [#42](https://github.com/kuokuant-oss/lead-reproduction/issues/42).

ADR: [ADR 0021](../adr/0021-powered-gate-as-transfer-confidence.md), which
supersedes only ADR 0019's powered-entry-gate clause. The transfer paradigm,
BDG2-only / GEPIII-overlap separation, no-supervised-metric rule, and no
absolute cross-dataset top-K rule remain in force.

A1 does not violate the transfer paradigm, does not touch the M3 numeric line,
and preserves and carries TabPFN forward.

### A2. Make Electricity The Entry Meter

Scope: make electricity the entry meter for the first transfer/FDD scoring path.
Chilledwater support is not deleted; it becomes a later Level-3
weather-conditioned path.

BDG2 paper fold: Miller et al. 2020 Fig 5 shows electricity weather sensitivity
is heterogeneous. ADR 0020 Level-3 `weather_response` evidence is therefore
partially usable per electricity meter. Unknown #25 stays a per-site/per-meter
caveat, not an electricity-wide Level-3 disqualifier.

Status: DONE.

Issue: [#44](https://github.com/kuokuant-oss/lead-reproduction/issues/44).

ADR: [ADR 0022](../adr/0022-electricity-entry-meter-for-bdg2-fdd.md).

A2 does not violate the transfer paradigm, does not touch the M3 numeric line,
and preserves and carries TabPFN forward.

### A4. Make Transfer/FDD Scoring Raw-First

Scope: make the transfer/FDD scoring path explicitly raw-first through a wrapper
or required argument. Do not flip `load_bdg2_frame`'s general default, and do
not break `test_bdg2_loader.py`. Cleaned data remains a sensitivity and
convergence companion.

BDG2 paper fold: this is a correctness precondition for M6.1. Miller et al.
2020, and the tracked repo note in
`docs/reference/papers/bdg2-miller-2020.md`, confirm that the cleaned BDG2
release applies Twitter AnomalyDetection, removes zero-reading runs longer than
24 hours, and removes electricity zeros. Scoring electricity FDD on cleaned data
removes the very candidates FDD targets.

A4 also opens unknown #27, recorded below.

Status: DONE.

Issue: [#45](https://github.com/kuokuant-oss/lead-reproduction/issues/45).

ADR: [ADR 0023](../adr/0023-raw-first-bdg2-transfer-scoring.md).

A4 does not violate the transfer paradigm, does not touch the M3 numeric line,
and preserves and carries TabPFN forward.

### A5. Decide Value-Change Regime Convergence

Scope: decide and document source/target value-change regime convergence before
multi-meter transfer expands. Preferred path: source and target score through a
meter-aware-equivalent value-change path. Alternative: commit to
one-detector-per-meter. Either choice must preserve the frozen M3 numeric line.

BDG2 paper fold: none specific to A5. The decision exists because the current
GEPIII source detector trains under `row_offset`, while BDG2 scoring uses
`row_offset_meter_aware`, justified only by single-meter equivalence.

Status: DONE.

Issue: [#48](https://github.com/kuokuant-oss/lead-reproduction/issues/48).

ADR: [ADR 0024](../adr/0024-value-change-regime-convergence.md).

Decision: choose the additive meter-aware-equivalent path. Future multi-meter
transfer must train and score through matching value-change semantics, using an
opt-in meter-aware source path paired with BDG2 target scoring through the same
regime. `row_offset` remains the M3 default and the M3 numeric line is not
moved. The one-detector-per-meter alternative remains a fallback only if later
implementation proves the additive path cannot be wired without violating the
frozen M3 line.

Deferred: the exact multi-meter source-training helper or script flag,
multi-meter detector artifacts, provenance schema, and any fallback decision for
a concrete later run. M6.1 single-meter electricity is unaffected because Phase E
Step 1 already proved single-meter equivalence.

A5 does not violate the transfer paradigm, does not touch the M3 numeric line,
and preserves and carries TabPFN forward.

### A3. Downgrade Swan Chilledwater Structural-Missingness Gating

Scope: move Swan chilledwater structural-missingness contiguity analysis from a
blocking gating task to optional future chilledwater work. Do not delete the
idea; remove it from the critical path once A1/A2 make it non-blocking.

BDG2 paper fold: none specific to A3. This cleanup follows from A1 and A2:
Swan contiguity only fed the old powered chilledwater gate.

Status: DONE.

Issue: [#49](https://github.com/kuokuant-oss/lead-reproduction/issues/49).

ADR: no new ADR. This is plan housekeeping that follows
[ADR 0021](../adr/0021-powered-gate-as-transfer-confidence.md) and
[ADR 0022](../adr/0022-electricity-entry-meter-for-bdg2-fdd.md).

Result: Swan chilledwater structural-missingness contiguity is optional future
chilledwater work, not a blocking gate. The idea and the EDA finding remain
available for a later Level-3 weather-conditioned chilledwater slice.

A3 does not violate the transfer paradigm, does not touch the M3 numeric line,
and preserves and carries TabPFN forward.

## Queued M6 Comparison Redesign

Status: QUEUED after Part A; do not start until A3 is approved by review.

Issue: [#47](https://github.com/kuokuant-oss/lead-reproduction/issues/47).

Scope: docs-only redesign of the M6 ladder plus a new M6-comparison ADR. This
future slice should compare GBDT and TabPFN on BDG2 instead of pre-assigning
roles from the M5 Phase D GEPIII result. The redesign must preserve independent
experiment phases: a run may go deep on one experiment while keeping others
light or skipped, and each phase needs its own ground-truth or comparison basis
and definition of done.

Queued design constraints for the future slice:

+ Forecasting RMSE/MAE is a forecasting metric, not FDD accuracy. Forecasting
  error is not a fault, though residuals may seed review candidates.
+ Synthetic-injection precision/recall/PR-AUC must be labeled as measured on
  injected synthetic anomalies only, not generalized to real BDG2 faults.
+ Human-audit labels remain triage judgments:
  `actionable_candidate`, `data_quality`, `ood_normal`, `not_interesting`, and
  `unknown`. No `confirmed` fault status is allowed.
+ No absolute cross-dataset top-K. Within-context ranking, BDG2-only versus
  GEPIII-overlap separation, raw-first scoring, and unknown #25/#27 caveats
  remain in force.
+ TabPFN remains a research focus and is not pre-assigned to auditor-only in the
  redesign. BDG2 evidence should decide whether GBDT wins, TabPFN wins, or a
  combined workflow is best. TabPFN license and about `6.3 ms/row` latency
  caveats stay in runtime comparisons.
+ The M3 numeric line and `lead.__all__` remain frozen unless a later slice
  explicitly satisfies the additive-plus-ADR-plus-test requirement.

Planned independent comparison phases for the future redesign:

+ M6.1 forecasting benchmark on BDG2 raw electricity with actual future meter
  reading as ground truth, comparing GBDT, TabPFN, and persistence baseline.
+ M6.2 synthetic anomaly injection benchmark on injected spike, flatline, drift,
  and extended-zero-run cases, reported as injected-set metrics only.
+ M6.3 union candidate pool plus shared event-packet comparison across GBDT
  top-K, TabPFN top-K, forecast residuals, raw-vs-cleaned disagreement,
  zero-run/missingness candidates, optional injected anomalies, and a stratified
  random baseline.
+ M6.4 human audit top-K comparison across GBDT-only, TabPFN-only, both, random,
  residual-only, and combined rankings, reported as triage utility rather than
  accuracy.

This queueing note does not add the ADR or change the current M6 ladder. The
actual comparison ADR and M6 ladder rewrite belong to the queued slice after
A3 approval and Part A review.

## Unknown #27

Unknown #27 is open in [unknowns.md](../reference/unknowns.md).

Question: how should Phase E/M6 carry the source-vs-target regime shift baked
into the GEPIII-to-BDG2 transfer paradigm?

The GEPIII/Kaggle source the detector trained on kept UTC weather timestamps and
unit-conversion errors left as-is, while BDG2 raw/cleaned uses local-time
weather and corrected units (Miller et al. 2020 Usage Notes). This does not
block M6 because outputs are within-context ranks/quantiles, not absolute risk,
and ADR 0019 forbids absolute-score risk claims. It must travel as a caveat with
every M6 transfer output.

The GEPIII-only `0.2931` correction (= Miller et al. Table 5 kBTU to kWh factor
`0.293071`) stays out of the BDG2 path to avoid double conversion.

## M6 Phase Ladder

M6 success means the repo produces a usable BDG2 anomaly review queue an
operator can open, plus triage value quantified. M6 is not another gate or
report-only milestone.

### M6.1. Full-Corpus Electricity Scan

Scope: run the GEPIII-trained GBDT detector on BDG2 electricity raw, per-site
and chunked. Store within-`(building, meter)` score rank/quantile, with BDG2-only
and GEPIII-overlap separated, under
`data/processed/m6_1_electricity_scan/`.

Dependency: A4 has landed. M6.1 is raw-only because cleaned electricity removes
zero readings and can remove FDD candidates. M6.1 is not opened or run by A4.

Definition of Done: at least one BDG2 site's electricity is fully scanned; the
result is non-quarantined; scan summary and provenance are written under
`data/processed/m6_1_electricity_scan/`; no BDG2 supervised metrics are
reported.

Folds: raw-only correctness from Miller et al. cleaned-release rules; within-
context ranks/quantiles only; BDG2-only and GEPIII-overlap separated.

### M6.2. Evidence Packets And Human-Readable Review Queue

Scope: implement ADR 0020's evidence-packet schema and produce real electricity
packets plus the human-readable review queue.

Dependency: M6.1 scan output.

Definition of Done: packets include `building_id`, `meter`, `site_id`,
interval, within-context score/rank/quantile, `why_suspicious` fields,
interpretation, and status in `{likely, data-quality, OOD-normal, unknown}`.
`confirmed` is not allowed.

Folds: zero-run logic is meter-type-aware. Heating, cooling, and irrigation
zeros lean `OOD-normal` unless other evidence contradicts; electricity zero-runs
lean `data-quality` or `likely` because the cleaned release specifically removes
electricity zeros. Optional `regime-shift` evidence may cite the paper's Twitter
BreakoutDetection / steady-state >=168h precedent if that field is implemented.

### M6.3. Enrichment-Vs-Random

Scope: quantify triage value by comparing top-K packet support rate against a
matched random baseline.

Dependency: M6.2 packets.

Definition of Done: report the share of top-K packets with independent
supporting evidence versus a matched random sample aligned on site, meter,
time coverage, and missingness/coverage eligibility. Prefer same-site neighbor
deviation as support. Report triage utility, not supervised accuracy.

Folds: no BDG2 supervised metrics; same-site neighbor support is preferred
because weather evidence may overlap with detector weather-lag features.

### M6.4. Raw-Vs-Cleaned Convergence

Scope: measure whether surfaced raw candidates are also removed or changed by
the cleaned BDG2 release.

Dependency: M6.2 packets and raw/cleaned companion access from A4.

Definition of Done: append to the M6.3 report a data-quality-candidate rate.
Explicitly state that this is not a fault rate.

Folds: raw-vs-cleaned convergence means agreement between GEPIII-GBDT
within-context surfacing and the BDG2 cleaned-release Twitter AnomalyDetection
plus zero-run/electricity-zero screen. It is agreement between data-quality
screens, not ground truth.

### M6.5. TabPFN Audit Re-Ranker

Scope: run TabPFN on real GBDT candidate cases for case-level re-ranking and
model-disagreement diagnostics.

Dependency: M6.2 candidate packets. Default plan includes a 50-200-case human
audit set; if unavailable at run time, deliver structural re-ranking plus
disagreement diagnostics and mark triage utility as `pending audit set`.

Definition of Done: TabPFN produces a re-rank/disagreement artifact on real GBDT
candidates. With a human audit set, evaluate triage utility through held-out
audit cases or within-audit cross-validation. Without the audit set, no
triage-utility claim is made.

Folds: TabPFN remains research/internal-use, offline, and audit-centered. It
keeps license and about `6.3 ms/row` latency caveats. It is not the BDG2
full-corpus scanner.

### M6.6. Milestone Close-Out

Scope: close M6 as the first usable BDG2 anomaly review queue.

Dependency: M6.1-M6.5.

Definition of Done: update README, `docs/plans/m6-plan.md`, relevant ADRs, and a
handoff. Position M6 as Phase E crossing from design into delivery.

Folds: M6 does not report BDG2 supervised metrics; does not claim `confirmed`;
does not use absolute cross-dataset top-K; does not fabricate labels; does not
touch the M3 numeric line. Chilledwater, other meters, and Level-3
weather-conditioned evidence are deferred to M7.

## Slice Tracker

| Slice | Issue | Status | ADR |
| --- | --- | --- | --- |
| Slice 0: archive Part A + M6 roadmap | [#43](https://github.com/kuokuant-oss/lead-reproduction/issues/43) | Done | n/a |
| A1: powered gate to confidence | [#42](https://github.com/kuokuant-oss/lead-reproduction/issues/42) | Done | [ADR 0021](../adr/0021-powered-gate-as-transfer-confidence.md) |
| A2: electricity entry meter | [#44](https://github.com/kuokuant-oss/lead-reproduction/issues/44) | Done | [ADR 0022](../adr/0022-electricity-entry-meter-for-bdg2-fdd.md) |
| A4: raw-first transfer/FDD scoring | [#45](https://github.com/kuokuant-oss/lead-reproduction/issues/45) | Done | [ADR 0023](../adr/0023-raw-first-bdg2-transfer-scoring.md) |
| A5: value-change regime convergence | [#48](https://github.com/kuokuant-oss/lead-reproduction/issues/48) | Done | [ADR 0024](../adr/0024-value-change-regime-convergence.md) |
| A3: Swan chilledwater off critical path | [#49](https://github.com/kuokuant-oss/lead-reproduction/issues/49) | Done | n/a |
| M6 comparison redesign | [#47](https://github.com/kuokuant-oss/lead-reproduction/issues/47) | Next queued after A3 approval; not started | To be added by redesign slice |
| M6.1: full-corpus electricity scan | Not opened | Raw-first precondition satisfied; not opened | To be decided |
| M6.2: evidence packets + review queue | Not opened | Queued after M6.1 | ADR 0020 plus possible follow-up |
| M6.3: enrichment-vs-random | Not opened | Queued after M6.2 | To be decided |
| M6.4: raw-vs-cleaned convergence | Not opened | Queued after M6.2/A4 | To be decided |
| M6.5: TabPFN audit re-ranker | Not opened | Queued after M6.2 | ADR 0020 plus possible follow-up |
| M6.6: milestone close-out | Not opened | Queued after M6.1-M6.5 | To be decided |
