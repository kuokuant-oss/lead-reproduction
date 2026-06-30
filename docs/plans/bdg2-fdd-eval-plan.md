# BDG2 FDD Audit-Yield Evaluation Plan

**Stage**: Phase E design follow-up after BDG2 EDA
**Status**: Draft for review; documentation only
**GitHub Issue**: [#41](https://github.com/kuokuant-oss/lead-reproduction/issues/41)
**Decision record**: [docs/adr/0020-bdg2-fdd-audit-yield-evaluation.md](../adr/0020-bdg2-fdd-audit-yield-evaluation.md)

## Scope

This plan turns ADR 0019's unlabeled BDG2 score-transfer contract into an
audit-yield / evidence-packet evaluation design. It does not implement scoring,
modeling, labels, feature generation, or a pipeline. It only defines the next
reviewable frame.

Guardrails:

+ Follow ADR 0017 for BDG2 schema and loader semantics.
+ Follow ADR 0018 for GEPIII-only assumption isolation.
+ Follow ADR 0019 for unlabeled transfer and BDG2-only / GEPIII-overlap
  separation.
+ Use candidate, plausibility level, packet, triage, enrichment, data-quality
  indicator, and OOD-normal language.
+ Do not report supervised BDG2 metrics, confirmed-fault%, readiness, or
  transfer-success claims.
+ GEPIII comparison remains read-only context.

## Evaluation Frame

### Candidate surfacing

Candidates must be ranked within a declared context:

+ `(site_id, meter)`
+ `(building_id, meter)`
+ held-out BDG2 site, if a later ADR or plan chooses one
+ BDG2-only and GEPIII-overlap strata, kept separate

Absolute cross-dataset top-K is forbidden because the BDG2 EDA showed reference
distribution shift large enough to contaminate score scale. The evaluation can
use rank, quantile, residual-like value, or score-like value only after the
context and eligibility frame are declared.

### Evidence levels

+ Level 1: surfaced candidate with a complete evidence packet.
+ Level 2: within-context statistical outlier evidence.
+ Level 3: weather-conditioned plausibility using BDG2 site weather; chilledwater
  has the strongest current premise because Stage 1 found median absolute
  temperature-load correlation around `0.73` and median best lag around `1`
  hour.
+ Level 4: multi-evidence convergence across at least two evidence families.

Level 5 and confirmed-fault% are explicitly out of scope. The current BDG2
release has no maintenance, BMS, work-order, or adjudicated review records in
the repo, and the multi-site archive is anonymous. A confirmed-fault headline is
therefore structurally unavailable.

### Triage enrichment

The quantitative success criterion is enrichment versus random, not supervised
accuracy:

+ top-K packets should contain independent supporting evidence more often than a
  matched random sample from the same eligibility frame;
+ the random baseline must match site, meter, available timestamps, and
  missingness/coverage eligibility as closely as possible;
+ same-site neighbor comparison is the preferred independent support;
+ weather support is useful but not fully independent if the detector also uses
  weather-lag features.

Report enrichment as evidence quality, not confirmation.

## Evidence-Packet Schema

Each packet should include:

+ `building_id`
+ `meter`
+ `site_id`
+ `interval_start`
+ `interval_end`
+ `context`
+ `within_context_score`
+ `within_context_rank`
+ `within_context_quantile`
+ `why_suspicious.missingness_context`
+ `why_suspicious.weather_response`
+ `why_suspicious.neighbor_comparison`
+ `why_suspicious.raw_visibility`
+ `interpretation`
+ `status`

Allowed `status` values:

+ `likely`
+ `data-quality`
+ `OOD-normal`
+ `unknown`

`confirmed` is not allowed for BDG2 under the current evidence contract.

## Raw-Vs-Cleaned Convergence

Raw-vs-cleaned agreement can support convergence, but not ground truth. The BDG2
cleaned release already applies data-cleaning rules including an unsupervised
detector and zero-run/electricity-zero removals. If a surfaced candidate is also
removed or changed by cleaned data, the correct interpretation is agreement
between two data-quality screens. The measurable output is a
data-quality-candidate rate, not a certified fault rate.

## Read-Only Gating Tasks

These tasks are intentionally listed but not executed in this design slice:

+ Characterize Swan chilledwater missingness time structure:
  contiguous-year blocks, seasonal concentration, and dispersed missingness.
+ Decide whether Swan has a within-site subwindow that can meet a powered pilot
  rule without relaxing the existing `missing_rate <= 0.50` gate globally.
+ Define the exact matched-random baseline for any future audit-yield run.
+ Define packet rendering and storage paths before implementation.
+ Review whether electricity can enter a later plan only after unknown #25's
  time-basis follow-up is handled.

## Stop Rule

Stop at this design review. Do not implement scoring, evidence-packet
generation, model transfer, a new meter scope, or a full BDG2 pipeline until
this ADR and plan are reviewed.
