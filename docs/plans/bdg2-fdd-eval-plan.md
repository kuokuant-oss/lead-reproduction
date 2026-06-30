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

## Two-Stage Scanner + Audit Re-Ranker

The design uses one evaluation frame, not a parallel TabPFN track:

```text
GEPIII-trained GBDT full-corpus scan
  -> within-context candidate generation
  -> evidence packet assembly
  -> diverse audit sample with matched random baseline
  -> TabPFN offline re-rank / few-shot calibration / disagreement review
  -> prioritized review queue
```

### Model roles

+ **GBDT primary scanner**: the GEPIII-trained detector remains the full-corpus
  scanner because Phase D kept GBDT as the real-time deployment candidate with
  sub-second inference.
+ **TabPFN second-stage tool**: TabPFN is offline, label-scarce, second-stage,
  and audit-centered. Its permitted roles are case-level re-ranking of GBDT
  top-K candidates, few-shot calibration after 50-200 reviewed cases,
  active-learning / audit-set selection, and model-disagreement diagnostics.
+ **Manual or external evidence**: higher plausibility comes only from review or
  external evidence. The BDG2 status contract still excludes `confirmed`.

TabPFN boundaries are explicit: Phase D measured about `6.3 ms/row`, roughly
`100x` slower than GBDT; the TabPFN-3.0 license is research/internal-use; and
minimal feature engineering was not a TabPFN win because raw-17-feature GBDT
ROC-AUC `0.9587` exceeded TabPFN `0.9499`. Value-change and meter-aware feature
engineering remain necessary.

### Methodological guardrails

+ Audit labels are reviewer triage judgments over `likely`, `data-quality`,
  `OOD-normal`, and `unknown`, not confirmed faults. TabPFN ranking utility can
  be evaluated only on held-out audit cases or within-audit cross-validation,
  and must be labeled triage-utility rather than accuracy or BDG2 performance.
+ A small review set can overfit the loop. Any re-ranker or calibrator must use
  a held-out review slice or within-audit cross-validation; it cannot validate
  itself on the same cases that selected or tuned it.
+ Case-level evidence features should separate data-quality/OOD candidates from
  operational candidates. Candidate features may include GBDT score,
  missingness, zero-run, flatline, raw-cleaned disagreement,
  weather-response mismatch, regime-shift, site-peer / neighbor deviation,
  OOD distance, meter type, site, primary use, and square feet. The re-ranker
  target should align to operational-anomaly candidate usefulness and must not
  merely restate distribution shift or missingness.

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
+ For electricity Level-3 `weather_response` evidence, treat unknown #25 as a
  per-site/per-meter weather-feature-validity caveat rather than an
  electricity-wide entry blocker.

## Stop Rule

Stop at this design review. Do not implement scoring, evidence-packet
generation, model transfer, a new meter scope, or a full BDG2 pipeline until
this ADR and plan are reviewed.
