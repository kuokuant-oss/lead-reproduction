# BDG2 FDD audit-yield evaluation framework

## Status

Proposed (2026-06-30)

## Context

ADR 0017 defines the real BDG2 loader contract: string building ids, BDG2 meter
names, site weather joined on `(site_id, timestamp)`, preserved GEPIII-overlap
fields, and no native per-row labels. ADR 0018 isolates GEPIII-only assumptions
before BDG2 use. ADR 0019 selects GEPIII-trained detector transfer to BDG2 as an
unlabeled cross-dataset baseline and forbids supervised BDG2 accuracy claims
without an external label source.

The BDG2 EDA in `docs/reports/bdg2-eda.md` adds two constraints for the next
evaluation design:

+ BDG2-only chilledwater is knife-edge gate-sensitive. At `missing_rate <= 0.50`
  only 3 BDG2-only buildings are sufficient; at `0.55` the count rises to 24,
  mainly because Swan has roughly 20 BDG2-only chilledwater columns clustered
  just above the `0.50` cut point.
+ Chilledwater reading magnitudes are not the main distance blocker: the
  per-meter chilledwater raw KS is `0.1177`, and log1p-zero-excluded KS is
  `0.06005`, the lowest per-meter distance in the EDA table.

The next design question is therefore not "which supervised BDG2 metric should
we report?" BDG2 has no native work-order, maintenance, BMS, or per-row review
record in this repo. The question is how to turn unlabeled score-transfer into
reviewable evidence packets and a measurable audit-yield frame without treating
BDG2 as labeled.

## Decision

Operationalize ADR 0019 as an **audit-yield / evidence-packet workflow**, not a
supervised metric workflow.

The primary BDG2 output is a ranked set of review candidates with evidence
packets. A candidate can be prioritized by within-context score position, but
it is not confirmed by the score. The workflow measures whether top-ranked
candidates are enriched for independent supporting evidence relative to random
sampling from the same eligible context.

### Within-context scoring

Candidate surfacing must be within context:

+ rank or quantile within `(site, meter)`, `(building, meter)`, or another
  predeclared comparable context;
+ residual-like or score-like quantities may be used only after the context,
  eligible rows, and missingness treatment are declared;
+ absolute cross-dataset top-K is prohibited because the EDA shows BDG2-vs-GEPIII
  distribution shift can dominate score scale.

GEPIII-overlap and BDG2-only packets remain separated. GEPIII-overlap packets
can serve as bridge/calibration context, but not as unqualified cross-dataset
evidence.

### Evidence levels

Use four attainable evidence levels:

+ **Level 1: surfaced candidate.** A row, interval, or building-meter segment is
  selected by the within-context rule and has a complete packet.
+ **Level 2: statistical outlier evidence.** The candidate is high-ranked within
  its declared context or has an unusual local residual/rank pattern.
+ **Level 3: weather-conditioned plausibility.** The candidate is inconsistent
  with a declared weather-conditioned expectation, using BDG2's own site weather
  where applicable. The Stage 1 chilledwater time-basis diagnostic supports this
  path for chilledwater: median absolute temperature-load correlation was about
  `0.73`, with median best lag about `1` hour.
+ **Level 4: multi-evidence convergence.** At least two evidence families agree,
  such as within-context rank, weather-conditioned behavior, same-site neighbor
  comparison, raw-vs-cleaned convergence, or missingness/coverage context.

Do not define Level 5 or report confirmed-fault%. In the current BDG2 release
structure, confirmed-fault% is not reachable because the repo has no maintenance,
BMS, work-order, or adjudicated review records, and BDG2 sites/buildings are
anonymous across multiple owners. Reporting Level 5 or confirmed-fault% would
overstate the evidence.

### Triage enrichment

The quantitative success measure is **triage enrichment versus random**:

+ sample top-K packets from the within-context ranking;
+ sample a matched random baseline from the same site/meter/building eligibility
  frame and time coverage;
+ compare the share of packets with independent supporting evidence.

This is enrichment evidence, not confirmation. Independence is imperfect when
weather evidence overlaps with detector weather-lag features. The strongest
available independent support is same-site neighbor comparison because it uses
nearby buildings under the same site/weather regime without relying on the same
building's score.

### Raw-vs-cleaned convergence

Raw-vs-cleaned agreement is a convergence signal, not ground truth.

BDG2 cleaned files are themselves produced by a release cleaning process that
includes an unsupervised detector and zero-run/electricity-zero rules. If a
candidate is also removed or changed by cleaned data, that is agreement between
two data-quality screens. It may support a data-quality-candidate rate, but it
must not be described as fault certification or supervised accuracy.

### Evidence-packet schema

Each packet should contain:

+ `building_id`
+ `meter`
+ `site_id`
+ interval start and end timestamps
+ within-context score, rank, or quantile
+ why-suspicious fields:
  `missingness_context`, `weather_response`, `neighbor_comparison`,
  `raw_visibility`, and any declared context notes
+ interpretation
+ status

Allowed status values are:

+ `likely`
+ `data-quality`
+ `OOD-normal`
+ `unknown`

`confirmed` is not an allowed BDG2 status under the current evidence contract.

## Consequences

+ BDG2 FDD evaluation remains aligned with ADR 0017, ADR 0018, and ADR 0019.
+ The next implementation slice, if approved, should produce review packets and
  enrichment summaries, not BDG2 supervised AUC/PR-AUC/precision/recall/F1.
+ Any future score calculation must first declare its context and random
  baseline, then report BDG2-only and GEPIII-overlap separately.
+ Swan chilledwater becomes a read-only gating question before any pilot:
  characterize whether its roughly half-missing coverage is contiguous enough
  for a within-Swan powered pilot.
+ The plan remains documentation-only until reviewed; no scoring, modeling, or
  pipeline implementation is authorized by this ADR.
