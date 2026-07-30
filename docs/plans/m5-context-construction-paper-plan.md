# M5 137-feature context mechanism plan

**Status:** Path A closed by failed TabPFN reproducibility gate; independent query remains unscored; Path B is deferred next candidate
**Updated:** 2026-07-30
**Scope:** GEPIII building-disjoint task; 137-feature temporal value-change representation; TabPFN-3 and matched-row tree ensembles.

## Decision question

This paper is not a generic context-size curve and does not ask whether 137
features beat the 17-feature baseline. The 137-feature representation is the
research object; 17 features are retained only as a representation
counterfactual when a targeted contrast is needed.

The first decision is whether larger contexts change the ordering of scores
between **within-meter** and **cross-meter** pairs for 137-feature TabPFN and
matched-row trees. If they do, we ask whether the movement is concentrated in a
meter × label × raw-reading regime or anomaly morphology, and which mechanism
best explains it:

| Candidate mechanism | Operational evidence |
| --- | --- |
| A. Label-specific support establishes a shared score reference | Stable label-specific score/rank movement, particularly across meters, that predicts the response to positive- and negative-support allocation. |
| B. A temporal feature family changes the usability of heterogeneous support | Stable direction or context-size slope change in a targeted temporal operator contrast. |
| C. Threshold/calibration only | Effects remain at one operating threshold and do not reproduce in pairwise ranking or fixed-FPR results. This is a stopping result, not a main claim. |

## Fixed boundary

- Completed evidence is treated as fixed: GEPIII building-disjoint task,
  17-feature baseline, 137-feature temporal representation, and full-data tree
  benchmark.
- Context-size evidence uses the existing natural-prevalence full-holdout
  predictions at 5k, 10k, 20k, 50k, and 100k only. No new GPU fit is permitted
  in Phase 1.
- Trees are matched to the same context rows. The full-data tree benchmark is a
  reference, not another context-construction arm.
- Site-transfer, site-only contexts, transfer matrices, site-property stories,
  retrieval, and 500k scaling are paused. Per-site views are diagnostic only,
  for building concentration and result stability.
- The temporal features include offline future operators; conclusions therefore
  concern the defined offline detection setting, not causal/online detection.

## Phase 1 — CPU-only mechanism analysis

The reproducible entry point is
`scripts/analyze_m5_137_context_mechanism.py`. It reads existing score files,
never imports or fits TabPFN, and writes row/cell/segment artifacts under
`data/processed/m5_context_mechanism_137/`.

For each model and context size, the analysis will produce:

1. paired row-level score movement from 5k to every larger context;
2. global and within-meter percentile-rank movement;
3. a within-meter/cross-meter pairwise-AUC decomposition;
4. meter × label × raw-reading-regime score distributions;
5. fixed-FPR recall, fixed-TPR FPR, and endpoint threshold crossings;
6. influential-building leave-one-building summaries and building-clustered
   bootstrap intervals; and
7. anomaly segments, formed by merging consecutive hourly anomaly rows within a
   building × meter stream. Segment outputs include duration, onset/middle/
   recovery phase, reading level and slope, 24h/168h deviations, diff/ratio
   statistics, context-size rank movement, and TabPFN/tree disagreement.

Priority diagnostic cuts are hotwater 0–1 reading behaviour, steam at 100k or
larger context, electricity misses newly introduced at larger contexts, and
non-monotone chilledwater movement. They are pre-specified diagnostic cuts, not
independent paper stories.

### Phase-1 decision gate

Use the convergence table in
`docs/reports/m5-137-context-mechanism-analysis.md` after the CPU run.

- Continue to **Path A** only if label-specific score/rank movement is stable
  across pairwise ranking, fixed-FPR metrics, and clustered uncertainty, and is
  not concentrated in one building or a few segments.
- Continue to **Path B** only if movement instead aligns with temporal-operator
  direction/family and a representation contrast yields a stable sign or slope
  change.
- Stop the mechanism claim if the pre-specified stopping conditions hold.

## Completed Phase 1: CPU-only mechanism analysis

`scripts/analyze_m5_137_context_mechanism.py` completed without a new model
fit. The full-holdout evidence establishes that, from 5k to 100k, both TabPFN
and matched-row trees improve in within-meter/cross-meter AUC and
recall@global-FPR=0.001. Anomaly within-meter rank rises in all four meters,
while normal movement is near zero. Hotwater 0–1 absolute scores decline for
both labels but more strongly for normal rows; steam-positive ×
hotwater-negative is the largest TabPFN cross-meter AUC movement; and
leave-one-building diagnostics do not show a single-building explanation.

These are selection criteria for a factorial intervention, **not** evidence
that positive or negative hotwater support has a causal role. The detailed
artifact inventory, numbers, and convergence table remain in
`docs/reports/m5-137-context-mechanism-analysis.md`.

## Exactly one active fit path: Path A label-role factorial

The only active experiment is a hotwater positive-support × negative-support
2×2 factorial. Site transfer, site-only contexts, retrieval, full transfer
matrices, 500k scaling, and representation ablation remain paused/deferred.

### Fixed first-round contract

- F4 only; N=20k; exact 50/50 labels; existing fixed screening-query artifact;
  no full holdout in this round.
- Context-draw seeds are 42, 123, and 999. Model seed is a separate fixed 42.
- Each seed has a pooled-reference context and the four cells `pos-present /
  neg-present`, `pos-present / neg-excluded`, `pos-excluded / neg-present`,
  and `pos-excluded / neg-excluded`.
- Excluded hotwater slots are replaced one-for-one by unique same-label,
  non-hotwater reserve rows. Total N, label count, slot order, and the
  conditional mix of other meters are audited. The preparation writes ordered
  raw-index digests, replacement maps, cell-overlap tables, and composition
  audits.
- TabPFN runs both a cell-specific StandardScaler and a per-seed frozen scaler
  fitted only on that seed's pooled-reference context. Trees use exactly the
  same F4 rows and the canonical tree pipeline; first run a scaler-invariance
  pilot, then reuse the canonical tree arm only if its predictions meet the
  predeclared tolerance.

`scripts/prepare_m5_hotwater_label_role_factorial.py` is the deterministic
pre-fit builder. Its outputs live under
`data/processed/m5_hotwater_label_factorial/`. All three seeds, replacement
maps, overlap/composition audits, two TabPFN scaler arms, and two tree scaler
arms completed. The tree scaler-invariance pilot failed (maximum absolute score
difference 0.163691 versus 1e-6), so both tree arms were retained.

### Pre-specified estimands

Primary estimands are: hotwater 0–1 anomaly-vs-normal within-meter rank gap and
pairwise AUC; steam-positive × hotwater-negative AUC; and global recall at
FPR=0.001. Secondary readouts are hotwater score/rank main effects, hotwater
anomaly segment score/rank/recall, four meter×label global/within-meter rank
movements, the complete cross-meter AUC matrix, and TabPFN/tree response
differences.

For every metric estimate positive-support and negative-support main effects,
their interaction, scaler-arm interaction, seed-level direction consistency,
and building- and segment-clustered uncertainty.

### Hypotheses and expansion gate

The hypotheses are that negative hotwater support lowers the relative scores of
low-reading hotwater normals and improves steam-positive versus
hotwater-negative ordering; positive support preserves hotwater-anomaly
morphology; the 2×2 interaction tests complementarity; and frozen scaling
tests preprocessing geometry.

The post-fit audit found that the original query has only three HW 0–1 negative
rows and that 176 normal rows cannot resolve FPR=0.001 (minimum resolution
0.00568). Local rank/AUC and low-FPR recall are therefore insufficient for an
expansion decision. A 192-row independent query with disjoint buildings and
cluster caps has been frozen. A complete recovery census found no factorial
TabPFN fitted state or tree booster model compatible with its manifest digest;
unrelated M6/smoke states and score-only artifacts are not reusable.

The authorized exact-design recovery refit completed: it rebuilt the original
12 manifests (three seeds × four cells), both scaler arms, and both learners
into a new recovery root while saving fitted states, all tree boosters, scalers,
environment provenance, and a query-independent scoring entry point. Its
original-352-row gate passed for trees but failed for TabPFN on the predeclared
score/rank/estimand/effect tolerances. No-fit state reload verification then
showed that tree reload is bit-exact but TabPFN portable-state inference itself
is variable. Path A is therefore **closed** without scoring the independent
query. There is no active model fit in this round: Path B is recorded only as
the deferred next candidate and must not start without a new instruction.
N={5k,100k}, full holdout, and more seeds remain prohibited. See the
artifact-recovery, reproduction, factorial, bootstrap, query-replication, and
tree-scaler reports.

## Deferred alternative

### Path B — targeted temporal representation contrasts

Run past-diff, past-ratio, future-diff, future-ratio, all-diff, all-ratio,
full-137, and full-137-without-meter. Start at N={5k, 100k}; add 20k and
multiple seeds only after a stable sign or slope change appears.

## Stopping conditions

Do not promote a mechanism claim when any required evidence fails because:

- the effect exists only at the 0.5 threshold;
- pairwise ranking and fixed-FPR results do not change stably;
- one building or very few anomaly segments dominate it;
- context seeds disagree in direction; or
- no intervention response can be predicted from the proposed mechanism.

## Submission completion thresholds

### Q2 minimum path

- one clear empirical finding;
- one intervention or representation contrast supporting the mechanism;
- at least three context seeds;
- building- and segment-clustered uncertainty;
- full-holdout and natural-prevalence confirmation.

### Q1 extension path

In addition to the Q2 path, develop the mechanism into a reusable
support-allocation diagnostic, cross-domain alignment metric, or context
selector, and confirm it with a second modern tabular learner or a second
usable dataset construction.
