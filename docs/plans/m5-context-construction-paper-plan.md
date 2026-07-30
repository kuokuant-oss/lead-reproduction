# M5 137-feature context mechanism plan

**Updated:** 2026-07-30
**Status:** Path A is retained as a measurement-aware composition experiment.
No model execution is authorized by this document update. The 192-row query is
frozen and unscored; Path B and all other deferred work remain paused.

## Research question and fixed boundary

This study concerns the 137-feature temporal value-change representation in the
GEPIII building-disjoint task. It is not a generic TabPFN context-size curve or
a claim that 137 features outperform 17; the 17-feature representation is only
a targeted counterfactual if later needed.

The completed CPU-only analysis of existing 5k/10k/20k/50k/100k full-holdout
predictions remains fixed Phase-1 evidence: TabPFN and matched-row trees both
improve within- and cross-meter AUC and recall@global-FPR=.001 from 5k to 100k;
anomaly within-meter rank rises in all meters while normal movement is near
zero; hotwater 0–1 normal scores fall more than anomaly scores; and the largest
TabPFN cross-meter movement is steam-positive versus hotwater-negative.
Building-influence diagnostics do not support a single-building explanation.
These results select a composition intervention; they do not establish the
roles of positive or negative support.

Site transfer, site-only contexts, transfer matrices, site properties,
retrieval, 500k scaling, full holdout refits, and representation ablations are
paused. Per-site slices are diagnostic only. Trees remain the matched-row
comparator; the completed tree context curves are retained Phase-1 evidence.

## Fixed scientific version policy

> **TabPFN 8.0.8 is the fixed scientific version for this study. TabPFN 8.1.0
> was used only as an isolated live-repeat diagnostic and is not used for
> factorial estimation or model comparison.**

All existing context-size results, Path-A screening/recovery artifacts, and
matched-tree comparisons use 8.0.8. The isolated 8.1.0 v2c run reused the
8.0.8 frozen arrays/scaler and demonstrated live-repeat variation of comparable
order; it is neither a version replacement nor formal factorial evidence.
Its retained R1–R3 artifacts belong only in the execution-audit appendix.

## Path A: repeated-inference estimand

Path A asks whether allocating hotwater positive and negative support changes
score ordering. For factorial cell \(c\), context seed \(s\), scaler arm \(a\),
and repeated inference from one fitted state \(r\), analyse

\[
Y[c,s,a,r] = \mu[c,s,a] + \epsilon[c,s,a,r].
\]

`mu` is the training-composition estimand; `epsilon` is observed
repeat-inference measurement variation. Bit-identical probabilities are not a
scientific eligibility criterion. Estimate cell means and the positive-support,
negative-support, positive×negative, and scaler-arm contrasts while retaining
the distribution over inference replicates. The interaction is
`(cell11 - cell10) - (cell01 - cell00)`.

Do not pair replicate IDs between cells unless a controlled shared random-state
mechanism is established. Use cell-level replicate distributions with
independent-replicate variance propagation or a hierarchical/cluster-bootstrap
analysis; context seed is a higher-level source of variation and inference
replicate is nested under its fitted state. Never average row probabilities
before calculating AUC, choose favourable calls, call inference calls new fits,
or ignore within-execution dependence. Existing tree states are bit-exact and
receive no artificial replicate noise.

## Next authorized design: 8.0.8 variance pilot (not running now)

The next model work, only after separate authorization, is a 8.0.8 repeated-
inference variance pilot. It is not a 24-cell recovery, independent-query
replication, Path B fit, tree refit, or context-curve run.

| Fixed item | Contract |
| --- | --- |
| Context/model | v3 checkpoint; F4 137 features; N=20,000; context seed 42; model seed 42; cell-specific scaler |
| Cells | `11` pos/neg present, `10` pos present/neg excluded, `01` pos excluded/neg present, `00` both excluded |
| Inputs | Original 352-row screening query, original ordered manifests, replacement maps, row/scaler/checkpoint digests, and TabPFN constructor |
| Fits | One 8.0.8 fit and persisted state per cell; no rebuild of F4 arrays |
| Main replicates | Same-process predictions from each fitted state; all cells are interleaved by one pre-generated, seeded randomized schedule |
| Diagnostic only | A limited fresh-process saved-state comparison, stored separately; it is not the main variance distribution |
| Prohibited | 192-row query, 24-cell grid, Path B, new version test, tree fit, full holdout, or context-size work |

Each replicate must preserve row probabilities, available per-estimator
probabilities/raw logits, global and within-meter ranks, raw identities,
process/device metadata, replicate index, randomized schedule position,
timestamps, and state/query/scaler/checkpoint digests. Per-replicate readouts
are HW 0–1 anomaly-vs-normal AUC and rank gap, steam-positive vs
hotwater-negative AUC, continuous steam-minus-HW-negative and HW-anomaly-minus-
normal margins, plus relevant segment summaries.

### Predeclared pilot sizing and stops

The pilot starts with **8 completed same-process replicates per cell**, executed
in complete four-cell schedule blocks, and may use at most **40 replicates per
cell**. After each complete block from the initial batch onward, use a 95% t
interval over a cell's inference replicates. A measurement target is met when
the half-width is at most **0.015** for every bounded AUC/rank primary readout
and at most **0.02 × the frozen fit-time pooled score IQR** for each continuous
margin. The IQR is recorded before repeats begin and is never re-estimated for
this threshold.

Stop early for a *clear measurable composition signal* only when the target is
met, at least one pre-specified factorial contrast has a 95% interval excluding
zero, its absolute contrast exceeds twice its Monte-Carlo standard error, and
its sign is unchanged in the two most recent complete schedule blocks. Stop at
the cap for *infeasible precision* when the target remains unmet for two or
more primary readouts, or when both ranking and continuous-margin contrasts are
indistinguishable from within-cell inference variation at the cap. These are
pilot feasibility decisions, not a scientific finding from one seed.

## Pilot decision paths

1. **Composition measurable:** freeze the 8.0.8 repeated-inference protocol,
   then plan the formal three-context-seed × four-cell × two-scaler-arm Path A
   study with replicate-aware clustered uncertainty.
2. **Local HW unreliable but cross-meter/margin readout measurable:** retain
   local HW as screening/diagnostic, make the pre-specified steam readout the
   primary candidate, and consider the 192-row independent query only after
   the full protocol is frozen.
3. **Measurement infeasibility:** only if within-cell variation is comparable
   to composition separation, the cap cannot meet precision, and both ranking
   and continuous margins fail. Report TabPFN+query measurement infeasibility
   separately; tree exactness does not prove a TabPFN mechanism.

The frozen 192-row query cannot choose pilot endpoints, directions, replicate
counts, or stopping rules. It remains unscored until the full repeated-
inference protocol is frozen.

## Deterministic-execution audit: engineering evidence only

The audit found live-repeat variation before saving under both 8.0.8
`low_memory` and 8.0.8 `fit_preprocessors`; same- and fresh-process reloads do
not remove it. The 8.1.0 diagnostic R1–R3 shows the same issue rather than a
direct version remedy. The earlier MAE/max/Spearman thresholds are retained as
engineering diagnostics only; none is a scientific gate for Path A, a reason
to block the 192-row query indefinitely, or a criterion for abandoning TabPFN.

## Mechanism alternatives and stopping conditions

Path B (past/future diff/ratio targeted contrasts) is deferred and must not be
started in this round. It becomes the next candidate only after Path A's
repeated-inference pilot makes one of the three decisions above. Do not promote
a mechanism claim if it exists only at a .5 threshold, lacks stable pairwise or
fixed-FPR support where resolution permits, is driven by one building/few
segments, reverses across context seeds, or has no predictable intervention
response.

## Submission thresholds

**Q2 minimum:** one clear empirical finding; an intervention or representation
contrast supporting its mechanism; at least three context seeds; building- and
segment-clustered uncertainty; full-holdout and natural-prevalence confirmation.

**Q1 extension:** the Q2 path plus a reusable support-allocation diagnostic,
cross-domain alignment metric, or context selector, confirmed with a second
modern tabular learner or usable second data construction.
