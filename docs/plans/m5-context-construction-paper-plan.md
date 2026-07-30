# M5 137-feature meter-specific mechanism plan

**Updated:** 2026-07-30
**Status:** documentation-only planning. No fit, inference replicate, 192-row
scoring, Path B, tree refit, context-curve rerun, or full-holdout refit is
authorized by this plan.
The 192-row independent query remains frozen and unscored.

## Research question and paper narrative

The paper asks:

> Why does TabPFN obtain meter-specific gains for steam and chilledwater
> anomalies under the 137-feature temporal representation, and do those gains
> arise from shared cross-meter score references, within-meter anomaly support,
> or distinct mechanisms?

This is not a generic context-size benchmark, a claim of unconditional TabPFN
superiority, a hotwater detection paper, or a TabPFN reproducibility paper.
The 137-feature representation is the research object. Matched-row trees are
the comparator for meter-specific learner response. Hotwater is a candidate
support source and intervention lever; its local outcomes are manipulation
checks and mechanism diagnostics, not the principal outcome.

Completed Phase-1 CPU analysis of existing 5k/10k/20k/50k/100k full-holdout
predictions establishes rank/score movement and selects hypotheses. It does not
by itself establish a steam or chilledwater mechanism. Site transfer, retrieval,
500k scaling, representation ablations, full transfer matrices, and per-site
stories remain paused; per-site slices are only robustness diagnostics.

## Fixed version and measurement policy

TabPFN 8.0.8 is the only scientific version. TabPFN 8.1.0 is an isolated
live-repeat diagnostic, excluded from factorial estimation, learner comparison,
Path-A decisions, and paper results.

For factorial cell c, context seed s, scaler arm a, and repeated inference r:

    Y[c,s,a,r] = mu[c,s,a] + epsilon[c,s,a,r]

mu is the training-composition estimand and epsilon is repeat-inference
measurement variation. The deterministic lifecycle thresholds remain engineering
artifact diagnostics only; they are not a scientific gate. Trees remain
bit-exact fixed comparators and receive no artificial replicate noise.
Bit-identical probabilities are not a scientific eligibility criterion.

## Experiment map

| Stage | Scientific role | Data | New fit | Primary outcome | Status |
| --- | --- | --- | --- | --- | --- |
| E0 | meter-specific learner-gap characterization | existing natural-prevalence full-holdout predictions, 5k-100k | no | steam/chilledwater TabPFN-minus-tree PR-AUC and ROC-AUC movement | planned CPU analysis |
| E1 | context mechanism localization | existing predictions and row/segment artifacts | no | within/cross-meter score and rank movement | completed |
| E2 / C1 | chilledwater localization | existing predictions and row/segment artifacts | no | candidate support or morphology source | planned CPU analysis |
| E3 | 8.0.8 variance pilot | original 352-row query | four fits, only after authorization | repeated-inference cell estimands | next fit candidate; not authorized |
| E4 | formal Path A | 3 seeds x 4 cells x 2 scaler arms | conditional | steam cross-meter response; chilledwater boundary readouts | conditional |
| E5 | independent replication | frozen 192-row query | no new fit if valid states exist | predeclared steam/chilledwater endpoints | locked |
| E6 | targeted confirmation | natural-prevalence targeted/full holdout | conditional | replicated mechanism | deferred |
| E7 | chilledwater intervention | only if C1 identifies a distinct mechanism | conditional | chilledwater mechanism | deferred |

### E0: meter-specific learner-gap characterization

Use only existing full-holdout predictions to report per-meter PR-AUC and
ROC-AUC for TabPFN and trees at each context size, their paired difference, and
the steam/chilledwater learner-gap slope. Include building-clustered bootstrap,
leave-one-building influence, and anomaly-segment concentration. Electricity is
a counterexample, not a claim of universal TabPFN advantage. Interpret observed
advantages without a formal significance claim until clustered uncertainty and
positive-ranking behaviour are available.

E0 is a CPU-only analysis specification. It must not trigger a fit if a required
artifact is absent; report the missing artifact instead.

### E1: completed context localization

E1 found broad rank improvement for both learners, positive anomaly
within-meter rank movement across meters, hotwater 0-1 normal score reduction
larger than anomaly score reduction, and a large steam-positive versus
hotwater-negative cross-meter movement. It supports a hotwater-support
intervention hypothesis, not a settled mechanism.

### E2 / C1: chilledwater CPU-only localization

C1 is fully specified in
docs/plans/m5-chilledwater-mechanism-localization.md. It uses existing
5k-100k prediction plus row/segment artifacts only. It must determine whether
chilledwater is more consistent with the hotwater-negative reference candidate,
a different support source, within-meter morphology/representation, or no stable
localization. It cannot launch a fit.

## Steam Path A: controlled support intervention

The future N=20k hotwater positive-support x negative-support 2x2 factorial is
organized around steam, not hotwater.

- Intervention: allocate hotwater positive and negative support.
- Principal mechanism outcome: steam-positive relative to hotwater-negative.
- Candidate mechanism: hotwater negative support supplies a cross-meter,
  normal-side score reference.
- Local hotwater AUC/rank gap: manipulation check and mechanism diagnostic.
- Scaler arm: preprocessing-geometry test.
- Trees: matched-row fixed comparator.
- TabPFN: 8.0.8 repeated-inference measurement process.

Primary steam readouts are pairwise AUC, continuous score margin, global and
within-meter rank, anomaly-segment movement, factorial contrasts, scaler-arm
interaction, context-seed consistency, and building-/segment-clustered
uncertainty. Do not require a local hotwater endpoint to carry the principal
claim.

## Chilledwater: mechanism-boundary readouts

If and only if the 352-row query has adequate pre-fit chilledwater strata after
a resolution audit, add these as secondary boundary readouts to the 8.0.8
variance pilot and then formal Path A:

- chilledwater-positive versus hotwater-negative pairwise AUC;
- continuous chilledwater-positive minus hotwater-negative score margin;
- chilledwater-positive versus other negative-meter groups;
- chilledwater within-meter anomaly-vs-normal AUC and rank gap;
- chilledwater positive/negative score and rank movement;
- chilledwater anomaly-segment summaries; and
- TabPFN-minus-tree factorial-response difference.

Before any pilot result is interpreted, the resolution audit must report rows,
buildings, segments, valid pair counts, AUC resolution, and building/segment
concentration for each chilledwater stratum. These endpoints distinguish a
shared hotwater-negative reference from a steam-only mechanism; they do not
make a chilledwater mechanism claim by themselves.

## Variance pilot and conditional continuation

The next fit candidate remains the fixed-8.0.8 variance pilot: v3 checkpoint,
F4 137 features, N=20,000, seed 42, model seed 42, cell-specific scaler, the
four original cells, original 352-row query, one fit per cell, persisted state,
and retained same-process inference replicates. Cells must follow a
pre-generated seeded interleaved schedule. Initial batch: 8 completed repeats
per cell; cap: 40. Fresh-process reload is limited device/process diagnostic.

The pilot is designed but unauthorized. It cannot score the 192-row query,
select its endpoints, alter C1, or run a 24-cell grid. Formal Path A follows
only if the measurement process supports replicate-aware composition estimates
with the predeclared precision and stopping rules in the policy JSON.

## Evidence chain and hard boundaries

A Q2-ready chain is: meter-specific empirical opening (E0), CPU localization
(E1/C1), controlled steam intervention, repeat-aware TabPFN estimation,
matched-tree comparison, three context seeds, building-/segment-clustered
uncertainty, frozen independent replication, and targeted natural-prevalence
confirmation. A hotwater intervention does not automatically establish a
chilledwater mechanism.

Do not run 8.1.0 as science, the 192-row query, a 24-cell grid, Path B, tree
refits, context-curve fits, site transfer, or manuscript conclusion changes in
this planning round.
