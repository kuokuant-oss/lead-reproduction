# M5 hotwater label-role factorial: exact-design reproduction gate

Date: 2026-07-30. This report evaluates the authorized recovery refit on the
original 352-row screening query before any independent-query scoring.

## Predeclared gate

The pre-fit tolerances are recorded in the artifact-recovery report and in
`recovery/reproduction_gate.json`: score MAE <= 0.005, Spearman >= 0.999,
primary-estimand absolute difference <= 0.02, factorial-effect absolute
difference <= 0.04, and matching effect sign whenever the original magnitude
is at least 0.01. Primary estimands exclude `recall@FPR=.001`.

## Result: failed — independent query remains unscored

The 48-cell check failed. Trees passed all 24 cell score and primary-estimand
checks. TabPFN failed all 24 score checks and 16 primary-estimand checks;
16 factorial effects exceeded the magnitude tolerance and one required effect
direction changed. The largest observed TabPFN score MAE was 0.006674, the
lowest reported Spearman correlation was 0.993613, and local HW effects changed
enough to invalidate a confirmation claim.

The exact results are machine-readable in:

- `data/processed/m5_hotwater_label_factorial/recovery/reproduction_gate.json`;
- `.../reports/reproduction_cell_comparison.csv`;
- `.../reports/reproduction_effect_comparison.csv`.

This is a reproducibility stop, not evidence against or for the support-role
hypothesis. The remaining permitted diagnostic is a no-fit reload check: load
each just-saved TabPFN state and tree ensemble, rescore the same screening
query, and compare with the recovery's fit-time score. If state reload is
stable, the discrepancy is a fresh-TabPFN-fit reproducibility issue; if not,
state serialization/inference is implicated. Neither outcome authorizes
independent-query scoring, Path B, or additional factorial refits in this
round.

## No-fit state reload result

All 48 states load and produce finite scores. Tree reload is bit-exact across
24 ensembles (MAE 0, maximum absolute difference 0, Spearman 1). TabPFN reload
is not numerically stable relative to the recovery fit-time score: across 24
states, maximum MAE is 0.006650, maximum absolute difference is 0.196808, and
minimum Spearman is 0.995948. Thus the failed reproduction gate cannot be
attributed solely to a fresh fit; the portable-state inference path itself is
also variable under the present environment.

This ends the allowed Path-A recovery work. The independent mechanism query is
not scored, and no claim is updated from it. A future attempt would require a
separately specified deterministic TabPFN execution investigation (including
backend/device controls) and a new predeclared reproduction gate; it is not
authorized in this round.
