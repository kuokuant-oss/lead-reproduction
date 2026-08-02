# M5 E6 — clustered uncertainty at 10.1M rows

## The problem is smaller than it looks

The co-primary contrast compares steam positives with hotwater negatives. At
natural prevalence that is **594,318 rows — 5.86% of the holdout** — not 10.1M.
Scoring is a 10.1M-row problem; the clustered statistics are a 594k-row problem.

## Exact cluster-weighted AUC

A cluster bootstrap draw resamples cluster names with replacement. The resampled
multiset differs from the original only by per-row integer multiplicities, and
**the score order does not depend on the draw**. So:

1. sort the 594,318-row subset once per unit — 47 ms
2. per draw, sweep weighted counts in score order — **23 ms**

The statistic is the cluster-weighted Mann-Whitney U:

```
AUC_w = Σ_{i∈pos} w_i · (W_below(i) + ½·W_tied(i))  /  (Σ_pos w · Σ_neg w)
```

where `w_i` is the multiplicity of row *i*'s cluster, `W_below` is the weighted
count of negatives strictly below, and `W_tied` the weighted count of negatives
at the same score. Ties contribute a half, exactly as `roc_auc_score` does.

This is **exact**, not an approximation: it equals the naive
materialise-and-recompute path by construction.

### Verified, not asserted

`tests/test_m5_e6_design.py` proves the equivalence to 1e−12 relative:

- on synthetic data with deliberately coarse scores so ties occur, across 40
  draws
- **on the real E5 192-row data**, across 25 draws, for both AUC and margin
- an all-ties case, where the answer must be exactly 0.5 and both paths give it

Cost for the whole of E6: 1000 draws × 2 clusterings × 24 units × 23 ms =
**19 minutes**, plus 24 sorts.

## Exact cluster-weighted margin

The margin is a difference of weighted means, so it needs no sort at all:

```
margin_w = Σ_pos w·s / Σ_pos w  −  Σ_neg w·s / Σ_neg w
```

Per-cluster score sums and counts are sufficient statistics, so a draw costs
O(#clusters). Also verified value-for-value against naive resampling.

## What was ruled out, and why

| Approach | Verdict |
|---|---|
| re-sort 10.1M rows per draw | never needed — the AUC subset is 594k and the order is draw-invariant |
| black-box PR/ROC per draw | would recompute a sort 48,000 times for no benefit |
| subsampling rows | changes the estimand; refused |
| approximate AUC | unnecessary once the sufficient statistic is used |

## The segment estimator needs a ruling

On the co-primary subset:

| Clustering | Clusters | Singletons |
|---|---:|---:|
| building | 215 | — |
| segment | 594,297 | **545,430 (91.8%)** |

Segments are contiguous anomaly runs, and every non-anomaly row is its own
cluster. All 545,430 hotwater negatives are non-anomaly, so at natural
prevalence the segment clustering is 91.8% singletons **on this contrast** and
behaves close to a row bootstrap on the negative side.

The estimator extends exactly — this is not a feasibility failure, and it is not
being removed. The problem is what its interval would *mean*. With 594,297
clusters against 215 building clusters, the segment interval will be
dramatically narrower and will not represent within-building correlation. In E5
the two were comparable (192 segment clusters vs 142 building); at natural
prevalence they are not.

Per the protocol this is raised rather than resolved unilaterally. It is not
silently dropped, not swapped for a row bootstrap, and not replaced with site
clusters.

### Options for the ruling

**Option 1 — report it as-is, with the degeneracy stated.** Keep both intervals,
and state that at natural prevalence the segment interval is close to a row
bootstrap for this contrast and is therefore a weaker check than the building
interval. Cheapest, and honest, but invites a reader to treat a narrow interval
as strong evidence.

**Option 2 — building-clustered only for the co-primary decision**, with the
segment interval reported as a secondary diagnostic and explicitly not part of
the confirmation bar. This changes the decision rule, which E5 froze with both
clusterings required, so it needs an explicit ruling.

**Option 3 — a segment definition that does not degenerate**, for example
extending segments to contiguous runs of *any* label within a building-meter
series rather than anomaly runs only. This would keep a genuine
within-series cluster on both sides, but it is a **new cluster definition**, not
the E4/E5 one, and would break comparability with both earlier stages.

**Recommendation: Option 1**, with Option 2 as the fallback if the reviewer
would rather not carry a near-degenerate interval into the confirmation bar.
Option 3 is not recommended: buying a better-behaved interval by redefining
clusters at the last stage of a three-stage evidence chain trades comparability
for a cosmetic gain, and the degeneracy is a real property of natural prevalence
rather than an artefact to engineer away.

Awaiting a human ruling. E6 is not launchable until it is settled.

## Seed mapping

```
SeedSequence([20260730, 6006, cluster_code[cluster_type], draw_id])
cluster_code = {"building": 1, "segment": 2}
draw_id = 0 .. 999
```

Namespace 6006 separates E6's draws from E4's 4004 and E5's 5005 while the
construction stays identical. Tested: the same draw id under different
namespaces gives different multiplicities, draws are reproducible individually,
and generating them in reverse order gives the same result as forward.

One draw's row multiset is shared across cells, arms, seeds, TabPFN and the
tree, exactly as in E4 and E5.
