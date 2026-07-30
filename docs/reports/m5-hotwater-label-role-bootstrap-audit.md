# M5 hotwater factorial bootstrap audit

The original 352-row query is clustered by building for the building bootstrap.
For each draw, sampled buildings are drawn with replacement and every query row
belonging to a selected building is included; repeated buildings duplicate all
of their rows. The same resample index is applied to all four factorial cells,
so factorial contrasts preserve cross-cell query pairing.

For the segment bootstrap, every anomaly row is assigned to a consecutive-hour
segment from the complete odd-building time stream. Normal rows are assigned a
unique `normal_<raw_index>` pseudo-segment, rather than being merged into an
unobserved anomaly episode. Segments/pseudo-segments are sampled with
replacement, again using the same index in all four cells. This preserves
within-segment dependence but cannot model a normal/anomaly cross-class segment
because normal rows do not belong to anomaly segments.

For AUC, both positive and negative classes are resampled together through the
cluster index. If a draw lacks either AUC class, its value is `NaN` and it is
excluded from the percentile calculation. Ties use the standard half-credit
AUC rule. When a building contains both classes they enter together, preserving
cross-class building dependence; this does not make the small cluster support
adequate.

The analysis writes 1,000-draw percentile intervals to
`factorial_cluster_bootstrap.csv`. Validity tests cover tie-aware AUC,
empty-class handling, and factorial-interaction algebra in
`tests/test_m5_hotwater_factorial_bootstrap.py`. The decisive limitation is the
local HW 0–1 negative class: it has only three clusters, so no bootstrap draw
count can make it confirmatory.
