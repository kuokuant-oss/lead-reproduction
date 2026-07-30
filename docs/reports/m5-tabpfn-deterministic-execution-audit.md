# M5 TabPFN execution-variation audit

**Date:** 2026-07-30
**Scope:** execution diagnostics only. No result here is a factorial estimate,
model comparison, Path-A decision, or independent-query result.

## Version policy and disposition

**TabPFN 8.0.8 is the fixed scientific version for this study. TabPFN 8.1.0
was used only as an isolated live-repeat diagnostic and is not used for
factorial estimation or model comparison.**

The 8.1.0 v2c run reused the frozen 8.0.8 arrays and scaler. It completed
before the requested stop could reach a live process; its atomic artifacts were
preserved and the monitor was paused. Only R1–R3 are interpreted: 8.1.0 did not
remove the 8.0.8 live-repeat variation, so changing versions is not a direct
solution. R4–R7 are retained as provenance artifacts but excluded from all
scientific analyses and will not be extended. No further version, CPU, backend,
or lifecycle sweep is authorized.

## What the audit established

Under 8.0.8, both low_memory and fit_preprocessors show variation on a
same-live-estimator GPU repeat before save. Same- and fresh-process reloads do
not remove it. Thus save/load is not the earliest identifiable cause.

For the 8.1.0 v2c diagnostic, R1→R2 had probability MAE 0.005061, maximum
absolute difference 0.105052, Spearman 0.997919, and 351/352 changed rows.
R1→R3 had MAE 0.005748, maximum 0.195849, Spearman 0.997304, and 350/352
changed rows. The raw-logit MAEs were 0.134341 and 0.135363 respectively.
These values are diagnostic measurements, not a pass/fail scientific gate.

The earlier MAE <= 1e-4, max <= .005, and Spearman >= .99999 thresholds remain
engineering diagnostics for artifact integrity. They do **not** block Path A,
invalidate TabPFN as a composition learner, authorize Path B, or delay the
frozen 192-row query indefinitely.

## Scientific handling

For cell c, context seed s, scaler arm a, and repeated inference r, the future
8.0.8 design estimates:

    Y[c,s,a,r] = μ[c,s,a] + ε[c,s,a,r]

μ is the composition effect; ε is repeated-inference measurement variation.
The response is replicate-aware estimation, not a bit-stability requirement.
The next permitted model design is the predeclared 8.0.8 four-cell
repeated-inference variance pilot in the research plan. It is not running under
this report.

## Artifact integrity

The historical low-memory artifacts remain untouched under
data/processed/m5_hotwater_label_factorial/deterministic_execution_audit/.
The v2c root retains its reused-array/state contracts, stage artifacts,
timestamps, and gate output. The gate is now historical engineering provenance,
not a scientific eligibility decision. No existing Path-A prediction or
recovery artifact was overwritten.

## Role in the revised research plan

This is an engineering/measurement appendix, not a paper narrative. It records
execution variation so that future 8.0.8 factorial estimates retain
repeat-inference uncertainty. It cannot choose the steam or chilledwater
mechanism, establish learner superiority, or block the planned CPU-only E0/C1
analyses. The canonical plan and repeated-inference policy define the scientific
endpoints; this report supplies only the measurement constraint.
