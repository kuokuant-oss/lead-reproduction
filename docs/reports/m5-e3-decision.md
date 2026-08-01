# M5 E3 — decision

**Artifact:** `data/processed/m5_e3_variance_pilot/e3_decision.json`

## Verdict

```
E3_MEASUREMENT_PROCESS_ACCEPTABLE
```

Criterion, quoted from the frozen protocol's `e3_decision_rule`: *"all four
cells pass both gate endpoints."*

## How it was selected

The protocol declares exactly four possible verdicts, and they are evaluated in
a fixed order. No other criterion was applied.

| Verdict | Condition | Cells |
|---|---|---|
| `E3_EXECUTION_INCOMPLETE` | fit, state, same-process lifecycle or provenance incomplete | none |
| `E3_MEASUREMENT_PROCESS_UNSTABLE` | any cell still failing at n=40 | none |
| `E3_MORE_REPEATS_REQUIRED` | predeclared continuation still pending under the cap | none |
| **`E3_MEASUREMENT_PROCESS_ACCEPTABLE`** | all four cells pass both gate endpoints | **00, 01, 10, 11** |

All four cells completed with status `COMPLETE_GATE_PASSED`, exactly one fit
each, four distinct state digests, and four distinct process UUIDs. No cell was
pending under the cap of 40, and none reached it — every cell passed at the
first batch of 8.

## What the verdict means

It means the measurement process is stable enough that repeat-level readouts can
be reported. Concretely: repeated inference on a fitted TabPFN 8.0.8 state
perturbs the gating endpoints by less than the predeclared precision targets in
all four cells.

## What it does not mean

- **It is not a scientific finding.** No claim about hotwater-positive or
  hotwater-negative context composition is established here.
- **It does not validate the between-cell differences.** Each cell has one fit,
  so any difference between cells confounds composition with fit-to-fit
  variation, which this pilot does not measure.
- **It authorises nothing downstream.** `authorises` is empty in the decision
  artifact. E4 formal Path A, Path B, representation ablation, the frozen
  192-row query, site transfer, 500k, full-holdout refit, tree refit,
  TabPFN 8.1.0 as science, and manuscript changes all remain prohibited and
  require separate human authorisation.

## Carried forward unresolved

| Item | Status | Why |
|---|---|---|
| chilledwater positive vs hotwater negative | `RESOLUTION_LIMITED_DIAGNOSTIC` | 16 hotwater-negative rows give AUC resolution 0.000977 against a 0.0043 effect — about 4.4x too coarse |
| onset / middle / recovery phase contrast | `UNRESOLVED_NOT_EXECUTED` | no frozen artifact assigns a phase to a row; producing one would require inventing a within-segment cutpoint rule |

Both were determined **before any fit** and neither is a consequence of the
results. See `m5-e3-query-phase-resolution-audit.md`.

## One thing to watch if E3 is ever extended

Cell 01's steam AUC is 1.000000 in all 8 repeats. Its half-width is exactly
zero because the metric is saturated, not because it is precisely measured. If
a later stage adds repeats or cells, that endpoint will keep passing its gate
while carrying no information about stability. Cell 01's margin endpoint —
half-width 0.000936 against a target of 0.001000, the tightest ratio in the
pilot — is the one doing real work there.

## Decision

E3 is complete and frozen at `E3_MEASUREMENT_PROCESS_ACCEPTABLE`. No further E3
execution is required or permitted. The next stage awaits explicit human
authorisation.
