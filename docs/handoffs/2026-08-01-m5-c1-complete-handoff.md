# M5 chilledwater C1 complete — active handoff

Supersedes [`2026-08-01-m5-e0-complete-handoff.md`](2026-08-01-m5-e0-complete-handoff.md)
as the active handoff. E0's state and carry-forward issues remain valid and are
not restated here except where C1 changes them.

## State

C1 chilledwater mechanism localization is **complete**. 236/236 atomic units,
0 `.tmp`, all shard stderr empty.

| Item | Value |
| --- | --- |
| base commit | `5e44479` (E0 completion) |
| branch / worktree | `m5-c1-localization` in `lead-reproduction-c1` |
| protocol SHA-256 | `fb6699d8ccf7fefda1213261fb1d64db36c71dcb6a6b96f879959c11a6b5ac1b` |
| units | 5 movement, 20 support, 11 morphology, 200 bootstrap blocks |
| bootstrap | 1,000 draws × 2 clusterings, 0 invalid |
| fits / inference / 192-row scoring | **none** |

## Decision

> **`WITHIN_METER_MORPHOLOGY`**

The chilledwater advantage exists only against chilledwater's own negatives.
Against hotwater-negative it is negative at all five contexts, refuting the
shared-reference hypothesis; electricity- and steam-negative are also negative
or ~zero.

The PR/ROC disagreement is a prevalence-and-saturation effect: ROC-AUC is
saturated at 0.98 with 6.7% prevalence, absolute score separation is unchanged
(+0.0011), and the advantage sits in the high-precision region.

**E0's classification "observed advantage but not stable" is unchanged.**

## Reports

- [`m5-c1-artifact-census.md`](../reports/m5-c1-artifact-census.md)
- [`m5-c1-query-resolution-audit.md`](../reports/m5-c1-query-resolution-audit.md)
- [`m5-c1-chilledwater-localization.md`](../reports/m5-c1-chilledwater-localization.md)
- [`m5-c1-decision.md`](../reports/m5-c1-decision.md)

Artifacts under `data/processed/m5_chilledwater_c1/`: `c1_protocol.json`,
`c1_artifact_census.json`, `c1_query_resolution_audit.json`, `c1_summary.json`,
`c1_decision.json`, and 236 unit checkpoints.

## Next step (requires human authorization)

The gate permits proposing **one targeted support or feature contrast**. The
proposal — chilledwater onset segments versus middle/recovery, negative support
held at chilledwater-negative — is written in the decision report. It is a
proposal only. Path B and representation ablation remain closed.

## Carry-forward issues

1. **The 352-row query cannot resolve a hotwater-negative contrast.** 16
   negative rows give 1,024 pairs and an AUC resolution of 9.8 × 10⁻⁴ against a
   0.0043 effect — 4.4×. Do not add that boundary readout to a pilot at this
   query size, and do not substitute a continuous margin for it.
2. **Frozen threshold defect.** `concentration_limit = 0.25` in the C1 protocol
   is vacuous for a four-bin split (0.25 is the uniform baseline). The decision
   used ≥1.5× each factor's own baseline instead and recorded the correction.
   Any future protocol should state concentration thresholds relative to the
   stratification's own baseline.
3. **Reading-quartile confounding.** 66.7% of absolute movement sits in the
   lowest reading quartile, but 90.4% of chilledwater positives are there. Do
   not quote this as a low-reading mechanism.
4. **Score and rank disagree in sign within-meter.** The within-meter anomaly
   rank gap is −0.0038 while the PR-AUC gap is +0.0486. Any claim phrased as
   "TabPFN ranks chilledwater anomalies higher" is wrong as stated.
5. **`pyarrow` still absent from the pinned formal lock.** C1 ran on the main
   repo venv for the same reason the E0 segment phase did. The pinned FORMAL_E0
   environment cannot reproduce any Parquet-reading phase end-to-end.

## Execution notes worth carrying

C1's compute is trivial once the code is right (the whole run is ~25 minutes on
8 shards), but three execution mistakes cost far more than the compute:

- **A worker exception was swallowed by `ProcessPoolExecutor`.** A `pd.qcut`
  failure in two morphology units propagated to the parent's `fut.result()`,
  which began pool shutdown and then waited on in-flight futures — leaving the
  parent at 0% CPU, workers at 100%, and no checkpoints written for 50 minutes
  with no visible error. Independent shard processes that write their own
  checkpoints surfaced the traceback within seconds. Prefer independent shards
  over an in-process pool for long runs, and **always read worker stderr before
  theorising about system-level causes.**
- **Parallelism was sized by core count, not by measured RSS.** 12 workers × up
  to 3.6 GB exhausted 32 GB and drove the pagefile to a 34 GB peak; throughput
  collapsed with no error.
- **Killing the parent did not kill pool children.** Twelve orphaned workers
  kept holding memory while the process check reported zero remaining. Enumerate
  by parent PID.

These are recorded in `C:\Users\tonykuo\remote-gpu-setup\AGENTS.md` alongside the
remote-execution lessons, since the same failure modes apply to tmux workers on
the GPU host.

## Remote host readiness (for the next stage, not started)

`gpu-host` was inventoried and is ready for Path A / E3 **without further
installation**: TabPFN 8.0.8, torch 2.12.1+cu130, CUDA available on an
RTX 5070 Ti (16 GB), 203 MB of TabPFN weights already cached, xgboost/lightgbm/
catboost present, and the 289 MB hotwater factorial inputs already on the host.
`pyarrow` 23.0.1 was installed into the remote **main** repo venv to match the
laptop; the pinned `lead-reproduction-e0-execution` venv was deliberately left
unchanged and still has no torch, tabpfn, or pyarrow.

The remaining cost of Path A is fit count (E4 = 3 seeds × 4 cells × 2 scaler
arms = 24 fits, plus E3's 4), not environment setup. Per-fit GPU time and
whether 16 GB holds an N=20k support context are **unmeasured** — measuring them
requires inference, which is not authorized.
