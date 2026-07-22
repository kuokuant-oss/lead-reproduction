# Handoff: M5 TabPFN single-context scaling

## Implementation

`scripts/run_m5_tabpfn_single_context_scaling.py` isolates preflight and each
budget in subprocesses. `src/lead/resource_guard.py` provides atomic writes,
GPU/RAM sampling, limit decisions, and recursive termination.

The experiment uses raw 17 features, disjoint mod-4/mod-2 building splits,
unique balanced nested contexts, fixed score rows, validation-selected recall
0.90 thresholding, and no external sharding or sample subsampling.

## Resume contract

State retains completed/failed budgets, running PID, last safe budget, and stop
reason. `--resume` skips completion, `--restart-budget` reruns from a selected
point, and a dead prior PID becomes `interrupted_previous_run`.

## Verification and next step

Unit/mock/fake-smoke verification does not initialize CUDA. Current Torch is
The local venv was updated to `torch 2.12.1+cu126`; CUDA availability and a
minimal RTX 4070 allocation were verified. Run the full no-fit preflight before
any formal attempt.
Do not infer 500K success from smoke or `.fit()` alone. Formal execution still
requires explicit operator approval.

Completed budgets also write atomic compressed prediction artifacts containing
only the fixed validation/test scoring rows. These make pooled and by-site
ROC/PR plots reproducible without another TabPFN fit. Use
`scripts/plot_m5_tabpfn_single_context_curves.py`; rare sites lacking both
classes in the natural-prevalence sample are reported as not estimable.

For operator-gated execution, pass `--max-budgets-this-run 1` and resume after
inspecting each completed budget. Optional `--site-curve-rows-per-class 16`
adds 512 balanced query rows only at `--site-curve-budget 500000`; these rows
are used solely for by-site curves and never replace natural-prevalence metrics.

Per operator decision, real model runs have no wall-time limit. The CLI default
is `--budget-timeout-minutes 0`; resource hard limits and stale-heartbeat
termination remain enabled.

The full-data no-fit preflight later completed `ready` with all nine checks
passing. It verified CUDA Torch 2.12.1+cu126, TabPFN 8.0.8, the local checkpoint
hash, disabled sample subsampling, the balanced unique 500K contract, 512
site-stratified scoring rows, and null model timeout. No model was fit.

A separate real-GPU 200-row gate completed after fixing a Windows heartbeat
atomic-write race and changing RSS monitoring to sum the launcher process tree.
It verified 200 effective context rows, one estimator, completed validation/test
predictions, atomic NPZ output, offline four-figure rendering, and full GPU
release after worker exit. Peak device GPU was 1,203 MiB.
