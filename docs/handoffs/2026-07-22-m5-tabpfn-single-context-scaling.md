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
CPU-only, so install a CUDA build and run preflight before any formal attempt.
Do not infer 500K success from smoke or `.fit()` alone. Formal execution still
requires explicit operator approval.

Completed budgets also write atomic compressed prediction artifacts containing
only the fixed validation/test scoring rows. These make pooled and by-site
ROC/PR plots reproducible without another TabPFN fit. Use
`scripts/plot_m5_tabpfn_single_context_curves.py`; rare sites lacking both
classes in the natural-prevalence sample are reported as not estimable.
