# 2026-06-25 Phase B Foundation-Model Planning

## Scope

Phase B added docs-only M5 planning for a TabPFN foundation-model track. The
slice is tracked by GitHub issue #27 and should close with `Closes #27`.

No TabPFN, torch, GPU, or BDG2 work was installed or run.

## Changes

+ Added `docs/plans/m5-plan.md` with an M5 model track for TabPFN benchmarking.
+ Added ADR 0015 as Proposed: evaluate TabPFN in M5 with success criteria based
  on rigorous benchmark and transfer evaluation, not beating tuned GBDT.
+ Added unknowns 19-21 for GEPIII row/feature fit, GPU/VRAM/license execution
  path, and real-time latency.
+ Updated README to expose the M5 plan and ADR range.

## Source notes

+ Prior Labs documents TabPFN-3 limits as `1,000,000 x 200`,
  `100,000 x 2,000`, or `1,000 x 20,000` rows x features.
+ The TabPFN-3 technical report is arXiv:2605.13986 and describes 1M-row scaling
  on H100-class hardware; this is not laptop-GPU evidence.
+ The current TabPFN-3 model weights use the TABPFN-3.0 License v1.0. Treat this
  as suitable for research/internal evaluation planning; production or
  business-decision use requires commercial licensing or API agreement.
+ Nature 2025 and the TabPFN-2.5 report are cited in the M5 plan and ADR 0015.

## Checklist state

+ Issue #27 was already open from Phase A.
+ README was updated because the change adds visible M5 planning and ADR 0015.
+ Plan tracker was updated through the new M5 plan.
+ ADR 0015 records the planning decision as Proposed.
+ Unknowns were updated.
+ No result JSON was produced because Phase B is docs-only.

## CJK encoding check

After editing `README.md` and `docs/reference/unknowns.md`, inspect:

```bash
git diff -- README.md docs/reference/unknowns.md
```

Expected persistent changes are only the new M5 README rows/tree entries and
the appended English unknowns 19-21. No encoding repair or mass CJK rewrite
should appear.

## Suggested next skills

+ `diagnose` for Phase C only if CUDA, token, or OOM behavior fails.
+ `handoff` after Phase C if the spike runs and produces local artifacts.
