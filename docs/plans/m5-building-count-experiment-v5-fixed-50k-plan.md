# M5 building-count V5: fixed 50K and non-electric evaluation

- Status: implementation and frozen-input preparation
- Experiment version: `m5_building_count_v5_fixed_50k`
- Execution policy: `docs/policies/long-running-research-execution.md`

## Scientific settings

- Reuse the exact five validated V4 nested building ladders for K=50, 100,
  200, and 400. No building is redrawn, repaired, or selected by labels.
- K=725 is one canonical support containing every even training building. It
  is not a building-seed replicate; `building_seed=725` is only an operational
  path sentinel. The scientific `building_draw_seed` is null.
- Every context contains 50,000 unique training rows: exactly 25,000 anomalies
  and 25,000 normals sampled uniformly without replacement from its support.
- Row draws are 0 and 1. V5 uses a new recorded PCG64 seed domain, so these are
  newly drawn 50K contexts rather than extensions of the V4 10K rows.
- Training may naturally contain all four meters. Evaluation contains only
  chilled water, steam, and hot water (meter IDs 1, 2, and 3); electricity is
  removed from the frozen canonical holdout before prediction.
- Both models receive the identical ordered rows, labels, 137-feature matrix,
  and non-electric holdout for each matched context. Model seed remains 42;
  TabPFN remains 8 estimators with subsampling disabled.

## Three-phase formal order

1. K=725 full support, row seeds 0 and 1: 2 contexts / 4 model cells.
2. Building seeds 0, 1, and 2, budget-major K=400, 200, 100, 50; within each K,
   building seed ascending and row seed 0 then 1: 24 contexts / 48 model cells.
3. Building seeds 3 and 4 in the same budget-major order: 16 contexts / 32
   model cells.

Each context runs Tree first and TabPFN second. Total: 42 matched contexts and
84 model cells. The scheduler writes an atomic gate after every model pair and
after each phase, reuses only provenance-compatible checkpoints, and blocks on
the first failure without retrying or starting a later phase.

## Runtime estimate

The 50K GPU probe measured approximately 31.37 seconds per 4,096 predictions.
The frozen non-electric holdout has 4,102,084 rows, implying about 8.7 hours of
TabPFN prediction per context before feature and checkpoint overhead. A matched
Tree/TabPFN context is budgeted at roughly 8.9--9.5 hours.

- Phase 1: roughly 18--20 hours.
- Phase 2: roughly 9--10 days.
- Phase 3: roughly 6--6.5 days.
- End-to-end: roughly 16--17 days on one sequential GPU queue.

These are planning estimates. After the first K=725 TabPFN has at least 50 new
prediction chunks, use its measured seconds/chunk to replace the estimate.
