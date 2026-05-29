# Handoff: M2 Milestone Closed

**Date**: 2026-05-29
**Status**: ✅ M2 milestone 完全 closed (M2.1 → M2.5 all done)
**Final Kaggle Private**: 0.98616 (vs 原作者 0.98661, gap 0.05% < noise floor ±0.0005)
**Final Commits**: 8b9af23 (Stage 1 report), 6a88fea (M2 detail fixes), a6c2252 (framing)

## What was Accomplished in M2

### M2.1: Baseline pipeline

- 57 raw features + downsampling + CV split + LightGBM
- Val AUC = 0.8952 (paper 0.9311, gap 3.86% < 5% pass)
- Lesson #1: tree monotonic invariance (cloud_coverage fix ΔAUC=0)
- Gap candidate: impute_nulls (unknown #10)

### M2.2: Feature engineering (5 sub-steps)

- M2.2.0: cloud_coverage fix (ΔAUC=0, tree invariant confirmed)
- M2.2.a: ClusterNo (ARI=1.0 after n_init=10 fix, lesson #2 sklearn n_init)
- M2.2.b: Value-change 120 features (60 shifts × 2 types)
- M2.2.c+d: SavGol + dayofyear (lessons #3 #4 #5)
- M2.2.e: 169 features → val AUC 0.9818 (gap 0.31%)

### M2.3: 4-model ensemble

- LGB+XGB+Cat+Hist equal-weight → val AUC 0.9830
- Cross-model importance divergence documented (lesson #6)
- Noise floor ±0.0005 measured (random_state=42)

### M2.4: Post-processing + Kaggle submission

- 3 hard rules (Rule 1 meter==1.0, Rule 2a dayofyear==1 filter, Rule 2b year-end)
- X_all dual-path refit pipeline (unknown #16 resolved by inference)
- **Kaggle Private = 0.98616** (gap 0.05% vs 0.98661 原作者, within noise floor)
- One-shot submission methodology purity achieved

### M2.5: In-notebook ablation + Kaggle validation

- 3 val-side ablations (A: gte_* removal / B: per-bldg mean impute / C: Rule 2a blanket)
- 3 additional Kaggle submissions (research data, not leaderboard probing)
- Quantified unknowns #5 (gte_* valid encoding), #10 candidate 1 (raw NaN > per-bldg mean in our pipeline), #15 (Rule 2a filter negligible at dataset level)
- Lesson #7: component interaction matters — cannot generalize single-component swap to paper design

## Final Metrics

| Layer | Result |
|---|---|
| Layer 1 (run through) | ✅ Kaggle Private 0.98616 |
| Layer 2 (understand) | ✅ 7 methodology lessons + 6 ADRs |
| Layer 3 (question) | ✅ 4.3 reproduction observations + 3 quantified ablations |

Paper overall design confirmed well-tuned: author's Private 0.98661 > ours 0.98616.

## Unknowns Final State (17 total)

- **Resolved (9)**: #1 (169 features), #3 (post-processing boundaries), #4 (downsampling seeds),
  #5 (gte_* target encoding), #6 (CatBoost 1000 iterations), #7 (LEAD upstream pipeline),
  #8 (anomaly rate), #9 (building_id range), #15 (Rule 2a filter)
- **Partially resolved (7)**: #2 (CV split single-fold confirmed, loop TBD), #10 (candidate 1 only),
  #11 (timestamp divergence), #12 (SavGol importance), #13 (cross-model importance),
  #14 (noise floor measured), #16 (X_all inferred)
- **Non-issue (1)**: #17 (Public < Private is normal Kaggle pattern)

## Files Final State

| File | Status |
|---|---|
| docs/reproduction-report.md | ~5500 words, 5 chapters, neutral framing |
| docs/workflow.md | ~2500 words, 8 sections |
| docs/unknowns.md | 17 entries with state tracking |
| docs/m2-plan.md | M2.1–M2.5 all ✅, Final Closure section added |
| docs/adr/ | 6 ADRs (0001–0006) |
| docs/handoffs/ | 5 handoffs (M2.2, M2.2a, M2.3, M2.4, this file) |
| notebooks/05-m2-integration.ipynb | 34 cells with engineer-level annotations |

## M3 Sync Prompt (for next session)

```
## Stage 0: 進入 M3 狀態

依序讀:
1. docs/handoffs/2026-05-29-m2-milestone-closed.md (本檔)
2. docs/reproduction-report.md Ch5.6 (M3 outlook)
3. notebooks/06-m3-baseline.ipynb (M3 framework, if exists)
4. docs/unknowns.md (M3 相關 unknowns)
5. git log --oneline -10

確認 M3 vs M2 差異後進 M3 Stage 1 (baseline run):
- 1449 buildings (vs M2 406), train rows ~20.2M (vs M2 1.75M)
- bad_meter_readings.csv positional join (not key join)
- Anomaly rate 6.5% (vs M2 2.13%, ~3×)
- No Kaggle leaderboard — reproducibility via random_state=42
- Rule 2a building_id filter must be re-designed for GEPIII building IDs
```

## Blocking Questions for M3

1. Should M3 use same 169-feature pipeline or start with fewer features given 20M rows?
2. Downsampling ratio: same 50:50 or adjust for 6.5% anomaly rate?
3. Validation split: same modulo-5 strategy or random building split?
4. CatBoost 1000 iters on 20M rows: acceptable training time?
