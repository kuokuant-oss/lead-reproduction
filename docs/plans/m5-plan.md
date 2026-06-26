# M5 Plan: FDD on BDG2

**Status**: Ready to implement
**Started**: 2026-06-25
**GitHub Issue**: [#27](https://github.com/kuokuant-oss/lead-reproduction/issues/27)
**References**:

+ M4 importable foundation: `docs/plans/m4-plan.md`
+ M3 cross-site diagnostic: site-held-out ensemble AUC `0.9774`
+ TabPFN Nature paper: Hollmann et al., "Accurate predictions on small data with
  a tabular foundation model", Nature 637, 319-326 (2025),
  https://doi.org/10.1038/s41586-024-08328-6
+ TabPFN-2.5 report: Grinsztajn et al., arXiv:2511.08667,
  https://arxiv.org/abs/2511.08667
+ TabPFN-3 technical report: Grinsztajn et al., arXiv:2605.13986,
  https://arxiv.org/abs/2605.13986
+ Prior Labs model limits and licensing: https://docs.priorlabs.ai/models

---

## M5 Goal

M5 moves from reproduction toward fault detection and diagnosis (FDD) on BDG2.
The milestone should reuse the importable `src/lead` foundation where possible,
name all feature and label assumptions explicitly, and preserve causal-feature
discipline for any real-time inference claim.

M5 planning does not download BDG2, run GPU jobs, or install foundation-model
dependencies. Those belong to later implementation slices.

---

## Entry Criteria

M5 starts from the frozen `src/lead` API recorded in the M4.5 handoff and
README.

+ Data interface: reuse `load_m3_frame`-style loaders for tabular frames. BDG2
  ingestion is M5's job, not M4's.
+ Label interface: BDG2 may have no per-row anomaly labels, so M5 interfaces
  must allow unlabeled or transfer evaluation and must not assume a 1:1 label
  column.
+ Split interface: use building-level and site-held-out splits through
  `split_mask` and `assert_no_building_overlap`. Site-held-out transfer is the
  motivating test, anchored by M3 site-held-out ensemble AUC `0.9774`.
+ Evaluation interface: use `classification_metrics` for AUC, precision,
  recall, and F1. Any real-time FDD claim must use `PAST_SHIFTS`-only causal
  features per ADR 0007 and ADR 0011.

---

## M5 model track: foundation-model benchmark

M5 will include a model track that evaluates TabPFN as a tabular foundation
model benchmark alongside the existing GBDT line.

### Model

Use TabPFN from Prior Labs as the candidate tabular foundation model. The current
default local OSS checkpoint is TabPFN-3. Prior Labs documents TabPFN-3 limits as
`1,000,000 x 200`, `100,000 x 2,000`, or `1,000 x 20,000` rows x features, with
row capacity trading off against feature count. The TabPFN-3 technical report
states that 1M-row scaling is demonstrated on an H100 with row chunking and a
reduced KV cache; this is not evidence that a laptop GPU can scale beyond the
LEAD feasibility spike.

TabPFN handles numerical, categorical, and missing values without the usual GBDT
preprocessing stack. The model documentation recommends a GPU; CPU use is
feasible only for small datasets around 1,000 samples. The TabPFN-3.0 License
v1.0 permits research and internal evaluation use, while production, competitive
vendor evaluation, or workflows influencing business decisions require a
commercial license or API agreement.

### Pipeline placement

TabPFN consumes the same tabular feature table as the GBDT step and slots in at
the model stage. It does not change upstream causal-feature discipline:

+ Feature construction must still respect `PAST_SHIFTS` for real-time FDD.
+ Split definitions and label provenance remain upstream responsibilities.
+ Any TabPFN comparison must use the same split, seed, downsample, and feature
  table as the paired GBDT anchor.

### Experiment ladder

1. Minimal LEAD subset benchmark: compare the existing GBDT local-validation
   line against TabPFN on the same LEAD feature table, split, seed, and
   downsample. This is a feasibility signal, not a benchmark.
2. Primary M5 contribution: few-shot transfer to held-out or unlabeled sites.
   Compare TabPFN in-context adaptation with GBDT retrain and GBDT transfer. The
   motivating M3 diagnostic is the cross-site drop to site-held-out AUC
   `0.9774`; the motivating BDG2 constraint is missing per-row labels.
3. Minimal-feature-engineering comparison: measure whether TabPFN retains useful
   performance with a smaller feature-engineering burden than the tuned GBDT
   line.

### Success criteria

Success is a rigorous benchmark and transfer evaluation, not beating tuned GBDT
on headline AUC. Tuned GBDT may still win raw AUC. The foundation-model value
claim must be tested in transfer, label scarcity, and low-feature-engineering
settings.

### Open constraints

+ Measure whether the downsampled GEPIII training table fits TabPFN-3's
  row/feature limits.
+ Record available GPU class and VRAM before any TabPFN run.
+ Treat real-time inference latency as unknown because each prediction batch
  recomputes against the in-context training set.
+ Prefer local GPU inference. Any TabPFN Client or cloud path that sends data
  off-machine requires explicit consent.

---

## Phase B close-out

+ [x] Planning only: no TabPFN install, no torch dependency, no BDG2 download.
+ [x] ADR 0015 records the proposed TabPFN evaluation decision.
+ [x] `docs/reference/unknowns.md` registers the new model-track unknowns.
+ [x] Phase A checklist applied.
+ [x] Commit with `Closes #27` after tests, ruff, markdownlint, and pre-commit
  pass.

---

## Phase C status

+ [x] Opened issue [#30](https://github.com/kuokuant-oss/lead-reproduction/issues/30).
+ [x] Added optional `m5` dependency group for `torch` and `tabpfn`; install
  with `uv sync --group m5`.
+ [x] Measured the M3.2 frozen-helper feature path: full downsample is
  `4,285,104 x 137`, exceeding the documented TabPFN-3 `1,000,000 x 200`
  limit.
+ [x] Archived local feasibility evidence to
  `data/processed/m5_phaseC_tabpfn_spike.json`.
+ [x] Completed local-weights TabPFN metric on the same reduced `1,000 x 137`
  table as the GBDT anchor: TabPFN AUC `0.9904`, GBDT AUC `0.9870`, TabPFN
  fit+predict `8.1954` seconds on RTX 4070 Laptop GPU.

---

## Phase D plan slice

**Status**: Ready to implement, not started
**GitHub Issue**: [#31](https://github.com/kuokuant-oss/lead-reproduction/issues/31)

Phase D is docs-only in this commit. It defines the next M5 contribution beyond
the LEAD/M3 feasibility spike and does not download BDG2 data, install
BDG2-specific dependencies, or run new experiments.

Phase D implementation should cover:

+ BDG2 ingestion design: a `load_m3_frame`-style loader for BDG2 tabular frames
  that explicitly allows unlabeled and transfer evaluation. BDG2 must not be
  assumed to provide a 1:1 per-row anomaly label column.
+ Primary contribution: few-shot transfer to held-out or unlabeled sites,
  comparing TabPFN in-context adaptation against GBDT retrain and GBDT transfer.
  The motivating anchor remains the M3 site-held-out ensemble AUC `0.9774`.
+ Minimal-feature-engineering comparison: evaluate TabPFN with a smaller feature
  set against the tuned GBDT line without overclaiming headline AUC.
+ Causal discipline: any real-time FDD claim must use `PAST_SHIFTS`-only
  features per ADR 0007 and ADR 0011.

---

## Issue Tracker Map (M5)

| Slice | GitHub issue | Status |
| --- | --- | --- |
| Phase B foundation-model planning | [#27](https://github.com/kuokuant-oss/lead-reproduction/issues/27) | Done |
| Phase C LEAD TabPFN feasibility spike | [#30](https://github.com/kuokuant-oss/lead-reproduction/issues/30) | Done |
| Phase D BDG2 transfer and minimal-feature plan | [#31](https://github.com/kuokuant-oss/lead-reproduction/issues/31) | Ready to implement, not started |
