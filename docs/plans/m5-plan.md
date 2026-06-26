# M5 Plan: FDD on BDG2

**Status**: FDD model-selection stage (model track, Phase A–D) complete; Phase E (BDG2) planned
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

## M5 model track: FDD model selection (TabPFN vs GBDT)

M5's model track is an **FDD model-selection stage**: it compares TabPFN (a
tabular foundation model, evaluated as one candidate FDD tool) against the
existing GBDT line to decide which model serves FDD on the GEPIII data, and
where each one is strong. TabPFN is a compared model here, **not** the goal of
the milestone and **not** an independent track — the milestone is FDD on BDG2.

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
  cold fit+predict `6.5070` seconds on RTX 4070 Laptop GPU.
+ [x] Phase C metric audit [#32](https://github.com/kuokuant-oss/lead-reproduction/issues/32)
  confirmed TabPFN threshold metrics are computed from TabPFN probabilities.
  The `0.5`-threshold confusion matrix matches the GBDT anchor on this slice,
  while AUC differs.

---

## Phase D plan slice

**Status**: Complete — foundation-vs-tree comparison on existing GEPIII data
**GitHub Issue**: [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35)

Phase D is the rigorous foundation-model (TabPFN) vs tree-model (GBDT) comparison
on the existing M3 GEPIII data. It is **not** docs-only and it does **not** add a
new dataset. It runs real paired experiments through the frozen `src/lead`
pipeline: `load_m3_frame`, `add_value_change_features`, the building-level
`80_20_mod5` split and the site-held-out `site_id % 5 == 4` split,
`downsample_indices`, and `classification_metrics` (PR-AUC computed additively in
the harness). Every paired cell reuses the same split, seed, downsample, and
feature table so the only variable is the model.

The data-scaling path is LEAD subset → full ASHRAE GEPIII (M3, done) → BDG2
(Phase E). BDG2 ingestion is M5's **next stage (Phase E)** — not a separate
milestone — with real data, a real schema, and real labels; the premature
`lead.bdg2` ingestion skeleton and its invented fixtures were retired under issue
[#34](https://github.com/kuokuant-oss/lead-reproduction/issues/34).

Phase D comparison axes (each reports AUC, precision, recall, F1, PR-AUC, and
fit+predict latency, with mean ± std across multiple seeds):

1. In-domain: TabPFN vs GBDT on the `80_20_mod5` split. The TabPFN fit set is
   pushed well beyond the 1,000-row spike, within documented TabPFN-3 limits
   (`1,000,000 × 200`); the subsample budget is recorded in the result JSON.
2. Primary axis — site transfer: the site-held-out `site_id % 5 == 4` split
   (M3 ensemble anchor AUC `0.9774`). TabPFN in-context adaptation vs GBDT
   retrain vs GBDT transfer-without-retrain.
3. Label scarcity / few-shot: shrink the labeled support set across several
   sizes and report how each model degrades as labels get scarce.
4. Minimal feature engineering: TabPFN on a reduced raw feature set vs the tuned
   GBDT 137-feature line, quantifying the feature-engineering-burden difference.

Per ADR 0015, TabPFN is judged on transfer, label scarcity, and minimal feature
engineering, not on a single headline AUC; tuned GBDT may still win raw AUC and
the report says so honestly. Any real-time FDD claim must use `PAST_SHIFTS`-only
features per ADR 0007 and ADR 0011, so offline and causal regimes stay explicit.

### Phase D results

Harness `scripts/run_m5_phaseD_foundation_vs_gbdt.py`; full numbers in
`data/processed/m5_phaseD_foundation_vs_gbdt.json`; analysis in
`docs/reports/m5-foundation-vs-gbdt.md`. Budget: 10,000 balanced fit rows, 4,000
fixed val rows, seeds `{42, 123, 999}` (mean ± std), RTX 4070 Laptop GPU,
`tabpfn==8.0.8` local weights, within the documented TabPFN-3 `1,000,000 × 200`
limit (no `ignore_pretraining_limits`).

+ **In-domain (80/20)**: TabPFN ROC-AUC `0.9925`, GBDT `0.9877`; TabPFN matches
  the tuned M3.4 ensemble `0.9928` at a 10k context but is ~100× slower at
  inference.
+ **Site transfer (PRIMARY, `site_id % 5 == 4`)**: among true cross-site models
  TabPFN-in-context ROC-AUC `0.9833` > GBDT-retrain `0.9797` (GBDT keeps PR-AUC).
  GBDT-transfer-without-retrain scores higher (`0.9882`) but is an easier setting
  (all-sites training), documented as such.
+ **Label scarcity**: TabPFN's clearest win — `+0.100` PR-AUC at 200 labels,
  narrowing as labels grow.
+ **Minimal FE**: hypothesis not supported — GBDT beats TabPFN on raw 17 features
  and TabPFN loses more without the engineered value-change lags.

Honest verdict: TabPFN adds real value in label-scarce and cross-site settings,
but does not dethrone the tuned GBDT headline, is far slower at inference, and
does not lower the feature-engineering burden. Real BDG2 cross-dataset transfer
and real-time FDD latency work are M5's next stage, Phase E (BDG2).

---

## M5 model-track close-out (Phase A–D complete)

The FDD model-selection stage is complete:

+ Phase A — readiness gate and frozen `src/lead` API (M4.5).
+ Phase B — model-track planning and ADR 0015 (issue #27).
+ Phase C — local TabPFN feasibility spike and metric audit (issues #30, #32).
+ Phase D — rigorous four-axis TabPFN-vs-GBDT comparison on GEPIII (issue #35).

Outcome for FDD model selection: GBDT remains the real-time deployment candidate
(sub-second inference); TabPFN is retained as an offline / label-scarce
bootstrapper where it wins (label scarcity, cross-site ROC-AUC), bounded by its
~6.3 ms/row inference latency (~100× slower than GBDT) and the TabPFN-3.0
research/internal-use license. The next stage is Phase E (FDD transfer to BDG2).

---

## Phase E: FDD transfer to BDG2

**Status**: Planned (docs-only); no data download, no implementation
**GitHub Issue**: to be opened at Phase E start

Phase E carries the selected FDD models from GEPIII to the BDG2 (Building Data
Genome 2) corpus. It is gated: each step below must clear before the next, so
ingestion does not start on an unverified schema or an unknown label situation.

1. **Labels (highest priority; registered as an unknown).** Confirm whether the
   BDG2 Fox subset carries a per-row anomaly/fault label. No ingestion begins
   until this is resolved — it decides the entire evaluation paradigm. See
   unknown #22.
2. **Evaluation-paradigm decision (record an ADR; one of three, per the label
   result).** (a) supervised AUC if per-row labels exist; (b)
   forecasting-residual scoring if only readings exist; (c) unsupervised /
   apply-a-GEPIII-trained detector with no BDG2 ground truth. See unknown #23.
3. **BDG2 ingestion contract.** Approve the data source, align the schema to
   `load_m3_frame` (site/building keys, weather, meter), define the site/building
   and site-held-out splits, and an optional label merge. This **rebuilds the
   retired skeleton but on real data**, owned by a single named owner. See
   unknown #24.
4. **Transfer-evaluation contract.** Treat the M3 site-held-out ensemble AUC
   `0.9774` and the Phase D cross-site TabPFN-in-context `0.9833` as **internal
   references only**, not BDG2 readiness claims. `site_id % 5 == 4` is a
   GEPIII-internal hold-out and is **not** cross-dataset transfer; BDG2 is the
   first true cross-dataset test.
5. **Causal discipline.** Any online / real-time FDD claim uses `PAST_SHIFTS`-only
   features per ADR 0007 and ADR 0011; offline and causal regimes stay explicit.
6. **Roles and limits.** TabPFN is bounded by the TabPFN-3.0 License
   (research / internal use only) and by inference latency (~6.3 ms/row, ~100×
   slower than GBDT); it is positioned as an offline / label-scarce bootstrapper.
   The real-time deployment candidate remains GBDT. These constraints are part of
   the Phase E plan, not afterthoughts.

---

## Issue Tracker Map (M5)

| Slice | GitHub issue | Status |
| --- | --- | --- |
| Phase B foundation-model planning | [#27](https://github.com/kuokuant-oss/lead-reproduction/issues/27) | Done |
| Phase C LEAD TabPFN feasibility spike | [#30](https://github.com/kuokuant-oss/lead-reproduction/issues/30) | Done |
| Phase C metric audit fix | [#32](https://github.com/kuokuant-oss/lead-reproduction/issues/32) | Done |
| Phase D BDG2 transfer and minimal-feature plan | [#31](https://github.com/kuokuant-oss/lead-reproduction/issues/31) | Superseded by #35 |
| Phase D slice 1 BDG2 ingestion skeleton | [#33](https://github.com/kuokuant-oss/lead-reproduction/issues/33) | Retired by #34 |
| Phase D retire premature BDG2 skeleton | [#34](https://github.com/kuokuant-oss/lead-reproduction/issues/34) | Done |
| Phase D TabPFN-vs-GBDT GEPIII comparison | [#35](https://github.com/kuokuant-oss/lead-reproduction/issues/35) | Done |
| M5 framing fix + model-track close-out + zh-TW report + Phase E plan | [#36](https://github.com/kuokuant-oss/lead-reproduction/issues/36) | Done |
| Phase E FDD transfer to BDG2 | _to be opened_ | Planned |
