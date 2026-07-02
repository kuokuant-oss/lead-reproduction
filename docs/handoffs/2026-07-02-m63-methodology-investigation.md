# M6.3 methodology preflight investigation

## Scope

This handoff closes the three-methodology-risk investigation opened in GitHub
issue #51 before M6.3 GBDT vs TabPFN comparison work.

The investigation is not a feature slice. It does not change the frozen M3
default, and it does not update README or milestone reports.

Artifacts:

+ `scripts/run_inv1_meter_aware_impact.py`
+ `scripts/run_m5_phaseD_foundation_vs_gbdt.py`
+ `scripts/run_inv3_scarcity_unique_support.py`
+ `data/processed/inv1_meter_aware_impact.json`
+ `data/processed/inv2_phaseD_val_variance.json`
+ `data/processed/inv3_scarcity_unique_support.json`

All result JSONs were written through `write_json_with_provenance`.

## INV-1: meter-crossing impact

Verdict: **material**.

The M3 `row_offset` default can cross meter series within multi-meter buildings.
INV-1 measured whether replacing it with `row_offset_meter_aware` or
`timestamp_merge` changes headline AUC or model ordering.

80/20 split:

+ `row_offset` single LightGBM AUC: `0.9920119520500562`.
+ `row_offset_meter_aware` single LightGBM AUC: `0.9925372152184203`, delta
  `+0.0005252631683640985` versus `row_offset`, outside the `0.0005` noise
  floor.
+ `row_offset` ensemble AUC: `0.9927886432126508`.
+ `row_offset_meter_aware` ensemble AUC: `0.9936946662228049`, delta
  `+0.0009060230101540778`, outside the noise floor.
+ `timestamp_merge` ensemble AUC: `0.9933764865287564`, delta
  `+0.0005878433161056185`, outside the noise floor.

50/50 split:

+ AUC deltas stayed within the noise floor.
+ `row_offset_meter_aware` changed the 4-model ensemble member ranking.

Feature-layer contamination was large:

+ 80/20 train: `62.84%` multi-meter rows; `57.85%` changed value-change cells;
  lag-diff median absolute delta `84.9662`, p95 `2171.3281`.
+ 80/20 validation: `64.13%` multi-meter rows; `59.27%` changed value-change
  cells; lag-diff median `99.8167`, p95 `2849.6069`.
+ 50/50 train: `62.78%` multi-meter rows; `57.55%` changed value-change cells;
  lag-diff median `85.9370`, p95 `2734.3799`.
+ 50/50 validation: `63.42%` multi-meter rows; `58.73%` changed value-change
  cells; lag-diff median `89.6330`, p95 `1961.6162`.

Conclusion:

+ Do not change the frozen M3 default.
+ For M6 comparison work, use an additive opt-in meter-aware comparison path.
+ Follow-up issue #52 records the opt-in M6 meter-aware path and ADR-format
  draft.

## INV-2: Phase D validation-resampling variance

Verdict: **mixed; material for wording and uncertainty framing**.

The original Phase D result used a single fixed 4000-row validation subsample
with `--val-seed 42`. INV-2 reran the existing Phase D axes with validation
seeds `42`, `123`, `999`, `2025`, and `7`, paired with model/fit seeds `42`,
`123`, and `999`.

GBDT-only validation-resampling scale:

+ In-domain GBDT ROC-AUC mean/std: `0.9882811921611052 +/- 0.0018962514030752822`.
+ Site-transfer GBDT-retrain ROC-AUC mean/std:
  `0.9757587919725298 +/- 0.004419861755850379`.
+ Low-support label-scarcity PR-AUC variance is large: support 200 GBDT PR-AUC
  std `0.05584288801407651`; support 500 std `0.05316778014388243`.

Scarcity non-monotonicity:

+ The fixed `val_seed=42` result where GBDT support 5000 exceeded support 10000
  is not stable.
+ Across validation seeds, support 5000 GBDT ROC-AUC is
  `0.9872253450473442 +/- 0.0019353071373120418`.
+ Support 10000 GBDT ROC-AUC is
  `0.9882811921611052 +/- 0.0018962514030752822`.
+ Treat the fixed-val non-monotonicity as validation-subsample noise.

TabPFN vs GBDT paired deltas:

+ In-domain ROC-AUC delta, TabPFN minus GBDT:
  `+0.002551642206091387 +/- 0.0023805273647723754`, with range
  `-0.0025065083989337378` to `+0.005909191315395024`.
+ In-domain PR-AUC delta:
  `+0.006828998399945611 +/- 0.005582204864494338`, with range
  `-0.00452337882834164` to `+0.021086334232934822`.
+ Site-transfer ROC-AUC delta:
  `+0.004382936585925834 +/- 0.0022838540764750313`, with all paired deltas
  positive.
+ Site-transfer PR-AUC delta:
  `-0.003256436472272132 +/- 0.010448280813250648`, crossing zero.

Label-scarcity PR-AUC paired deltas, TabPFN minus GBDT:

+ support 200: mean `+0.11619185827670259`, bootstrap CI
  `[0.09532175321796252, 0.13883616141357472]`.
+ support 500: mean `+0.06252050885003807`, CI
  `[0.049121291025626755, 0.07731614803371029]`.
+ support 1000: mean `+0.06969123367004233`, CI
  `[0.05686577962417218, 0.08286003795488733]`.
+ support 2000: mean `+0.025993885456296602`, CI
  `[0.020888874515285843, 0.03207188223304701]`.
+ support 5000: mean `+0.006634524792164983`, CI
  `[0.0033653229590709373, 0.009919131053682669]`.
+ support 10000: mean `+0.008082066028811237`, CI
  `[0.005490749773818245, 0.010726094240497391]`.

Conclusion:

+ Phase D fixed-validation headline wording should be downgraded. Validation
  resampling is a real uncertainty source.
+ Low-support TabPFN PR-AUC advantage remains robust and can be carried into
  M6.3.
+ In-domain average TabPFN lift is directional, not a strong robust claim.
+ Site-transfer ROC-AUC advantage is stable; site-transfer PR-AUC advantage is
  not.

## INV-3: unique support rows

Verdict: **immaterial**.

The full M3-compatible `ds_idx_full` intentionally contains duplicates because
`downsample_indices` preserves `[negs1, pos, negs2, pos]`:

+ `ds_idx_full_rows`: `4285104`.
+ `ds_idx_full_unique_rows`: `3137276`.
+ `ds_idx_full_duplicate_rate`: `0.2678646772633756`.

However, the actual Phase D support samples are effectively literal at the
requested sizes:

+ support 200: unique rows `200-200`, max duplicate rate `0.0`.
+ support 500: unique rows `500-500`, max duplicate rate `0.0`.
+ support 1000: unique rows `1000-1000`, max duplicate rate `0.0`.
+ support 2000: unique rows `1999-2000`, max duplicate rate `0.0005`.
+ support 5000: unique rows `4997-5000`, max duplicate rate `0.0006`.
+ support 10000: unique rows `9988-9992`, max duplicate rate `0.0012`.

The materiality threshold was duplicate rate `> 0.01`. No support-size/seed cell
crossed it.

Conclusion:

+ No additive unique-row scarcity rerun is required.
+ Existing Phase D support-size wording is literal enough for M6.3.
+ Do not change `downsample_indices` or Phase D defaults.

## External issue status

+ Issue #51 was created at slice start.
+ INV-1 and INV-2 verdict comments were posted to #51.
+ Follow-up issue #52 was created for the material INV-1 opt-in meter-aware
  M6 comparison path.
+ The INV-3 issue comment was prepared but not posted during this local handoff
  because `gh issue comment` was blocked by the tool usage limit. Post the INV-3
  verdict to #51 before final commit/closeout.

## Verification so far

Local checks run during the investigation:

+ `uv run python -m py_compile scripts/run_inv1_meter_aware_impact.py`
+ `uv run ruff check scripts/run_inv1_meter_aware_impact.py`
+ `uv run python scripts/run_inv1_meter_aware_impact.py --out data/processed/inv1_meter_aware_impact.json`
+ `uv run python -m py_compile scripts/run_m5_phaseD_foundation_vs_gbdt.py`
+ `uv run ruff check scripts/run_m5_phaseD_foundation_vs_gbdt.py`
+ `uv run python scripts/run_m5_phaseD_foundation_vs_gbdt.py --smoke --skip-tabpfn --axes in_domain --val-seeds 42 123 --out data/processed/inv2_phaseD_val_variance_smoke.json`
+ `uv run python scripts/run_m5_phaseD_foundation_vs_gbdt.py --skip-tabpfn --axes in_domain site_transfer label_scarcity --val-seeds 42 123 999 2025 7 --out data/processed/inv2_phaseD_val_variance_gbdt_only.json`
+ `uv run python scripts/run_m5_phaseD_foundation_vs_gbdt.py --axes in_domain site_transfer label_scarcity --val-seeds 42 123 999 2025 7 --out data/processed/inv2_phaseD_val_variance.json`
+ `uv run python -m py_compile scripts/run_inv3_scarcity_unique_support.py`
+ `uv run ruff check scripts/run_inv3_scarcity_unique_support.py`
+ `uv run python scripts/run_inv3_scarcity_unique_support.py --out data/processed/inv3_scarcity_unique_support.json`

Full pre-commit closeout gate has not yet been run.
