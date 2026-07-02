# M3 validation credibility investigation

## Scope

This handoff covers GitHub issue #53: an additive investigation into whether the
M3 `0.992`-level validation result on GEPIII is credible, overfit, or overly
dependent on validation design.

The investigation asks whether validation is independent enough. It does not
change M3 defaults, golden metrics, `load_m3_frame`, `downsample_indices`,
StandardScaler behavior, or `lead.__all__`.

All model-training investigations use opt-in `row_offset_meter_aware` features
to avoid the INV-1 meter-crossing confound.

Artifacts:

+ `scripts/run_gate_label_join_integrity.py`
+ `scripts/run_inv4_shuffle_ablation.py`
+ `scripts/run_inv5_time_holdout.py`
+ `scripts/run_inv6_train_val_gap.py`
+ `scripts/run_inv7_per_building_distribution.py`
+ `scripts/run_inv8_sampling_fragility.py`
+ `data/processed/gate_label_join_integrity.json`
+ `data/processed/inv4_shuffle_ablation.json`
+ `data/processed/inv5_time_holdout.json`
+ `data/processed/inv6_train_val_gap.json`
+ `data/processed/inv7_per_building_distribution.json`
+ `data/processed/inv8_sampling_fragility.json`

All result JSONs were written with `write_json_with_provenance`.

## GATE: label positional-join integrity

Verdict: **credible**.

The overall anomaly rate is `0.065021`, matching the expected `6.50%` within
the `0.002` tolerance.

The real anomaly labels show strong contiguous runs, unlike the within-building
shuffle null:

+ Real positive rows in runs of at least 24: `0.938716931639576`.
+ Max null positive rows in runs of at least 24: `0.01167463182991828`.
+ Real p95 run length: `96.0`.
+ Max null p95 run length: `3.0`.
+ Real max run length: `8784`.

Conclusion: the positional label join does not look randomly scattered. No
GATE red flag.

## INV-4: label-shuffle root cause

Verdict: **needs report caveat, not red flag**.

The full row-offset-meter-aware feature line still has unstable shuffled-label
AUC, but feature-group ablation isolates the leading source as value-change
features.

Mean shuffled-label AUC across 8 shuffle seeds:

+ full: `0.5197251329669951` with std `0.06538536558209732`.
+ remove `meter_reading`: `0.5250535941081068`.
+ remove value-change features: `0.5092436400785258`.
+ remove building metadata: `0.5278672206845696`.
+ remove weather: `0.5252785242887497`.

Removing value-change moves the distribution closest to `0.5`. The result does
not identify a hard leakage path, but it does make the old generic phrase
`residual structure` too vague. Report wording should attribute the shuffled
signal specifically to value-change residual structure under shuffled labels.

## INV-5: causal time-held-out split

Verdict: **credible for this check**.

The causal same-building time holdout did not degrade relative to the causal
building split. Both used `row_offset_meter_aware` and `PAST_SHIFTS`.

+ Building split causal single LightGBM AUC: `0.990737`.
+ Time-holdout causal single LightGBM AUC: `0.992828`.
+ Single delta, time minus building: `+0.0020911375218332084`.
+ Building split causal ensemble AUC: `0.991491`.
+ Time-holdout causal ensemble AUC: `0.993677`.
+ Ensemble delta, time minus building: `+0.0021856592620101978`.

Conclusion: no evidence from this check that the headline depends on
same-building time-neighborhood leakage. The time-holdout result is easier than
the building split on this M3 slice.

## INV-6: train/validation gap

Verdict: **needs report caveat**.

Fit-set and full-train scores are near perfect, while validation remains lower.
The full-train rows and fit-set rows are close to each other, so the model is
not only memorizing duplicated fit-set rows. But the full-train-vs-validation
gap is real.

LightGBM:

+ fit-set AUC: `0.998340`.
+ full train-buildings AUC: `0.998312`.
+ validation AUC: `0.992537`.
+ full train minus validation: `+0.005774593985864662`.

4-model ensemble:

+ fit-set AUC: `0.9996557661879552`.
+ full train-buildings AUC: `0.9996221399428573`.
+ validation AUC: `0.9936946662228049`.
+ full train minus validation: `+0.005927473720052423`.

Conclusion: report the train/validation gap as a capacity/stability caveat.
This is not a red flag that invalidates the validation score by itself.

## INV-7: per-building distribution and bootstrap CI

Verdict: **needs report caveat**.

The row-level aggregate remains high, but the building-level distribution shows
why a single aggregate AUC is too compressed.

LightGBM:

+ row aggregate AUC: `0.9925372152184203`.
+ effective validation buildings with both classes: `234`.
+ per-building median AUC: `0.9995647581456906`.
+ p10/p90: `0.9750640799239463` / `1.0`.
+ min/max: `0.40607199817726136` / `1.0`.
+ building-bootstrap mean CI: `[0.9802077951870567, 0.9928124898369466]`.

4-model ensemble:

+ row aggregate AUC: `0.9936946662228049`.
+ effective validation buildings with both classes: `234`.
+ per-building median AUC: `0.9998622797484491`.
+ p10/p90: `0.9893052339616705` / `1.0`.
+ min/max: `0.8041707369482783` / `1.0`.
+ building-bootstrap mean CI: `[0.992786798321831, 0.9970437085348297]`.

The row aggregate delta, ensemble minus LightGBM, is `+0.0011574510043845798`,
but the building-bootstrap CIs overlap. Treat the ensemble lift as directionally
positive, not clearly separated at building level.

High-score small primary-use slices were flagged for `primary_use_enc` values
`3`, `7`, `8`, `11`, `12`, `13`, `14`, and `15`, all with `<= 4` effective
buildings and median AUC near `0.999` or `1.0`.

Conclusion: report per-building distribution and CI instead of relying only on
the single row aggregate. The small-slice high scores need caveat language.

## INV-8: sampling fragility

Verdict: **needs report caveat**.

M3-style downsample seed sweep:

+ mean AUC: `0.9924050920217852`.
+ std AUC: `0.00024248931628974223`.
+ min/max range: `0.0008054886404132988`.
+ canonical seeds `(10, 20)` AUC: `0.9925372152184203`.

Clean 50:50 without positive duplication:

+ mean AUC: `0.9924425432222953`.
+ std AUC: `0.00026737581264994114`.
+ min/max range: `0.0009007584369150612`.
+ clean mean minus M3-style mean: `+0.00003745120051013018`.

Conclusion: the result is not dependent on positive duplication in the mean,
but validation AUC varies by more than the `0.0005` noise floor across sampling
seeds. Add a sampling-stability caveat.

## Overall verdict

No investigation produced a red flag that invalidates the M3 headline. The GATE
passed, time holdout did not collapse, and support from multiple checks remains
consistent with a strong GEPIII validation result.

However, the report should not present the headline as a single unqualified
number. A separate report-fix slice should add:

+ value-change-specific explanation for shuffled-label residual structure;
+ train/validation gap caveat;
+ per-building AUC distribution and building-bootstrap CI;
+ note that the ensemble-vs-LightGBM lift is not clearly separated by
  building-bootstrap CI;
+ sampling-seed stability caveat.

No ADR, golden update, or public API change is warranted from this investigation.

## Verification so far

Local checks run during the investigation:

+ `uv run python -m py_compile scripts/run_gate_label_join_integrity.py`
+ `uv run ruff check scripts/run_gate_label_join_integrity.py`
+ `uv run python scripts/run_gate_label_join_integrity.py --out data/processed/gate_label_join_integrity.json`
+ `uv run python -m py_compile scripts/run_inv4_shuffle_ablation.py`
+ `uv run ruff check scripts/run_inv4_shuffle_ablation.py`
+ `uv run python scripts/run_inv4_shuffle_ablation.py --out data/processed/inv4_shuffle_ablation.json`
+ `uv run python -m py_compile scripts/run_inv5_time_holdout.py`
+ `uv run ruff check scripts/run_inv5_time_holdout.py`
+ `uv run python scripts/run_inv5_time_holdout.py --out data/processed/inv5_time_holdout.json`
+ `uv run python -m py_compile scripts/run_inv6_train_val_gap.py`
+ `uv run ruff check scripts/run_inv6_train_val_gap.py`
+ `uv run python scripts/run_inv6_train_val_gap.py --out data/processed/inv6_train_val_gap.json`
+ `uv run python -m py_compile scripts/run_inv7_per_building_distribution.py`
+ `uv run ruff check scripts/run_inv7_per_building_distribution.py`
+ `uv run python scripts/run_inv7_per_building_distribution.py --out data/processed/inv7_per_building_distribution.json`
+ `uv run python -m py_compile scripts/run_inv8_sampling_fragility.py`
+ `uv run ruff check scripts/run_inv8_sampling_fragility.py`
+ `uv run python scripts/run_inv8_sampling_fragility.py --out data/processed/inv8_sampling_fragility.json`

Full pre-commit closeout gate has not yet been run.
