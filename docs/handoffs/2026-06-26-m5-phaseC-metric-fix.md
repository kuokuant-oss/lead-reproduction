# M5 Phase C handoff: metric audit fix

## Scope

This slice closes issue #32 before Phase D implementation. It audits the Phase C
TabPFN metric path, regenerates the Phase C evidence JSON, and documents timing
semantics. It does not change the frozen `src/lead` path, M3.2 80/20 mod5 split,
offline row-offset regime, M3 downsampling seeds, shared scaler, local
checkpoint path, or no-cloud/no-token execution rule.

## Finding

`scripts/run_m5_phaseC_tabpfn_spike.py` already computed TabPFN
precision/recall/F1 from TabPFN `predict_proba` output through
`classification_metrics`; the identical threshold metrics were not copied from
the GBDT anchor. A regression test now stubs the TabPFN classifier and verifies
that `fit_tabpfn` derives metrics from the stubbed TabPFN probabilities.

The regenerated bounded validation slice still gives both models the same
`0.5`-threshold confusion matrix: `TP=53`, `FP=49`, `TN=896`, `FN=2`. AUC differs
because the probability rankings differ: GBDT AUC `0.986955266955267`, TabPFN
AUC `0.9904377104377105`.

## Evidence

+ Result archive: `data/processed/m5_phaseC_tabpfn_spike.json`.
+ GBDT anchor: precision `0.5196078431372549`, recall
  `0.9636363636363636`, F1 `0.6751592356687898`, fit+predict
  `0.8529972999822348` seconds.
+ TabPFN: precision `0.5196078431372549`, recall `0.9636363636363636`,
  F1 `0.6751592356687898`, cold fit+predict `6.507047699997202` seconds.
+ TabPFN timing breakdown: model initialization `1.5884365998208523` seconds,
  fit `1.0767969998996705` seconds, `predict_proba`
  `3.8418111000210047` seconds.

## Hygiene

+ `git check-ignore data/processed/m5_phaseC_tabpfn_spike.json` produced no
  output, so the evidence JSON remains tracked.
+ `uv.lock` and `pyproject.toml` stayed clean; CUDA torch remains a manual local
  environment step rather than a pinned dependency.
+ `.tabpfn-cache/` remains ignored and the checkpoint is not staged.
