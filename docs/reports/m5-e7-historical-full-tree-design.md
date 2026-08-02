# M5 E7 Historical-Full-Budget Support-Aware Tree Strategy

Protocol frozen on 2026-08-02 from Phase 0 source-map commit `2b4c9bc77c274912da9786c3f6b1cfaa43cfc84c`.

E7 is a pre-specified, local-CPU Tree study of whether the established
steam/hotwater support response generalises to a historical-full-budget model.
It uses only even buildings for fitting and building-disjoint even folds for
selection. The odd-building steam holdout is not used by the fitting or model
selection path.

The fixed F4 representation has 137 float32 columns built with
`timestamp_merge`. Each expert uses the historical four-component Tree
ensemble (LightGBM, XGBoost, CatBoost, HistGradientBoosting), one CPU thread
per component, fixed seed 42, and an equal probability mean. Historical
downsampling is preserved exactly as `[negative(seed=10), positive,
negative(seed=20), positive]`.

For each outer fold, support experts `s00`, `s01`, `s10`, and `s11` are paired
with full-support neutral experts `n00`–`n11`. Every neutral pair is required
to exactly match total, positive, and negative training-row counts. The OOF
meta learners use only five factorial score features, an isolated
StandardScaler plus L2 LogisticRegression, and C selected from
`0.001, 0.01, 0.1, 1, 10` with the pre-specified smaller-C tie rule.

The census is fixed at 160 OOF component fits plus 32 all-even component fits:
192 total. Each component runs in an explicit below-normal-priority subprocess
and writes a model, scaler, prediction, stdout/stderr, digest-validated
completion marker, and atomic status record. Invalid or partial units are
quarantined rather than resumed.

Formal scoring is a fresh process with an odd-label firewall. Evaluation is
post-score-freeze only, with exact weighted AP building bootstrap (1,000 draws,
SeedSequence namespace 7007) and all 162 leave-one-building omissions.
