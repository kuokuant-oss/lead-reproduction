# BDG2 evaluation paradigm

## Status

Accepted (2026-06-29)

## Context

Phase E transfers the FDD work from GEPIII to the real BDG2 corpus. Stage 0
measured the local BDG2 archive and found raw and cleaned meter files, metadata,
weather, and no native per-row anomaly labels. Unknown #22 is therefore
resolved as "no native labels", while unknown #23 owns the evaluation paradigm.

ADR 0017 defines `load_bdg2_frame` as an unlabeled real-schema loader. ADR 0018
isolates GEPIII-only assumptions, keeps the M3 row-offset path unchanged, and
adds BDG2-safe branches such as meter-aware value-change features. Phase E Step
1 also validated the weather join time basis empirically: chilledwater
temperature-load diagnostics found about `0.73` median absolute correlation and
`1` hour median best lag, supporting `(site_id, timestamp)` joins for the tested
sites. Unknown #25 remains open as a non-blocking follow-up for electricity
because its temperature-load diagnostic was weaker (`5.5` hour median best lag,
about `0.28` median absolute correlation).

The remaining decision is what a BDG2 "result" means before any model is trained
or any BDG2 headline metric is reported.

## Decision

Use **GEPIII-trained detector transfer to BDG2 as an unlabeled cross-dataset
baseline** as Phase E's primary evaluation paradigm.

The contract is:

+ Train or select the detector on GEPIII only. Apply it to BDG2 without using
  BDG2 native anomaly labels, because none exist in the measured archive.
+ Treat the primary BDG2 output as unlabeled score transfer, not supervised
  accuracy. Valid primary outputs are score coverage, score distributions,
  rank stability, site/meter/building stratification, runtime, and documented
  qualitative or operational review slices.
+ Report GEPIII-overlap and BDG2-only rows separately. `is_gepiii_overlap`
  distinguishes the 1,449 GEPIII-overlap buildings from the 187 BDG2-only
  buildings measured in Stage 0.
+ Headline cross-dataset results must be reported at least on BDG2-only
  buildings (`is_gepiii_overlap == False`) or on a held-out BDG2 site. A mixed
  all-BDG2 aggregate can be secondary, but it cannot be the sole headline.
+ GEPIII-overlap results are not true cross-dataset evidence by themselves:
  applying a GEPIII-trained detector to the 1,449 overlap buildings risks
  evaluating buildings whose 2017 GEPIII/Kaggle representation was part of the
  source domain. These rows are useful for bridge audits and calibration checks,
  not for an unqualified transfer claim.
+ `site_id % k` inside one dataset is a within-dataset split and must not be
  called cross-dataset transfer. The M3 site-held-out ensemble AUC `0.9774` and
  Phase D TabPFN-in-context site-transfer ROC-AUC `0.9833` are internal
  references only, not BDG2 readiness claims.
+ Any real-time FDD claim must use `PAST_SHIFTS`-only features per ADR 0007 and
  ADR 0011. Offline scoring and causal/online scoring must remain explicitly
  separated.

Raw/cleaned pseudo-labels may be used only as a **secondary sensitivity
analysis**. If used, metrics such as ROC-AUC, PR-AUC, precision, recall, or F1
must be labeled as pseudo-label metrics, for example "raw/cleaned pseudo-label
ROC-AUC". They must not be described as ground-truth BDG2 anomaly AUC.

If the selected transfer path includes electricity in anomaly scoring, unknown
unknown 25 must be handled before an electricity-specific headline. The Step 1
chilledwater result supports the general weather join, but electricity still
needs a per-site time-basis review if it becomes part of the scored BDG2 path.

## Alternatives Considered

### Forecasting-residual scoring

Forecasting residuals can score unexpected readings without anomaly labels and
fit naturally with meter time series. However, residual magnitude is a model
error signal, not a fault label. It is sensitive to occupancy, seasonality,
missingness, meter type, and forecast horizon. Without an external label source
or a pseudo-label definition, it can rank anomalies but cannot produce true
supervised AUC, precision, recall, or F1.

This remains a possible future BDG2-only unsupervised/forecasting branch, but it
is not the primary Phase E transfer paradigm because it does not directly test
the GEPIII-to-BDG2 detector transfer question.

### Unsupervised detection

Unsupervised detection avoids fabricating labels and can be run entirely on BDG2.
Its limitation is semantic: anomaly scores are only as meaningful as the
assumptions in the detector, and there is no native BDG2 ground truth for
supervised AUC. Different detectors may disagree for reasons unrelated to real
faults. It is useful as a baseline or triangulation method, not as the primary
answer to whether the GEPIII FDD detector transfers.

### GEPIII-trained detector transfer

This is selected because Phase E is explicitly about carrying the selected FDD
models from GEPIII to BDG2. It preserves the source-domain training contract and
makes the no-label reality explicit. The cost is that BDG2 outputs are unlabeled
scores unless paired with a separate review or pseudo-label layer. The contract
therefore prohibits calling those outputs supervised BDG2 AUC.

The `is_gepiii_overlap` field is a key constraint. GEPIII-overlap buildings are
not pure cross-dataset evidence, because the GEPIII-trained detector may have
seen the same buildings in the source domain. BDG2-only or held-out-site results
are required for any headline cross-dataset claim.

### Raw/cleaned difference pseudo-labels

Raw/cleaned differences are the closest measured analogue to a BDG2 cleaning
signal: cells present in raw meter files but set to missing in cleaned files can
be treated as candidate "cleaning removed this reading" events. This is useful
for sensitivity analysis, but it is noisy. The cleaned release can reflect unit
handling, gap treatment, coverage changes, and other data-cleaning decisions
that are not necessarily faults or anomalies. It can also miss abnormal readings
that were retained.

Therefore raw/cleaned pseudo-label metrics are allowed only with explicit
labeling as pseudo-label metrics. They cannot be used as BDG2 ground truth and
cannot by themselves justify a headline supervised BDG2 anomaly claim.

## Metric Contract

For unlabeled BDG2 transfer:

+ Do not report ROC-AUC, PR-AUC, precision, recall, or F1 as BDG2 ground-truth
  metrics.
+ Report score coverage, missing-score rate, score distribution summaries,
  runtime/latency, site/meter/building stratification, and stability checks.
+ Separate BDG2-only, GEPIII-overlap, held-out-site, and all-BDG2 aggregates.

For raw/cleaned pseudo-label sensitivity:

+ Metrics may include pseudo-label ROC-AUC, pseudo-label PR-AUC, precision,
  recall, and F1.
+ Every table, JSON key, figure, and prose reference must include
  `pseudo-label` or an equivalent explicit qualifier.
+ Pseudo-label metrics are secondary evidence and must be interpreted as
  agreement with BDG2 cleaning behavior, not agreement with true anomaly labels.

For any later external-label or human-review slice:

+ The label source, sampling frame, adjudication process, and eligible meters
  must be documented before supervised metrics are reported.

## Consequences

+ Unknown #23 is resolved: Phase E has an evaluation paradigm and metric
  contract before any BDG2 headline metric.
+ Phase E may proceed to implementation planning for unlabeled GEPIII-to-BDG2
  detector scoring, subject to the BDG2-only / held-out-site reporting rule.
+ No model training, BDG2 scoring, or BDG2 metric is created by this ADR.
+ ADR 0017's unlabeled loader contract and ADR 0018's GEPIII-assumption
  isolation remain in force.
+ Unknown #25 remains open and non-blocking unless electricity enters the
  anomaly-scoring path.
