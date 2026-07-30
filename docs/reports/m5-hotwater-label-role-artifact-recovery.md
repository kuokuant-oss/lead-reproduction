# M5 hotwater label-role factorial: artifact recovery

Date: 2026-07-30. Scope: Path A only. No paper manuscript, Path B, site
transfer, retrieval, transfer matrix, 500k scaling, or representation-ablation
work is changed by this recovery.

## Search census and stopping rule

The following locations were searched recursively for `tabpfn_fit`, portable
state, checkpoint, `joblib`/pickle, booster/model, cache, resume, and scratch
artifacts: the repository; `data/processed/m5_hotwater_label_factorial`; the
repository `.scratch`, `.cache`, `.tabpfn-cache`, `models`; `C:\tmp`; the user
cache; and the local temporary-job directory. Source and runbook references to
state persistence were searched separately.

| Location / artifact | Finding | Completeness | Usable for these 12 cells? |
|---|---|---:|---:|
| `data/processed/m5_hotwater_label_factorial/predictions/` | 48 fixed-query score files and result metadata | scoring only | no |
| `.../scalers/` | 3 pooled-reference scalers | partial | no model state |
| `.../manifests/` | 12 exact ordered-row designs and audits | complete design | yes, for exact refit only |
| `.scratch/m6_tabpfn_context/` | persisted TabPFN states for M6 contexts | different context/data contract | no |
| `.scratch/codex-tabpfn-fit-persistence-gate-*` | save/load preflight state | smoke/preflight only | no |
| `models/`, factorial tree paths, job/cache roots | no factorial booster model or portable fitted state | absent | no |

The recovery search stops here: a candidate is reusable only if it has the
matching factorial manifest digest, model/scaler arm, F4 feature contract,
model seed, and a loadable state that scores an arbitrary query. No candidate
met all criteria. Score files, unrelated M6 states, and checkpoint weights do
not meet this criterion.

## Authorized exact-design recovery

The refit writes only to `data/processed/m5_hotwater_label_factorial/recovery/`
and preserves the original predictions as the comparison baseline. It consumes
the original 12 manifests (three context seeds × four cells), preserves their
ordered raw-index digests, uses model seed 42 and the original TabPFN
hyperparameters, and reconstructs both scaler arms. Each completed cell saves:

- TabPFN `model.tabpfn_fit` or a joblib tree ensemble containing all four base
  boosters, plus the exact scaler;
- fit metadata, model/scaler digests, ordered context digest, environment and
  checkpoint provenance;
- query-independent loader/scorer entry point.

There is deliberately no wall-time timeout. The state and query score artifacts
are durable per cell; rerunning resumes from a valid saved state.

## Predeclared reproduction gate

Before the frozen independent query can be scored, recovery predictions on the
original 352-row screening query must meet all of these criteria across all 48
model × seed × arm × cell values:

| Check | Tolerance |
|---|---:|
| score mean absolute difference | <= 0.005 |
| score Spearman correlation | >= 0.999 |
| each primary-estimand absolute difference | <= 0.02 |
| factorial-effect absolute difference | <= 0.04 |
| sign consistency | required when original effect abs >= 0.01 |

The primary estimands for this gate are HW 0–1 within-meter rank gap, HW 0–1
pairwise AUC, and steam-positive × HW-negative AUC. `recall@FPR=.001` remains
excluded because the original normal-query resolution is inadequate. Any failed
gate stops independent-query scoring and triggers reproducibility diagnosis.

## Recovery outcome

The recovery completed all 24 TabPFN states and all 24 tree ensembles (each
with four saved boosters), scalers, manifests, state digests, and model-specific
environment provenance. The gate failed solely in the TabPFN comparison:
trees passed 24/24 cell score and primary-estimand checks, whereas TabPFN failed
24/24 score checks, 16/24 primary-estimand checks, 16 factorial-effect
magnitude checks, and one required effect-direction check. The independent
query is therefore locked. A no-fit load-and-rescore verification is the final
permitted reproducibility diagnostic before any new scientific decision.

The no-fit verification is now complete. All 48 states are loadable and finite;
tree reload is bit-exact. In contrast, TabPFN reload versus its own recovery
fit-time predictions has maximum MAE 0.006650, maximum absolute difference
0.196808, and minimum Spearman 0.995948 across 24 states. Portable-state
availability has therefore been recovered and repeated-inference variation is
measured. This supersedes the former deterministic stopping rule: Path A is not
closed on bit instability. The frozen independent query remains unscored while
a separately predeclared 8.0.8 repeated-inference variance pilot determines
whether composition contrasts are estimable with adequate precision. Path B is
still deferred.
