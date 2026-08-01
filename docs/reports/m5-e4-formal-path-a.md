# M5 E4 — formal Path A

The design, what was frozen before any fit ran, and how the run was executed.
Results are in `m5-e4-factorial-results.md`; the verdict is in
`m5-e4-decision.md`.

## Design

| | |
|---|---|
| Decomposition | 3 context seeds × 4 cells × 2 scaler arms = **24 fits** |
| Fits per unit | exactly 1 |
| Repeats per fit | 8 same-process inference repeats → **192** `predict_proba` calls |
| Context seeds | 42, 123, 999 |
| Cells | 00, 01, 10, 11 (hotwater positive-support × negative-support) |
| Scaler arms | `cell_specific`, `frozen_reference` |
| Model | TabPFN 8.0.8, v3 checkpoint, F4/137 features, N=20,000, model seed 42 |
| Ensemble | `n_estimators=8`, `auto_scale_n_estimators=False` |
| Query | the original 352-row screening query, `06874156…7718575` |
| Comparator | matched-row fixed trees, not refit, not tuned |
| Base commit | `e7aa8f72a26eceb926038c33c29043ec0c3ce2aa` |

`r` is repeated *inference* on one fitted state, per the canonical policy's
estimand `Y[c,s,a,r] = mu[c,s,a] + epsilon[c,s,a,r]`. It is not a fit replicate,
so E4 is 24 fits and not 192. No model-seed factor was added.

## The ensemble contract

E3's `n_estimators=8` was correct but had never been pinned: it came from an
argparse default, and E3 recorded only the *requested* value. Before E4 was
authorised, the four E3 states were audited by reading `init_params.json` and
`fitted_attrs.joblib` out of each persisted archive and by re-reading
`n_estimators_` after a fresh GPU reload. All four were 8, and so were the
runtime containers the low-memory executor actually iterates.

E4 pins the contract and enforces it in three independent places:

| Where | What |
|---|---|
| protocol | `requested=8`, `auto_scale=False`, `required effective n_estimators_=8`, mismatch is a hard failure |
| runner, after every fit | `n_estimators_`, `len(ensemble_configs_)`, and the executor's `configs`/`pipelines`/`pipeline_seeds`/`subsample_feature_indices` must all be 8 |
| importer | re-reads the effective value out of the persisted state and refuses to trust the runner's own record |

All 24 fits reported effective `n_estimators_ = 8`, and the importer confirmed
it independently from all 24 states.

Two dimensions that must not be confused: the 8 internal ensemble members are
combined *inside* one `predict_proba` call; the 8 inference repeats are eight
separate calls on one fitted state. Internal members are not replicates of any
kind and never enter a mean, an interval, or a bootstrap.

## Randomised execution schedule

The first frozen order was deterministic, which left each unit's identity
perfectly collinear with its position in the run. A later ruling required
randomisation, and the schedule was re-frozen without touching anything
scientific.

One `Generator(PCG64, 42)` is consumed in a fixed sequence: permute the 12
`(context_seed, cell)` blocks, then permute the 2 scaler arms within each block.

Blocking rather than full shuffling is what makes randomisation affordable. Both
arms of a block run consecutively, so that block's raw feature matrix is built
and digest-verified once and then transformed twice. The measured cost of that
choice: a cache-miss unit took 573 s at 16.2 GB, its cache-hit partner 69 s at
2.7 GB. Randomising at unit level would have doubled the expensive half of the
run for no additional protection.

Realised order digest: `63ca76f1167768252b29992fd791c450ba33447f5908b8938f1b67d0ecc732e3`

The order is frozen in the artifact, not re-derived at run time; resume replays
it and skips only completed units.

## What was frozen before the first fit

`e4_protocol.json` (`6efdc937…`), `e4_fit_manifest.json` (`8af4a541…`),
`e4_repeat_manifest.json` (`220de1c6…`) and `e4_input_manifest.json`, covering:
the ensemble contract, the schedule and its digest, all 24 context manifests,
all 24 tree comparators, the 3 frozen scalers, the query, the factorial
formulas, the clustered-uncertainty rulings recorded verbatim, the output
schema, the completion census, and the resume/fail-closed rules.

Source files are digested over **newline-normalised** content. A raw byte digest
taken on the Windows checkout disagreed with the Linux clone for three files
nobody had edited, purely because of CRLF versus LF. No tracked file was
rewritten to make the check pass; binary inputs keep raw byte digests.

### Execution provenance versus analysis provenance

The protocol was **not** re-frozen after the run. Re-freezing would have replaced
the document the run executed under, so the executed artifact is left exactly as
it was, and the split is recorded instead:

| | Commit | Files |
|---|---|---|
| **executed under** | `ac0310d5160398ec8d611ad9d049a72e120dbb71` | `m5_e4_runner.py`, `m5_e4_protocol.py` — both unchanged since |
| **analysed under** | `a85297b65bc5a9209dd2cd62653f6065e830c991` and later | `m5_e4_endpoints.py`, `m5_e4_clustered.py` |

Verifying the frozen `source_digests` against the working tree today shows
exactly two drifted files, both on the analysis side, and none on the execution
side. The change was the addition of `endpoint_value()`, a single-endpoint fast
path for the clustered bootstrap, which
`test_endpoint_value_matches_the_full_dict` pins to `endpoints()` across random
inputs — so the quantity being computed did not change, only the cost of
computing it.

Nothing that produced a score, a state, or a repeat record moved after the run.

## A structural fact about the scaler axis

A context seed's `frozen_reference` scaler is fitted on that seed's
`hw_pos_present__hw_neg_present` matrix — which *is* cell 11. So at cell 11 the
two scaler arms are the same transform.

The 24 fixed tree comparators carry only 21 distinct digests, and the three
collisions are exactly the cell-11 arm pairs, one per seed. All 24 units still
ran; the census requires at least 21 distinct states and permits collisions only
there.

The run then produced something the protocol did not anticipate: **all 24
TabPFN states are distinct**, including the cell-11 pairs whose fit inputs are
byte-identical. See `m5-e4-factorial-results.md` — it turns cell 11 into a null
control for the scaler axis.

## Execution

Single GPU worker, strictly sequential, one subprocess per unit, no
`ProcessPoolExecutor`. See `m5-e4-remote-execution.md` for the engineering
record and the measurements behind the single-worker decision.

## What E4 does not do

The frozen 192-row query was not scored. The 10,137,155-row holdout was not
scored. No tree was refit, no representation ablation was run, TabPFN 8.1.0 was
not used, and the manuscript was not touched.
