# M5 E5 — frozen 192-row independent replication

The design, what was frozen before any 192-row score existed, and how the run
was executed. Results are in `m5-e5-factorial-results.md`, the verdict in
`m5-e5-decision.md`, and the engineering record in `m5-e5-remote-execution.md`.

## What E5 asks

> Does the steam negative-support response E4 established reproduce on a
> completely independent, never-scored 192-row query with zero overlap with the
> 352-row screening query?

E5 re-explores nothing. It re-selects no endpoint, no cell, no seed, and it did
not adjust its protocol after seeing any result. The verdict vocabulary and
every threshold were frozen at commit `c5d38c4b…`, before the query had ever
been scored.

## The query

| | |
|---|---|
| Path | `data/processed/m5_hotwater_label_factorial/independent_query/` |
| `queries.npz` SHA-256 | `d780f0f8a96c47f49ffe061a72906728f1301056555350cabd979348aa41a2a0` |
| `raw_index` SHA-256 | `2fc4a638a2a0880f2b4d7feac87875c941d155f5fe5172b75b13d041b654fa16` |
| Rows | 192, all unique |
| Strata | `steam_positive` 64, `hw01_negative` 64, `hw01_positive` 64 |
| Overlap with the 352-row query | **0 rows** |
| Chilledwater rows | 0 |
| Previously scored | never — no prediction artifact existed anywhere for it |

The protocol also checks something easy to overlook: the endpoint code selects
strata by `meter` and `anomaly`, not by stratum name. On this query
`meter==2 & anomaly==1` must be exactly `steam_positive` and
`meter==3 & anomaly==0` exactly `hw01_negative`, or the co-primary endpoints
would quietly mean something different than they did in E4. Both hold.

## Pure re-scoring

E5 reuses E4's 24 persisted states. Nothing is fitted.

Rather than rely on the runner simply not calling `fit`, `m5_e5_guard` replaces
every fit entry point on TabPFN and on the tree estimators with a raising stub
before any model is loaded, and the runner refuses to score unless the guard is
armed. Three tests call `fit` through the guard and require the exception.

Recorded coverage: `fits_performed = 0` across all 24 units, re-verified by the
importer from the unit records rather than taken on trust.

## Per-state lifecycle

Each of the 24 states, in its own fresh process, in E4's frozen execution order
(`63ca76f1…`, never reshuffled):

1. verify the state digest against the frozen manifest
2. reload with `load_fitted_tabpfn_model`
3. verify TabPFN 8.0.8, requested `n_estimators=8`, `auto_scale=False`,
   effective `n_estimators_=8`, and the low-memory executor's four runtime
   containers all 8
4. 8 same-process inference repeats, each producing one length-192 score vector
5. atomic completion marker before the next state

Every E5 score is therefore already a fresh-process reload; no separate
fresh-process diagnostic applies.

## The scaler, verified rather than assumed

Each unit inherits E4's arm; no arm was re-selected.

`frozen_reference` loads the persisted
`scalers/seed{S}_pooled_reference.joblib` and checks its digest.
`cell_specific` was never persisted by E4 — it was fitted on the fly — so E5
rebuilds it with E4's exact code and then **proves** the rebuild by transforming
the same raw context and comparing to the scaled `X_train` stored inside the E4
state, which is what E4 actually fitted on. Scoring is refused unless the
reproduction is exact. All 24 units passed.

Two dtype details are load-bearing and both cost a debugging cycle:

- The context and query matrices are used in the dtype E4 used (float32).
  Upcasting to float64 before `transform` changes the arithmetic and the
  comparison fails on the last bit for reasons unrelated to the scaler.
- The tree half initially scored through a float64 path while its gate
  certified a float32 one. That was caught by re-derivation, not by a digest,
  and is described in `m5-e5-remote-execution.md`.

### A representation difference inside the treatment

Verifying the scaler surfaced a fact that constrains how E5's result may be
read. In cell 00 the context contains no hotwater rows, so `meter` takes three
values instead of four and TabPFN classifies it as **CATEGORICAL** and
ordinal-encodes it after our scaling; in the other three cells all 137 features
stay NUMERICAL.

So removing hotwater support does not only remove rows: for the reference cell
it also flips a feature's modality. **E5 therefore tests independent
reproduction of E4's negative-support intervention as a whole. A successful
replication may not be described as having isolated the hotwater-normal
reference as the sole mechanism.** This is stated again in the decision.

## Execution split, and why

TabPFN scored on gpu-host, as E4 did. The fixed tree comparator scored on the
laptop, under a human override recorded in `e5_tree_execution_override.json`,
because the reloaded ensembles reproduce E4's frozen comparator bit for bit only
in the environment that fitted them. No tree was refit, and no gpu-host tree
output entered the analysis. The full evidence is in
`m5-e5-remote-execution.md`.

This is an execution-provenance limitation, not a scientific factor.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `e5_protocol.json` | `f417ca5c58e5607085e00cc910b55de91b6f373d2c2922d724e1485a5005a4d0` |
| `e5_state_manifest.json` | `9e6f50835470d946e658ff0a4e643453b11c5a2ed5a73374d783fddb7b35a1e6` |
| `e5_repeat_manifest.json` | `f93df07fe5c0daa33616897475aef4e515656036ce953c5aae39c672181b1eb2` |
| `e5_query_audit.json` | `a84eccf5ff83d7c2c948a1161ef7acc2a8ef15c068f685773b01a8db29b85b68` |
| `e5_tree_execution_override.json` | `79c0ced5b3c3601ee5961a0c441a92a9fe1ce0c559053fa4b96c0ffbea947f4a` |
| shared 192×137 feature matrix | `e6b44c9ccc902cd6dfa6f1fce07ad98d9af1af52dab32faaf36b148d12ab0482` |

## What E5 does not do

The 10,137,155-row holdout was not scored. No model was fitted, no tree refit,
no representation ablation, no Path B, no TabPFN 8.1.0, and no manuscript
change.
