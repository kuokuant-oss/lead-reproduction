# M5 E6 natural-prevalence factorial artifacts

Byte-identical copies of the artifacts in `data/processed/m5_e6_protocol/`,
which is gitignored with the rest of `data/processed/`.

`e6_protocol.json` is **frozen and human-authorised for execution**. It was
frozen only after the non-holdout throughput probe passed the 336-hour launch
gate, and it pins the exact inputs the run may use: the 24 E4 state digests, the
context caches, the 5.56 GB raw feature matrix, the microbatch partition, the
decision rules, and the newline-normalised digest of every source file that
participates.

The draft that preceded it is kept as `e6_protocol.DRAFT.json` so the difference
between what was proposed and what was authorised stays visible.

## Frozen protocol

| File | What it is |
|---|---|
| `e6_protocol.json` | the frozen protocol: question, wording constraints, repeat policy, estimand, grid, holdout, census, topology, fail-closed rules, prohibitions, source digests |
| `e6_input_manifest.json` | one digest per frozen artifact and per state, the single thing preflight checks against |
| `e6_state_manifest.json` | the 24 units in E4's frozen execution order, with state, context and comparator digests |
| `e6_microbatch_manifest.json` | the 516-microbatch partition per state; every call census is counted from here, never divided out of a row count |
| `e6_row_manifest.json` | full-holdout identity, verified; records which columns were read and that the score column was not |
| `e6_shard_manifest.json` | 12 row shards that tile the holdout |
| `e6_sentinel_manifest.json` | the 352-row sentinel: 8 repeats per state, and the rule that it may never enter an endpoint |
| `e6_tree_manifest.json` | the 24 fixed comparators, the laptop-only environment contract, and the bit-exact identity gate |
| `e6_bootstrap_manifest.json` | addressable RNG in namespace 6006, the co-primary subset, and the segment-degeneracy disclosure |
| `e6_decision_rules.json` | the four-term vocabulary, the confirmation conditions, and the mandatory E4/E5/E6 comparison columns |
| `e6_cost_model.json` | measured throughput and the R1 / R8 / R1_PLUS_SENTINEL comparison |
| `e6_throughput_probe.json` | the non-holdout probe that decided the launch gate |
| `e6_protocol.DRAFT.json` | the pre-freeze draft, kept for comparison; not a launch artifact |

## Two things worth reading before the results

**The estimand is not E4's or E5's.** Each unit contributes one *canonical
single-process batched pass*: one state, one process UUID, a fixed microbatch
partition in a fixed order, every row scored exactly once. E4 and E5 averaged 8
repeats per fit before forming contrasts; E6 does not. The clustered intervals
are conditional on that realised pass and do not cover same-state full-holdout
inference-repeat variation.

**The segment clustering is 91.8% singletons** on the co-primary subset
(545,430 of 594,297 clusters). It is kept unchanged, and kept in the decision
rule, but it must not be described as cluster-level corroboration equal in
independence to the building interval.
