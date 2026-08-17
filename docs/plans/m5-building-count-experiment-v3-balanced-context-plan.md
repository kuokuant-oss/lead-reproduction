# M5 building-count experiment V3 plan: balanced matched contexts

- Status: training contexts and dedicated scheduler prepared; bounded validation
  pending; conditional V2-to-V3 switch authorized by the user
- Date: 2026-08-17
- Planning branch: `m5-building-count-v3-balanced-context-plan`
- Experiment version: `m5_building_count_v3_balanced_context`
- Context builder: `scripts/prepare_m5_building_count_v3_balanced_contexts.py`
- Preparation gate:
  `experiments/m5_building_count_v3_balanced_context/audit/training_context_gate.json`
- Dedicated scheduler: `scripts/run_m5_building_count_v3.py`
- Parent protocol: `docs/plans/m5-building-count-experiment-v2-plan.md`
- Balance reference: `docs/reports/m5-matched-context-breakdown.md`
- Execution policy: `docs/policies/long-running-research-execution.md`

Training-context preparation and scheduler implementation are complete. The
user authorized stopping V2 and starting V3 after V3 implementation tests and
bounded validation are ready. Validation outputs remain non-scientific, and
formal execution is mechanically blocked unless the 40/40 validation gate
passes and the explicit formal authorization flag is present.

## 1. Decision summary

Treat this as V3, not as a replacement or silent revision of V2. V2 measures
performance under the natural class prevalence produced by its frozen
building ladders and label-blind row cap. V3 will measure performance with an
exact 50:50 training context drawn from the same frozen building support.

V3 freezes:

- building seeds 42--51;
- the ordered, strict-nested K=10/20/50/100 building identities from V2;
- context totals 5,000/10,000/25,000/50,000 rows;
- the 137-feature contract, model seeds, tree contract, and TabPFN contract;
- the canonical building-disjoint odd-building holdout and its natural
  prevalence;
- the model-pair requirement that Tree and TabPFN receive byte-identical
  ordered training rows.

V3 changes only the training-row policy intentionally: each context contains
exactly half unique anomalies and half unique normals, without replacement.
Building selection is never rerun and class labels never enter building
identity selection.

Nesting is a hard invariant at two levels. For every seed, `B10` must be an
exact ordered prefix of `B20`, `B20` of `B50`, and `B50` of `B100`. The V3
ordered training-row vectors must follow the same prefix relation. Failure at
either level blocks preparation; it never triggers an independent draw,
building replacement, or redraw.

The precise estimand is therefore **the effect of a balanced context drawn
by seeded class-stratified random sampling from the same selected building
support**. V3 does not preserve, prefer, optimize, or constrain V2 row
identities or per-building row quotas.

## 2. Research question and hypotheses

Primary question:

> Holding the selected building identities, total context size, features,
> models, seeds, and holdout fixed, how does exact 50:50 class balance alter
> the Tree-versus-TabPFN building-count curve?

Primary hypotheses:

1. Balancing will reduce the portion of seed-to-seed variance caused by very
   different natural anomaly prevalence.
2. The effect will be largest at K=10 and K=20, where both building
   composition and natural anomaly prevalence vary most across seeds.
3. The effect may be meter-specific; no direction is assumed in advance.

A negative result is informative: if seed variance remains large after
balancing, building/meter/site composition rather than the global label ratio
is the more likely driver.

## 3. Existing feasibility evidence

The current V2 composition audits already show that every one of the 40
seed-by-K building pools has enough full unique rows for the proposed fixed
total 50:50 context:

| K | context rows | required per class | minimum full anomalies across seeds 42--51 | limiting seed | feasible |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | 5,000 | 2,500 | 5,662 | 42 | yes |
| 20 | 10,000 | 5,000 | 10,089 | 43 | yes |
| 50 | 25,000 | 12,500 | 36,127 | 42 | yes |
| 100 | 50,000 | 25,000 | 84,739 | 44 | yes |

Normals are not limiting. These counts are a planning observation, not a
substitute for the formal raw-row feasibility gate.

The V2 capped row vectors and per-building quotas are not sampling inputs to
V3. They may be compared after the draw for interpretation, but they cannot
change row priority, eligibility, acceptance, or rejection.

## 4. Frozen building identity

The following V2 manifests are immutable inputs:

- seeds 42--46:
  `data/processed/m5_building_curve/sensitivity/building_candidate_pilot/building_ladder_seed{seed}.json`
- seeds 47--51:
  `experiments/m5_building_count_v2_seed47_51/audit/building_ladder_seed{seed}.json`

V3 must record and verify the exact source-manifest SHA-256, the ordered
building vector, and the per-K building-vector digest. A mismatch is a hard
failure. No missing class support, meter imbalance, or undesirable context
composition may trigger a building redraw or ladder rejection.

## 5. Balanced row contract

### 5.1 Targets

| K | total rows | anomalies | normals | prevalence |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 5,000 | 2,500 | 2,500 | 0.5000 |
| 20 | 10,000 | 5,000 | 5,000 | 0.5000 |
| 50 | 25,000 | 12,500 | 12,500 | 0.5000 |
| 100 | 50,000 | 25,000 | 25,000 | 0.5000 |

The candidate pool for `(building_seed, K)` is every eligible unique training
row from the exact K selected even-numbered buildings. It is not limited to
the already capped V2 row vector, because that vector intentionally preserves
natural prevalence and is insufficient for 50:50 in many cells.

No oversampling, duplicated anomaly, synthetic row, class weight, or balanced
holdout is permitted.

### 5.2 Deterministic nested construction

The implementation follows the semantics of `nested_balanced_indices()` from
`scripts/run_m5_tabpfn_single_context_scaling.py`: exact balance, a frozen
seed, strict prefixes, and sampling without replacement. The only additional
eligibility restriction is membership in the frozen K-building set.

Use `balance_seed=42`, separate from `building_seed` and `model_seed`. Stable
row priority must be a platform-independent function of `(raw_index,
balance_seed)` with `raw_index` as the final tie-breaker. Row priority must not
use V2 membership, building, site, meter, anomaly severity, or any composition
diagnostic.

For each building seed, process K in 10, 20, 50, 100 order:

1. Freeze the exact ordered building prefix `B[s,K]` from the V2 manifest.
2. Partition all eligible unique rows from `B[s,K]` into anomaly and normal
   pools using only the binary label.
3. Keep the complete smaller-K V3 row prefix unchanged.
4. Exclude rows already selected, rank the remaining rows independently
   within each class by seeded row priority, and take exactly the additional
   class count required at K.
5. Append the new rows anomaly/normal in deterministic alternation. Because
   each K target adds the same number from both classes, every completed
   prefix remains exactly 50:50.

Hard outcomes at every K:

- the smaller-K ordered vector is an exact prefix of the larger-K vector;
- all rows are unique and belong to one of the exact selected buildings;
- row count and label counts exactly match the target table;
- Tree and TabPFN consume the exact same ordered row vector.

If any hard outcome is infeasible, preparation fails. It must not redraw
buildings, duplicate rows, lower a target silently, or fall back to natural
prevalence. There is no per-building minimum, V2-row preference, quota repair,
or composition-based acceptance rule.

#### 5.2.1 Formal definition

For building seed `s` and budget `K`, define:

- `B[s,K]`: the first K ordered building IDs in the frozen V2 ladder;
- `U+[s,K]`: all unique anomaly-row IDs from buildings in `B[s,K]`;
- `U-[s,K]`: all unique normal-row IDs from buildings in `B[s,K]`;
- `S[v3,s,K]`: the ordered V3 balanced context;
- `C[K]`: 5,000, 10,000, 25,000, or 50,000;
- `Q[K] = C[K] / 2`: the required count for each class.

The balance requirement is global within the complete context:

- `|S[v3,s,K]| = C[K]`;
- `count(anomaly=1) = Q[K]`;
- `count(anomaly=0) = Q[K]`;
- `prevalence = 0.5` exactly;
- `unique(S[v3,s,K]) = S[v3,s,K]`;
- every row's building is in `B[s,K]`.

It does **not** require each building, site, or meter to be 50:50. Enforcing
any of those marginal distributions would introduce another intervention and
would answer a different question.

#### 5.2.2 Seeded random selection

Within each class, rank eligible rows solely by:

1. platform-independent seeded pseudo-random priority from `(raw_index,
   balance_seed)`;
2. ascending `raw_index` as the final tie-breaker.

Take the first required number from each class after excluding the prior-K
prefix. This is a reproducible class-stratified random draw without
replacement, conditional only on the frozen building support and nesting
requirement. V2 row membership and all building/site/meter diagnostics are
ignored.

#### 5.2.3 Worked K10-to-K20 example

At K=10, `C[10]=5,000`, so preparation randomly selects exactly 2,500 unique
anomalies and 2,500 unique normals from the ten frozen buildings. These 5,000
ordered rows become an immutable prefix.

At K=20, those 5,000 rows are kept exactly. From all eligible rows in the full
twenty-building pool, excluding the K10 rows, the sampler randomly adds 2,500
unique anomalies and 2,500 unique normals. The result is exactly 10,000 rows
with 5,000 of each class, and the original K=10 vector remains its first 5,000
rows.

Additional K=20 rows may come from either the ten old or ten new buildings.
The intervention controls the eligible building support, exact class counts,
and total context size; it imposes no building, site, or meter quota.

#### 5.2.4 Why one fixed balance seed

Use the same `balance_seed=42` for all building seeds. The experimental
replicates are the ten independently drawn building ladders; varying the row
seed inside the primary experiment would add a second source of Monte Carlo
variation and obscure the building-seed estimand. A row-seed sensitivity study
may be run later as a separate analysis, but it must not be mixed into the
primary V3 average.

### 5.3 Row-allocation disclosure

Per-building, site, and meter composition are outcomes of the seeded random
draw, not sampling constraints. For every seed/K/building, record:

- full rows and full anomalies;
- V3 selected rows and anomalies;
- V3 positive and negative contribution shares;
- selected-row SHA-256.

Also report positive-row concentration by seed/K: anomaly-source building
count, top-1 share, top-3 share, HHI, and effective source-building count
`1 / HHI`. These values may be compared with V2 after selection, but they
must never affect eligibility, row order, acceptance, rejection, or redraw.

## 6. Matched-context and provenance gates

Preparation must emit one strict gate artifact covering all 40 seed-by-K
pairs. It passes only if all of the following hold:

1. Source V2 manifest bytes and ordered building IDs match their pinned
   digests.
2. K building sets are strict nested prefixes and contain exactly K IDs.
3. Every V3 raw row is unique, even-building training data, and belongs to
   the selected K IDs.
4. Exact total, anomaly, normal, and prevalence targets pass.
5. V3 row vectors are strict ordered prefixes across K.
6. Tree and TabPFN context row IDs, row order, labels, and feature-matrix
   digests match for every pair.
7. The holdout raw-row vector and digest equal the canonical V2 holdout
   (`6cfebd1cb2bb818f69806c0f14d66a84b81c53d37a716badd48c17b86210d893`).
8. Fit and holdout buildings have zero overlap.
9. Feature count is exactly 137 and model contracts match V2.
10. Validation outputs are isolated and cannot be finalized as scientific
    results.

The gate should include `expected_pairs=40`, `checked_pairs=40`, and a
per-pair failure reason. Missing pairs are failures.

## 7. Model and evaluation contract

Do not change the V2 model hyperparameters. In particular:

- Tree uses the frozen no-early-stopping ensemble and all K buildings for fit.
- TabPFN uses the V2 estimator count, fit mode, memory mode, and model seed.
- Both models score the complete canonical odd-building holdout in its
  natural prevalence.
- Prediction chunking is operational only and must not change scores.

The primary reported metrics are per-meter PR-AUC and ROC-AUC. Overall and
per-site metrics remain audit artifacts or secondary appendices, not headline
tables.

## 8. Analysis plan

The comparison is paired by `(building_seed, K, model, meter)`:

- V2: natural-prevalence capped context;
- V3: exact 50:50 balanced context from the same building support.

For each model, K, and meter, report:

- V2 mean and sample SD across seeds 42--51;
- V3 mean and sample SD across the same seeds;
- the seed-paired delta `V3 - V2`, summarized as mean and sample SD;
- all ten raw paired values in a machine-readable CSV.

A paired bootstrap or exact paired randomization interval may be secondary,
but it must operate on seed-level deltas and be declared before inspecting V3
scores. With ten seeds, uncertainty and effect size take priority over a
binary significance claim.

Interpretation must separate:

- reduced variance associated with class balance;
- residual variance associated with building/site/meter composition;
- any effect plausibly mediated by anomaly concentration in a few buildings.

## 9. Artifact layout

Use new roots; never overwrite V2:

- audit and frozen contexts:
  `experiments/m5_building_count_v3_balanced_context/audit/`
- formal runs:
  `data/processed/m5_building_curve/v3_balanced_context/`
- bounded validation:
  `data/processed/m5_building_curve/NON_SCIENTIFIC_VALIDATION_v3_balanced_context/`
- final report:
  `docs/reports/m5-building-count-experiment-v3-balanced-context.md`

Required audit artifacts include a source-manifest registry, feasibility CSV,
balanced-context manifest per seed, per-building contribution CSV,
matched-context gate JSON, holdout contract JSON, unit census, and a formal
run plan.

Every model cell must record the experiment version, Git commit, source
manifest digest, ordered context digest, holdout digest, all three seed roles,
model contract, environment versions, and completion state.

## 10. Implementation and execution phases

### Phase 0: protocol review

Approve this plan and freeze the exact V3 name, artifacts, balance seed, and
reporting contract. Do not run models.

### Phase 1: CPU-only feasibility and context preparation

Build the 40 balanced contexts from frozen manifests and raw data. Emit all
gates and concentration audits. Re-running preparation must reproduce every
ordered row digest byte-for-byte.

### Phase 2: implementation tests

Tests must cover determinism, no replacement, exact class counts, selected
building membership, strict K prefixes, label-only random priority,
manifest-drift rejection, Tree/TabPFN row identity, holdout identity, atomic
writes, corrupt-checkpoint rejection, deterministic resume, missing-unit
finalization failure, mode guards, validation caps, and heartbeat freshness.

### Phase 3: bounded non-scientific validation

Run the complete 40-pair/80-model-cell validation census with small explicit
caps for both context and holdout. A validation context must remain balanced
and include both classes. Expensive phases, including feature construction and
prediction, must obey the caps.

Validation passes only with exit code 0 and a strict 40/40 matched-context
gate. Validation scores are never scientific results.

### Phase 4: formal authorization and execution

Formal execution requires a successful validation gate and the explicit
`--authorize-formal` launch flag. The user has authorized the conditional
V2-to-V3 switch once those gates are satisfied. The fixed order is group-first,
budget-major within each group:

1. seeds 42--46: K10 all five seeds;
2. seeds 42--46: K20 all five seeds;
3. seeds 42--46: K50 all five seeds;
4. seeds 42--46: K100 all five seeds;
5. only after that group is complete, repeat K10, K20, K50, K100 for seeds
   47--51.

Within each seed/K pair, run Tree and TabPFN against the already frozen shared
context. Stop the V2 supervisor and active V2 child only after the V3 code,
focused tests, and launch preflight are ready. Confirm those exact PIDs have
stopped before starting V3 validation/formal work so the two scientific queues
never compete for CPU, RAM, or GPU.

The scheduler defaults to `--mode plan` and cannot launch models in that mode.
Validation uses the isolated
`NON_SCIENTIFIC_VALIDATION_v3_balanced_context` root. Formal launch additionally
requires `--mode formal --authorize-formal`, a clean committed implementation,
the 40/40 frozen-context gate, and the 40/40 validation matched-context gate.

Use bounded prediction chunks whose measured p95 runtime is at most ten
minutes, atomic per-unit checkpoints, deterministic resume, and a heartbeat
artifact. No automatic wall-clock timeout, kill, restart, or silent retry may
change scientific work.

### Phase 5: finalization

Finalize only after all 80 formal model cells and all 40 model-pair gates are
complete. Generate raw results, paired V2/V3 deltas, seed summaries, meter
tables, concentration diagnostics, and the report from artifacts rather than
from terminal logs.

## 11. Non-goals and prohibited shortcuts

V3 does not:

- redraw or reject buildings using labels;
- modify any V2 manifest or result;
- prefer V2 row identities or preserve V2 per-building quotas;
- require a minimum selected-row count for any building;
- balance or constrain rows by building, site, or meter;
- balance the holdout or change evaluation prevalence;
- duplicate anomalies or sample with replacement;
- add synthetic data, class weights, or per-meter label balancing;
- tune model hyperparameters after seeing V3 results;
- suppress seeds or meters whose result is unfavorable;
- treat validation output as evidence.

Per-meter 50:50 and equal-per-building row allocation are distinct follow-up
interventions. They should not be mixed into this primary V3 contrast.

## 12. Acceptance criteria

The V3 protocol is ready for bounded validation only when:

- all ten source ladders and their digests are pinned;
- all 40 balanced contexts pass exact-size, exact-balance, uniqueness,
  building-membership, and nesting gates;
- per-building and anomaly-concentration metrics are audited as outcomes only;
- Tree and TabPFN input digests match for all 40 pairs;
- the canonical holdout digest matches V2;
- protocol and long-running execution tests pass;
- validation and formal output roots are provably isolated;
- no active V2 process was stopped, restarted, or modified to prepare V3.
