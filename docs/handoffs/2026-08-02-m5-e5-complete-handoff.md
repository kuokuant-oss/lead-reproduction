# M5 E5 independent replication complete — handoff (2026-08-02)

## 0. Status

E5 is **complete and frozen**. Nothing is running. There is no resume command.
The next stage requires explicit human authorisation and is not started here.

| | |
|---|---|
| Branch | `m5-e5-independent-replication` |
| Base | E4 completion `9a336faa5f80ab7acb68533057d908d0723a90cf` |
| Protocol artifact | `f417ca5c58e5607085e00cc910b55de91b6f373d2c2922d724e1485a5005a4d0` |
| Tree execution override | `79c0ced5b3c3601ee5961a0c441a92a9fe1ce0c559053fa4b96c0ffbea947f4a` |
| Shared 192×137 feature matrix | `e6b44c9ccc902cd6dfa6f1fce07ad98d9af1af52dab32faaf36b148d12ab0482` |
| Result root | `data/processed/m5_e5_independent_replication/` |

## 1. Result

**A. Response replication: REPLICATED.**
**B. TabPFN-specific replication: REPLICATED.**

Adding hotwater-negative support raises steam separation by **+0.4049 AUC** and
**+0.5992 margin** on a 192-row query that shares no row with the 352-row query
E4 used, with 3/3 seeds positive, both scaler arms positive, and all eight
clustered intervals excluding zero. The TabPFN-minus-tree gap (**+0.083 AUC**,
**+0.101 margin**) clears the same bar.

## 2. Six things a successor should not have to rediscover

**The gap shrank, and that is not hidden.** E4's TabPFN-minus-tree gap was
+0.124 AUC / +0.152 margin; E5's is +0.083 / +0.101 — about a third smaller. It
still meets every pre-declared condition, and E5 tested direction and
significance rather than magnitude agreement, but do not report E5 as having
reproduced the gap *size*.

**The AUC does not saturate on this query.** E4 had six of 24 fits pinned at
exactly 1.000000 with zero variance, all cell 01. Here cell 01 sits at
0.985–0.993 and no state has zero variance, so the ceiling that qualified E4's
AUC does not apply to E5.

**The interaction now contradicts itself across endpoints.** AUC +0.288 and
margin −0.158, both excluding zero — opposite signs. E4 had them disagreeing
only about significance. Compatible with "not established", and firmer.

**TabPFN's inference is still not bitwise reproducible.** 192 of 192 repeats
gave distinct score digests, a third independent confirmation.

**Removing hotwater support also flips a feature's modality.** In cell 00 the
context has no hotwater rows, `meter` drops to three levels, and TabPFN
classifies it as categorical and ordinal-encodes it after scaling; the other
three cells keep all 137 features numerical. So E5 replicates the intervention
**as a whole** and does not isolate the hotwater-normal reference as the sole
mechanism. This constraint is written into the results report and the decision.

**The fixed tree comparator only reproduces where it was fitted.** On the laptop
all 24 ensembles reproduce E4's frozen 352-row comparator bit for bit; on
gpu-host none do, differing by a mean of 8.1e−03 — comparable to the gap under
test. E5 therefore scored trees on the laptop under a recorded human override
and prohibited gpu-host tree output. TabPFN scored entirely on gpu-host, as in
E4. No tree was refit. This is an execution-provenance limitation, not a
scientific factor.

## 3. Coverage

| | |
|---|---|
| states reloaded | 24 / 24 |
| same-process repeats | 192 / 192 |
| score vector length | 192 |
| tree score vectors | 24 / 24, all bit-exact against E4 |
| **fits performed** | **0** |
| distinct states / process UUIDs | 24 / 24 |
| effective `n_estimators_` | 8 in all 24, re-read from the persisted states |
| scaler verified exact | 24 / 24 |
| clustered draws valid | 1000/1000 in all four bootstraps |
| stderr / interrupted / temp files | 0 / 0 / 0 |

## 4. Measured throughput

| Quantity | Measured |
|---|---:|
| state reload | 0.9–1.2 s |
| one unit (reload + 8 repeats of 192 rows) | 69.7 s |
| whole TabPFN run | ~28 min |
| 192-row feature build (once, gpu-host) | 242 s |
| 24 tree gates + 24 tree scorings (laptop) | a few minutes |
| clustered bootstrap, 1000 draws | 118–121 s per AUC endpoint per clustering; 2 s per margin |

E5 is ~5× faster than E4 because it builds no feature matrix — 89% of E4's
wall-clock — and scores 192 rows rather than 352.

## 5. E6, for when it is authorised

E6 reloads the same 24 states and scores 10,137,155 rows — four orders of
magnitude more than E5. Size it from the existing sharded full-test logs, not
by extrapolating E5's 8 s per repeat, which is dominated by fixed overhead.

Two things E5 makes concrete for E6:

- **The tree comparator will have the same problem at full-test scale.** If E6
  needs a tree comparison, the trees must again score in the laptop environment
  or a bit-exact-reproducing equivalent, and 10.1M rows there is a very
  different proposition from 192.
- **If E6 is split across machines, give each machine a row shard and have it
  score all 24 states**, never a subset of states. Sharding by state makes
  "machine" confounded with every state contrast. The laptop and gpu-host differ
  in CUDA build (cu126 vs cu130) and GPU architecture, so this is real.

## 6. Not authorised by this handoff

E6 full test, any refit, Path B, representation ablation, 500k, site transfer,
tree refit, TabPFN 8.1.0 as science, manuscript changes, and scoring the
10,137,155-row holdout.

**The 10,137,155-row holdout was not scored in this round.** Every E5 score
vector is length 192, verified by the importer.

## 7. Reproducing the result

```bash
# re-validate the canonical root against itself; recomputes every TabPFN
# endpoint and re-derives all 24 tree vectors from the persisted ensembles
python scripts/m5_e5_import.py \
  --staged     data/processed/m5_e5_independent_replication \
  --canonical  data/processed/m5_e5_independent_replication \
  --repo-root  . \
  --feature-npz <the shared 192x137 artifact>

python scripts/m5_e5_analysis.py \
  --canonical data/processed/m5_e5_independent_replication --repo-root .

python scripts/m5_e5_decision.py \
  --canonical data/processed/m5_e5_independent_replication
```

The importer is read-only without `--apply`. Clustered draws are addressable:
draw *d* of cluster type *t* comes from
`SeedSequence([20260730, 5005, code[t], d])`, so a single draw reproduces on its
own and no result depends on loop order. Namespace 5005 keeps E5's draws
separate from E4's 4004 while the construction stays identical.

## 8. Artifact digests

| Artifact | SHA-256 |
|---|---|
| `e5_protocol.json` | `f417ca5c58e5607085e00cc910b55de91b6f373d2c2922d724e1485a5005a4d0` |
| `e5_state_manifest.json` | `9e6f50835470d946e658ff0a4e643453b11c5a2ed5a73374d783fddb7b35a1e6` |
| `e5_repeat_manifest.json` | `f93df07fe5c0daa33616897475aef4e515656036ce953c5aae39c672181b1eb2` |
| `e5_query_audit.json` | `a84eccf5ff83d7c2c948a1161ef7acc2a8ef15c068f685773b01a8db29b85b68` |
| `e5_tree_execution_override.json` | `79c0ced5b3c3601ee5961a0c441a92a9fe1ce0c559053fa4b96c0ffbea947f4a` |
| `e5_summary.json` | `9f7bb274dec26cbbb95f0d92846cf2085997c30901d75a27f895a30e2760b7ce` |
| `e5_factorial.json` | `da10ea169d7d565235c6f47aa37cf10a108857bcc3041c01f5780d10328b1fcc` |
| `e5_clustered.json` | `67af305e2c47d79b296fe9e6f2fe361f045badff0cf75380447e9b39543029fa` |
| `e5_decision.json` | `8a40b882ed5330cd69d852b32e0299fffc9a162c6189d38405498bfd3cf16458` |
| tree manifest (laptop) | `0a32a54b9130259b1e12ee4e2b9e473d1ebe8b8eabfd88804a3bfc028d31c9bf` |
| TabPFN archive (gpu-host) | `845cf5704017875daf507b6ecc567fb47162c2f5e24e9cb834291b597434871f` |

## 9. Evidence order

```
E4 formal Path A        (done)
  → E5 frozen 192-row independent replication   (done, REPLICATED)
    → E6 complete other-half natural-prevalence full-test confirmation
```

E6 remains a factorial confirmation on rows that prior context-curve runs
already scored — new states, not new rows. See
`docs/reports/m5-final-holdout-identity-audit.md`.
