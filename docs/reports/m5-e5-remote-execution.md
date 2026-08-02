# M5 E5 — execution record

Engineering record. No scientific finding here; results are in
`m5-e5-factorial-results.md` and the verdict in `m5-e5-decision.md`.

## Two hosts, and why

| Half | Host | Environment |
|---|---|---|
| TabPFN, 24 states × 8 repeats | **gpu-host** | WSL2 Ubuntu, Python 3.12.13, RTX 5070 Ti, TabPFN 8.0.8, torch 2.12.1+cu130, CUDA 13.0, numpy 2.4.6, pandas 3.0.3, sklearn 1.8.0, scipy 1.17.1 |
| 24 fixed tree comparators | **laptop** | Windows-11-10.0.26200-SP0, Python 3.13.13, lightgbm 4.6.0, xgboost 3.2.0, catboost 1.2.10, sklearn 1.8.0, numpy 2.4.6, joblib 1.5.3 |

The protocol required all scientific scoring on gpu-host. Execution showed that
rule cannot be satisfied for the tree comparator, and a human ruled on it. The
protocol was not rewritten; the ruling is a supplemental artifact,
`e5_tree_execution_override.json` (`79c0ced5…`).

### The failure that forced the ruling

The comparator identity gate stopped on the first unit:

```
HARD FAILURE seed42__cell11__frozen_reference:
  the reloaded tree does not reproduce E4's 352-row comparator (max |diff| 1.245e-01)
```

Four candidates were eliminated before concluding:

| Candidate | Test | Result |
|---|---|---|
| the scaler | recompute with recovery, frozen, and refitted scalers | all three gave the **identical** error → not the scaler |
| the query matrix | E4's cached 352-row matrix vs a fresh build | **bit-identical** → not the input |
| threshold flipping | per-row comparison | **352/352 rows differ, none exact** → systemic, not a few rows crossing a split |
| tree reliability | E4's own `reproduction_gate.json` | **24/24 tree rows passed**; its 13 failures were all TabPFN |

The cause is recorded in `recovery/environment_provenance_trees.json`: the
ensembles were fitted on the laptop under Windows 11 and Python 3.13.13.
Gradient-boosting inference is not bit-reproducible against gpu-host's
Linux/Python 3.12.13 build, whose compiled lightgbm/xgboost/catboost packages
were reinstalled on 2026-08-01.

| Host | bit-exact rows | max abs diff | mean abs diff |
|---|---:|---:|---:|
| laptop (fitted the trees) | **352 / 352** | **0.000e+00** | **0.000e+00** |
| gpu-host | 0 / 352 | 1.245e-01 | 8.149e-03 |

The ruling was OPTION_A: TabPFN stays on gpu-host, trees score where they were
fitted, no refit, and no gpu-host tree output may enter the analysis. Nothing
scientific moved — endpoints, states, seeds, arms, the clustered estimator and
both thresholds are unchanged, and the override asserts each of those
explicitly so a later edit that quietly relaxed one fails its test.

**This is an execution-provenance limitation, not a scientific factor.**

## One shared input

Both hosts scored the same 192×137 feature artifact,
`e6b44c9ccc902cd6dfa6f1fce07ad98d9af1af52dab32faaf36b148d12ab0482`, built once
on gpu-host in 242 s and transferred with a digest check. The laptop scorer
refuses to run if the artifact it is given does not match the digest the
override fixed, so the two halves cannot drift onto different inputs.

## The tree gate

Before any 192-row tree score was produced, all 24 units had to reproduce E4's
frozen 352-row comparator with `max_abs_diff == 0` and 352/352 rows exact.
Sampling and tolerances are both refused; one failure stops the whole of E5.

**Result: 24/24 bit-exact.**

## TabPFN run

Single tmux session, single GPU worker, strictly sequential, one subprocess per
state, no `ProcessPoolExecutor`, in E4's frozen order (`63ca76f1…`, never
reshuffled).

| | |
|---|---|
| Started | 09:29 |
| Completed | 24/24 units, 192/192 repeats |
| Mean interval per unit | 69.7 s |
| State reload | 0.9–1.2 s |
| Elapsed | ~28 min |
| Peak VRAM | ~3.06 GB |
| stderr | 0 bytes |
| interrupted / stray temp files | 0 / 0 |
| swap | 0 throughout |

Far faster than E4's 2 h 10 m because E5 builds no feature matrix — that was 89%
of E4's wall-clock — and scores 192 rows rather than 352.

## Three defects caught, all by verification rather than by luck

**The E4 states were not on gpu-host.** E4's results were imported to the
laptop; the remote's canonical E4 root held only protocol files. Installed from
the remote's own `~/outputs/m5-e4-formal-path-a/` and verified against all 24
digests in the frozen E5 state manifest.

**The E5 runner upcast the context to float64.** E4 handed the cached float32
matrix to the scaler directly. Transforming a float64 copy does not give the
same float32 result in the last bit, so the scaler check failed by ~1e−7 on a
unit whose scaler was the persisted frozen one. The loader now keeps the stored
dtype and `verify_scaler` refuses a context that is not float32. The protocol
was re-frozen before any scoring, since nothing had been scored yet.

**The tree gate certified float32 while the scoring ran float64.** The laptop
scorer built its 352-row gate matrix as float32 and passed 24/24, then scored
the 192 rows through an upcast float64 copy — so the vectors were not produced
by the path the gate had certified. This was caught by the importer
re-deriving the tree scores, not by any digest, and the trees were rescored.

## Return and import

| | |
|---|---|
| TabPFN archive | 242 files, `845cf5704017875daf507b6ecc567fb47162c2f5e24e9cb834291b597434871f` |
| per-file re-verification after extraction | 242/242 |
| tree manifest | `0a32a54b9130259b1e12ee4e2b9e473d1ebe8b8eabfd88804a3bfc028d31c9bf` |

Packaging refused to run until it had verified 24 units, 192 repeats, zero
interrupted, zero temp files, **`fits_performed = 0`**, 24/24 scaler gates, and
**zero gpu-host tree outputs**.

The importer trusts neither half. It recomputes every TabPFN endpoint from the
raw score vectors, re-reads the effective ensemble size from each persisted E4
state, and **re-derives all 24 tree vectors** from the persisted ensembles and
the shared feature artifact rather than believing the tree manifest.

### Fault injection

Eleven faults were injected into copies of the real staged results; **all eleven
were rejected**. Eight had a manifest regenerated over the tampering so digests
alone could not catch them.

| Injected fault | Manifest rebuilt | Result |
|---|---|---|
| perturbed TabPFN repeat endpoint | no | rejected |
| falsified TabPFN endpoint | yes | rejected |
| a fit recorded (`fits_performed = 1`) | yes | rejected |
| scaler gate marked inexact | yes | rejected |
| one TabPFN repeat deleted | yes | rejected |
| tree bit-exact gate falsified | yes (tree) | rejected |
| **tree score vector edited** | yes (tree) | rejected — only by re-derivation |
| tree half claims a different feature artifact | yes (tree) | rejected |
| tree half claims it ran on gpu-host | yes (tree) | rejected |
| tree given artificial replicates | yes (tree) | rejected |
| one tree unit removed | — | rejected |

The seventh case is the one that justified the re-derivation: on the first pass
it was **accepted**, because a tampered vector plus a regenerated tree manifest
is internally consistent. Digest checks cannot catch that; recomputation can.

Import then ran as one directory-level `os.replace` per unit after a dry run
against a throwaway root, and the canonical root re-validated in place.
