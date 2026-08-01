# M5 E4 — remote execution record

Engineering record. No scientific finding here; the results are in
`m5-e4-factorial-results.md` and the verdict in `m5-e4-decision.md`.

## Where it ran

| | |
|---|---|
| Host | gpu-host, MS-7E58, 32 GB RAM, 24 logical CPUs |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16,303 MiB |
| Guest | WSL2 Ubuntu (24 GB via `.wslconfig`), Python 3.12.13 |
| Stack | TabPFN 8.0.8, torch 2.12.1+cu130, CUDA 13.0, numpy 2.4.6, pandas 3.0.3, sklearn 1.8.0, scipy 1.17.1 |
| Checkpoint | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` |
| Deployment | git bundle into an independent clean clone; gitignored inputs via leaf symlinks |
| Clone HEAD | `ac0310d5160398ec8d611ad9d049a72e120dbb71`, worktree changes 0 |

## Timeline

| | |
|---|---|
| First launch | 02:24, failed on unit 1 |
| Relaunch | 02:37 |
| Complete | 04:47:07, 24/24 fits, 192/192 repeats |
| Elapsed | 2 h 10 m against a 2 h 09 m estimate |

## The first launch failed, and the failure was worth the eight minutes

Unit 1 built its feature matrix and then died renaming the cache file:

```
FileNotFoundError: '.../seed42__cell11.npz.tmp' -> '.../seed42__cell11.npz'
```

`np.savez` appends `.npz` to a *path* argument that does not already end in it,
so the write landed on `seed42__cell11.npz.tmp.npz` and the rename had nothing
to move. Confirmed directly rather than inferred:

```python
np.savez(Path("x.npz.tmp"), ...)   # writes x.npz.tmp.npz; x.npz.tmp does not exist
```

The cache now writes through an open file object, which `np.savez` does not
rename, and a test exercises the whole write-rename-reload path.

The failure was fail-safe: it happened before `fit_start.json`, so no state, no
repeats and no `INTERRUPTED_INCOMPLETE` marker were written, and the unit simply
re-ran. Cleanup refused to delete anything until it had verified that zero fits,
zero repeats and zero states existed.

## Why one worker, measured rather than assumed

| Phase | Share of wall-clock | Peak RSS | GPU |
|---|---:|---:|---:|
| feature-matrix build (12 cache misses) | 89% | 16.2 GB | **0%** |
| fit + 8 repeats (all 24) | 11% | 2.7 GB | 78–97% |

Peak RSS was read from `/proc/PID/status` `VmHWM`, the kernel's own high-water
mark, not from a sampler. That mattered: a 1-minute sampler on the E3 run had
reported 8.0 GB and finer sampling 12.7 GB, while the true peak here is
**16.2 GB**. A sampler can only miss a peak downward, so its error is not noise
but a bias toward "this fits, run two."

The parallelism gate then decides itself:

```
2 × 16.2 = 32.5 GB   vs   70% of 23 GB = 16.1 GB      → fails
32.5 GB              vs   23 GB total                 → exceeds the whole VM
```

A single unit's peak already consumes the entire 70% budget. And the phase that
*could* be parallelised — fit and inference, 2.7 GB, GPU-bound — is 11% of the
run, so parallelising it perfectly would save about seven minutes.

The GPU was idle for 89% of the wall-clock. Adding GPU workers would have
contended for a device that was doing nothing during the part that took the
time.

Splitting units across the laptop and the remote was considered and rejected on
scientific grounds, not capacity: the laptop is capable (RTX 4070 Laptop,
31.6 GB), but it runs torch 2.12.1+**cu126** on Ada against the remote's
2.12.1+**cu130** on Blackwell. Assigning units to two machines would make
"machine" a factor perfectly confounded with unit assignment, in the one
experiment whose purpose is controlled attribution — and would reintroduce a
larger confounder than the execution-order one the randomised schedule had just
removed.

## What the cache bought

| | cache miss | cache hit |
|---|---:|---:|
| units | 12 | 12 |
| elapsed | 570–579 s | 65–70 s |
| peak RSS | 16.22–16.26 GB | 2.70–2.73 GB |

Blocking both scaler arms of a `(seed, cell)` together made the second arm **8×
faster and 6× smaller**. Total saving against an unblocked schedule: about
57 minutes, roughly half the run — considerably more than parallelism could have
delivered, and with no confounding cost.

Spread across the 24 units was under 0.5% on both time and memory. Swap stayed
at zero throughout.

## Discipline that held

- One unit = one subprocess. A failure stops the run rather than being retried,
  because a silent retry is how a fit gets duplicated.
- Per-unit atomic writes; `INTERRUPTED_INCOMPLETE` on failure; reload-backfill
  of same-process repeats refused by construction.
- The ensemble contract checked after every fit: `n_estimators_`,
  `len(ensemble_configs_)`, and the low-memory executor's four runtime
  containers. All 24 reported 8.
- Monitors ran the whole time with alert conditions covering the failure paths,
  not just progress. One condition was wrong — a glob that could not match
  `effective_n_estimators_=8` fired a false alarm — and was fixed to alert on
  any value that is not 8 rather than to pattern-match the good case.

Final remote state: 0 live processes, 0 tmux sessions, 0 bytes of stderr, 0
interrupted markers, 0 stray temp files.

## Return path

Packaging refused to run until it had verified 24 fits, 192 repeats, 24 states,
0 interrupted, 0 temp files, and at least 21 distinct state digests. It then
digested all 266 files into a manifest carried inside the archive.

| | |
|---|---|
| files | 266 |
| uncompressed | 144,109,572 bytes |
| archive | 142,157,313 bytes |
| archive digest | `81bbd57e772ab292a6a15f611519237c047188c371e9ad4dfc2d10f2f41f8612` |

The digest matched on the laptop and all 266 files were re-verified
individually after extraction.

## Fault injection

The importer recomputes every endpoint from the raw score vectors and re-reads
the effective ensemble size from each persisted state, so it does not depend on
anything the runner said about itself. Seven faults were injected into copies of
the real results; **all seven were rejected**, and six of them had the manifest
regenerated over the tampering so that digests alone could not catch them:

| Injected fault | Manifest rebuilt | Result |
|---|---|---|
| perturbed repeat endpoint | no | rejected |
| perturbed repeat endpoint | **yes** | rejected |
| effective `n_estimators_` falsified to 16 | **yes** | rejected |
| state file swapped between two units | **yes** | rejected |
| one repeat deleted | **yes** | rejected |
| process UUID mismatched | **yes** | rejected |
| score vector edited directly | **yes** | rejected |

Import then ran as one directory-level `os.replace` per unit after a dry run
against a throwaway root.
