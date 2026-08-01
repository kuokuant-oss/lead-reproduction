# M5 E3 — remote execution record

Engineering record for the E3 variance pilot. Nothing here is a scientific
finding; the science is in `m5-e3-variance-pilot.md` and the verdict in
`m5-e3-decision.md`.

## Where it ran

| | |
|---|---|
| Host | gpu-host, MS-7E58, 32 GB RAM, 24 logical CPUs |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16,303 MiB |
| Guest | WSL2 Ubuntu, Python 3.12.13 |
| Stack | TabPFN 8.0.8, torch 2.12.1+cu130, CUDA 13.0 |
| Checkpoint | `tabpfn-v3-classifier-v3_default.ckpt` |
| Deployment | git bundle of `m5-e3-variance-pilot` into an independent clean clone; gitignored inputs reached by leaf symlinks |

CUDA was verified available before every fit and every reload. CPU fallback is
prohibited and was never exercised.

## Timeline

| Event | Result |
|---|---|
| Cells 00 → 01 → 10 → 11, sequential | 4 fits, 32 same-process repeats, all gates passed at n=8 |
| Cell-11 fresh-process diagnostic | 2 independent processes |
| Archive and return | 68 files, 23,425,511 bytes, digest matched end to end |

No cell escalated past the first batch of 8, so batches 16/24/32/40 were never
reached and the cap was never approached.

## Two engineering problems worth recording

### The remote's usable RAM was a config default, not the hardware

Early sizing used `free -g` from inside WSL, which reports the VM's allocation
rather than the machine. With no `.wslconfig` present, WSL2 defaults to roughly
half of host RAM, so a 32 GB host presented ~15 GB to Linux. Every decision
built on that number was wrong by 2x, and the symptom — swapping during feature
matrix construction — was misread as a genuine memory limit.

Ground truth had to come from the Windows side. After pinning
`C:\Users\User\.wslconfig` to `memory=24GB, swap=8GB`, WSL reported 23 GiB,
swap went to zero, and context loading fell from about 25 minutes to about 4.

The correction was prompted by the user, not by the monitor. A monitor that
reports "memory is tight" cannot tell you that the ceiling itself is wrong.

### Controlled parallelism was authorised, measured, and then declined

Two independent `tmux` workers were permitted (never `ProcessPoolExecutor`),
gated on measured resources: `2 x peak VRAM + 2 GB <= GPU VRAM`, `2 x peak RSS
<= 70% of physical RAM`, and no swap, OOM, or throughput loss.

VRAM was never the constraint — peak was 0.425 GB per cell, identical across all
four. RAM was. An initial peak-RSS estimate of 8.0 GB came from 1-minute
sampling; finer sampling showed cell 01 alone reaching **12.7 GB**, so
`2 x 12.7 = 25.4 GB` exceeded the 16.1 GB threshold. The second worker was
killed and execution reverted to sequential.

That was the right call for the wrong reason at first — the RAM ceiling was
itself misconfigured at the time. It stayed the right call after the ceiling was
raised: `2 x 12.7 = 25.4 GB` still exceeds 70% of 32 GB. The lesson is not "do
not parallelise" but **do not size from a coarsely sampled peak**.

## Discipline that held throughout

- One cell = one fit, then that cell's same-process repeats, then the next cell.
  No cross-cell interleaving. No reload-backfill of same-process repeats.
- Per-unit atomic writes (temp + `os.replace`), heartbeats, and a resume guard
  that writes `INTERRUPTED_INCOMPLETE` rather than silently continuing a fit
  from another process.
- Process-UUID lifecycle enforcement: each cell's `CELL_COMPLETE.json` records
  the UUID of the process that produced it, and all four differ.
- Monitors ran for the whole execution, with alert conditions that could
  actually fire. Two earlier conditions (`*INSTALL_EXIT*` and `swapMB!=0`) were
  permanently true and were replaced with `*INSTALL_EXIT=[1-9]*` and
  swap-growth-against-baseline — a monitor that always alerts is a monitor
  nobody reads.

Final remote state at archive time: 0 live worker processes, 0 `tmux` sessions,
0 bytes of worker stderr, 0 `INTERRUPTED` markers, 0 stray `.tmp` files.

## Return path

Packaging refused to run while any worker process, `tmux` session, or `.tmp`
file remained. It then digested all 68 files into a manifest carried inside the
archive, and digested the archive itself:

```
47547b1f8d64ab42def4599798b0a9467b5726eb1a77fc66d741d717cda7407c
```

That digest matched on the laptop, and every one of the 68 files was
re-verified individually after extraction — the archive digest alone would not
have caught a manifest that omitted a file.

Import ran in three stages: validate against a throwaway root, negative-test the
validator, then import for real with one directory-level `os.replace` per cell.

The validator does not trust the runner's own verdict. It re-derives every gate
statistic from the raw per-repeat records, and takes the targets from the frozen
protocol rather than from the recorded result. Two negative tests confirmed it
works:

| Injected fault | Caught by |
|---|---|
| one repeat's endpoint perturbed by +0.05 | per-file digest check |
| `auc_half_width` falsified to 0.001, **manifest regenerated over it** | gate recomputation (`0.001 -> 0.013179`) |

The second test is the one that matters. The first is caught by digests and so
proves nothing about the recomputation.

The staged protocol artifact digest
(`679f8211…4c9bff5672abe5c1628fb322a1f1f33f1419b14`) matched the canonical one
exactly, so the remote ran under the same frozen protocol that the laptop holds.
