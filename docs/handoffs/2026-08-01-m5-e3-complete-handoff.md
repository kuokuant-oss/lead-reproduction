# M5 E3 complete — handoff (2026-08-01)

## 0. Status

E3 is **complete and frozen**. Verdict: `E3_MEASUREMENT_PROCESS_ACCEPTABLE`.

There is no resume command. Nothing is running, remotely or locally. The next
stage requires explicit human authorisation and is not started here.

| | |
|---|---|
| Branch | `m5-e3-variance-pilot` |
| Base commit | `e9cb59b9bbf7977f4bda2dfbb9779fb0659d9168` |
| Protocol artifact | SHA-256 `679f82114e6806572f61a32454c9bff5672abe5c1628fb322a1f1f33f1419b14` |
| Result root | `data/processed/m5_e3_variance_pilot/` (69 files) |

## 1. What was run

Four factorial cells, one fit each, 8 same-process repeats each, in the frozen
order `00 → 01 → 10 → 11`. All four passed both gate endpoints at the first
batch; the cap of 40 was never approached. Two fresh-process reload runs on
cell 11, kept entirely separate from the repeat statistics.

| Cell | AUC half-width (≤0.015) | Margin half-width | Margin target | State digest |
|---|---:|---:|---:|---|
| 00 | 0.013179 | 0.005507 | 0.016183 | `75f0537c` |
| 01 | 0.000000 | 0.000936 | 0.001000 | `14422d3d` |
| 10 | 0.006094 | 0.005606 | 0.008743 | `3438ecb6` |
| 11 | 0.002035 | 0.005547 | 0.012071 | `45666736` |

## 2. Reports

- `docs/reports/m5-e3-query-phase-resolution-audit.md` — pre-fit, read-only
- `docs/reports/m5-e3-remote-execution.md` — engineering record
- `docs/reports/m5-e3-variance-pilot.md` — the measurement
- `docs/reports/m5-e3-decision.md` — the verdict

## 3. Three facts a successor should not have to rediscover

**Repeated inference is not bitwise reproducible.** 32 repeats produced 32
distinct score-vector digests, zero collisions. Any future stage that assumes a
fitted TabPFN state returns identical output on re-scoring is wrong. The
endpoint-level consequence is small — that is the pilot's result, not an
assumption.

**Cell 01's AUC gate is saturated, not precise.** Its half-width is exactly
0.000000 because the metric is pinned at 1.000000 in all 8 repeats. It will keep
passing while carrying no information. Cell 01's margin endpoint (0.000936
against a target of 0.001000, the tightest ratio in the pilot) is the one doing
real work.

**The between-cell differences are not E3's finding.** Steam AUC runs 0.473 to
1.000 across cells against half-widths of 0.006 and 0.000. With one fit per
cell, that spread confounds composition with fit-to-fit variation, which E3
does not measure at all.

## 4. Carried forward unresolved

| Item | Status |
|---|---|
| chilledwater positive vs hotwater negative | `RESOLUTION_LIMITED_DIAGNOSTIC` — 16 hotwater-negative rows, AUC resolution 0.000977 against a 0.0043 effect |
| onset / middle / recovery phase contrast | `UNRESOLVED_NOT_EXECUTED` — no frozen artifact assigns a phase to a row |

Both were settled before any fit; neither is a consequence of the results.

## 5. Measured throughput, for sizing the next stage

Taken from the E3 artifacts, not estimated:

| Quantity | Measured |
|---|---:|
| Fit, 20,000 context rows, F4_137 | 0.76 – 3.40 s |
| One inference repeat, 352 query rows | **8.2 s** (very stable; first repeat of each cell ~9–18 s from warm-up) |
| One cell, fit + 8 repeats | 67 – 75 s |
| Whole pilot, GPU compute | ≈ 5 minutes |
| Peak VRAM per cell | 0.425 GB (identical across all four) |
| Peak RSS per cell | up to 12.7 GB |

**The GPU was never the bottleneck.** Context/feature-matrix construction
dominated wall-clock: ~4 minutes per cell after the WSL memory ceiling was
raised, ~25 minutes before. Any plan that sizes the next stage by GPU seconds
will be wrong about where the time goes.

## 6. Path A preparation (investigation only — not authorised, not started)

Path A was selected as the only sanctioned future fit direction in
`m5-137-context-mechanism-analysis.md`: *"F4 hotwater positive-support ×
negative-support 2×2, N=20k, fixed query, matched TabPFN/trees."* E3 has now
executed that 2×2 geometry — but as a **variance pilot**, one fit per cell.

What formal Path A additionally requires, per that report's convergence table:

1. **Replicate-aware factorial estimation** — more than one fit per cell, so
   composition can be separated from fit-to-fit variation. E3 deliberately does
   not measure the latter, and it is the gap that blocks every between-cell
   claim.
2. **≥3 context seeds** — E3 used seed 42 only.
3. **Frozen-scaler control** — E3 used the `cell_specific` scaler arm.
4. **Building- and segment-clustered uncertainty.**
5. **Full-holdout / natural-prevalence confirmation.**

Cost structure, from the numbers in §5:

- The 2×2 × 3 seeds × R fits arm is **cheap**. At R=8 that is 96 fits; even with
  8 repeats each it is on the order of 2 hours of GPU compute. Data loading, not
  the GPU, sets the wall-clock, so the useful optimisation is to build each
  cell's feature matrix **once** and fit repeatedly against it — not to add GPU
  workers.
- The **full-holdout arm is a different order of problem**. The holdout is
  10,137,155 rows against E3's 352-row query — roughly 4 orders of magnitude
  more scoring. Do not extrapolate from the 8.2 s figure, which is dominated by
  fixed overhead at 352 rows. Size it from the existing sharded full-test run
  logs, which are already in `data/processed/`.

Remote readiness, given what E3 established: the clone, symlinked inputs,
`.wslconfig` at 24 GB, monitors, and the archive/validate/import chain are all
in place and proven. The open engineering question for Path A is **not**
capacity but the peak-RSS ceiling — at 12.7 GB per unit, `2 × 12.7 = 25.4 GB`
still exceeds 70% of 32 GB, so a second GPU worker remains unavailable unless
the feature matrix is built once and shared rather than loaded per worker.

None of this is authorised. It is recorded so the next authorisation decision
can be made against measured numbers.

## 7. Explicitly not authorised by this handoff

E4 formal Path A, Path B, representation ablation, the frozen 192-row query,
site transfer, 500k, full-holdout refit, tree refit, TabPFN 8.1.0 as science,
manuscript changes, changing N or the cells, the 24-cell grid, and undocumented
reruns. The base policy file
(`docs/reports/m5-tabpfn-repeated-inference-policy.json`) was **not modified** —
its historical `designed_not_running` status and its `forbidden_this_round` list
are preserved as written. Authorisation for E3 is recorded inside the E3
protocol artifact instead.

## 8. Verification, if you want to re-check the result

```bash
# re-validate the canonical result root against itself (recomputes every gate)
python scripts/m5_e3_import.py \
  --staged   data/processed/m5_e3_variance_pilot \
  --canonical data/processed/m5_e3_variance_pilot

# regenerate the summary and decision from the raw records
python scripts/m5_e3_summary.py --root data/processed/m5_e3_variance_pilot
```

Both are read-only unless `--apply` is passed. The importer takes its precision
targets from the frozen protocol, not from the recorded results, so a falsified
verdict fails validation even if the file digests are consistent.
