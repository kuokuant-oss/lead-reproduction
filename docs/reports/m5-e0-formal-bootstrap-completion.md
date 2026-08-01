# M5 E0 formal bootstrap — 4,000/4,000 completion record

## Scope and identity

This is an execution and verification record for the completion of the formal
E0 bootstrap. Repository `kuokuant-oss/lead-reproduction`, branch
`m5-tabpfn-repro-audit`, pinned HEAD
`d8e59da2c40cb5102367d6a73299e807680f6ca6`. Execution mode `FORMAL_E0`
throughout; no non-scientific validation provenance is present in the canonical
tree.

Scientific interpretation lives in the synthesis report, not here.

## Coverage census

Coverage was established by enumerating every unit on disk, not by trusting any
summary count. All twenty checks passed.

| Check | Result |
| --- | --- |
| bootstrap unit files | 4,000 |
| distinct unit IDs | 4,000 |
| exact 4-meter × draw 0–999 grid | missing 0, extra 0 |
| draws per meter | 1,000 each for electricity, chilledwater, steam, hotwater |
| `.tmp` under formal checkpoints | 0 |
| provenance mode | `FORMAL_E0` |
| distinct provenance digests across 4,000 units | 1 |
| provenance digest | `38d22eecd913cf1a96cabb86e56ed4dc77295e3bd65d1ce2ff6ed6453bd1b71a` |
| `content_sha256` recomputation | 4,000/4,000 reproduce |
| seed mapping `[20260730, meter_code, draw]` | consistent, 0 bad |
| analyzer source digest | consistent, 0 bad |
| frozen input digests | consistent, 0 bad |
| formal manifest digest | consistent, 0 bad |
| units outside the assignment | 0 |

Formal manifest SHA-256
`9ab425b7c41a53c03114bf1c8e47db26e161336f010f8ea8fcaa6360fa92e9c0`.

## Pre-existing 525 checkpoints are unmodified

A SHA-256 + `mtime_ns` + size snapshot of the 525 pre-existing units was taken
before the import and re-compared afterwards, and again after the completion
tranche ran:

| Comparison | SHA changed | size changed | mtime changed | missing |
| --- | ---: | ---: | ---: | ---: |
| after formal import | 0 | 0 | 0 | 0 |
| after completion tranche | 0 | 0 | 0 | 0 |

This is structural, not incidental: the importer iterates only the 3,475
assigned units, so the pre-existing 525 are never opened for write.

## Six shard sources

The 3,475 missing units were produced by six independent shard roots under one
pinned assignment, SHA-256
`3d0b5a0f047718dddbef76894c485ba1f477e949bd07a24d7a304413d559b0e9`. The
assignment was never regenerated or renumbered.

| Source | Machine | Units |
| --- | --- | ---: |
| `remote_forward` | gpu-host | 868 |
| `remote_backward` | gpu-host | 868 |
| `local_forward` `[0:138]` | laptop | 138 |
| `local_backward` `[0:138]` | laptop | 138 |
| `local_forward_tail` `[138:871]` | gpu-host | 733 |
| `local_backward_tail` `[138:868]` | gpu-host | 730 |
| **Total** | | **3,475** |

Ownership was expressed as **disjoint index ranges into the pinned ordered
assignment**, so duplicate units are structurally impossible rather than merely
unlikely. Union check: 3,475 units, 3,475 distinct, 0 duplicates, set equal to
the assignment set, 0 overlap with the pre-existing canonical 525, and
3,475 + 525 = 4,000 covering the grid exactly.

## Importer validation before any canonical write

The importer was exercised on a throwaway `NON_SCIENTIFIC_VALIDATION` root
seeded with copies of the 525 canonical units, before it was allowed near the
formal tree.

| Test | Expected | Result |
| --- | --- | --- |
| unmodified inputs | pass | pass |
| duplicate unit across roots | reject | `duplicate shard unit` |
| tampered payload | reject | `payload digest mismatch` |
| incomplete shard set | reject | `incomplete shard set` |
| conflicting canonical checkpoint | refuse | `refusing to overwrite canonical checkpoint` |
| seeded 525 after full import | untouched | SHA/size/mtime all unchanged |

An additional identity check: the importer's `canonical_record()` re-derived all
525 existing canonical units **byte-identically** from their own payloads
(525/525), proving the 3,475 new records carry the same on-disk format.

## Formal import

Single atomic import, 2026-08-01 14:06:20 → 14:06:32 (+08:00).

```json
{"assigned_units": 3475, "validated_units": 3475,
 "written": 3475, "reused": 0, "dry_run": false,
 "canonical_provenance_digest": "38d22eec...b1b71a"}
```

## Completion marker

The marker was issued by the repository's own formal tranche launcher, not
hand-written. Preflight reported `selected_units: []` and the tranche reported
`checkpointed 0 bootstrap units; 4000/4000 complete`, confirming **no unit was
recomputed** — the run only verified and signed.

- `checkpoints/bootstrap/COMPLETE.json`: phase `bootstrap`, 4,000 expected
  units, provenance digest `38d22eec…b1b71a`
- heartbeat moved from `running` to `status=completed`, `completed=4000`,
  `pending=0`, with `phase_completion_marker` populated
- `formal_tranche_summary.json`: `status=BOOTSTRAP_COMPLETE`,
  `newly_computed_units=0`

### Known cost of this step

`complete_phase` took 145 minutes of single-core CPU. The cause is a quadratic
in the pinned checkpoint utility:

```python
def missing_units(self, expected):
    return [unit for unit in expected if unit not in self.completed_units(expected)]
```

`completed_units(expected)` is re-evaluated inside the comprehension, once per
unit — 4,000² ≈ 16 million `read_unit` calls, each parsing a ~6.6 KB JSON and
recomputing its SHA (measured 497 µs per call over the full set). It is correct,
just expensive, and it is invisible for the 40-unit phases. **The pinned source
was not modified**, because its digest is part of this run's provenance. Future
phases with far fewer units (LOO at 1,196; segment at 4) pay a negligible
fraction of this cost. If a later milestone is free to change the utility,
hoisting the call out of the comprehension is a one-line fix.

## Formal bootstrap summary

Assembled directly from the 4,000 completed units (a linear read), using the
committed `bootstrap_intervals` estimator unchanged.

- 128,000 draw-level records (4,000 units × 32 records)
- every `meter × context × quantity × metric` cell has exactly 1,000 distinct
  draws; 0 incomplete cells
- **invalid draws: 0; single-class events: 0** for all four meters
- each draw carries its own metric; row probabilities are never averaged and
  then scored once
- learner gap is always TabPFN minus the matched tree **within the same draw**

Artifacts: `data/processed/m5_meter_specific_learner_gap/formal/
formal_bootstrap_summary.{json,csv}`.

## Cross-machine execution provenance

| Item | Value |
| --- | --- |
| laptop | Windows, execution clone `.venv`, Python 3.11.9 |
| gpu-host | WSL2 Ubuntu, clone-local `.venv`, Python 3.12.13 |
| numeric stack (both) | pandas 3.0.3, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.8.0 |
| remote validation | draw 0 recomputed for all four meters, byte-identical to canonical baselines |
| remote wall time | `remote_forward`/`remote_backward` 5.24 h each; tail workers 3.1 h |
| measured unit cost | electricity 57.5 s, chilledwater 15.9 s, steam 9.7 s, hotwater 4.0 s |

### Checkpoint-utility digest across platforms

`scripts/_research_checkpoint.py` is checked out CRLF on Windows
(`3ded546f…`, 7,566 B) and LF on Linux (`86cdb47c…`, 7,367 B) from the *same*
pinned blob, so raw working-tree digests cannot agree across platforms. The
shard workers therefore compare the **canonical content of the pinned Git blob**
(`git cat-file blob <commit>:<path>`, then `CRLF→LF` and lone `CR→LF` on both
sides). A canonical difference is a hard failure; a line-ending-only difference
is recorded, never ignored:

```json
{"actual_raw_sha256": "86cdb47c...", "expected_raw_sha256": "3ded546f...",
 "canonical_sha256": "86cdb47c...", "line_ending_only_mismatch": true}
```

3,199 remote-computed records carry this flag; the 276 laptop-computed records
do not, as expected. The file is never imported by the worker, so this is
orchestration provenance, not payload identity. The assignment manifest was not
modified, so its SHA remains `3d0b5a0f…`.

## Explicitly not done

No LOO interpretation, no finalization, no C1, no Path A/B, no 192-row query,
no site transfer, no 500k, no full-holdout refit, no fit/refit, no inference,
no GPU work, and no manuscript edits are part of this record.
