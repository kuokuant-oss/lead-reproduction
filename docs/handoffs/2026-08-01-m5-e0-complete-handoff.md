# M5 E0 complete — active handoff

Supersedes
[`2026-07-31-m5-e0-four-way-remote-execution-handoff.md`](2026-07-31-m5-e0-four-way-remote-execution-handoff.md),
whose execution plan is now finished.

## State

Formal E0 is **complete**: bootstrap 4,000/4,000, exact LOO 1,196/1,196, segment
concentration 4/4, all three phases carrying `COMPLETE.json` markers and
`status=completed` heartbeats.

| Phase | Units | Marker | Provenance digest |
| --- | ---: | :-: | --- |
| identity | 10 | yes | (unchanged) |
| base_metrics | 40 | yes | (unchanged) |
| bootstrap | 4,000 | yes | `38d22eec…b1b71a` |
| leave_one_building | 1,196 | yes | `7fb67090…955362` |
| segment | 4 | yes | (segment phase body) |

Pinned HEAD `d8e59da2c40cb5102367d6a73299e807680f6ca6`, branch
`m5-tabpfn-repro-audit`, execution mode `FORMAL_E0` throughout. Assignment SHA
`3d0b5a0f047718dddbef76894c485ba1f477e949bd07a24d7a304413d559b0e9`, unchanged.
`.tmp` count 0 everywhere. The 525 pre-existing bootstrap checkpoints are
unmodified in SHA, size and mtime, verified twice.

## Results

| Meter | E0 classification | Manuscript role |
| --- | --- | --- |
| steam | stable empirical advantage | principal outcome |
| chilledwater | observed advantage but not stable | second empirical outcome; C1 required |
| electricity | counterexample | counterexample |
| hotwater | no supported advantage | supporting/context lever |

Full evidence, decision rule and caveats: [`m5-e0-final-synthesis.md`](../reports/m5-e0-final-synthesis.md).

## Reports

- [`m5-e0-formal-bootstrap-completion.md`](../reports/m5-e0-formal-bootstrap-completion.md)
- [`m5-e0-exact-loo.md`](../reports/m5-e0-exact-loo.md)
- [`m5-e0-segment-concentration.md`](../reports/m5-e0-segment-concentration.md)
- [`m5-e0-final-synthesis.md`](../reports/m5-e0-final-synthesis.md)

Result artifacts are under
`data/processed/m5_meter_specific_learner_gap/formal/`:
`formal_bootstrap_summary.{json,csv}`,
`formal_loo_influence_summary.{json,csv}`,
`formal_segment_concentration_summary.{json,csv}`,
`formal_e0_final_classification.json`.

## Next step decision (not taken here)

**Whether chilledwater enters C1 is an open human decision.** E0 establishes
that chilledwater is precisely the borderline case C1 exists to resolve: PR-AUC
excludes zero, direction is stable at 97–98%, no building flips the sign across
252 omissions, the effect is diffuse across 2,604 segments — but the ROC-AUC
interval fails to exclude zero by 2.7 × 10⁻⁵. That is not enough to claim a
stable advantage nor weak enough to drop.

Nothing in this handoff authorizes C1, Path A, Path B, the 192-row frozen query,
site transfer, 500k, full-holdout refit, fit/refit, inference, GPU work, or
manuscript edits.

## Carry-forward issues for the next operator

1. **Quadratic in the pinned checkpoint utility.** `missing_units` re-evaluates
   `completed_units(expected)` inside its list comprehension, so finalizing the
   4,000-unit bootstrap phase cost 145 minutes of single-core CPU (~16M
   `read_unit` calls at a measured 497 µs each). It is correct, and the pinned
   source was deliberately **not** modified because its digest is part of this
   run's provenance. Any milestone free to change that file should hoist the
   call out of the comprehension. Re-running `complete_phase` on an already
   marked phase is cheap — it validates and reuses the existing marker.
2. **`pyarrow` is not in the pinned `uv.lock`.** The segment phase therefore ran
   on the main repo venv (identical pandas/numpy/sklearn; `pyarrow` only decodes
   Parquet). The pinned environment cannot currently reproduce that phase
   end-to-end. Decide whether to add `pyarrow` to the lock before any future
   segment work.
3. **Cross-platform digest handling.** `scripts/_research_checkpoint.py` is CRLF
   on Windows and LF on Linux from the same pinned blob, so raw working-tree
   digests never agree across machines. Workers compare the canonical content of
   the pinned Git blob and record `line_ending_only_mismatch` rather than
   ignoring it. Do not "fix" this with `unix2dos` on tracked source — it dirties
   the clone and re-trips the clean guard.
4. **Steam magnitude sensitivity.** One building carries ~80% of steam's
   ROC-AUC and positive-rank magnitude without ever flipping its sign. Quote
   steam's direction confidently; do not quote those magnitudes as precise.
5. **The main working tree's analyzer is NOT what was executed.**
   `scripts/analyze_m5_meter_specific_learner_gap.py` in this repository has
   uncommitted modifications (+91/−19 vs `d8e59da`), digest `4adbc43b…`. All
   formal work ran in the separate execution clone
   `lead-reproduction-e0-execution`, a clean checkout at `d8e59da` whose
   analyzer digest is the pinned `3733649628…`, and every worker re-verified
   that digest before computing. These local edits pre-date this session and
   were deliberately left untouched. Do not assume the working-tree analyzer
   reproduces the E0 results; use the pinned commit.
6. **`--formal` cannot run LOO or segment.** The committed analyzer permits only
   identity/base_metrics/bootstrap under `--formal`; the checkpointed LOO and
   segment phases exist only under `--validation-mode`, which stamps
   NON_SCIENTIFIC_VALIDATION provenance and truncates LOO to the first N
   buildings. Both phases were therefore driven by external orchestration
   calling the committed functions unchanged. If formal LOO/segment is to be
   repeatable from the CLI, the analyzer needs a formal path.

## Remote execution notes

`gpu-host` did 1,736 bootstrap units, 1,463 tail units, and 1,180 LOO units.
Operational lessons — the WSL distro dying ~15s after the last SSH session
disconnects, the monitoring pattern, worker sizing, and how to split work across
machines without duplicates — are recorded in
`C:\Users\tonykuo\remote-gpu-setup\AGENTS.md` and summarised in that repo's
`README.md`.

Measured per-unit costs, for planning: bootstrap electricity 57.5 s,
chilledwater 15.9 s, steam 9.7 s, hotwater 4.0 s; LOO electricity 24.5 s,
chilledwater 7.3 s (4 parallel workers, ~2.5 GB RSS each).
