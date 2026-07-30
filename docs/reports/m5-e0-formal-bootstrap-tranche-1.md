# M5 E0 formal bootstrap — Tranche 1 execution record

## Scope and identity

This is a checkpointed execution record, not a scientific result report.  The
formal run used repository `kuokuant-oss/lead-reproduction`, branch
`m5-tabpfn-repro-audit`, HEAD `d8e59da2c40cb5102367d6a73299e807680f6ca6`.
It executed only identity, base metrics, and the authorized first bootstrap
tranche.  It did not run LOO, segment analysis, finalization, fit/refit,
inference, GPU work, the frozen 192-row query, C1, Path B, site transfer, or
manuscript work.

## Preflight and provenance

Formal preflight passed at 2026-07-31 04:21:05 +08:00.  It validated ten
frozen input artifacts, a 4,000-unit formal `meter x draw` manifest (SHA-256
`9ab425b7c41a53c03114bf1c8e47db26e161336f010f8ea8fcaa6360fa92e9c0`), source
identity, isolated formal roots, and write access without computing metrics.
All checkpoint provenance declares `FORMAL_E0`; it is incompatible with the
non-scientific validation provenance and roots.

## Completed work and checkpoint state

| Phase | State | Units |
| --- | --- | ---: |
| identity | completed marker and heartbeat | 10/10 computed |
| base_metrics | completed marker and heartbeat | 40/40 computed |
| bootstrap | partial heartbeat; no completion marker | 168/4,000 computed; 3,832 pending |

The bootstrap tranche selected draw IDs 0–41 for each of electricity,
chilledwater, steam, and hotwater, in deterministic meter round-robin order.
Thus 42 draws per meter were checkpointed.  The 42-draw limit is an execution
tranche, not a scientific stopping rule; the formal target remains 1,000 draws
per meter.  No bootstrap intervals, confidence intervals, or interpretation
are supported by this partial work.

The bootstrap heartbeat records `status=running`, `completed=168`,
`pending=3832`, `computed=168`, `reused=0`, a null current unit, and no phase
completion marker.  There are 168 valid bootstrap unit checkpoints, no `.tmp`
files, and no bootstrap `COMPLETE.json`; only identity and base_metrics have
completion markers.

## Runtime, artifact map, and resume point

Execution started 2026-07-31 04:20:04 +08:00 and wrote the partial summary at
05:36:42 +08:00: elapsed 4,534.70 seconds (1:16:38).  Runtime events retain
per-unit compute, checkpoint-write, checkpoint-validation, rows, and RSS.

- Output/summary: `data/processed/m5_meter_specific_learner_gap/formal/`
- Checkpoints/heartbeats: `data/processed/m5_meter_specific_learner_gap/formal_checkpoints/`
- Stdout/stderr logs: `data/processed/m5_meter_specific_learner_gap/formal_logs/`

These generated, ignored artifacts are not staged or committed.  The exact
authorized resume point is the same formal root, manifest, seed/draw mapping,
and source provenance, selecting the next missing draw ID 42 per meter.  A
further tranche requires explicit authorization.  Valid checkpoints must not
be removed or rewritten.
