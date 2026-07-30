# Long-running research execution policy

## Mandatory rule

**NO CHECKPOINT, NO LAUNCH.** Treat runtime as long-running whenever it may
exceed ten minutes, is uncertain, spans multiple scientific units, repeats
evaluation, or would make interrupted work expensive to repeat. An
interruption may lose only the active checkpoint unit; completed valid units
must remain reusable.

## Before launch

Create a computational-unit census that identifies inputs, units, expected
counts, expensive operations, reusable intermediates, memory risk, output
growth, and a checkpoint layout. Implement and test explicit bounded units,
atomic unit-level checkpoints, deterministic resume, result-affecting
provenance validation, flushed progress logging, completed/total counters,
throughput and ETA, atomic heartbeat/status, and phase-completion markers.

Checkpoint outputs are complete only after they are calculated, written to a
temporary file, flushed, closed, schema- and digest-validated, and atomically
renamed. Temporary, corrupt, incompatible, or partial outputs are never
complete. Do not delete valid checkpoints merely to simplify a rerun.

Record provenance for the repository and committed source identity, source
digest, command and mode, input paths and digests, row/label/group identity
digests, scientific settings and seeds, software versions, and unit identity.
Resume must hard-fail if result-affecting provenance differs.

The measured p95 duration of a checkpoint unit must be at most ten minutes.
Subdivide a unit before formal execution when it exceeds that target.

## Execution modes and authorization

Implementation and formal execution are separate authorizations. Default
commands must not launch full work. A bounded non-scientific validation mode
must require deterministic limits for *every* expensive phase and must use an
isolated output root with clearly non-scientific provenance. Reducing one
parameter, such as bootstrap draws, does not permit unbounded work elsewhere.

Formal execution requires a separately explicit human authorization, a clean
committed implementation, validated input provenance, passing focused tests,
and successful bounded validation. Finalization must refuse to assemble an
output while expected units are missing, temporary, corrupt, incompatible, or
unvalidated.

## No-timeout rule

Never use automatic wall-clock termination for research computation: no shell
or subprocess timeout, watchdog, scheduler deadline, CI limit, auto-kill, or
automatic restart. Bounded validation is bounded by deterministic units, not
time. Diagnose a suspected stall read-only first; progress, checkpointing,
resume, and heartbeat replace timeouts.

## Required tests

Before launch, tests must cover atomic writes, temporary/corrupt checkpoint
rejection, provenance mismatches, resume and reuse, missing-unit finalization
refusal, mode guards, bounded validation caps, heartbeat/log flushing, and
the scientific computation's identity and optimized/reference equivalence.
