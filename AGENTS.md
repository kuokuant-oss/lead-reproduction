# Repository instructions

## Long-running research execution policy

Before writing or launching research code, read and follow
[`docs/policies/long-running-research-execution.md`](docs/policies/long-running-research-execution.md).

**NO CHECKPOINT, NO LAUNCH.** Any potentially long-running or uncertain
research computation must have explicit bounded units, atomic checkpoints,
deterministic resume, provenance validation, flushed progress logging,
heartbeat/status, phase completion markers, bounded non-scientific validation,
and tests before execution. Do not configure automatic timeouts or auto-kill
mechanisms. Formal scientific runs require explicit human authorization after
the implementation and bounded validation have been completed.
