# M5 E0 checkpointed execution validation

## Ground truth before implementation

E0 asks whether the existing natural-prevalence, GEPIII building-disjoint,
full-holdout F4/137-feature predictions show a meter-specific TabPFN-minus-tree
performance difference across 5k, 10k, 20k, 50k, and 100k contexts. It reads
only the fixed prediction artifacts. TabPFN 8.0.8 is the scientific boundary;
8.1.0 is diagnostic-only. Trees are matched-row comparators.

Completed evidence is E1's CPU-only score/rank localization, including broad
rank movement and the steam/chilledwater observations. E0, exact clustered
uncertainty, influence, and segment concentration are planned but have no
accepted E0 scientific result. Steam is the principal empirical outcome,
chilledwater the second principal outcome for later CPU-only localization,
electricity the required counterexample, and hotwater a supporting/contextual
lever.

The frozen 192-row query is unscored and must not be read. Path B, site
transfer, 500k, retrieval, full-holdout refit, context-curve reruns, model fit,
model refit, inference, GPU use, manuscript changes, and legacy-output
overwrites are prohibited. Formal E0 remains unauthorized pending the exact
instruction `AUTHORIZE E0 FORMAL RUN`.

## Starting state

On 2026-07-31, branch `m5-tabpfn-repro-audit` started at
`5ab0ea98c727ef8156d866c9c2b51d41ab3353f9`. The repository had unrelated
modified/deleted documentation assets and untracked files, including an
untracked pre-checkpoint E0 script and test. No E0 Python process was found by
the permitted process query; Windows denied full command-line enumeration, so
that limitation is recorded. No active process was stopped and no incomplete
result is accepted as scientific evidence.

## Policy and implementation status

The repository now has `AGENTS.md` and the detailed long-running policy at
`docs/policies/long-running-research-execution.md`; the active M5 handoff links
to it. The shared `scripts/_research_checkpoint.py` utility provides canonical
unit IDs, atomic validated JSON checkpoints, provenance mismatch rejection,
completion markers that reject missing units, atomic heartbeat writes, and
flushed progress logs.

`scripts/analyze_m5_meter_specific_learner_gap.py` is now safe by default: it
will not launch without `--validation-mode`, a named phase, and explicit
positive limits for bootstrap draws, LOO buildings, and segment draws. `--formal`
unconditionally refuses to start pending `AUTHORIZE E0 FORMAL RUN`. Validation
uses separate checkpoint namespaces for identity, base metrics, bootstrap,
leave-one-building, and segment work. Bootstrap checkpoints are meter scoped;
bounded LOO and segment validation are meter scoped. No automatic timeout or
watchdog was added.

Focused static/unit verification passed:

```powershell
.venv\Scripts\python.exe -m py_compile scripts\_research_checkpoint.py scripts\analyze_m5_meter_specific_learner_gap.py
.venv\Scripts\python.exe -m unittest tests.test_research_checkpoint tests.test_m5_meter_specific_learner_gap tests.test_m5_tabpfn_repeated_inference_plan
```

The command completed successfully with 14 tests passing.

Repository-wide static verification is not yet clean: existing research-related
code outside the authorized E0 change set still contains timeout controls, for
example `scripts/supervise_m5_tabpfn_recovery.py`,
`scripts/run_m5_tabpfn_single_context_scaling.py`, and related tests/deployment
wrappers. The new E0 path contains no timeout or auto-kill control, but these
pre-existing paths prevent a truthful repository-wide no-timeout certification.

## Validation infrastructure block

The required bounded validation was prepared with all explicit limits set to
one bootstrap draw per meter, three LOO buildings per meter, and one segment
draw per meter. This environment imposes a hard approximately ten-second
command termination. Its PowerShell process launcher is also unavailable
because the inherited environment has conflicting `Path` and `PATH` variables;
the fallback detached launch was likewise terminated at the runner limit. No
Python process, checkpoint, or partial scientific output remained after the
attempt.

The policy prohibits retrying this potentially longer artifact-backed analysis
under an automatic timeout. Therefore bounded validation, runtime measurement,
the computational census from frozen inputs, interruption/resume evidence, and
the Stage I commit remain pending a persistent terminal/session with automatic
termination disabled, plus a decision on remediating the pre-existing timeout
paths. This report makes no scientific claim from the failed launch attempt.

## Stage I blocker-resolution update

### Execution environment capability

`scripts/probe_foreground_persistent_session.py` is a foreground-only,
non-scientific probe. It runs for 20 seconds, emits immediately flushed output
every two seconds, and atomically updates a temporary heartbeat. It neither
reads frozen artifacts nor computes E0 metrics or checkpoints.

The Codex runner terminated this probe at 10.4 seconds. Its stdout contained
ticks at 0, 2, 4, 6, and 8 seconds; the last heartbeat recorded PID 19048,
tick 4, elapsed 8.019 seconds, and `running`. No Python process survived the
termination. The verified `.scratch/m5-e0-foreground-probe` temporary directory
was then removed. This is direct evidence that this Codex path is not a
persistent foreground session and must not launch artifact-backed validation.

### Repository timeout audit and compliance gate

`scripts/check_long_running_timeout_policy.py` produces the tracked
`docs/reports/repository-long-running-timeout-audit.md` report and exits
nonzero for any active category-A research-process wall-clock timeout or
auto-kill. It classifies external service/request bounds as B, cleanup/no-timeout
observability as C, test fixtures as D, and a legacy path with an unconditional
launch block as E.

The current audit passed with `A=0, B=6, C=7, D=15, E=10`. The M5 137-batch,
canonical full-test, and single-context-scaling legacy launchers are now
unconditionally blocked—without an override—until they are migrated to the
checkpoint/resume execution contract. The audit excludes external request
timeouts, cleanup waits, and test bounds from the research-process prohibition.

### Foreground validation handoff

`scripts/run_m5_e0_stage1_validation.ps1` is the manual foreground launcher.
It has no timeout, background job, detached process, or auto-kill. It runs
unbuffered Python, streams a timestamped combined log, checks the timeout
policy, specifies every bounded work-unit limit, and reports the heartbeat and
resume command. Run it only from a persistent PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_m5_e0_stage1_validation.ps1
```

The fixed bounded limits are one bootstrap draw per meter, three deterministic
LOO buildings per meter, and one segment draw per meter. Outputs and
checkpoints remain under `data/processed/m5_e0_validation/` with
`NON_SCIENTIFIC_VALIDATION` provenance.

### Verification update

The compliance scan passed (exit 0), and the focused test command passed with
21 tests:

```powershell
.venv\Scripts\python.exe scripts\check_long_running_timeout_policy.py --write-report docs\reports\repository-long-running-timeout-audit.md
.venv\Scripts\python.exe -m unittest tests.test_long_running_timeout_policy tests.test_m5_e0_stage1_validation_harness tests.test_research_checkpoint tests.test_m5_meter_specific_learner_gap tests.test_m5_tabpfn_repeated_inference_plan
```

The E0 default launch guard and formal guard were also checked: default exit 1,
formal exit 1, and bounded provenance-only inspection exit 0. Artifact-backed
validation, runtime measurements, interruption/resume evidence, and the
focused commit remain blocked until a persistent terminal executes the handoff
command successfully.

## Artifact-backed bounded-validation evidence audit

A manual persistent-terminal validation subsequently completed on 2026-07-31
using the foreground launcher with one bootstrap draw per meter, three LOO
buildings per meter, and one segment draw per meter. The combined launcher log
is `data/processed/m5_e0_validation/stage1-validation-20260731-021339.log`.
It shows all ten frozen artifacts loading between 02:13:50 and 02:14:41
(UTC+08:00), the four bounded bootstrap units completing between 02:17:17 and
02:17:57, and final bounded-validation completion at 02:18:22.

The ignored validation root contains valid atomic completion evidence:

| Phase | Expected/completed units | Checkpoint bytes | Completion time (UTC+08:00) |
| --- | ---: | ---: | --- |
| identity | 10/10 | 22,582 | 02:16:06 |
| base metrics | 40/40 | 51,178 | 02:16:07 |
| bootstrap | 4/4 | 37,288 | 02:17:57 |
| leave-one-building | 4/4 | 33,778 | 02:17:58 |
| segment | 4/4 | 17,230,128 | 02:18:22 |

All 62 checkpoint unit content digests were revalidated, every completion
marker matched the phase provenance digest, and no temporary checkpoint file
was present. Every phase records `NON_SCIENTIFIC_VALIDATION`, the ten frozen
input identities, and the current E0 source digest
`7c07fa63bd3f797d4d35b805082d89b9d19dfcbc6c2f042202d0f66b16af6c94`.
The ten identity records each have 10,137,155 rows; their row and label digests
agree across all artifacts and all scores were finite.

This is **not** a commit-ready validation record. There is no artifact-backed
deliberate interruption/resume or checkpoint-reuse run, no validation runtime
summary with memory/checkpoint-overhead measurements, and no evidence of a
separate stderr log. The existing checkpoint utility tests cover temporary,
corrupt, provenance-mismatch, and incomplete-finalization behavior on fixtures,
but that does not replace the required artifact-backed reuse evidence. The
valid checkpoints must remain in place. Do not commit or launch formal E0 until
a persistent terminal supplies the missing evidence.

## EvidenceSuite implementation update (pending local execution)

The prior validation evidence was produced by an earlier E0 source digest and
is not commit-gate evidence after this implementation update. It remains an
ignored diagnostic record and must not be deleted. A fresh local
`-EvidenceSuite` run is now required.

LOO checkpoints are now explicitly `meter × building`: with the validation
limit of three buildings per meter, the expected manifest is 12 independent
units (`4 × 3`), not four meter-level payloads. Bootstrap checkpoints are
`meter × draw` (four units at one draw per meter); phase completion markers are
not computational units. Segment validation remains meter × bounded-summary;
the new per-unit duration evidence is required before asserting that its formal
granularity meets the ten-minute recovery target.

`--validation-stop-after-units N` is validation-only. It stops only after an
atomic checkpoint write and revalidation, writes
`EXPECTED_VALIDATION_INTERRUPTION.json`, returns exit code 75, and does not
create the current phase completion marker. Each computational unit now records
compute, checkpoint-write, checkpoint-validation, total time, rows, meter,
computed/reused state, and RSS when available. The run summary reports phase
distributions (p95 only when at least 20 observations exist), checkpoint
overhead, resume startup, finalization, and peak RSS.

The foreground PowerShell launcher now has `-EvidenceSuite`. It creates a new
ignored root and writes separate stdout/stderr logs, exit JSON, checkpoint
census, heartbeat root, per-run runtime summary, and suite summary for: expected
interruption (75), resume (0), and reuse (0 with zero computed units and
unchanged checkpoint SHA256).

## Windows PowerShell expected-interruption fix

The first local EvidenceSuite attempt correctly reached the deliberate
interruption after 12 atomic units, but Windows PowerShell promoted its stderr
message to a terminating `NativeCommandError` because the launcher had
`$ErrorActionPreference = "Stop"`. The interruption itself was not a timeout or
scientific failure. Expected interruption reporting now uses stdout, and the
launcher temporarily uses `Continue` only while collecting the native Python
exit code, then restores its original error policy. A fresh timestamped
EvidenceSuite root is required because the source digest changed; preserve the
partial prior root for diagnostics and do not attempt to resume it.

## Heartbeat contract correction (pending replacement evidence)

The subsequent three-run local EvidenceSuite correctly proved checkpoint
preservation, resume, and full reuse, but exposed one remaining Stage I
blocker: an all-reused phase wrote its startup heartbeat and never advanced it.
The final heartbeat could therefore remain `starting` with zero completed
units even though its phase completion marker and unit checkpoints were valid.
That evidence root is diagnostic only and is not the final Stage I evidence.

Heartbeat publication now scans and validates existing units before the first
write. It records total, completed, computed, reused, pending, current and
last-completed unit, timestamp, elapsed time, throughput, and ETA. Every
computed or reused unit atomically advances the same per-phase status. After
the completion marker has been validated or created, a final atomic heartbeat
records `status=completed`, `completed=total`, `pending=0`, a null current
unit, the marker path, and the actual computed/reused totals. Reused completion
markers are validated rather than rewritten, preserving their timestamps.

The replacement EvidenceSuite also verifies Run 3 unit-checkpoint SHA256 and
modification times are unchanged, and hard-fails if any of the three stderr
logs is non-empty. A new timestamped validation root is required because this
source digest changed.

## Per-run heartbeat snapshot correction (pending replacement evidence)

Each EvidenceSuite process shares the live heartbeat paths, so a later resume
or reuse process necessarily replaces the live status. The launcher now saves
`run1.heartbeat.json`, `run2.heartbeat.json`, and `run3.heartbeat.json`
immediately after the corresponding process exits and before the next process
starts. Each atomic snapshot contains the unmodified full contents of every
phase heartbeat present for that run, source path and SHA-256, matching
provenance path and SHA-256, run exit code, timestamp, and runtime counter
totals.

Snapshot creation rejects missing or invalid heartbeat JSON, missing or
incompatible provenance, invalid counters, absent runtime summary, duplicate
snapshot paths, and any run-specific state that cannot satisfy interruption,
resume, or full-reuse acceptance rules. This launcher source change requires a
fresh timestamped evidence root.

## Final Stage I bounded-validation evidence (passed)

The final non-scientific evidence root is
`data/processed/m5_e0_validation/evidence-suite-20260731-033842`. It was run
in a local foreground PowerShell session without a research-process timeout,
job, detached process, or auto-kill. It is ignored by Git and is not a
scientific result.

| Run | Exit | Unit / marker / temporary-file census | Execution evidence |
| --- | ---: | --- | --- |
| `run1-interruption` | 75 | 12 / 1 / 0 | Expected interruption after 12 completed units; its snapshot retains a partial `base_metrics` heartbeat with 2/40 complete and 38 pending. |
| `run2-resume` | 0 | 70 / 5 / 0 | Reused the 12 Run 1 units and computed exactly the remaining 58 units. |
| `run3-reuse` | 0 | 70 / 5 / 0 | Computed 0 units and reused all 70; unit checkpoint SHA-256 and modification times did not change. |

`run1.heartbeat.json`, `run2.heartbeat.json`, and `run3.heartbeat.json` are
distinct atomic snapshots. They contain complete per-phase heartbeat payloads,
the source heartbeat and provenance paths/SHA-256 values, exit code, snapshot
timestamp, and run-level computed/reused counters. Run 2 records five
completed phases with 58 computed and 12 reused units. Run 3 records five
completed phases, each with `completed=total`, `pending=0`, `computed=0`,
`reused=total`, a null current unit, and an existing completion marker.

All three stderr logs are empty. The five phase provenances declare
`NON_SCIENTIFIC_VALIDATION`; the identity provenance records all ten frozen
input artifacts with SHA-256 and positive row counts. LOO contains twelve
independent `meter x building` checkpoints, three per meter. The repository
timeout scan passed with no active research-process wall-clock timeout, and
the focused test suite passed. Stage I execution-contract gates are therefore
complete. No formal E0 computation, scientific finalization, or scientific
result interpretation was performed.
