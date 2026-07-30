# Repository long-running timeout audit

This static audit distinguishes research-process wall-clock termination from external-service bounds, cleanup waits, and test fixtures.

| Category | Path:line | Context | Disposition | Required action |
| --- | --- | --- | --- | --- |
| B | `scripts/deploy_m5_tabpfn_colab_head.ps1:41` | `"--timeout", "$SetupTimeout"` | external_service_request_timeout | External setup/request bound; not a scientific-process kill. |
| B | `scripts/deploy_m5_tabpfn_site_shard.ps1:72` | `"--timeout", "$SetupTimeout"` | external_service_request_timeout | External setup/request bound; not a scientific-process kill. |
| C | `scripts/m6_tabpfn_context_protocol.py:239` | `self._thread.join(timeout=max(5.0, 2 * self.interval_seconds))` | cleanup_wait | Cleanup wait; it does not terminate research work. |
| C | `scripts/recover_m5_hotwater_label_role_factorial.py:4` | `the first-pass predictions.  It has no wall-time timeout: completed cell states` | explicit_no_timeout_declaration | Documentation/observability only; it does not terminate research work. |
| E | `scripts/run_m5_tabpfn_137_batches.ps1:11` | `# wall-clock timeout. It cannot be run until migrated to the repository` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| E | `scripts/run_m5_tabpfn_137_batches.ps1:13` | `throw "Blocked: migrate this legacy long-running research path to atomic checkpoints, resume, provenance, and no-timeout execution first."` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| E | `scripts/run_m5_tabpfn_137_batches.ps1:97` | `Write-Log "batch_timeout=$batch after ${BatchTimeoutMinutes}m; moving on"` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| E | `scripts/run_m5_tabpfn_canonical_full_test.py:113` | `"long model call is never treated as a wall-time timeout."` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| E | `scripts/run_m5_tabpfn_canonical_full_test.py:939` | `"checkpoints, resume, provenance, and no-timeout execution first."` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| C | `scripts/run_m5_tabpfn_portable_shard.py:5` | `wall-time timeout; resource pressure reduces only the query microbatch, while` | explicit_no_timeout_execution | Reviewed no-timeout path; the match is explanatory text, not a kill control. |
| E | `scripts/run_m5_tabpfn_single_context_scaling.py:112` | `"--budget-timeout-minutes",` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| E | `scripts/run_m5_tabpfn_single_context_scaling.py:146` | `raise ValueError("budget timeout minutes must be non-negative")` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| C | `scripts/run_m5_tabpfn_single_context_scaling.py:610` | `self._thread.join(timeout=2)` | cleanup_wait | Cleanup wait; it does not terminate research work. |
| E | `scripts/run_m5_tabpfn_single_context_scaling.py:1252` | `"--budget-timeout-minutes",` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| E | `scripts/run_m5_tabpfn_single_context_scaling.py:1478` | `exit_code = worker.wait(timeout=max(1.0, args.termination_grace_seconds * 2))` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| E | `scripts/run_m5_tabpfn_single_context_scaling.py:1673` | `"checkpoints, resume, provenance, and no-timeout execution first."` | legacy_long_running_path | Launch blocked until checkpoint/resume migration removes termination logic. |
| C | `scripts/run_m6_tabpfn_context_suite.ps1:67` | `Write-Host "START (no timeout): $Stem" -ForegroundColor Cyan` | explicit_no_timeout_execution | Reviewed no-timeout path; the match is explanatory text, not a kill control. |
| C | `scripts/run_m6_tabpfn_context_suite.ps1:71` | `# wall-time or output-silence timeout.` | explicit_no_timeout_execution | Reviewed no-timeout path; the match is explanatory text, not a kill control. |
| C | `scripts/run_m6_tabpfn_context_suite.ps1:144` | `Write-Host "TabPFN context $ContextRows suite completed without a wall-time timeout." -ForegroundColor Green` | explicit_no_timeout_execution | Reviewed no-timeout path; the match is explanatory text, not a kill control. |
| B | `scripts/supervise_m5_tabpfn_recovery.py:287` | `timeout=timeout_seconds,` | external_service_request_timeout | External setup/request bound; not a scientific-process kill. |
| B | `scripts/supervise_m5_tabpfn_recovery.py:294` | `f"supervisor subprocess timeout after {timeout_seconds:g}s",` | external_service_request_timeout | External setup/request bound; not a scientific-process kill. |
| B | `scripts/supervise_m5_tabpfn_recovery.py:446` | `"--timeout",` | external_service_request_timeout | External setup/request bound; not a scientific-process kill. |
| B | `scripts/supervise_m5_tabpfn_recovery.py:727` | `"--timeout",` | external_service_request_timeout | External setup/request bound; not a scientific-process kill. |
| D | `tests/test_long_running_timeout_policy.py:20` | `"subprocess.run(command, timeout=10)",` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_long_running_timeout_policy.py:31` | `"timeout=10",` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_long_running_timeout_policy.py:46` | `"--timeout",` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_long_running_timeout_policy.py:54` | `"thread.join(timeout=2)",` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_long_running_timeout_policy.py:60` | `classify(Path("tests/test_example.py"), "timeout=1", blocked=False)[0], "D"` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_long_running_timeout_policy.py:66` | `"worker.wait(timeout=10)",` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_long_running_timeout_policy.py:74` | `[Finding("scripts/x.py", 5, "timeout=1", "E", "legacy", "migrate")]` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_m5_e0_stage1_validation_harness.py:21` | `self.assertNotIn("Start-Job", source)` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_m5_e0_stage1_validation_harness.py:23` | `self.assertNotIn("Wait-Process -Timeout", source)` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_m5_e0_stage1_validation_harness.py:71` | `self.assertNotIn("Start-Job", source)` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_m5_meter_specific_learner_gap.py:212` | `self.assertNotIn("timeout=", source)` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_m5_tabpfn_recovery_supervisor.py:568` | `self.assertIn("--timeout", captured["command"])` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_m5_tabpfn_recovery_supervisor.py:570` | `captured["command"].index("--timeout") + 1` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_m6_tabpfn_context_protocol.py:108` | `self.assertNotIn("Wait-Process -Timeout", source)` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_tabpfn_single_context_scaling.py:227` | `def wait(processes, timeout):` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_tabpfn_single_context_scaling.py:323` | `timeout=60,` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_tabpfn_single_context_scaling.py:367` | `timeout=60,` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |
| D | `tests/test_tabpfn_single_context_scaling.py:407` | `timeout=60,` | test_fixture_or_test_timeout | Not a research launch; retain only as a bounded test fixture. |

## Certification

Findings: A=0, B=6, C=7, D=18, E=10.
No category-A research-process timeout/auto-kill finding may remain. Category-E paths are not launchable and require migration before use. Network/request timeouts, cleanup waits, and tests are explicitly excluded from the research-process prohibition.
