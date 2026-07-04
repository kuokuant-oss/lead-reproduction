# Legacy processed outputs

These files are retained for provenance and regime-sensitivity comparisons.

- `m5_phaseD_foundation_vs_gbdt_row_offset_baseline_superseded.json`: row_offset baseline, superseded by timestamp_merge for the current M5/M6 model-comparison matrix, retained for the value-change regime ladder.
- `m5_phaseC_tabpfn_spike_row_offset_baseline_superseded.json`: row_offset Phase C baseline, superseded by timestamp_merge, retained for comparison only.
- `m4_5_readiness_check.json` remains in `data/processed/` for path compatibility but is marked `superseded_by`; it records the 2026-06-26 row_offset readiness gate, not the current timestamp_merge golden line.
- `m6_phaseD_meter_aware.json`: superseded row_offset_meter_aware comparison residue with contradictory default metadata; moved out of active processed outputs because current reports use regime-specific timestamp_merge artifacts.
