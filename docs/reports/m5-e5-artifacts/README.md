# M5 E5 machine-readable artifacts

Byte-identical copies of the small machine-readable artifacts from the canonical
result root `data/processed/m5_e5_independent_replication/`, which is gitignored
along with the rest of `data/processed/`.

Only the summaries are here. The per-repeat records carrying the raw 192-element
score vectors, the 24 tree score vectors, and the run logs stay out of version
control. E5 persists no model state of its own -- it reloads E4's.

| File | What it is |
|---|---|
| `e5_protocol.json` | the frozen design and the pre-declared decision rules |
| `e5_tree_execution_override.json` | the human ruling on where the fixed trees may score, with the evidence that forced it |
| `e5_input_manifest.json` | every input the results depend on, by digest |
| `e5_state_manifest.json` | the 24 E4 states E5 reloaded |
| `e5_repeat_manifest.json` | the 192 same-process inference repeats |
| `e5_query_audit.json` | identity of the 192-row query, including zero overlap with the 352-row query |
| `e5_summary.json` | per-state repeat statistics and coverage |
| `e5_factorial.json` | factorial contrasts, per seed and overall |
| `e5_clustered.json` | building- and segment-clustered percentile intervals, namespace 5005 |
| `e5_decision.json` | the two verdicts and the bar each was judged against |
| `e5_tree_manifest.json` | the laptop tree half: 24/24 bit-exact gate and per-unit digests |

Digests are listed in `docs/handoffs/2026-08-02-m5-e5-complete-handoff.md` §8.
