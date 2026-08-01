# M5 E4 machine-readable artifacts

Byte-identical copies of the small machine-readable artifacts from the canonical
result root `data/processed/m5_e4_formal_path_a/`, which is gitignored along
with the rest of `data/processed/`.

Only the summaries are here. The 24 fitted states (5.9 MB each), the per-repeat
records that carry the raw 352-element score vectors, and the run logs stay out
of version control.

| File | What it is |
|---|---|
| `e4_protocol.json` | the frozen design, including the human rulings recorded verbatim |
| `e4_input_manifest.json` | every input the results depend on, by digest |
| `e4_fit_manifest.json` | the 24 external fitted states |
| `e4_repeat_manifest.json` | the 192 same-process inference repeats |
| `e4_summary.json` | per-fit repeat statistics and coverage |
| `e4_factorial.json` | full-query factorial contrasts, per seed and overall |
| `e4_clustered.json` | building- and segment-clustered percentile intervals |
| `e4_decision.json` | the verdict and the bar it was judged against |

Digests are listed in `docs/handoffs/2026-08-02-m5-e4-complete-handoff.md` §9.
