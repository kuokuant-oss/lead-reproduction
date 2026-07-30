# M5 context composition audit

CPU-only audit of the exact context rows used by the Story A/E probe.
The contexts are label-balanced, fit-building-only, and identified by an ordered raw-index digest.

| context | rows | positive | negative | buildings | raw-index digest |
| --- | ---: | ---: | ---: | ---: | --- |
| `pooled_reference` | 20,000 | 10,000 | 10,000 | 725 | `94e62e1dafeb0a34` |
| `meter_balanced` | 20,000 | 10,000 | 10,000 | 720 | `89630b5265c9e6a6` |
| `meter_heavy:hotwater:0.5` | 20,000 | 10,000 | 10,000 | 721 | `78c72b46dc174369` |
| `meter_excluded:hotwater` | 20,000 | 10,000 | 10,000 | 725 | `703d6b894610f2d3` |

Gates: `holdout_overlap=0`, `duplicate_raw_index=0`, exact 50/50 label counts, and the full ordered digest is stored in each manifest.

CSV artifact: `data/processed/m5_context_stories/reports/context_composition_audit.csv`
