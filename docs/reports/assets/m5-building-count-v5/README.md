# 2026-09-01 — Experiment B (source-building scarcity, fixed 50k context)

Snapshot taken after K=100 completed (b2 r1 finished 09-01 07:03). K=50 TabPFN
was still running, so every artifact here is provisional at K=50.

Schedule state at generation: 74 of 84 model cells complete, no `FAILED.json`.

## Contents

| File | What it is |
| --- | --- |
| `m5_exp_b_building_count_meter_pr_auc_v5_fixed50k_provisional_2026-09-01.png` | Three-panel PR-AUC curve across K, Tree vs TabPFN |
| `..._2026-09-01_data.json` | Audit data behind the figure: plot points, per-cell metrics, identity gates |
| `m5_exp_b_building_count_v5_fixed50k_setup.{tex,pdf,png}` | Protocol table |
| `m5_exp_b_building_count_meter_pr_auc_detail.{tex,pdf,png}` | Meter-level PR-AUC detail table, K=50-725 |
| `m5_exp_b_host_run_performance_cost.md` | Predictive performance vs computational cost on host-run cells |
| `scripts/` | Generators for the two files that are not produced by the figure script |

## Regenerating

The figure and its data JSON come from the figure script, which stays with its
metric cache in the 2026-08-26 folder:

```
/mnt/c/Users/User/Documents/Codex/2026-08-26/wsl-ubuntu-m5-building-count-v5/outputs/plot_m5_v5_fixed50k_building_scarcity.py
```

Run it in WSL against the experiment root, pointing `--output` and `--summary`
at this directory. It is read-only with respect to the formal experiment and
reuses `.m5_v5_fixed50k_plot_cache`, so only newly completed cells are computed.

The detail table:

```powershell
uv --directory C:\Users\User\projects\lead-analysis run python scripts\make_meter_table.py
pdflatex -interaction=nonstopmode -output-directory=. m5_exp_b_building_count_meter_pr_auc_detail.tex
```

The cost report (needs the experiment root, so run it in WSL):

```bash
/home/kuant_kuo/projects/lead-reproduction/.venv/bin/python scripts/make_report.py
```

PNGs are rasterised from the PDFs at 200 dpi with
`uvx --from pymupdf python -c "..."`.

## Status of the numbers

- K=100, 200, 400 are complete at five building seeds with both row seeds.
- K=725 uses five row seeds and has no building seed.
- K=50 has all ten Tree cells but no TabPFN result; the detail table leaves
  those two columns as `--` and the figure omits the TabPFN point.
- The protocol table and both result tables report the three non-electric
  meters only, with no pooled or macro summary.
