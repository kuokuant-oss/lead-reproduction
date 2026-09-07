"""Generate the host-run performance/efficiency markdown report. Read-only inputs.

Run inside WSL (needs the experiment root):
    /home/kuant_kuo/projects/lead-reproduction/.venv/bin/python make_report.py

Reads the published COMPLETE.json cells plus the plot metric cache; writes
m5_exp_b_host_run_performance_cost.md next to this script's parent directory.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(
    "/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/"
    "data/processed/m5_building_curve/v5_fixed_50k"
)
RUNS = ROOT / "model_runs"
# The metric cache lives with the figure script it belongs to.
CACHE = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-08-26/wsl-ubuntu-m5-building-count-v5/"
    "outputs/.m5_v5_fixed50k_plot_cache"
)
OUT = Path(
    "/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5/"
    "outputs/m5_exp_b_host_run_performance_cost.md"
)
TPE = timezone(timedelta(hours=8))
HOLDOUT = 4_102_084
METERS = (("chilled_water", "Chilled water"), ("steam", "Steam"), ("hot_water", "Hot water"))


def ts(e):
    return datetime.fromtimestamp(e, TPE).strftime("%m-%d %H:%M")


def cell(K, b, r, model):
    d = "tree_no_es" if model == "tree" else "tabpfn"
    return RUNS / f"building_seed{b}/row_seed{r}/{d}_k{K}_f137"


def info(d):
    ch = d / "prediction_chunks"
    chunks = sorted(ch.glob("*.npz"), key=lambda p: p.stat().st_mtime) if ch.is_dir() else []
    mt = [p.stat().st_mtime for p in d.iterdir() if p.is_file()]
    return {"chunks": len(chunks), "start": min(mt), "end": max(mt), "span": max(mt) - min(mt)}


host, colab = [], []
for K in (725, 400, 200, 100, 50):
    seeds = [725] if K == 725 else [0, 1, 2, 3, 4]
    for b in seeds:
        for r in (0, 1):
            t = cell(K, b, r, "tabpfn")
            if not (t / "COMPLETE.json").is_file():
                continue
            (host if info(t)["chunks"] else colab).append((K, b, r))

rows = []
for K, b, r in host:
    e = {"K": K, "b": b, "r": r}
    for m in ("tree", "tabpfn"):
        e[m] = info(cell(K, b, r, m))
        e[f"{m}_score"] = json.loads(
            (CACHE / f"k{K}_b{b}_r{r}_{'ensemble' if m == 'tree' else 'tabpfn'}.json").read_text()
        )["meter_pr_auc"]
    rows.append(e)
rows.sort(key=lambda x: (-x["K"], x["b"], x["r"]))
BUDGETS = sorted({x["K"] for x in rows}, reverse=True)


def sel(K):
    return [x for x in rows if x["K"] == K]


L = []
A = L.append
A("# Experiment B — Tree vs TabPFN on host-run cells")
A("")
A(f"Generated {datetime.now(TPE).strftime('%Y-%m-%d %H:%M')} (Asia/Taipei). "
  "All inputs read-only from published `COMPLETE.json` cells.")
A("")
A("## Scope")
A("")
A(f"{len(rows)} matched contexts in which **both** models ran on this host. "
  f"The {len(colab)} TabPFN cells executed on Colab are excluded, together with their "
  "paired Tree cells, so every number comes from the same machine.")
A("")
A("Colab cells are identified by the absence of a `prediction_chunks` directory: cells "
  "computed here stream 206 prediction chunks to disk, Colab cells were published as a "
  "finished artifact only.")
A("")
A("Excluded (Colab TabPFN): " + ", ".join(f"K={K} b{b} r{r}" for K, b, r in colab) + ".")
A("")
A("Hardware: NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120), 24 CPU cores, WSL2 Ubuntu. "
  "Every cell uses the same 50,000-row context (25k anomaly / 25k normal), 137 features, "
  f"model seed 42, and the same frozen {HOLDOUT:,}-row holdout.")
A("")
A("Cost is the span of artifact mtimes inside each cell directory, the same definition "
  "for both models. For the four Tree cells the scheduler timed directly this span runs "
  "about 4% below the recorded elapsed time, so the ratios are mild underestimates.")
A("")
A("Results are reported per budget K. Surviving contexts are unevenly spread across K "
  "(" + ", ".join(f"K={K}: {len(sel(K))}" for K in BUDGETS) + "), so a context-weighted "
  "average would silently weight some budgets more than others. Where a single figure is "
  "given it is the mean of the per-budget means, weighting each K equally.")
A("")
A("## Computational efficiency")
A("")
A("| K | n | Tree mean | TabPFN mean | Ratio | Tree rows/s | TabPFN rows/s |")
A("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
ratios = []
for K in BUDGETS:
    s = sel(K)
    t = np.mean([x["tree"]["span"] for x in s])
    p = np.mean([x["tabpfn"]["span"] for x in s])
    ratios.append(p / t)
    A(f"| {K} | {len(s)} | {t:,.0f} s | {p:,.0f} s ({p/3600:.2f} h) | {p/t:.0f}x | "
      f"{HOLDOUT/t:,.0f} | {HOLDOUT/p:,.0f} |")
A(f"| **Equal-weight** | | | | **{np.mean(ratios):.0f}x** | | |")
A("")
A(f"**TabPFN costs {np.mean(ratios):.0f}x the wall-clock time of the tree ensemble**, "
  f"ranging from {min(ratios):.0f}x at K={BUDGETS[int(np.argmin(ratios))]} to "
  f"{max(ratios):.0f}x at K={BUDGETS[int(np.argmax(ratios))]}.")
A("")
tab_all = np.array([x["tabpfn"]["span"] for x in rows])
tree_by_k = [np.mean([x["tree"]["span"] for x in sel(K)]) for K in BUDGETS]
A(f"The ratio narrows as K grows because the two models scale differently. TabPFN run "
  f"time is essentially independent of K: across all {len(rows)} cells it spans "
  f"{tab_all.min():,.0f}-{tab_all.max():,.0f} s, a range of "
  f"{(tab_all.max()-tab_all.min())/tab_all.mean()*100:.1f}%. Tree run time grows with the "
  f"source pool, from {min(tree_by_k):,.0f} s to {max(tree_by_k):,.0f} s.")
A("")
A("The two models also spend their time differently. The tree ensemble fits four boosters "
  "in 9-188 s and spends the rest building the holdout feature matrix and scoring it, "
  "producing five score arrays. TabPFN fits its context in roughly 530-660 s and spends "
  "about 98% of the run streaming 206 prediction chunks, producing one score array.")
A("")
A("## Predictive performance")
A("")
A("PR-AUC, Tree / TabPFN (difference). Difference is TabPFN minus Tree.")
A("")
A("| K | n | Chilled water | Steam | Hot water |")
A("| ---: | ---: | ---: | ---: | ---: |")
diffs = {m: [] for m, _ in METERS}
for K in BUDGETS:
    s = sel(K)
    out = []
    for m, _ in METERS:
        t = np.mean([x["tree_score"][m] for x in s])
        p = np.mean([x["tabpfn_score"][m] for x in s])
        diffs[m].append(p - t)
        out.append(f"{t:.4f} / {p:.4f} ({p-t:+.4f})")
    A(f"| {K} | {len(s)} | " + " | ".join(out) + " |")
A("| **Equal-weight difference** | | " + " | ".join(
    f"**{np.mean(diffs[m]):+.4f}**" for m, _ in METERS) + " |")
A("")
A("Per-budget direction:")
A("")
for m, ml in METERS:
    signs = [f"K={K} {d:+.4f}" for K, d in zip(BUDGETS, diffs[m])]
    pos = sum(d > 0 for d in diffs[m])
    A(f"- **{ml}**: TabPFN ahead at {pos} of {len(BUDGETS)} budgets ({'; '.join(signs)}).")
A("")
A("## Cost-effectiveness")
A("")
A("| K | Extra TabPFN time | Chilled water | Steam | Hot water |")
A("| ---: | ---: | ---: | ---: | ---: |")
extras = []
for i, K in enumerate(BUDGETS):
    s = sel(K)
    extra = (np.mean([x["tabpfn"]["span"] for x in s])
             - np.mean([x["tree"]["span"] for x in s])) / 3600
    extras.append(extra)
    out = []
    for m, _ in METERS:
        d = diffs[m][i]
        out.append(f"{extra * 0.1 / d:.0f} h" if d > 0 else "no gain")
    A(f"| {K} | {extra:.2f} h | " + " | ".join(out) + " |")
mean_extra = float(np.mean(extras))
out = []
for m, _ in METERS:
    d = float(np.mean(diffs[m]))
    out.append(f"**{mean_extra * 0.1 / d:.0f} h**" if d > 0 else "**no gain**")
A("| **Equal-weight** | " + f"{mean_extra:.2f} h | " + " | ".join(out) + " |")
A("")
A("Hours of extra compute per +0.1 PR-AUC. \"No gain\" means the extra compute produced "
  "a worse detector than the tree ensemble.")
A("")
A("## Overall")
A("")
A(f"Across the {len(BUDGETS)} budgets, TabPFN costs {np.mean(ratios):.0f}x the wall-clock "
  f"time of the tree ensemble, {mean_extra:.1f} hours more per context. It returns "
  f"{np.mean(diffs['chilled_water']):+.4f} PR-AUC on chilled water and "
  f"{np.mean(diffs['steam']):+.4f} on steam, and loses "
  f"{abs(np.mean(diffs['hot_water'])):.4f} on hot water. Buying a tenth of a point of "
  f"PR-AUC therefore costs {mean_extra*0.1/np.mean(diffs['chilled_water']):.0f} hours on "
  f"chilled water and {mean_extra*0.1/np.mean(diffs['steam']):.0f} hours on steam, and is "
  "not available at any price on hot water.")
A("")
A("## Detailed records")
A("")
A("### Per-cell timing")
A("")
A("| K | b | r | Model | Chunks | Span (s) | Hours | Start | End |")
A("| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |")
for x in rows:
    for m, label in (("tree", "Tree"), ("tabpfn", "TabPFN")):
        i = x[m]
        A(f"| {x['K']} | {x['b']} | {x['r']} | {label} | {i['chunks']} | {i['span']:,.0f} | "
          f"{i['span']/3600:.2f} | {ts(i['start'])} | {ts(i['end'])} |")
A("")
A("### Per-context PR-AUC")
A("")
A("| K | b | r | Chilled water | Steam | Hot water |")
A("| ---: | ---: | ---: | --- | --- | --- |")
for x in rows:
    c = [f"{x['tree_score'][m]:.4f} / {x['tabpfn_score'][m]:.4f} "
         f"({x['tabpfn_score'][m]-x['tree_score'][m]:+.4f})" for m, _ in METERS]
    A(f"| {x['K']} | {x['b']} | {x['r']} | " + " | ".join(c) + " |")
A("")
A("Each cell reads Tree / TabPFN (difference).")
A("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L), encoding="utf-8", newline="\n")
print("wrote", OUT)
