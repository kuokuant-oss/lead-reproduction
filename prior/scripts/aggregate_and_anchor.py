"""Aggregate per-cell metrics with the OFFICIAL aggregation, and anchor-check.

Official aggregation (from the figure JSON's own `aggregation` block):
  scarcity K=50/100/200/400 : mean over row seeds within a building seed,
                              then mean +- sampleSD/sqrt(n) across building seeds
  K=725                     : mean +- sampleSD/sqrt(5) across row seeds 0..4

Anchor check is done at the STRICTEST level available: per-building-seed
(scarcity) and per-row-seed (K=725) values, not just the plotted means.
"""
from __future__ import annotations
import json, math
from pathlib import Path

BASE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/analysis_2026-09-08")
CELLS = json.loads((BASE / "data/cell_metrics.json").read_text(encoding="utf-8"))
OFFDIR = Path("/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5/outputs/scripts")
OFF_PR = json.loads((OFFDIR / "m5_exp_b_building_count_meter_pr_auc_v5_fixed50k_provisional_data.json").read_text(encoding="utf-8"))
OFF_ROC = json.loads((OFFDIR / "m5_exp_b_building_count_meter_roc_auc_v5_fixed50k_provisional_data.json").read_text(encoding="utf-8"))
BUDGETS = (50, 100, 200, 400, 725)
MODELS = ("tree", "tabpfn")
OFFMODEL = {"tree": "ensemble", "tabpfn": "tabpfn"}


def mean(v): return sum(v) / len(v)
def se(v):
    if len(v) < 2: return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))


def group_values(budget, model, getter):
    """-> {group_id: value} after averaging row seeds inside each group."""
    per = {}
    for c in CELLS["cells"]:
        if c["budget"] == budget and c["model"] == model and not c.get("missing"):
            per.setdefault(c["group"], []).append(getter(c))
    return {g: mean(v) for g, v in sorted(per.items())}


def summarize(budget, model, getter):
    gv = group_values(budget, model, getter)
    vals = list(gv.values())
    return {"mean": mean(vals), "se": se(vals), "n": len(vals), "by_group": gv}


# ---------------------------------------------------------------- anchor check
def official_raw(off, budget, model, metric_key):
    for p in off["plot_points"]:
        if p["meter"] != "hot_water" or int(p["budget"]) != budget or p["model"] != OFFMODEL[model]:
            continue
        if "raw_by_building_seed" in p:
            return {f"b{k}": (v[metric_key] if isinstance(v, dict) else v)
                    for k, v in p["raw_by_building_seed"].items()}, p[f"mean_{metric_key}"]
        return {f"r{k}": v for k, v in p["raw_by_row_seed"].items()}, p[f"mean_{metric_key}"]
    raise KeyError((budget, model))


print("=" * 100)
print("ANCHOR CHECK  --  my pooled hot-water values vs the OFFICIAL figure JSON")
print("             (compared per building seed / per row seed, not just the mean)")
print("=" * 100)
worst = 0.0
for metric, off, getter in (("pr_auc", OFF_PR, lambda c: c["all"]["pr"]),
                            ("roc_auc", OFF_ROC, lambda c: c["all"]["roc"])):
    print(f"\n--- {metric} ---")
    print(f"{'K':>4} {'model':7} {'group':>6} {'mine':>10} {'official':>10} {'diff':>12}")
    for budget in BUDGETS:
        for model in MODELS:
            oraw, omean = official_raw(off, budget, model, metric)
            mine = group_values(budget, model, getter)
            for g in sorted(mine):
                d = mine[g] - oraw[g]
                worst = max(worst, abs(d))
                print(f"{budget:>4} {model:7} {g:>6} {mine[g]:10.6f} {oraw[g]:10.6f} {d:+12.2e}")
            mmean = mean(list(mine.values()))
            dm = mmean - omean
            worst = max(worst, abs(dm))
            print(f"{budget:>4} {model:7} {'MEAN':>6} {mmean:10.6f} {omean:10.6f} {dm:+12.2e}  <<")
print(f"\nWORST ABSOLUTE DISCREPANCY ACROSS ALL {2*len(BUDGETS)*len(MODELS)} points+means: {worst:.3e}")
print("=> anchor", "PASS" if worst < 1e-9 else "FAIL")

# ------------------------------------------------------------ subset summaries
GET = {
    ("all", "pr"): lambda c: c["all"]["pr"], ("all", "roc"): lambda c: c["all"]["roc"],
    ("r0", "pr"): lambda c: c["r0"]["pr"],   ("r0", "roc"): lambda c: c["r0"]["roc"],
    ("rN", "pr"): lambda c: c["rN"]["pr"],   ("rN", "roc"): lambda c: c["rN"]["roc"],
}
res = {}
for budget in BUDGETS:
    res[budget] = {}
    for model in MODELS:
        res[budget][model] = {f"{sub}_{met}": summarize(budget, model, g)
                              for (sub, met), g in GET.items()}

print("\n" + "=" * 100)
print("STANDARD METRICS BY SUBSET  (mean +- SE over the official replicate level)")
print("  r0 = meter_reading == 0  (n=169,386; 80,776 anomalies -> prevalence 47.7%)")
print("  rN = meter_reading != 0  (n=466,735;  9,915 anomalies -> prevalence  2.1%)")
print("=" * 100)
for met, name in (("pr", "PR-AUC"), ("roc", "ROC-AUC")):
    print(f"\n### {name}")
    print(f"{'K':>4} | {'pooled Tree':>15} {'pooled TabPFN':>15} | {'r0 Tree':>15} {'r0 TabPFN':>15} | {'rN Tree':>15} {'rN TabPFN':>15}")
    print("-" * 118)
    for budget in BUDGETS:
        cells = []
        for sub in ("all", "r0", "rN"):
            for model in MODELS:
                d = res[budget][model][f"{sub}_{met}"]
                s = f"{d['mean']:.4f}" + (f"±{d['se']:.4f}" if d["se"] is not None else "")
                cells.append(f"{s:>15}")
        print(f"{budget:>4} | {cells[0]} {cells[1]} | {cells[2]} {cells[3]} | {cells[4]} {cells[5]}")

# ------------------------------------------------------- paired Tree - TabPFN
print("\n" + "=" * 100)
print("PAIRED GAP  Tree - TabPFN  (paired inside each building seed / row seed, then mean +- SE)")
print("=" * 100)
paired = {}
for met, name in (("pr", "PR-AUC"), ("roc", "ROC-AUC")):
    print(f"\n### {name} gap (Tree - TabPFN)")
    print(f"{'K':>4} | {'pooled':>18} | {'reading=0':>18} | {'reading!=0':>18}")
    print("-" * 70)
    for budget in BUDGETS:
        paired.setdefault(budget, {})
        cells = []
        for sub in ("all", "r0", "rN"):
            t = res[budget]["tree"][f"{sub}_{met}"]["by_group"]
            p = res[budget]["tabpfn"][f"{sub}_{met}"]["by_group"]
            gs = sorted(set(t) & set(p))
            diffs = [t[g] - p[g] for g in gs]
            m, s = mean(diffs), se(diffs)
            paired[budget][f"{sub}_{met}"] = {"mean": m, "se": s, "n": len(diffs), "diffs": diffs}
            txt = f"{m:+.4f}" + (f"±{s:.4f}" if s is not None else "")
            cells.append(f"{txt:>18}")
        print(f"{budget:>4} | {cells[0]} | {cells[1]} | {cells[2]}")

(BASE / "data/aggregated.json").write_text(json.dumps(
    {"subset": {str(k): v for k, v in res.items()},
     "paired_gap": {str(k): v for k, v in paired.items()}}, indent=1), encoding="utf-8")
print(f"\nwrote {BASE / 'data/aggregated.json'}")
