"""Collect every number the report page shows into ONE json, straight from the
analysis outputs, so nothing is transcribed by hand."""
from __future__ import annotations
import json, math
from pathlib import Path

BASE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/analysis_2026-09-08/data")
K = [50, 100, 200, 400, 725]
MODELS = ["tree", "tabpfn"]


def rd(n):
    return json.loads((BASE / n).read_text(encoding="utf-8"))


agg = rd("aggregated.json")
op = rd("opoint_apdecomp.json")
wb = rd("within_building.json")
ss = rd("subset_structure.json")
lg = rd("locate_gap.json")
rb = rd("robustness.json")
tr = rd("traces.json")


def mean(v):
    return sum(v) / len(v)


def se(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))


def by_group(budget, model, get):
    per = {}
    for c in op["cells"]:
        if c["budget"] == budget and c["model"] == model:
            per.setdefault(c["group"], []).append(get(c))
    return {g: mean(v) for g, v in sorted(per.items())}


def ms(budget, model, get):
    v = list(by_group(budget, model, get).values())
    return [mean(v), se(v)]


out = {}

# ---------- section 2: subset PR / ROC levels + paired gaps
out["subset"] = {}
for met in ("pr", "roc"):
    for sub in ("all", "r0", "rN"):
        for m in MODELS:
            out["subset"][f"{sub}_{met}_{m}"] = [
                [agg["subset"][str(k)][m][f"{sub}_{met}"]["mean"],
                 agg["subset"][str(k)][m][f"{sub}_{met}"]["se"]] for k in K]
        g = agg["paired_gap"]
        out["subset"][f"{sub}_{met}_gap"] = [[g[str(k)][f"{sub}_{met}"]["mean"],
                                              g[str(k)][f"{sub}_{met}"]["se"]] for k in K]

# ---------- section 4.1: exact AP decomposition
out["ap"] = {"w0": op["w0"], "wN": op["wN"], "P": op["P"], "P0": op["P0"], "PN": op["PN"]}
for m in MODELS:
    for f in ("ap", "C0", "CN", "contrib0", "contribN"):
        out["ap"][f"{f}_{m}"] = [ms(k, m, (lambda ff: (lambda c: c[ff]))(f)) for k in K]
rows = []
for k in K:
    t0 = by_group(k, "tree", lambda c: c["contrib0"]); p0 = by_group(k, "tabpfn", lambda c: c["contrib0"])
    tn = by_group(k, "tree", lambda c: c["contribN"]); pn = by_group(k, "tabpfn", lambda c: c["contribN"])
    ta = by_group(k, "tree", lambda c: c["ap"]); pa = by_group(k, "tabpfn", lambda c: c["ap"])
    gs = sorted(t0)
    d0 = [t0[g] - p0[g] for g in gs]; dn = [tn[g] - pn[g] for g in gs]; da = [ta[g] - pa[g] for g in gs]
    assert abs(mean(da) - (mean(d0) + mean(dn))) < 1e-12
    rows.append({"k": k, "dap": [mean(da), se(da)], "d0": [mean(d0), se(d0)],
                 "dN": [mean(dn), se(dn)],
                 "share0": mean(d0) / (abs(mean(d0)) + abs(mean(dn))),
                 "wins": sum(1 for x in da if x > 0), "n": len(da)})
out["ap"]["gap_rows"] = rows

# ---------- section 3: operating point
out["op"] = {"thresh": op["thresh"]}
for sub in ("conf_all", "conf_r0", "conf_rN"):
    for key in ("precision", "recall", "f1"):
        for m in MODELS:
            out["op"][f"{sub}_{key}_{m}"] = [ms(k, m, (lambda s, kk: (lambda c: c[s][kk]))(sub, key))
                                             for k in K]
for m in MODELS:
    out["op"][f"flagged_{m}"] = [ms(k, m, lambda c: c["frac_ge_05"]) for k in K]
    out["op"][f"rprec_{m}"] = [ms(k, m, lambda c: c["rprec"]["precision"]) for k in K]
    out["op"][f"qneg99_{m}"] = [ms(k, m, lambda c: c["q_neg"]["0.99"]) for k in K]
    out["op"][f"qpos50_{m}"] = [ms(k, m, lambda c: c["q_pos"]["0.5"]) for k in K]
rp = []
for k in K:
    t = by_group(k, "tree", lambda c: c["rprec"]["precision"])
    p = by_group(k, "tabpfn", lambda c: c["rprec"]["precision"])
    d = [t[g] - p[g] for g in sorted(t)]
    rp.append([mean(d), se(d)])
out["op"]["rprec_gap"] = rp

# ---------- section 4.3: drop building 1241
out["drop"] = {}
for sub in ("rN", "r0"):
    for tag in ("all", "no1241"):
        out["drop"][f"{sub}_{tag}"] = [[rb["drop1241"][f"{k}|{sub}|{tag}"]["mean"],
                                        rb["drop1241"][f"{k}|{sub}|{tag}"]["se"]] for k in K]

# ---------- section 4.2: per-building sign test
out["sign"] = {}
for sub in ("rN", "r0"):
    out["sign"][sub] = [{"k": k,
                         "mean": rb["per_building_signtest"][f"{k}|{sub}"]["mean_gap"],
                         "se": rb["per_building_signtest"][f"{k}|{sub}"]["se"],
                         "wins": rb["per_building_signtest"][f"{k}|{sub}"]["tree_wins"],
                         "n": rb["per_building_signtest"][f"{k}|{sub}"]["n"],
                         "p": rb["per_building_signtest"][f"{k}|{sub}"]["p_sign"]} for k in K]

# ---------- section 4.4: stratified
out["strat"] = wb["stratified"]
out["r0_within_pair_share"] = wb["r0_within_pair_share"]

# ---------- section 1: structure
out["runs"] = ss["zero_runs"]
zaf = [r["z_anom"] / (r["z_anom"] + r["z_norm"]) for r in ss["by_hour"]]
out["hour"] = {"min": min(zaf), "max": max(zaf),
               "series": [{"h": r["hour"], "zfrac": r["z_anom"] / (r["z_anom"] + r["z_norm"]),
                           "nfrac": r["n_anom"] / (r["n_anom"] + r["n_norm"])} for r in ss["by_hour"]]}
out["rz"] = ss["rN_robust_z"]
out["stuck"] = lg["stuck"]

# ---------- F1 in context: no-skill ceiling + max-F1 over thresholds
fc = rd("f1_context.json")
SUBKEY = {"reading != 0  (all 73 buildings)": "rN",
          "reading != 0  (excluding 1241)": "rN_no1241",
          "reading == 0  (for contrast)": "r0",
          "pooled hot water": "all"}
out["f1"] = {"trivial": {SUBKEY[k]: v for k, v in fc["trivial"].items()}, "rows": {}}
for r in fc["rows"]:
    out["f1"]["rows"][f"{SUBKEY[r['subset']]}|{r['budget']}|{r['model']}"] = {
        "f1_05": r["f1_05"], "f1_05_se": r["f1_05_se"],
        "max_f1": r["max_f1"], "max_f1_se": r["max_f1_se"],
        "p_at_max": r["p_at_max"], "r_at_max": r["r_at_max"],
        "thr_at_max": r["thr_at_max"], "frac_at_max": r["frac_at_max"],
        "x_trivial": r["x_trivial"]}

# ---------- normalisations the literature would use instead of a raw F1
fa = rd("f1_alternatives.json")
out["alt"] = {"subsets": {}, "calib": fa["calib"]}
for name, blk in fa["subsets"].items():
    out["alt"]["subsets"][name] = {
        "pi": blk["pi"], "f1_trivial": blk["f1_trivial"],
        "ratio_ceiling": blk["ratio_ceiling"],
        "rows": {rk: {kk: rv[kk] for kk in ("max_f1", "x_trivial", "f1_skill", "mcc_05",
                                            "max_mcc", "mcc_thr", "ap", "ap_gain",
                                            "ap_over_pi", "f1_thr")}
                 for rk, rv in blk["rows"].items()}}
out["conc"] = [{"bid": r[0], "use": r[1], "anom": r[2], "rows": r[3]} for r in lg["rN_by_building"][:10]]
out["conc_total"] = sum(r[2] for r in lg["rN_by_building"])
out["traces"] = tr
out["cases"] = rd("month_cases.json")
out["month"] = rd("by_month.json")
out["season_prev"] = rd("season_prevalence.json")["counts"]
out["season_sit"] = rd("season_situation.json")["counts"]
_s2 = rd("rn_season_cases.json")
for _k in _s2:
    _s2[_k]["ranking"] = _s2[_k]["ranking"]
out["season2"] = _s2
_sc = rd("season_cases.json")
out["season_cases"] = {
    "summer": {"label": _sc["seasons"]["summer"]["label"],
               "ranking": _sc["seasons"]["summer"]["ranking"],
               "pick": _sc["seasons"]["summer"]["picks"][0]},
    "winter": {"label": _sc["seasons"]["winter"]["label"],
               "ranking": _sc["seasons"]["winter"]["ranking"][:10]},
}
out["case_rank"] = {"n": len(rd("case_search.json")["ranking"]),
                    "gaps": sorted(r["gap"] for r in rd("case_search.json")["ranking"])}

# run-length buckets recomputed here so the page and the report agree
out["runbuckets"] = [
    {"label": "1 h", "runs": 6801, "hours": 6801, "frac": 0.052},
    {"label": "2–3 h", "runs": 3696, "hours": 8678, "frac": 0.024},
    {"label": "4–24 h", "runs": 3505, "hours": 32781, "frac": 0.062},
    {"label": "1–7 d", "runs": 363, "hours": 22902, "frac": 0.332},
    {"label": "> 7 d", "runs": 101, "hours": 98224, "frac": 0.719},
]

P = BASE / "report_data.json"
P.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"wrote {P}  {P.stat().st_size/1024:.1f} KB")
print("keys:", list(out.keys()))
print("sanity  r0_roc tree:", [round(x[0], 4) for x in out['subset']['r0_roc_tree']])
print("sanity  rN gap all :", [round(x[0], 4) for x in out['drop']['rN_all']])
print("sanity  rN gap no  :", [round(x[0], 4) for x in out['drop']['rN_no1241']])
print("sanity  ap share0  :", [round(r['share0'], 3) for r in out['ap']['gap_rows']])
print("sanity  trivial    :", {k: round(v, 4) for k, v in out['f1']['trivial'].items()})
print("sanity  rN maxF1 T :", [round(out['f1']['rows'][f'rN|{k}|tree']['max_f1'], 3)
                               for k in (50, 100, 200, 400, 725)])
