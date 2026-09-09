from __future__ import annotations
import json, math
from pathlib import Path

BASE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/analysis_2026-09-08")
J = json.loads((BASE / "data/opoint_apdecomp.json").read_text(encoding="utf-8"))
CELLS = J["cells"]
w0, wN = J["w0"], J["wN"]
BUDGETS = (50, 100, 200, 400, 725)
MODELS = ("tree", "tabpfn")


def mean(v): return sum(v) / len(v)
def se(v):
    if len(v) < 2: return None
    m = mean(v); return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))


def by_group(budget, model, get):
    per = {}
    for c in CELLS:
        if c["budget"] == budget and c["model"] == model:
            per.setdefault(c["group"], []).append(get(c))
    return {g: mean(v) for g, v in sorted(per.items())}


def ms(budget, model, get):
    v = list(by_group(budget, model, get).values())
    return mean(v), se(v)


def fmt(m, s, prec=4, sign=False):
    f = f"{{:+.{prec}f}}" if sign else f"{{:.{prec}f}}"
    return f.format(m) + (f"±{s:.{prec}f}" if s is not None else "")


print("=" * 108)
print("A.  EXACT DECOMPOSITION OF POOLED HOT-WATER AP  (=PR-AUC)")
print(f"    identity:  AP = w0*C0 + wN*CN   with w0={w0:.4f} (r0 holds {J['P0']:,} of {J['P']:,} positives),"
      f" wN={wN:.4f} ({J['PN']:,})")
print("    C_S = mean precision-at-own-threshold over the positives in subset S")
print("=" * 108)
print(f"{'K':>4} {'model':7} | {'AP':>16} | {'C0':>16} {'w0*C0':>16} | {'CN':>16} {'wN*CN':>16}")
print("-" * 108)
for b in BUDGETS:
    for m in MODELS:
        ap = ms(b, m, lambda c: c["ap"]); c0 = ms(b, m, lambda c: c["C0"])
        cn = ms(b, m, lambda c: c["CN"]); k0 = ms(b, m, lambda c: c["contrib0"])
        kn = ms(b, m, lambda c: c["contribN"])
        print(f"{b:>4} {m:7} | {fmt(*ap):>16} | {fmt(*c0):>16} {fmt(*k0):>16} | {fmt(*cn):>16} {fmt(*kn):>16}")
    print("-" * 108)

print("\n" + "=" * 108)
print("B.  WHERE THE POOLED AP GAP COMES FROM   dAP = w0*dC0 + wN*dCN   (paired Tree-TabPFN)")
print("=" * 108)
print(f"{'K':>4} | {'dAP (pooled)':>17} | {'w0*dC0 (r=0 pos)':>19} {'share':>7} | {'wN*dCN (r!=0 pos)':>19} {'share':>7} | wins")
print("-" * 108)
for b in BUDGETS:
    t0 = by_group(b, "tree", lambda c: c["contrib0"]); p0 = by_group(b, "tabpfn", lambda c: c["contrib0"])
    tn = by_group(b, "tree", lambda c: c["contribN"]); pn = by_group(b, "tabpfn", lambda c: c["contribN"])
    ta = by_group(b, "tree", lambda c: c["ap"]); pa = by_group(b, "tabpfn", lambda c: c["ap"])
    gs = sorted(t0)
    d0 = [t0[g] - p0[g] for g in gs]; dn = [tn[g] - pn[g] for g in gs]; da = [ta[g] - pa[g] for g in gs]
    m0, s0 = mean(d0), se(d0); mn, sn = mean(dn), se(dn); ma, sa = mean(da), se(da)
    tot = abs(m0) + abs(mn)
    wins = f"{sum(1 for x in da if x > 0)}/5 Tree"
    print(f"{b:>4} | {fmt(ma, sa, 4, True):>17} | {fmt(m0, s0, 4, True):>19} {m0/tot:>6.0%} |"
          f" {fmt(mn, sn, 4, True):>19} {mn/tot:>6.0%} | {wins}")
    chk = abs(ma - (m0 + mn))
    assert chk < 1e-12, chk

print("\n" + "=" * 108)
print("C.  FIXED OPERATING POINT  score >= 0.50   (precision / recall / F1)")
print("=" * 108)
for sub, lbl, npos in (("conf_all", "POOLED (n=636,121, 90,691 pos)", 90691),
                       ("conf_r0", "reading == 0 (n=169,386, 80,776 pos -> prev 47.7%)", 80776),
                       ("conf_rN", "reading != 0 (n=466,735,  9,915 pos -> prev  2.1%)", 9915)):
    print(f"\n### {lbl}")
    print(f"{'K':>4} | {'precision T/P':>27} | {'recall T/P':>27} | {'F1 T/P':>27}")
    print("-" * 96)
    for b in BUDGETS:
        row = []
        for key in ("precision", "recall", "f1"):
            t = ms(b, "tree", lambda c: c[sub][key]); p = ms(b, "tabpfn", lambda c: c[sub][key])
            row.append(f"{t[0]:.3f}±{t[1]:.3f} / {p[0]:.3f}±{p[1]:.3f}")
        print(f"{b:>4} | {row[0]:>27} | {row[1]:>27} | {row[2]:>27}")

print("\n" + "=" * 108)
print("D.  IS THE 0.50 THRESHOLD COMPARABLE?  fraction of hot-water rows flagged at >=0.5")
print("    (true anomaly rate = 14.26%).  Also score quantiles.")
print("=" * 108)
print(f"{'K':>4} | {'%flagged T/P':>21} | {'median score T/P':>21} | {'p99 neg T/P':>21} | {'median pos T/P':>21}")
print("-" * 100)
for b in BUDGETS:
    fl = (ms(b, "tree", lambda c: c["frac_ge_05"]), ms(b, "tabpfn", lambda c: c["frac_ge_05"]))
    md = (ms(b, "tree", lambda c: c["q"]["0.5"]), ms(b, "tabpfn", lambda c: c["q"]["0.5"]))
    n99 = (ms(b, "tree", lambda c: c["q_neg"]["0.99"]), ms(b, "tabpfn", lambda c: c["q_neg"]["0.99"]))
    mp = (ms(b, "tree", lambda c: c["q_pos"]["0.5"]), ms(b, "tabpfn", lambda c: c["q_pos"]["0.5"]))
    print(f"{b:>4} | {f'{fl[0][0]:.2%} / {fl[1][0]:.2%}':>21} | {f'{md[0][0]:.3f} / {md[1][0]:.3f}':>21} |"
          f" {f'{n99[0][0]:.3f} / {n99[1][0]:.3f}':>21} | {f'{mp[0][0]:.3f} / {mp[1][0]:.3f}':>21}")

print("\n" + "=" * 108)
print("E.  THRESHOLD-FREE OPERATING POINT: R-precision (flag exactly 90,691 rows = top 14.26%)")
print("    precision == recall by construction, so one number per model.")
print("=" * 108)
print(f"{'K':>4} | {'R-precision T/P':>25} | {'paired gap':>18}")
print("-" * 66)
for b in BUDGETS:
    t = by_group(b, "tree", lambda c: c["rprec"]["precision"])
    p = by_group(b, "tabpfn", lambda c: c["rprec"]["precision"])
    gs = sorted(t); d = [t[g] - p[g] for g in gs]
    tm, ts = mean(list(t.values())), se(list(t.values()))
    pm, ps = mean(list(p.values())), se(list(p.values()))
    print(f"{b:>4} | {f'{tm:.4f}±{ts:.4f} / {pm:.4f}±{ps:.4f}':>25} | {fmt(mean(d), se(d), 4, True):>18}")
