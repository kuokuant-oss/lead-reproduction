import json
from pathlib import Path
j = json.loads(Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff"
                    "/analysis_2026-09-08/data/aggregated.json").read_text(encoding="utf-8"))
S = j["subset"]
print("Per-seed detail. Cell = Tree/TabPFN. Group = building seed (K<=400) or row seed (K=725).")
for met, lbl in (("all_pr", "pooled PR-AUC"), ("r0_pr", "PR-AUC reading=0"), ("rN_pr", "PR-AUC reading!=0"),
                 ("all_roc", "pooled ROC-AUC"), ("r0_roc", "ROC-AUC reading=0"), ("rN_roc", "ROC-AUC reading!=0")):
    print(f"\n### {lbl}")
    hdr = " ".join(f"{g:>14}" for g in ("g0", "g1", "g2", "g3", "g4"))
    print(f"{'K':>4} | {hdr} |{'MEAN gap':>12}")
    for K in ("50", "100", "200", "400", "725"):
        t = S[K]["tree"][met]["by_group"]; p = S[K]["tabpfn"][met]["by_group"]
        gs = sorted(t)
        cells = " ".join(f"{t[g]:.3f}/{p[g]:.3f}".rjust(14) for g in gs)
        gap = sum(t[g] - p[g] for g in gs) / len(gs)
        print(f"{K:>4} | {cells} |{gap:+12.4f}")
