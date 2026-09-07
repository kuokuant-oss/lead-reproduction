from pathlib import Path
import numpy as np, pandas as pd
SCRATCH = Path("/mnt/c/Users/User/AppData/Local/Temp/claude/C--Users-User-projects-lead-reproduction-temp/4888b221-2e78-4bf8-bfd6-7f21071cc05b/scratchpad")
META = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/m3/building_metadata.csv")
side = np.load(SCRATCH / "hotwater_sideinfo.npz", allow_pickle=True)
y, bid, sid = side["y"], side["bid"], side["sid"]
bmeta = pd.read_csv(META).set_index("building_id")
sqft = bmeta["square_feet"].reindex(bid).to_numpy(dtype=float)
uniq = pd.DataFrame({"b": bid, "v": sqft}).drop_duplicates("b")
edges = np.percentile(uniq["v"], [0, 25, 50, 75, 100])
print("=== 各面積四分位的 site 組成 ===")
for i in range(4):
    lo, hi = edges[i], edges[i+1]
    m = (sqft >= lo) & (sqft <= hi) if i == 3 else (sqft >= lo) & (sqft < hi)
    nb = pd.Series(sid[m]).groupby(pd.Series(bid[m])).first().groupby(lambda x: 0).size()
    bl = pd.DataFrame({"b": bid[m], "s": sid[m]}).drop_duplicates("b")
    comp_b = bl.s.value_counts().sort_index()
    pos = pd.Series(sid[m & (y == 1)]).value_counts()
    tot = pos.sum()
    print(f"\n  {lo:,.0f}–{hi:,.0f} sq ft  ({len(bl)} 棟, {tot:,} 異常)")
    print("    建物數: " + ", ".join(f"site {s}: {n}" for s, n in comp_b.items()))
    print("    異常占比: " + ", ".join(f"site {s}: {v/tot:.0%}" for s, v in pos.sort_index().items()))
