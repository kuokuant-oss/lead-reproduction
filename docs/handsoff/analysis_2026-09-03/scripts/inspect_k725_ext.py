import json
from pathlib import Path

import numpy as np

BASE = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/data/processed"
            "/m5_building_curve/v5_fixed_50k_k725_colab/model_results")
for r in (2, 3, 4):
    d = BASE / f"row_seed{r}"
    print(f"== row_seed{r} ==")
    print("  files:", sorted(p.name for p in d.iterdir()))
    with np.load(d / "predictions.npz") as z:
        print("  npz keys:", sorted(z.files))
        for k in sorted(z.files):
            print(f"    {k}: shape={z[k].shape} dtype={z[k].dtype}")
    for name in ("cell.json", "COMPLETE.json"):
        p = d / name
        if p.is_file():
            meta = json.loads(p.read_text())
            keep = {k: meta.get(k) for k in
                    ("experiment_version", "building_budget", "building_seed", "row_seed",
                     "features", "model_seed", "context_rows", "holdout_rows",
                     "holdout_row_sha256", "mode", "score_names", "n_estimators",
                     "evaluation_meter_ids", "excluded_evaluation_meter_ids")}
            print(f"  {name}: {json.dumps(keep, ensure_ascii=False)}")
            prov = meta.get("provenance")
            if prov:
                print(f"    provenance: {prov}")
summary = BASE / "k725_five_seed_summary.json"
if summary.is_file():
    s = json.loads(summary.read_text())
    print("\n== five_seed_summary keys ==")
    print(list(s.keys()))
    print(json.dumps(s, ensure_ascii=False)[:1500])
