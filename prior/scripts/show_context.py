import json
from pathlib import Path
D = Path("/mnt/c/Users/User/Documents/Codex/2026-09-01/wsl-ubuntu-m5-building-count-v5/outputs/scripts")
j = json.loads((D / "m5_exp_b_building_count_meter_pr_auc_v5_fixed50k_provisional_data.json").read_text(encoding="utf-8"))
for k in ("context_rows", "class_balance", "holdout_rows", "selection", "experiment_version",
          "evaluation_meter_ids", "excluded_evaluation_meter_ids", "status", "generated_at_utc"):
    print(f"{k}: {json.dumps(j.get(k))[:700]}")
