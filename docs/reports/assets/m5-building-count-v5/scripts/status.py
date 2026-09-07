"""Read-only status snapshot for m5_building_count_v5_fixed_50k.

Run in WSL:
    /home/kuant_kuo/projects/lead-reproduction/.venv/bin/python status.py

Touches nothing. Safe to run at any time while the formal schedule is running.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(
    "/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/"
    "data/processed/m5_building_curve/v5_fixed_50k"
)
TPE = timezone(timedelta(hours=8))
now = time.time()


def ts(e):
    return datetime.fromtimestamp(e, TPE).strftime("%m-%d %H:%M")


print(f"now  {ts(now)} (Asia/Taipei)")

st = json.loads((ROOT / "supervisor/status.json").read_text())
print(f"\nsupervisor: {st['status']}  completed={st['completed']}  pending={st['pending']}"
      f"  (updated {ts(st['timestamp'])}, {(now-st['timestamp'])/3600:.1f}h ago)")
ci = st["current_identity"]
print(f"current unit {st['current_unit']}: K={ci['K']} b{ci['building_seed']} "
      f"r{ci['row_seed']} {ci['model']}")

print("\nmodel-cell grid")
print(f"  {'K':>5} {'Tree':>7} {'TabPFN':>7}")
for K in (725, 400, 200, 100, 50):
    seeds = [725] if K == 725 else [0, 1, 2, 3, 4]
    counts = {}
    for model, d in (("tree", f"tree_no_es_k{K}_f137"), ("tabpfn", f"tabpfn_k{K}_f137")):
        counts[model] = sum(
            (ROOT / f"model_runs/building_seed{b}/row_seed{r}/{d}/COMPLETE.json").is_file()
            for b in seeds for r in (0, 1)
        )
    tot = len(seeds) * 2
    print(f"  {K:>5} {counts['tree']:>4}/{tot} {counts['tabpfn']:>5}/{tot}")

failed = list(ROOT.rglob("FAILED.json"))
print(f"\nFAILED.json: {len(failed)}" + ("" if not failed else f" -> {failed}"))

print("\nlast 4 events")
for line in (ROOT / "supervisor/events.jsonl").read_text().splitlines()[-4:]:
    ev = json.loads(line)
    i = ev.get("identity") or {}
    extra = f" elapsed={ev['elapsed_seconds']:.0f}s" if "elapsed_seconds" in ev else ""
    who = (f"K={i.get('K')} b{i.get('building_seed')} r{i.get('row_seed')} {i.get('model')}"
           if i else ev.get("stage", ""))
    print(f"  {ts(ev['timestamp'])}  {ev['event']:<17} {who}{extra}")

cell = (ROOT / f"model_runs/building_seed{ci['building_seed']}/row_seed{ci['row_seed']}"
        f"/tabpfn_k{ci['K']}_f137")
hb_path = cell / "heartbeat.json"
if hb_path.is_file():
    hb = json.loads(hb_path.read_text())
    done, total = hb["completed_units"], hb["total_units"]
    print(f"\nactive cell {cell.parent.parent.name}/{cell.parent.name}/{cell.name}")
    print(f"  chunks {done}/{total}  heartbeat age {now-hb['timestamp']:.0f}s  status={hb['status']}")
    # ETA from this cell's own chunk rate plus the mean of finished K=100/50 cells
    chunks = sorted((cell / "prediction_chunks").glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if len(chunks) > 1:
        rate = (chunks[-1].stat().st_mtime - chunks[0].stat().st_mtime) / (len(chunks) - 1)
        finish = chunks[-1].stat().st_mtime + (total - len(chunks)) * rate
        print(f"  chunk rate {rate:.1f} s/chunk -> finishes ~{ts(finish)}")
        remaining_cells = st["pending"] - 1
        print(f"  {remaining_cells} further cells at ~8.2 h each -> all done ~"
              f"{ts(finish + remaining_cells * 29500)}")
else:
    print(f"\nactive cell {cell.name}: no heartbeat yet (context fit phase, ~8-11 min)")
