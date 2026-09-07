import json
from pathlib import Path

R = Path("/home/kuant_kuo/projects/lead-reproduction-v4-fixed-10k/data/processed"
         "/m5_building_curve/v5_fixed_50k")

st = R / "supervisor" / "status.json"
print("--- supervisor/status.json ---")
print(st.read_text() if st.is_file() else "（不存在）")

comp = sorted(p for p in (R / "model_runs").rglob("COMPLETE.json"))
print(f"\nCOMPLETE.json 總數: {len(comp)}")

k50 = sorted(str(p).split("model_runs/")[1] for p in comp if "tabpfn_k50_f137" in str(p))
print(f"\nK=50 TabPFN 完成 {len(k50)}/10:")
for p in k50:
    print("  " + p)

failed = list(R.rglob("FAILED.json"))
print(f"\nFAILED.json: {len(failed)}")
for f in failed:
    print("  " + str(f))

ev = R / "supervisor" / "events.jsonl"
if ev.is_file():
    lines = [l for l in ev.read_text().splitlines() if l.strip()]
    print(f"\n最後 6 筆事件:")
    for l in lines[-6:]:
        try:
            d = json.loads(l)
            print("  " + json.dumps({k: d.get(k) for k in
                                     ("timestamp", "event", "budget", "building_seed",
                                      "row_seed", "model", "elapsed_seconds")
                                     if d.get(k) is not None}, ensure_ascii=False))
        except Exception:
            print("  " + l[:160])
