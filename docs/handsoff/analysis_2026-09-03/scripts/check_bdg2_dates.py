from pathlib import Path

D = Path("/home/kuant_kuo/projects/lead-reproduction/data/raw/bdg2")
files = sorted(D.iterdir()) if D.is_dir() else []
print(f"bdg2 目錄檔案數: {len(files)}")
for f in files:
    print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")
for f in files:
    if f.suffix.lower() != ".csv":
        continue
    with f.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip()
        first = fh.readline().strip()
        last = first
        for line in fh:
            if line.strip():
                last = line.strip()
    print(f"\n--- {f.name} ---")
    print(f"  header: {header[:160]}")
    print(f"  first : {first[:160]}")
    print(f"  last  : {last[:160]}")
