from pathlib import Path


root = Path("/content/lead_tabpfn_head")
(root / "work" / "chunks").mkdir(parents=True, exist_ok=True)
print(root)
