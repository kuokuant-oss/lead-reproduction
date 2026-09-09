"""Substitute report_data.json into report.template.html -> report.html."""
from pathlib import Path

BASE = Path("/mnt/c/Users/User/projects/lead_reproduction_temp/docs/handsoff/analysis_2026-09-08")
tpl = (BASE / "report.template.html").read_text(encoding="utf-8")
data = (BASE / "data/report_data.json").read_text(encoding="utf-8")
assert "/*__DATA__*/" in tpl, "placeholder missing"
out = tpl.replace("/*__DATA__*/", data)
assert "/*__DATA__*/" not in out
p = BASE / "report.html"
p.write_text(out, encoding="utf-8")
print(f"wrote {p}  {p.stat().st_size/1024:.1f} KB  (data {len(data)/1024:.1f} KB)")
