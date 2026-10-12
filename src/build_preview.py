"""Make a single-file copy of the site with the data inlined (for sharing a snapshot)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
site = ROOT / "site"
data = {p.stem: json.loads(p.read_text()) for p in (site / "data").glob("*.json")}
html = (site / "index.html").read_text()
inline = "<script>window.__DATA__=" + json.dumps(data) + ";</script>\n"
html = html.replace("<script>\nconst pct", inline + "<script>\nconst pct", 1)
out = ROOT / "preview.html"
out.write_text(html)
print(out, len(html) // 1024, "KB")
