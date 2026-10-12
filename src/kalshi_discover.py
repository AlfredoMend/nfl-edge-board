"""Find Kalshi NFL player-prop series and save a sample of their markets (format discovery)."""
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
SKIP = ("KXNFLGAME", "KXNFLSPREAD")


def main():
    series = []
    for params in ({"category": "Sports"}, {}):
        try:
            r = requests.get(f"{API}/series", params=params, timeout=30)
            r.raise_for_status()
            series = r.json().get("series") or []
            if series:
                break
        except Exception as e:
            print("series list failed:", params, e)
    nfl = [s for s in series if str(s.get("ticker", "")).startswith("KXNFL")]
    out = {"n_series_total": len(series),
           "nfl_series": [{k: s.get(k) for k in ("ticker", "title", "category", "tags")} for s in nfl],
           "samples": {}}
    for s in nfl:
        t = s["ticker"]
        if t in SKIP or "SEASON" in t or "MVP" in t or "DRAFT" in t:
            continue
        try:
            r = requests.get(f"{API}/markets", params={"series_ticker": t, "status": "open", "limit": 200}, timeout=30)
            r.raise_for_status()
            mk = r.json().get("markets", [])
        except Exception as e:
            print("markets failed", t, e)
            continue
        if mk:
            out["samples"][t] = {"count": len(mk), "markets": mk[:6]}
    (ROOT / "data" / "kalshi_props_discovery.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"{len(nfl)} NFL series, {len(out['samples'])} with open markets")


if __name__ == "__main__":
    main()
