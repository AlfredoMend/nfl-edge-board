"""Price Kalshi NFL player-prop markets with the model's simulated distributions.

Kalshi lists props as threshold contracts, for example "Puka Nacua: 70+ receiving yards"
(YES wins if he reaches 70). For each open contract on an upcoming game, the model's chance
of YES comes from its 10,000 simulations. Expected value per $1 contract, after Kalshi's taker fee:
    buy YES at ask a:  EV = p - a - fee(a)
    buy NO  at ask b:  EV = (1 - p) - b - fee(b)
Writes site/data/kalshi_props.json and appends a price log to data/kalshi_props_history.csv.
"""
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
TAKER_FEE = 0.07
# series -> (model stat, label). PROT = Kalshi's "protected" version of the same prop.
SERIES = {
    "KXNFLPASSYDS": ("pass_yds", "passing yards"), "KXNFLPASSYDSPROT": ("pass_yds", "passing yards"),
    "KXNFLPASSATT": ("pass_att", "pass attempts"), "KXNFLPASSCOMP": ("comp", "completions"),
    "KXNFLPASSTDS": ("pass_td", "passing TDs"), "KXNFLPASSTDSPROT": ("pass_td", "passing TDs"),
    "KXNFLPASSINT": ("ints", "interceptions"),
    "KXNFLRSHYDS": ("rush_yds", "rushing yards"), "KXNFLRSHYDSPROT": ("rush_yds", "rushing yards"),
    "KXNFLRSHATT": ("rush_att", "rush attempts"),
    "KXNFLREC": ("rec", "receptions"), "KXNFLRECPROT": ("rec", "receptions"),
    "KXNFLRECYDS": ("rec_yds", "receiving yards"), "KXNFLRECYDSPROT": ("rec_yds", "receiving yards"),
    "KXNFLRRYDS": ("rush_rec_yds", "rush + rec yards"),
    # TD series are ladders (1+, 2+, 3+ touchdowns); the threshold comes from each contract
    "KXNFLANYTD": ("tds", "TDs"), "KXNFLTD": ("tds", "TDs"), "KXNFLTDPROT": ("tds", "TDs"),
    "KXNFL2TD": ("tds", "TDs"),
}
KALSHI_CODE = {"LA": ["LAR", "LA"], "JAX": ["JAC", "JAX"], "WAS": ["WSH", "WAS"], "LV": ["LV", "LVR"]}
MIN_EDGE = 0.05          # show plays with at least 5 cents of value per contract (covers the model's
                         # remaining calibration error of a few percentage points on props)
CHECK_EDGE = 0.15        # edges this large usually mean news the model has not seen; flagged "check news"


def _price(m, key):
    v = m.get(f"{key}_dollars")
    if v not in (None, ""):
        return float(v)
    v = m.get(key)
    return v / 100 if isinstance(v, (int, float)) and v > 0 else None


def _fee(p):
    return TAKER_FEE * p * (1 - p)


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    return re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s).strip()


def fetch(series: str) -> list:
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{API}/markets", params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        out += js.get("markets", [])
        cursor = js.get("cursor")
        if not cursor:
            return out


def _game_for_event(event: str, games: pd.DataFrame):
    code = re.sub(r"^\d{2}[A-Z]{3}\d{2}", "", event.split("-", 1)[-1])
    for g in games.itertuples():
        for a in KALSHI_CODE.get(g.away_team, [g.away_team]):
            for h in KALSHI_CODE.get(g.home_team, [g.home_team]):
                if code == a + h:
                    return g
    return None


def p_at_least(q: list, threshold: float) -> float:
    """P(stat > threshold) from the 101 simulated percentiles (threshold is N - 0.5 for 'N+')."""
    q = np.asarray(q, dtype=float)
    below = np.searchsorted(q, threshold, side="right")      # percentiles at or below the line
    if 0 < below < len(q) and q[below] > q[below - 1]:       # interpolate inside a continuous stretch
        frac = (threshold - q[below - 1]) / (q[below] - q[below - 1])
        cdf = (below - 1 + frac) / (len(q) - 1)
    else:
        cdf = below / len(q)
    return float(min(max(1 - cdf, 0.0), 1.0))


def main():
    preds = pd.read_json(ROOT / "site" / "data" / "predictions.json")
    dist_path = ROOT / "site" / "data" / "props_dist.json"
    if not dist_path.exists() or preds.empty:
        print("no prop distributions yet")
        return
    dist = json.loads(dist_path.read_text())
    games = preds[preds.status == "upcoming"]
    kick = pd.to_datetime(games.gameday.astype(str).str[:10] + " " + games.gametime.fillna("13:00"))
    kick = kick.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    games = games[kick > pd.Timestamp.now(tz="UTC")]

    # name lookup per game: normalized name -> (side, player id)
    lookup = {}
    for gid, sides in dist.items():
        for side, players in sides.items():
            for pid, p in players.items():
                lookup.setdefault(gid, {})[_norm(p["name"])] = (side, pid)

    rows, seen = [], 0
    for series, (stat, label) in SERIES.items():
        try:
            mk = fetch(series)
        except Exception as e:
            print("skip", series, type(e).__name__)
            continue
        for m in mk:
            seen += 1
            g = _game_for_event(m.get("event_ticker", ""), games)
            if g is None or g.game_id not in dist:
                continue
            title = m.get("title") or ""
            name = (m.get("yes_sub_title") or title).split(":")[0].strip()
            hit = lookup.get(g.game_id, {}).get(_norm(name))
            if hit is None:
                continue
            side, pid = hit
            pl = dist[g.game_id][side][pid]
            fs = m.get("floor_strike")
            if fs is None:
                suffix = m["ticker"].rsplit("-", 1)[-1]
                if stat == "tds" and suffix.isdigit():
                    fs = int(suffix) - 0.5
                elif series == "KXNFL2TD":
                    fs = 1.5
                elif series == "KXNFLANYTD":
                    fs = 0.5
                else:
                    continue
            line = int(round(float(fs) + 0.5))
            if stat == "tds":
                exact = {1: pl.get("anytime_td"), 2: pl.get("td_2plus"), 3: pl.get("td_3plus")}.get(line)
                if exact is None:
                    if "tds" not in pl["dist"]:
                        continue
                    exact = p_at_least(pl["dist"]["tds"], float(fs))
                p = float(exact)
            else:
                if stat not in pl["dist"]:
                    continue
                p = p_at_least(pl["dist"][stat], float(fs))
            ya, na, yb = _price(m, "yes_ask"), _price(m, "no_ask"), _price(m, "yes_bid")
            ev_yes = p - ya - _fee(ya) if ya else None
            ev_no = (1 - p) - na - _fee(na) if na else None
            best = max([(ev_yes, "YES", ya), (ev_no, "NO", na)], key=lambda x: -9 if x[0] is None else x[0])
            team = g.home_team if side == "home" else g.away_team
            rows.append({"game_id": g.game_id, "ticker": m["ticker"], "player": pl["name"], "team": team,
                         "market": label, "stat": stat, "line": line,
                         "contract": (f"{pl['name']}: anytime TD" if (stat == "tds" and line == 1)
                                      else f"{pl['name']}: {line}+ {label}"),
                         "protected": series.endswith("PROT"), "model_p": round(p, 4),
                         "yes_bid": yb, "yes_ask": ya, "no_ask": na,
                         "kalshi_mid": round((yb + ya) / 2, 3) if yb and ya else None,
                         "ev_yes": None if ev_yes is None else round(ev_yes, 4),
                         "ev_no": None if ev_no is None else round(ev_no, 4),
                         "best_side": best[1], "best_ev": None if best[0] is None else round(best[0], 4),
                         "best_price": best[2], "wide": bool(yb is None or (ya and yb and ya - yb > 0.10)),
                         "check": bool(best[0] is not None and best[0] >= CHECK_EDGE)})
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("best_ev", ascending=False)
        log = df.assign(snapshot_utc=now)[["snapshot_utc", "game_id", "ticker", "model_p", "yes_bid", "yes_ask", "no_ask"]]
        hp = ROOT / "data" / "kalshi_props_history.csv"
        log.to_csv(hp, mode="a", header=not hp.exists(), index=False)
    payload = {"updated": now, "markets_seen": seen, "priced": int(len(df)), "min_edge": MIN_EDGE,
               "markets": df.to_dict(orient="records") if len(df) else []}
    (ROOT / "site" / "data" / "kalshi_props.json").write_text(json.dumps(payload, default=str))
    good = df[(df.best_ev >= MIN_EDGE) & ~df.wide] if len(df) else df
    print(f"kalshi props: {seen} markets seen, {len(df)} priced, {len(good)} with {MIN_EDGE*100:.0f}c+ edge")


if __name__ == "__main__":
    main()
