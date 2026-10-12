"""Snapshot Kalshi NFL game-winner prices and line them up with model predictions.

Uses Kalshi's public market-data endpoints (no API key needed for reading prices).
Writes site/data/kalshi_latest.json and appends to data/kalshi_history.csv so the
model can later be scored against the price that existed before each game.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNFLGAME"
# Kalshi team codes that differ from nflverse codes
ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV"}
TAKER_FEE = 0.07  # Kalshi fee ~ 0.07 * P * (1 - P) per contract


def _price(m: dict, key: str):
    """Kalshi returns cents (yes_bid) and/or dollar strings (yes_bid_dollars)."""
    if m.get(f"{key}_dollars") not in (None, ""):
        return float(m[f"{key}_dollars"])
    if m.get(key) not in (None, 0):
        return m[key] / 100
    return None


def fetch_markets() -> list[dict]:
    markets, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{API}/markets", params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        markets += js.get("markets", [])
        cursor = js.get("cursor")
        if not cursor:
            return markets


def snapshot() -> pd.DataFrame:
    rows = []
    taken = datetime.now(timezone.utc).isoformat(timespec="minutes")
    for m in fetch_markets():
        team = m["ticker"].rsplit("-", 1)[-1]
        team = ALIASES.get(team, team)
        bid, ask, last = _price(m, "yes_bid"), _price(m, "yes_ask"), _price(m, "last_price")
        mid = (bid + ask) / 2 if bid and ask else last
        rows.append({"snapshot_utc": taken,
                     "event_ticker": m.get("event_ticker") or m["ticker"].rsplit("-", 1)[0], "ticker": m["ticker"],
                     "team": team, "yes_bid": bid, "yes_ask": ask, "last": last, "mid": mid,
                     "volume": m.get("volume"), "close_time": m.get("close_time")})
    return pd.DataFrame(rows)


def match_to_games(snap: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """One row per predicted game with Kalshi home-team prices attached."""
    out = []
    events = {ev: set(d.team) for ev, d in snap.groupby("event_ticker")}
    for g in preds.itertuples():
        want = {g.home_team, g.away_team}
        ev = next((e for e, teams in events.items() if teams == want), None)
        if ev is None:
            continue
        d = snap[snap.event_ticker == ev]
        home = d[d.team == g.home_team]
        if home.empty:
            continue
        h = home.iloc[0]
        p_k = h["mid"]
        cost_home = h["yes_ask"]
        away = d[d.team == g.away_team]
        cost_away = away.iloc[0]["yes_ask"] if len(away) else None
        rec = {"game_id": g.game_id, "kalshi_event": ev, "kalshi_home_mid": p_k,
               "kalshi_home_ask": cost_home, "kalshi_away_ask": cost_away,
               "snapshot_utc": h["snapshot_utc"]}
        # Expected value per $1 contract after fees, buying the side the model likes
        if cost_home:
            fee = TAKER_FEE * cost_home * (1 - cost_home)
            rec["ev_home"] = g.p_model - cost_home - fee
        if cost_away:
            fee = TAKER_FEE * cost_away * (1 - cost_away)
            rec["ev_away"] = (1 - g.p_model) - cost_away - fee
        out.append(rec)
    return pd.DataFrame(out)


def main():
    preds = pd.read_json(ROOT / "site" / "data" / "predictions.json")
    snap = snapshot()
    hist = ROOT / "data" / "kalshi_history.csv"
    snap.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    matched = match_to_games(snap, preds) if len(snap) else pd.DataFrame()
    payload = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
               "markets_seen": int(len(snap)), "games": matched.to_dict(orient="records")}
    (ROOT / "site" / "data" / "kalshi_latest.json").write_text(json.dumps(payload, default=str))
    print(f"{len(snap)} markets, matched {len(matched)} of {len(preds)} games")


if __name__ == "__main__":
    main()
