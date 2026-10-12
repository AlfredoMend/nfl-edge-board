"""Snapshot Kalshi NFL game-winner and spread prices and line them up with model predictions.

Uses Kalshi's public market-data endpoints (no API key needed for reading prices).
Writes site/data/kalshi_latest.json and appends to data/kalshi_history.csv so the
model can later be scored against the price that existed before each game.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNFLGAME"
SPREAD_SERIES = "KXNFLSPREAD"  # e.g. KXNFLSPREAD-26SEP13BUFHOU-BUF3 = "Buffalo wins by over 2.5"
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


def fetch_markets(series: str = SERIES) -> list[dict]:
    markets, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "limit": 1000}
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


def spread_snapshot() -> pd.DataFrame:
    rows = []
    taken = datetime.now(timezone.utc).isoformat(timespec="minutes")
    for m in fetch_markets(SPREAD_SERIES):
        mt = re.match(r"^([A-Z]+)(\d+)$", m["ticker"].rsplit("-", 1)[-1])
        if not mt:
            continue
        team = ALIASES.get(mt.group(1), mt.group(1))
        line = m.get("floor_strike")
        line = float(line) if line not in (None, "") else int(mt.group(2)) - 0.5
        yes_bid, yes_ask = _price(m, "yes_bid"), _price(m, "yes_ask")
        no_ask = _price(m, "no_ask") or ((1 - yes_bid) if yes_bid else None)
        rows.append({"snapshot_utc": taken,
                     "event_ticker": m.get("event_ticker") or m["ticker"].rsplit("-", 1)[0],
                     "ticker": m["ticker"], "team": team, "line": line,
                     "yes_bid": yes_bid, "yes_ask": yes_ask, "no_ask": no_ask,
                     "title": m.get("yes_sub_title") or m.get("title"), "volume": m.get("volume")})
    return pd.DataFrame(rows)


def _fee(price):
    return TAKER_FEE * price * (1 - price)


def match_spreads(spr: pd.DataFrame, preds: pd.DataFrame, sd: float) -> dict:
    """For each game, price every Kalshi spread contract with the model and pick the best one."""
    out = {}
    if spr.empty or "pred_margin" not in preds:
        return out
    events = {ev: set(d.team) for ev, d in spr.groupby("event_ticker")}
    for g in preds.itertuples():
        want = {g.home_team, g.away_team}
        ev = next((e for e, teams in events.items() if _event_has_both(e, g) or teams == want), None)
        if ev is None:
            continue
        contracts = []
        for c in spr[spr.event_ticker == ev].itertuples():
            margin = g.pred_margin if c.team == g.home_team else -g.pred_margin
            p_yes = float(norm.cdf((margin - c.line) / sd))
            opp = g.away_team if c.team == g.home_team else g.home_team
            rec = {"team": c.team, "line": c.line, "p_yes": p_yes,
                   "yes_ask": c.yes_ask, "no_ask": c.no_ask,
                   "yes_label": f"{c.team} wins by {int(c.line + 0.5)}+",
                   "no_label": f"{opp} +{c.line:g} (covers)"}
            rec["ev_yes"] = p_yes - c.yes_ask - _fee(c.yes_ask) if c.yes_ask else None
            rec["ev_no"] = (1 - p_yes) - c.no_ask - _fee(c.no_ask) if c.no_ask else None
            contracts.append(rec)
        if not contracts:
            continue
        best = None
        for c in contracts:
            for side in ("yes", "no"):
                ev_ = c[f"ev_{side}"]
                if ev_ is not None and (best is None or ev_ > best["ev"]):
                    best = {"side": side, "label": c[f"{side}_label"], "price": c[f"{side}_ask"],
                            "p_win": c["p_yes"] if side == "yes" else 1 - c["p_yes"], "ev": ev_,
                            "team": c["team"], "line": c["line"]}
        out[g.game_id] = {"spread_event": ev, "best": best, "contracts": len(contracts)}
    return out


def _code_tail(ev: str) -> str:
    return ev.split("-", 1)[-1]


def _event_has_both(ev: str, g) -> bool:
    """Event codes look like 26OCT12BUFLAR: date, then away and home Kalshi codes."""
    code = re.sub(r"^\d{2}[A-Z]{3}\d{2}", "", _code_tail(ev))
    rev = {}
    for k, v in ALIASES.items():
        rev.setdefault(v, set()).add(k)
    homes = {g.home_team} | rev.get(g.home_team, set())
    aways = {g.away_team} | rev.get(g.away_team, set())
    return any(code == a + h for a in aways for h in homes)


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
    if "status" in preds:
        preds = preds[preds.status == "upcoming"]
    snap = snapshot()
    hist = ROOT / "data" / "kalshi_history.csv"
    snap.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    matched = match_to_games(snap, preds) if len(snap) else pd.DataFrame()
    games = {r["game_id"]: r for r in matched.to_dict(orient="records")}
    try:
        meta = json.loads((ROOT / "site" / "data" / "meta.json").read_text())
        spr = spread_snapshot()
        if len(spr):
            sh = ROOT / "data" / "kalshi_spread_history.csv"
            spr.to_csv(sh, mode="a", header=not sh.exists(), index=False)
        for gid, rec in match_spreads(spr, preds, meta.get("cover_sd", 13.0)).items():
            games.setdefault(gid, {"game_id": gid}).update(rec)
        print(f"{len(spr)} spread markets")
    except Exception as e:  # spreads are extra: never let them break the moneyline snapshot
        print("spread snapshot failed:", e)
    payload = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
               "markets_seen": int(len(snap)), "games": list(games.values())}
    (ROOT / "site" / "data" / "kalshi_latest.json").write_text(json.dumps(payload, default=str))
    print(f"{len(snap)} markets, matched {len(matched)} of {len(preds)} games")


if __name__ == "__main__":
    main()
