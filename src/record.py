"""Results tracker: log the model's pregame calls every run, then grade them once games are final.

Game picks (straight-up, spread lean, total lean) use the last snapshot taken before kickoff.
Games played before the tracker existed use the walk-forward backtest prediction (a model trained
only on earlier seasons) and are labeled "reconstructed".

Kalshi paper bets: the first time a contract shows at least MIN_EDGE of value after fees before
kickoff (and, for props, in a tight market), one $1 contract is "bought" at that moment's ask.
  P&L per contract = (1 if it wins else 0) - price - fee
  Closing line value = the contract's last pre-kickoff mid price minus the entry price
    (positive means the market moved toward the model after the bet).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, SITE = ROOT / "data", ROOT / "site" / "data"
PICK_LOG = DATA / "pick_log.csv"
PROP_LOG = DATA / "kalshi_props_log.csv"
MIN_EDGE = 0.05
TAKER_FEE = 0.07
RECORD_START = "2026-10-13T00:28"   # first run with the current prop model and visible edges


def fee(p):
    return TAKER_FEE * p * (1 - p)


def kickoff_utc(gameday, gametime) -> pd.Series:
    t = pd.Series(gametime).fillna("13:00").astype(str)
    ts = pd.to_datetime(pd.Series(gameday).astype(str).str[:10] + " " + t.values)
    return ts.dt.tz_localize("America/New_York").dt.tz_convert("UTC")


# ---------------------------------------------------------------- logging
def log_snapshot(now: str):
    """Append the current pregame calls for every game that has not kicked off."""
    preds = pd.read_json(SITE / "predictions.json")
    if preds.empty:
        return
    up = preds[preds.status == "upcoming"].copy()
    up["kick"] = kickoff_utc(up.gameday, up.gametime).set_axis(up.index)
    up = up[up.kick > pd.Timestamp(now)]
    if up.empty:
        return
    kal = {}
    kp = SITE / "kalshi_latest.json"
    if kp.exists():
        kal = {g["game_id"]: g for g in json.loads(kp.read_text()).get("games", [])}
    rows = []
    for g in up.itertuples():
        k = kal.get(g.game_id, {})
        b = k.get("best") or {}
        rows.append({"snapshot_utc": now, "game_id": g.game_id, "p_model": g.p_model, "pred_margin": g.pred_margin,
                     "pred_total": g.pred_total, "p_home_cover": g.p_home_cover, "spread_line": g.spread_line,
                     "total_line": g.total_line, "p_market": g.p_market,
                     "kalshi_home_mid": k.get("kalshi_home_mid"), "kalshi_home_ask": k.get("kalshi_home_ask"),
                     "kalshi_away_ask": k.get("kalshi_away_ask"), "ev_home": k.get("ev_home"), "ev_away": k.get("ev_away"),
                     "sp_label": b.get("label"), "sp_side": b.get("side"), "sp_team": b.get("team"),
                     "sp_line": b.get("line"), "sp_price": b.get("price"), "sp_ev": b.get("ev")})
    pd.DataFrame(rows).to_csv(PICK_LOG, mode="a", header=not PICK_LOG.exists(), index=False)


# ---------------------------------------------------------------- grading helpers
def _wl(series):
    s = pd.Series(series).dropna()
    w, l = int((s == 1).sum()), int((s == 0).sum())
    p = int((s == 0.5).sum())
    return {"w": w, "l": l, "p": p, "pct": round(w / (w + l), 4) if w + l else None}


def grade_games(sched: pd.DataFrame, now: str) -> pd.DataFrame:
    season = int(sched.season.max())
    final = sched[(sched.season == season) & sched.result.notna()].copy()
    final["kick"] = kickoff_utc(final.gameday, final.gametime).set_axis(final.index)
    rows = []
    logged = pd.read_csv(PICK_LOG) if PICK_LOG.exists() else pd.DataFrame(columns=["game_id"])
    if len(logged):
        logged["snap"] = pd.to_datetime(logged.snapshot_utc, utc=True)
    bt = pd.read_csv(DATA / "backtest_games.csv", low_memory=False)
    bt = bt[bt.season == season].set_index("game_id")
    for g in final.itertuples():
        src, row = None, None
        if len(logged):
            L = logged[(logged.game_id == g.game_id) & (logged.snap < g.kick)]
            if len(L):
                row, src = L.iloc[-1], "live"
        if row is None and g.game_id in bt.index:
            row, src = bt.loc[g.game_id], "reconstructed"
        if row is None:
            continue
        margin, total = g.home_score - g.away_score, g.home_score + g.away_score
        su = 1.0 if (row.p_model > 0.5) == (margin > 0) else 0.0
        if margin == 0:
            su = 0.5
        ats = None
        if pd.notna(row.spread_line) and pd.notna(row.p_home_cover):
            if margin == row.spread_line:
                ats = 0.5
            else:
                ats = 1.0 if (row.p_home_cover > 0.5) == (margin > row.spread_line) else 0.0
        ou = None
        if pd.notna(row.total_line) and pd.notna(row.pred_total):
            if total == row.total_line:
                ou = 0.5
            else:
                ou = 1.0 if (row.pred_total > row.total_line) == (total > row.total_line) else 0.0
        pick = g.home_team if row.p_model > 0.5 else g.away_team
        rows.append({"game_id": g.game_id, "week": int(g.week), "source": src, "away": g.away_team, "home": g.home_team,
                     "score": f"{g.away_team} {int(g.away_score)}, {g.home_team} {int(g.home_score)}",
                     "pick": pick, "p_pick": float(max(row.p_model, 1 - row.p_model)), "su": su, "ats": ats, "ou": ou})
    return pd.DataFrame(rows)


def grade_kalshi_games(sched: pd.DataFrame) -> list:
    """Moneyline and spread paper bets from the pick log."""
    if not PICK_LOG.exists():
        return []
    L = pd.read_csv(PICK_LOG)
    L = L[L.snapshot_utc >= RECORD_START]
    if L.empty:
        return []
    L["snap"] = pd.to_datetime(L.snapshot_utc, utc=True)
    g = sched.set_index("game_id")
    clv = pd.read_csv(DATA / "clv_log.csv") if (DATA / "clv_log.csv").exists() else pd.DataFrame()
    bets = []
    for gid, d in L.groupby("game_id"):
        if gid not in g.index:
            continue
        info = g.loc[gid]
        kick = kickoff_utc([info.gameday], [info.gametime]).iloc[0]
        d = d[d.snap < kick].sort_values("snap")
        if d.empty:
            continue
        done = pd.notna(info.result)
        margin = info.home_score - info.away_score if done else None
        # moneyline
        evs = d[["ev_home", "ev_away"]].max(axis=1)
        hit = d[evs >= MIN_EDGE]
        if len(hit):
            r = hit.iloc[0]
            home = r.ev_home >= r.ev_away
            price = r.kalshi_home_ask if home else r.kalshi_away_ask
            team = info.home_team if home else info.away_team
            close = None
            started = pd.Timestamp.now(tz="UTC") >= kick
            c = clv[(clv.game_id == gid)] if (len(clv) and started) else pd.DataFrame()
            if len(c):
                c = c[pd.to_datetime(c.snapshot_utc, utc=True) < kick]
                if len(c):
                    m = float(c.iloc[-1].kalshi_home_mid)
                    close = m if home else 1 - m
            won = None if not done or margin == 0 else ((margin > 0) == home)
            bets.append(_bet("moneyline", gid, info, f"{team} to win", "YES", price, r.ev_home if home else r.ev_away,
                             r.snapshot_utc, won, close))
        # spread
        hit = d[d.sp_ev >= MIN_EDGE]
        if len(hit):
            r = hit.iloc[0]
            team_margin = None
            if done:
                team_margin = margin if r.sp_team == info.home_team else -margin
            won = None if team_margin is None else ((team_margin > r.sp_line) == (r.sp_side == "yes"))
            bets.append(_bet("spread", gid, info, r.sp_label, str(r.sp_side).upper(), r.sp_price, r.sp_ev,
                             r.snapshot_utc, won, None))
    return bets


def grade_props(sched: pd.DataFrame) -> list:
    if not PROP_LOG.exists():
        return []
    L = pd.read_csv(PROP_LOG)
    L = L[(L.snapshot_utc >= RECORD_START) & (L.best_ev >= MIN_EDGE) & (~L.wide.astype(bool))]
    if L.empty:
        return []
    full = pd.read_csv(PROP_LOG)
    full["snap"] = pd.to_datetime(full.snapshot_utc, utc=True)
    pg = pd.read_csv(DATA / "player_games.csv")
    g = sched.set_index("game_id")
    stat_col = {"pass_yds": "pass_yds", "pass_att": "pass_att", "comp": "comp", "pass_td": "pass_td", "ints": "ints",
                "rush_yds": "rush_yds", "rush_att": "carries", "rec": "rec", "rec_yds": "rec_yds"}
    bets = []
    for ticker, d in L.groupby("ticker"):
        r = d.sort_values("snapshot_utc").iloc[0]
        if r.game_id not in g.index:
            continue
        info = g.loc[r.game_id]
        kick = kickoff_utc([info.gameday], [info.gametime]).iloc[0]
        if pd.Timestamp(r.snapshot_utc) >= kick:
            continue
        side = r.best_side
        price = r.yes_ask if side == "YES" else r.no_ask
        hist = full[(full.ticker == ticker) & (full.snap < kick)]
        close = None
        if len(hist) and pd.Timestamp.now(tz="UTC") >= kick:
            h = hist.iloc[-1]
            if pd.notna(h.yes_bid) and pd.notna(h.yes_ask):
                mid = (h.yes_bid + h.yes_ask) / 2
                close = mid if side == "YES" else 1 - mid
        won = None
        if pd.notna(info.result):
            a = pg[(pg.game_id == r.game_id) & (pg.player_id == r.player_id)]
            if len(a):
                a = a.iloc[0]
                if r.stat == "tds":
                    val = a.rec_td + a.rush_td
                elif r.stat == "rush_rec_yds":
                    val = a.rush_yds + a.rec_yds
                else:
                    val = a[stat_col[r.stat]]
                yes = val >= r.line
                won = bool(yes) if side == "YES" else bool(not yes)
            else:
                won = "void"
        b = _bet("prop", r.game_id, info, r.contract, side, price, r.best_ev, r.snapshot_utc,
                 None if won == "void" else won, close)
        if won == "void":
            b["status"] = "void"
        bets.append(b)
    return bets


def _bet(kind, gid, info, label, side, price, ev, when, won, close):
    price = float(price) if price is not None and pd.notna(price) else None
    pnl = None
    status = "open"
    if won is not None and price is not None:
        pnl = (1.0 if won else 0.0) - price - fee(price)
        status = "won" if won else "lost"
    return {"type": kind, "game_id": gid, "game": f"{info.away_team} at {info.home_team}", "week": int(info.week),
            "contract": label, "side": side, "price": price, "ev": None if ev is None or pd.isna(ev) else round(float(ev), 4),
            "flagged_utc": when, "status": status, "pnl": None if pnl is None else round(pnl, 4),
            "clv": None if close is None or price is None else round(close - price, 4)}


def summarize_bets(bets: list) -> dict:
    out = {}
    for kind in ("all", "moneyline", "spread", "prop"):
        b = [x for x in bets if kind == "all" or x["type"] == kind]
        settled = [x for x in b if x["status"] in ("won", "lost")]
        clvs = [x["clv"] for x in b if x["clv"] is not None]
        staked = sum(x["price"] for x in settled)
        pnl = sum(x["pnl"] for x in settled)
        out[kind] = {"bets": len(b), "settled": len(settled), "open": sum(1 for x in b if x["status"] == "open"),
                     "won": sum(1 for x in settled if x["status"] == "won"),
                     "pnl": round(pnl, 2), "roi": round(pnl / staked, 4) if staked else None,
                     "avg_clv": round(float(np.mean(clvs)), 4) if clvs else None,
                     "beat_close": round(float(np.mean([c > 0 for c in clvs])), 4) if clvs else None}
    return out


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data import load_schedule
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    log_snapshot(now)
    sched = load_schedule()
    games = grade_games(sched, now)
    bets = grade_kalshi_games(sched) + grade_props(sched)
    out = {"updated": now, "record_start": RECORD_START, "min_edge": MIN_EDGE, "season": int(sched.season.max())}
    if len(games):
        for src in ("all", "live", "reconstructed"):
            d = games if src == "all" else games[games.source == src]
            out[f"games_{src}"] = {"n": int(len(d)), "su": _wl(d.su), "ats": _wl(d.ats), "ou": _wl(d.ou)}
        out["by_week"] = [{"week": int(w), "source": d.source.iloc[0], "su": _wl(d.su), "ats": _wl(d.ats), "ou": _wl(d.ou)}
                          for w, d in games.groupby("week")]
        out["recent_games"] = games.sort_values(["week", "game_id"], ascending=False).head(16).to_dict(orient="records")
    out["kalshi"] = summarize_bets(bets)
    out["bets"] = sorted(bets, key=lambda x: x["flagged_utc"], reverse=True)[:60]
    (SITE / "record.json").write_text(json.dumps(out, default=str))
    k = out["kalshi"]["all"]
    print(f"record: {len(games)} games graded, {k['bets']} Kalshi paper bets ({k['settled']} settled, P&L {k['pnl']})")


if __name__ == "__main__":
    main()
