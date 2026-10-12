"""Walk-forward backtest against sportsbook closing lines.

For every season from 2012 on, all models are trained only on earlier seasons and the
stacking weights only on earlier seasons' out-of-sample predictions. Results are scored
with Brier score and log loss (win probability), RMSE and MAE (spread and total), hit rate
against the closing spread and total, and closing line value on logged Kalshi prices.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import train as T

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FROM = 2007        # 2006 has no prior-form history
FIRST_OOF = 2009         # first season with out-of-sample base predictions (stacker training)
FIRST_TEST = 2012        # first season reported
SPREAD_SD = 13.45
BREAKEVEN_110 = 110 / 210


def american_to_prob(ml):
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml > 0, 100 / (ml + 100), -ml / (-ml + 100))


def market_prob(df: pd.DataFrame) -> pd.Series:
    """No-vig closing home win probability from moneylines, else from the spread."""
    ph, pa = american_to_prob(df.home_moneyline), american_to_prob(df.away_moneyline)
    novig = ph / (ph + pa)
    return pd.Series(np.where(np.isfinite(novig), novig, norm.cdf(df.spread_line / SPREAD_SD)), index=df.index)


def payout_per_dollar(ml):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml > 0, ml / 100, 100 / -ml)


# ---------------------------------------------------------------- walk-forward
def walk_forward(f: pd.DataFrame, current_season: int):
    done = f[f.home_win.notna() & (f.season >= TRAIN_FROM)].copy()
    scored = f[f.home_score.notna() & (f.season >= TRAIN_FROM)].copy()
    oof_parts, hfa_by_season = [], {}
    for s in range(FIRST_OOF, current_season + 1):
        test = done[done.season == s]
        if test.empty:
            continue
        models = T.fit_bases(done[done.season < s])
        oof_parts.append(pd.concat([test, T.predict_bases(models, test)], axis=1))
    oof = pd.concat(oof_parts)

    parts = []
    for s in range(FIRST_TEST, current_season + 1):
        te = oof[oof.season == s].copy()
        if te.empty:
            continue
        stack = T.fit_stacker(oof[oof.season < s])
        hfa_by_season[s] = stack[2]
        te = T.finalize(te, stack)
        tm = T.fit_totals(scored[scored.season < s])
        te["pred_total"] = T.predict_totals(tm, te)
        parts.append(T.add_scores(te))
    bt = pd.concat(parts)
    bt["p_market"] = market_prob(bt)
    return bt, oof, hfa_by_season


# ---------------------------------------------------------------- scoring
def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def summarize(bt: pd.DataFrame, stack, hfa_by_season: dict, edge: float = 0.03) -> dict:
    y = bt.home_win.to_numpy()

    def score(c):
        p = bt[c].to_numpy()
        return {"brier": float(np.mean((p - y) ** 2)), "logloss": T.logloss(p, y),
                "accuracy": float(np.mean((p > 0.5) == y))}

    overall = {"model": score("p_model"), "vegas": score("p_market")}
    members = [{"key": b, "name": T.BASE_LABELS[b], "weight": float(w), **score(f"b_{b}")}
               for b, w in zip(T.BASES, stack[0])]

    # Spread vs closing line
    sp = bt[bt.spread_line.notna()]
    ats = sp[sp.result != sp.spread_line]
    conf = np.maximum(ats.p_home_cover, 1 - ats.p_home_cover)
    won = np.where(ats.p_home_cover > 0.5, ats.result > ats.spread_line, ats.result < ats.spread_line)
    spread = {"model_rmse": _rmse(sp.result, sp.pred_margin), "vegas_rmse": _rmse(sp.result, sp.spread_line),
              "model_mae": float(np.mean(np.abs(sp.result - sp.pred_margin))),
              "vegas_mae": float(np.mean(np.abs(sp.result - sp.spread_line))),
              "breakeven": BREAKEVEN_110,
              "tiers": [{"min_confidence": t, "picks": int((conf > t).sum()),
                         "win_rate": float(won[conf > t].mean()) if (conf > t).sum() else 0.0}
                        for t in (0.5, 0.55, 0.58)]}

    # Totals vs closing line
    tot = bt.home_score + bt.away_score
    tl = bt[bt.total_line.notna()]
    tt = tl.home_score + tl.away_score
    ou = tl[tt != tl.total_line]
    ou_won = np.where(ou.pred_total > ou.total_line, (ou.home_score + ou.away_score) > ou.total_line,
                      (ou.home_score + ou.away_score) < ou.total_line)
    totals = {"model_rmse": _rmse(tot, bt.pred_total), "vegas_rmse": _rmse(tt, tl.total_line),
              "model_mae": float(np.mean(np.abs(tot - bt.pred_total))),
              "vegas_mae": float(np.mean(np.abs(tt - tl.total_line))),
              "team_pts_mae": float(np.mean(np.abs(np.r_[bt.home_score - bt.pred_home_pts,
                                                           bt.away_score - bt.pred_away_pts]))),
              "ou_win_rate": float(ou_won.mean()), "ou_picks": int(len(ou))}

    by_season = []
    for s, d in bt.groupby("season"):
        yy = d.home_win.to_numpy()
        by_season.append({"season": int(s), "games": int(len(d)),
                          "model_brier": float(np.mean((d.p_model - yy) ** 2)),
                          "vegas_brier": float(np.mean((d.p_market - yy) ** 2)),
                          "model_acc": float(np.mean((d.p_model > 0.5) == yy)),
                          "vegas_acc": float(np.mean((d.p_market > 0.5) == yy)),
                          "hfa": hfa_by_season.get(int(s))})

    bins = np.linspace(0, 1, 11)
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (bt.p_model >= lo) & (bt.p_model < hi)
        if m.sum() >= 20:
            cal.append({"bin": f"{lo:.1f}-{hi:.1f}", "predicted": float(bt.p_model[m].mean()),
                        "actual": float(bt.home_win[m].mean()), "n": int(m.sum())})

    d = bt[bt.home_moneyline.notna() & bt.away_moneyline.notna()]
    bet_home = d.p_model - d.p_market > edge
    bet_away = d.p_market - d.p_model > edge
    pnl = np.where(bet_home, np.where(d.home_win == 1, payout_per_dollar(d.home_moneyline), -1.0), 0.0)
    pnl += np.where(bet_away, np.where(d.home_win == 0, payout_per_dollar(d.away_moneyline), -1.0), 0.0)
    n_bets = int(bet_home.sum() + bet_away.sum())
    betting = {"edge_threshold": edge, "bets": n_bets, "profit_units": float(pnl.sum()),
               "roi": float(pnl.sum() / n_bets) if n_bets else 0.0,
               "seasons": f"{int(d.season.min())}-{int(d.season.max())}"}

    return {"overall": overall, "members": members, "ats": spread, "totals": totals,
            "by_season": by_season, "calibration": cal, "betting": betting,
            "hfa": {"by_season": {int(k): v for k, v in hfa_by_season.items()},
                    "current": float(stack[2]), "range": [float(T.HFA_GRID[0]), float(T.HFA_GRID[-1])]},
            "clv": kalshi_clv(), "test_games": int(len(bt))}


# ---------------------------------------------------------------- closing line value
def kalshi_clv() -> dict:
    """Closing line value on Kalshi: for each game, take the side the model favored at the first
    logged pre-kickoff price, then measure how far the price moved toward that side by the last
    pre-kickoff snapshot. Positive means the model was ahead of the market."""
    path = ROOT / "data" / "clv_log.csv"
    if not path.exists():
        return {"games": 0}
    log = pd.read_csv(path).dropna(subset=["kalshi_home_mid"])
    rows = []
    for gid, d in log.groupby("game_id"):
        if len(d) < 2:
            continue
        first, last = d.iloc[0], d.iloc[-1]
        side = 1 if first.p_model > first.kalshi_home_mid else -1
        rows.append(side * (last.kalshi_home_mid - first.kalshi_home_mid))
    if not rows:
        return {"games": 0}
    return {"games": len(rows), "avg_cents": float(np.mean(rows) * 100),
            "beat_close_rate": float(np.mean(np.array(rows) > 0))}
