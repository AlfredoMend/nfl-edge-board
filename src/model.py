"""Walk-forward training, backtest vs Vegas, and predictions for upcoming games."""
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data"
FIRST_TEST = 2012       # earlier seasons are training-only
TRAIN_FROM = 2007       # 2006 has no prior-form history
SPREAD_SD = 13.45       # NFL margin standard deviation vs the spread


def american_to_prob(ml):
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml > 0, 100 / (ml + 100), -ml / (-ml + 100))


def market_prob(df: pd.DataFrame) -> pd.Series:
    """No-vig home win probability from closing moneylines, else from the spread."""
    ph, pa = american_to_prob(df.home_moneyline), american_to_prob(df.away_moneyline)
    novig = ph / (ph + pa)
    from_spread = norm.cdf(df.spread_line / SPREAD_SD)
    return pd.Series(np.where(np.isfinite(novig), novig, from_spread), index=df.index)


def make_models():
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7,
                             min_child_samples=60, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    return logit, gbm


def fit_predict(train, test):
    logit, gbm = make_models()
    X, y = train[FEATURES], train.home_win
    logit.fit(X, y)
    gbm.fit(X, y)
    p_l = logit.predict_proba(test[FEATURES])[:, 1]
    p_g = gbm.predict_proba(test[FEATURES])[:, 1]
    return p_l, p_g, 0.5 * p_l + 0.5 * p_g


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def payout_per_dollar(ml):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml > 0, ml / 100, 100 / -ml)


def backtest(f: pd.DataFrame, current_season: int) -> pd.DataFrame:
    done = f[f.home_win.notna() & (f.season >= TRAIN_FROM)].copy()
    rows = []
    for s in range(FIRST_TEST, current_season + 1):
        train = done[done.season < s]
        test = done[done.season == s].copy()
        if test.empty:
            continue
        test["p_logit"], test["p_gbm"], test["p_model"] = fit_predict(train, test)
        rows.append(test)
    bt = pd.concat(rows)
    bt["p_market"] = market_prob(bt)
    return bt


def summarize(bt: pd.DataFrame, edge: float = 0.03) -> dict:
    y = bt.home_win.to_numpy()
    overall = {k: {"brier": brier(bt[c].to_numpy(), y), "logloss": logloss(bt[c].to_numpy(), y),
                   "accuracy": float(np.mean((bt[c] > 0.5) == y))}
               for k, c in [("model", "p_model"), ("logit", "p_logit"), ("gbm", "p_gbm"),
                            ("vegas", "p_market"), ("elo_only", "p_elo")]}
    by_season = []
    for s, d in bt.groupby("season"):
        yy = d.home_win.to_numpy()
        by_season.append({"season": int(s), "games": int(len(d)),
                          "model_brier": brier(d.p_model.to_numpy(), yy),
                          "vegas_brier": brier(d.p_market.to_numpy(), yy),
                          "model_acc": float(np.mean((d.p_model > 0.5) == yy)),
                          "vegas_acc": float(np.mean((d.p_market > 0.5) == yy))})
    # Calibration: 10 bins
    bins = np.linspace(0, 1, 11)
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (bt.p_model >= lo) & (bt.p_model < hi)
        if m.sum() >= 20:
            cal.append({"bin": f"{lo:.1f}-{hi:.1f}", "predicted": float(bt.p_model[m].mean()),
                        "actual": float(bt.home_win[m].mean()), "n": int(m.sum())})
    # Paper betting: bet $1 on a side when model beats no-vig market by `edge`.
    has_ml = bt.home_moneyline.notna() & bt.away_moneyline.notna()
    d = bt[has_ml].copy()
    bet_home = d.p_model - d.p_market > edge
    bet_away = d.p_market - d.p_model > edge
    pnl = np.where(bet_home, np.where(d.home_win == 1, payout_per_dollar(d.home_moneyline), -1.0), 0.0)
    pnl += np.where(bet_away, np.where(d.home_win == 0, payout_per_dollar(d.away_moneyline), -1.0), 0.0)
    n_bets = int(bet_home.sum() + bet_away.sum())
    betting = {"edge_threshold": edge, "bets": n_bets, "profit_units": float(pnl.sum()),
               "roi": float(pnl.sum() / n_bets) if n_bets else 0.0,
               "win_rate": float(((pnl > 0).sum()) / n_bets) if n_bets else 0.0,
               "seasons": f"{int(d.season.min())}-{int(d.season.max())}"}
    return {"overall": overall, "by_season": by_season, "calibration": cal, "betting": betting,
            "test_games": int(len(bt))}


def upcoming(f: pd.DataFrame, days: int = 10) -> pd.DataFrame:
    done = f[f.home_win.notna() & (f.season >= TRAIN_FROM)]
    now = pd.Timestamp.now(tz="America/Los_Angeles").tz_localize(None).normalize()
    fut = f[f.home_win.isna() & (f.gameday >= now) & (f.gameday <= now + pd.Timedelta(days=days))].copy()
    if fut.empty:
        fut = f[f.home_win.isna()].sort_values("gameday").head(16).copy()
    if fut.empty:
        return fut
    fut["p_logit"], fut["p_gbm"], fut["p_model"] = fit_predict(done, fut)
    fut["p_market"] = market_prob(fut)
    return fut


def run(f: pd.DataFrame, current_season: int):
    f = f.copy()
    f["p_elo"] = 1 / (1 + 10 ** (-f.elo_diff / 400))
    bt = backtest(f, current_season)
    summary = summarize(bt)
    fut = upcoming(f)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["game_id", "season", "week", "gameday", "gametime", "away_team", "home_team",
            "home_moneyline", "away_moneyline", "spread_line", "p_model", "p_logit", "p_gbm", "p_market"]
    preds = fut[cols].copy() if len(fut) else pd.DataFrame(columns=cols)
    preds["gameday"] = preds.gameday.astype(str)
    recent = bt[bt.season == current_season][cols + ["home_score", "away_score", "home_win"]].copy()
    recent["gameday"] = recent.gameday.astype(str)
    meta = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "season": current_season, "features": FEATURES}
    (OUT / "predictions.json").write_text(preds.to_json(orient="records"))
    (OUT / "season_results.json").write_text(recent.to_json(orient="records"))
    (OUT / "backtest.json").write_text(json.dumps(summary, indent=1))
    (OUT / "meta.json").write_text(json.dumps(meta))
    bt.to_csv(ROOT / "data" / "backtest_games.csv", index=False)
    return summary, preds
