"""Walk-forward training, backtest vs Vegas, and predictions for upcoming games."""
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data"
FIRST_TEST = 2012       # earlier seasons are training-only
FIRST_OOF = 2009        # out-of-sample base predictions start here (to train the stacker)
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


BASES = ["logit", "gbm", "margin", "trees", "mlp", "elo"]
BASE_LABELS = {"logit": "Logistic regression", "gbm": "LightGBM", "margin": "Margin regression",
               "trees": "Extra trees", "mlp": "Neural net", "elo": "Elo"}


def base_preds(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Six different models, each giving a home-win probability."""
    X, y, Xt = train[FEATURES], train.home_win, test[FEATURES]
    out = {}
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)).fit(X, y)
    out["logit"] = m.predict_proba(Xt)[:, 1]
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=60,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                           verbose=-1).fit(X, y)
    out["gbm"] = m.predict_proba(Xt)[:, 1]
    # Predict the point margin, then turn it into a win probability.
    r = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(X, train.result)
    sd = float((train.result - r.predict(X)).std())
    out["margin"] = norm.cdf(r.predict(Xt) / sd)
    m = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=40, max_features=0.5,
                             n_jobs=-1, random_state=0).fit(X, y)
    out["trees"] = m.predict_proba(Xt)[:, 1]
    m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=500,
                                                      early_stopping=True, random_state=0)).fit(X, y)
    out["mlp"] = m.predict_proba(Xt)[:, 1]
    out["elo"] = test.p_elo.to_numpy()
    return out


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_stacker(oof: pd.DataFrame):
    """Non-negative weights on the models (summing to 1) plus a calibration scale,
    chosen to minimize log loss on out-of-sample predictions."""
    Z = np.column_stack([_logit(oof[f"b_{b}"]) for b in BASES])
    y = oof.home_win.to_numpy()

    def loss(th):
        w = np.exp(th[:-1]); w /= w.sum()
        return logloss(1 / (1 + np.exp(-th[-1] * (Z @ w))), y)

    th = minimize(loss, np.r_[np.zeros(len(BASES)), 1.0], method="L-BFGS-B").x
    w = np.exp(th[:-1]); w /= w.sum()
    return w, float(th[-1])


def apply_stacker(df: pd.DataFrame, w, scale) -> np.ndarray:
    Z = np.column_stack([_logit(df[f"b_{b}"]) for b in BASES])
    return 1 / (1 + np.exp(-scale * (Z @ w)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def payout_per_dollar(ml):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml > 0, ml / 100, 100 / -ml)


def backtest(f: pd.DataFrame, current_season: int):
    """Walk-forward: each season's base models train only on earlier seasons, and each
    season's ensemble weights come only from earlier seasons' out-of-sample predictions."""
    done = f[f.home_win.notna() & (f.season >= TRAIN_FROM)].copy()
    rows = []
    for s in range(FIRST_OOF, current_season + 1):
        test = done[done.season == s].copy()
        if test.empty:
            continue
        for k, v in base_preds(done[done.season < s], test).items():
            test[f"b_{k}"] = v
        rows.append(test)
    oof = pd.concat(rows)
    parts = []
    for s in range(FIRST_TEST, current_season + 1):
        te = oof[oof.season == s].copy()
        if te.empty:
            continue
        w, a = fit_stacker(oof[oof.season < s])
        te["p_model"] = apply_stacker(te, w, a)
        parts.append(te)
    bt = pd.concat(parts)
    bt["p_market"] = market_prob(bt)
    weights, scale = fit_stacker(oof)  # used for this week's games
    return bt, (weights, scale)


def summarize(bt: pd.DataFrame, stack, edge: float = 0.03) -> dict:
    y = bt.home_win.to_numpy()
    score = lambda c: {"brier": brier(bt[c].to_numpy(), y), "logloss": logloss(bt[c].to_numpy(), y),
                       "accuracy": float(np.mean((bt[c] > 0.5) == y))}
    overall = {"model": score("p_model"), "vegas": score("p_market")}
    members = [{"key": b, "name": BASE_LABELS[b], "weight": float(w), **score(f"b_{b}")}
               for b, w in zip(BASES, stack[0])]
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
    return {"overall": overall, "members": members, "stack_scale": stack[1], "by_season": by_season, "calibration": cal, "betting": betting,
            "test_games": int(len(bt))}


def current_week(f: pd.DataFrame, season: int):
    """The NFL week in progress: the earliest week that still has an unplayed game."""
    left = f[(f.season == season) & f.result.isna()]
    return int(left.week.min()) if len(left) else None


def week_games(f: pd.DataFrame, bt: pd.DataFrame, stack, season: int) -> tuple[pd.DataFrame, int | None]:
    """All games of the current week. Unplayed games get fresh predictions trained on
    everything played so far. Finished games keep the prediction made before kickoff."""
    week = current_week(f, season)
    if week is None:
        return f.iloc[0:0].copy(), None
    wk = f[(f.season == season) & (f.week == week)].copy()
    todo = wk[wk.result.isna()].copy()
    done = f[f.home_win.notna() & (f.season >= TRAIN_FROM)]
    if len(todo):
        for k, v in base_preds(done, todo).items():
            todo[f"b_{k}"] = v
        todo["p_model"] = apply_stacker(todo, *stack)
        todo["p_market"] = market_prob(todo)
        todo["status"] = "upcoming"
    final = bt[(bt.season == season) & (bt.week == week)].copy()
    final["status"] = "final"
    return pd.concat([todo, final], ignore_index=True), week


def run(f: pd.DataFrame, current_season: int):
    f = f.copy()
    f["p_elo"] = 1 / (1 + 10 ** (-f.elo_diff / 400))
    bt, stack = backtest(f, current_season)
    summary = summarize(bt, stack)
    fut, week = week_games(f, bt, stack, current_season)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["game_id", "season", "week", "game_type", "gameday", "gametime", "away_team", "home_team",
            "home_moneyline", "away_moneyline", "spread_line", "p_model", "p_market",
            "status", "home_score", "away_score", "home_win"] + [f"b_{b}" for b in BASES]
    preds = fut[cols].copy() if len(fut) else pd.DataFrame(columns=cols)
    preds["gameday"] = preds.gameday.astype(str)
    meta = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "season": current_season, "week": week, "features": FEATURES}
    (OUT / "predictions.json").write_text(preds.to_json(orient="records"))
    (OUT / "season_results.json").unlink(missing_ok=True)
    (OUT / "backtest.json").write_text(json.dumps(summary, indent=1))
    (OUT / "meta.json").write_text(json.dumps(meta))
    bt.to_csv(ROOT / "data" / "backtest_games.csv", index=False)
    return summary, preds
