"""Full pipeline: data -> features -> walk-forward backtest -> this week's predictions -> site data."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import train as T
from backtest import market_prob, summarize, walk_forward
from data import build_tables, load_schedule
from features import FEATURES, TOTAL_FEATURES, build_features, set_qb_priors
from injuries import _get, BASE, build_injury_table, injury_notes
from weather import fill_forecasts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "data"


def current_week(f: pd.DataFrame, season: int):
    left = f[(f.season == season) & f.result.isna()]
    return int(left.week.min()) if len(left) else None


def predict_week(f, bt, oof, season):
    """Games in the current week. Unplayed games use models fit on everything played so far;
    finished games keep the prediction made before kickoff (from the walk-forward run)."""
    week = current_week(f, season)
    if week is None:
        return pd.DataFrame(), None, None
    done = f[f.home_win.notna() & (f.season >= 2007)]
    scored = f[f.home_score.notna() & (f.season >= 2007)]
    models = T.fit_bases(done)
    stack = T.fit_stacker(oof)

    wk = f[(f.season == season) & (f.week == week)]
    todo = wk[wk.result.isna()].copy()
    if len(todo):
        todo = pd.concat([todo, T.predict_bases(models, todo)], axis=1)
        todo = T.finalize(todo, stack)
        todo["pred_total"] = T.predict_totals(T.fit_totals(scored), todo)
        todo = T.add_scores(todo)
        todo["p_market"] = market_prob(todo)
        todo["status"] = "upcoming"
    final = bt[(bt.season == season) & (bt.week == week)].copy()
    final["status"] = "final"
    info = {"stack": stack}
    return pd.concat([todo, final], ignore_index=True), week, info


def main():
    refresh = "--refresh" in sys.argv
    sched = fill_forecasts(load_schedule())
    season = int(sched.season.max())
    tg, qg = build_tables(season)
    players = pd.read_parquet(_get("players.parquet", f"{BASE}/players/players.parquet", refresh),
                              columns=["gsis_id", "position", "draft_round", "rookie_season"])
    set_qb_priors(players)
    inj = build_injury_table(sched, refresh=refresh)
    f = build_features(sched, tg, qg, inj)

    bt, oof, hfa_by_season = walk_forward(f, season)
    week_df, week, info = predict_week(f, bt, oof, season)
    stack = info["stack"] if info else T.fit_stacker(oof)
    summary = summarize(bt, stack, hfa_by_season)

    cols = ["game_id", "season", "week", "game_type", "gameday", "gametime", "away_team", "home_team",
            "home_moneyline", "away_moneyline", "spread_line", "total_line", "p_model", "p_market",
            "status", "home_score", "away_score", "home_win", "pred_margin", "p_home_cover",
            "pred_total", "pred_home_pts", "pred_away_pts", "temp", "wind", "weather_source", "roof",
            "home_qb_name", "away_qb_name", "d_inj_off", "d_inj_def",
            "home_inj_off_total", "away_inj_off_total", "home_inj_def_total", "away_inj_def_total"] + [f"b_{b}" for b in T.BASES]
    preds = week_df[cols].copy() if len(week_df) else pd.DataFrame(columns=cols)
    preds["gameday"] = preds.gameday.astype(str)

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "season": season,
            "week": week, "features": FEATURES, "total_features": TOTAL_FEATURES,
            "hfa_points": stack[2], "sigma": T.SIGMA, "cover_sd": T.COVER_SD}
    (OUT / "predictions.json").write_text(preds.to_json(orient="records"))
    notes = injury_notes(sched, season, week) if week else {}
    (OUT / "injuries.json").write_text(json.dumps(notes))
    (OUT / "backtest.json").write_text(json.dumps(summary, indent=1))
    (OUT / "meta.json").write_text(json.dumps(meta))
    bt.to_csv(ROOT / "data" / "backtest_games.csv", index=False)

    o = summary["overall"]
    print(f"win prob  Brier {o['model']['brier']:.4f} (Vegas {o['vegas']['brier']:.4f})  "
          f"log loss {o['model']['logloss']:.4f} (Vegas {o['vegas']['logloss']:.4f})  "
          f"acc {o['model']['accuracy']:.3f}")
    a, t = summary["ats"], summary["totals"]
    print(f"spread    RMSE {a['model_rmse']:.2f} (Vegas {a['vegas_rmse']:.2f})  ATS {a['tiers'][0]['win_rate']:.3f}")
    print(f"totals    RMSE {t['model_rmse']:.2f} (Vegas {t['vegas_rmse']:.2f})  O/U {t['ou_win_rate']:.3f}")
    print(f"HFA       {stack[2]:.1f} pts (by season {summary['hfa']['by_season']})")
    print("weights  ", {m["key"]: round(m["weight"], 2) for m in summary["members"]})
    if len(preds):
        print(preds[["game_id", "status", "p_model", "p_market", "pred_away_pts", "pred_home_pts"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
