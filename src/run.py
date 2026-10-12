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


TEAM_STATS = [  # (group, label, column suffix, higher is better, format)
    ("Offense", "Opponent-adjusted EPA per play", "adj_off_epa", True, "epa"),
    ("Offense", "EPA per dropback", "f_off_pass_epa", True, "epa"),
    ("Offense", "EPA per rush", "f_off_rush_epa", True, "epa"),
    ("Offense", "Success rate", "f_off_sr", True, "pct"),
    ("Offense", "Early-down EPA", "f_off_early_epa", True, "epa"),
    ("Offense", "Explosive play rate", "f_off_explosive", True, "pct"),
    ("Offense", "CPOE", "f_off_cpoe", True, "num"),
    ("Offense", "Avg depth of target", "f_off_adot", True, "num"),
    ("Offense", "Pressure rate allowed", "f_off_press_rate", False, "pct"),
    ("Offense", "Turnover-worthy pass rate", "f_off_twp_rate", False, "pct"),
    ("Offense", "Red-zone TD rate", "f_rz_off", True, "pct"),
    ("Offense", "Plays per game", "f_plays_total", True, "num"),
    ("Defense", "Opponent-adjusted EPA allowed", "adj_def_epa", False, "epa"),
    ("Defense", "EPA per dropback allowed", "f_def_pass_epa", False, "epa"),
    ("Defense", "EPA per rush allowed", "f_def_rush_epa", False, "epa"),
    ("Defense", "Success rate allowed", "f_def_sr", False, "pct"),
    ("Defense", "Explosive rate allowed", "f_def_explosive", False, "pct"),
    ("Defense", "Pressure rate", "f_def_press_rate", True, "pct"),
    ("Defense", "Turnover-worthy passes forced", "f_def_twp_rate", True, "pct"),
    ("Defense", "Red-zone TD rate allowed", "f_rz_def", False, "pct"),
    ("Overall", "Elo rating", "elo", True, "int"),
    ("Overall", "Starting QB EPA per dropback", "qb_rating", True, "epa"),
]


def team_stats(f: pd.DataFrame, season: int) -> dict:
    """Pregame ratings for every team at its next game, with league ranks (1 = best)."""
    nxt = f[(f.season == season) & f.result.isna()].sort_values(["gameday", "game_id"])
    rows = {}
    for side in ("home", "away"):
        for r in nxt.itertuples():
            team = getattr(r, f"{side}_team")
            if team in rows:
                continue
            rows[team] = {s[2]: float(getattr(r, f"{side}_{s[2]}")) for s in TEAM_STATS}
    df = pd.DataFrame(rows).T
    out = {"stats": [{"group": g, "label": l, "key": k, "higher_better": hb, "fmt": fm} for g, l, k, hb, fm in TEAM_STATS],
           "teams": {}}
    for team in df.index:
        out["teams"][team] = {}
    for g, l, k, hb, fm in TEAM_STATS:
        rank = df[k].rank(ascending=not hb, method="min")
        for team in df.index:
            out["teams"][team][k] = {"v": round(float(df.at[team, k]), 4), "rank": int(rank[team])}
    out["n_teams"] = int(len(df))
    return out


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
            "hfa_points": stack[2], "sigma": T.SIGMA, "cover_sd": T.COVER_SD,
            "weather_forecasts": int((sched.weather_source == "forecast").sum())}
    (OUT / "predictions.json").write_text(preds.to_json(orient="records"))
    (OUT / "teamstats.json").write_text(json.dumps(team_stats(f, season)))
    upcoming = preds[preds.status == "upcoming"]
    if len(upcoming):
        try:
            from props import project_week
            pr = project_week(sched, upcoming, refresh=refresh)
            (OUT / "props.json").write_text(json.dumps(pr))
            print(f"props: simulated {len(pr)} games x {pr[next(iter(pr))]['n_sims']:,} sims")
        except Exception as e:  # props are an add-on; never block the main board
            import traceback
            traceback.print_exc()
            print("props failed:", e)
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
