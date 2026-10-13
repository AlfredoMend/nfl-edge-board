"""Walk-forward check of the prop medians on a past season.

For every regular-season game from week 3 on, players are projected using only earlier games,
that week's injury report and the listed starting QBs, with the game model's out-of-sample
margin and total. A well-calibrated median should land above the actual result about half the
time. The naive baseline is each player's average over his last four games.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import props as P
from data import load_schedule
from players import build_player_tables, player_info

ROOT = Path(__file__).resolve().parents[1]
MARKETS = {  # market: (stat key in sims, stat column in box score, who)
    "pass_yds": ("pass_yds", "pass_yds", "QB"), "pass_att": ("pass_att", "pass_att", "QB"),
    "comp": ("comp", "comp", "QB"), "pass_td": ("pass_td", "pass_td", "QB"),
    "rush_yds": ("rush_yds", "rush_yds", "RUSH"), "rush_att": ("carries", "carries", "RUSH"),
    "rec": ("rec", "rec", "REC"), "rec_yds": ("rec_yds", "rec_yds", "REC"),
}


def run(season: int, first_week: int = 3, last_week: int = 18, n_sims: int = 4000):
    sched = load_schedule()
    pg, tm = build_player_tables(int(sched.season.max()))
    pg, tm = P.prepare(pg, tm, sched, player_info())
    bt = pd.read_csv(ROOT / "data" / "backtest_games.csv", low_memory=False).set_index("game_id")
    hist0 = tm[tm.season < season]
    lg = P.league_rates(hist0[hist0.season >= season - 3], pg[(pg.season < season) & (pg.season >= season - 3)])
    vol = P.fit_volume(hist0[hist0.season >= season - P.VOLUME_SEASONS])
    rows = []
    games = sched[(sched.season == season) & (sched.game_type == "REG") & sched.week.between(first_week, last_week)]
    for g in games.itertuples():
        if g.game_id not in bt.index:
            continue
        cut = g.gameday
        pgh, tmh = pg[pg.gameday < cut], tm[tm.gameday < cut]
        inj = P.injury_status(season, g.week)
        home = P.build_side(pgh, tmh, g.home_team, g.away_team, g.home_qb_id, inj, lg)
        away = P.build_side(pgh, tmh, g.away_team, g.home_team, g.away_qb_id, inj, lg)
        b = bt.loc[g.game_id]
        sims = P.raw_samples({"pred_margin": b.pred_margin, "pred_total": b.pred_total}, home, away, vol, lg, n=n_sims)
        actual = pg[pg.game_id == g.game_id].set_index("player_id")
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            plist = (home if side == "home" else away)["players"]
            for p in plist:
                if p["id"] not in actual.index or not actual.loc[p["id"]].appeared:
                    continue   # props are voided when a player does not play
                a = actual.loc[p["id"]]
                past = pgh[(pgh.player_id == p["id"]) & pgh.appeared].tail(4)
                s = sims[side][p["id"]]
                for mk, (sk, col, who) in MARKETS.items():
                    if who == "QB" and p["pos"] != "QB":
                        continue
                    if who == "REC" and (p["pos"] == "QB" or p["tgt"] < 0.05):
                        continue
                    if who == "RUSH" and p["car"] < 0.10 and p["pos"] != "QB":
                        continue
                    if sk not in s:
                        continue
                    x = s[sk]
                    act = float(a[col])
                    pit = float((x < act).mean() + 0.5 * (x == act).mean())
                    rows.append({"pit": pit, "game_id": g.game_id, "week": g.week, "player": p["name"], "pos": p["pos"], "market": mk,
                                 "median": float(np.median(x)), "mean": float(np.mean(x)), "actual": float(a[col]),
                                 "naive": float(past[col].mean()) if len(past) else np.nan,
                                 "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75))})
                td = (s["rec_td"] + s["rush_td"]) > 0
                rows.append({"game_id": g.game_id, "week": g.week, "player": p["name"], "pos": p["pos"], "market": "anytime_td",
                             "median": float(td.mean()), "mean": float(td.mean()),
                             "actual": float((a.rec_td + a.rush_td) > 0), "naive": np.nan, "p25": np.nan, "p75": np.nan})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> dict:
    out = {}
    for mk, d in df[df.market != "anytime_td"].groupby("market"):
        # Book lines sit on a half point, so test the line just above and below the median
        over_hi = float((d.actual > np.floor(d["median"]) + 0.5).mean())
        under_lo = float((d.actual < np.ceil(d["median"]) - 0.5).mean())
        out[mk] = {"n": int(len(d)),
                   "over_median": float((d.actual > d["median"]).mean()),
                   "under_median": float((d.actual < d["median"]).mean()),
                   "over_mean": float((d.actual > d["mean"]).mean()),
                   "in_25_75": float(((d.actual >= d.p25) & (d.actual <= d.p75)).mean()),
                   # PIT: where the actual result falls in the simulated distribution (0.5 = centered)
                   "pit_mean": float(d.pit.mean()), "pit_mid_half": float(d.pit.between(0.25, 0.75).mean()),
                   "mae_model": float((d.actual - d["median"]).abs().mean()),
                   "mae_naive": float((d.actual - d.naive).abs().mean()) if d.naive.notna().any() else None}
    # by projection tier: are starters (top tiers) centered too?
    out["tiers"] = {}
    for mk in ("rec", "rec_yds", "rush_att", "rush_yds"):
        d = df[df.market == mk].copy()
        if len(d) < 50:
            continue
        d["tier"] = pd.qcut(d["median"].rank(method="first"), 5, labels=["lowest", "low", "mid", "high", "top"])
        out["tiers"][mk] = {str(k): {"over": round(float((v.actual > v["median"]).mean()), 3),
                                     "under": round(float((v.actual < v["median"]).mean()), 3),
                                     "pit": round(float(v.pit.mean()), 3)} for k, v in d.groupby("tier", observed=True)}
    t = df[df.market == "anytime_td"]
    bins = pd.cut(t["median"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    out["anytime_td"] = {"n": int(len(t)), "brier": float(((t["median"] - t.actual) ** 2).mean()),
                         "base_brier": float(((t.actual.mean() - t.actual) ** 2).mean()),
                         "calibration": [{"bin": str(k), "pred": round(float(v["median"].mean()), 3),
                                          "actual": round(float(v.actual.mean()), 3), "n": int(len(v))}
                                         for k, v in t.groupby(bins, observed=True)]}
    return out


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    for kv in sys.argv[3:]:            # e.g. SHARE_GAMMA=1.3
        k, v = kv.split("=")
        setattr(P, k, float(v))
    df = run(season, n_sims=int(P.__dict__.get("BT_SIMS", 4000)))
    df.to_csv(ROOT / "data" / f"props_backtest_{season}{tag}.csv", index=False)
    rep = report(df)
    (ROOT / "data" / f"props_backtest_{season}{tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
