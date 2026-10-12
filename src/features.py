"""Pregame features. Every value for a game uses only games played before it.

- Situation-neutral efficiency comes from data.py.
- Recent form: exponential moving averages with a 7-game half-life, weighted by how many
  neutral plays each game had, and regressed halfway to league average each offseason.
- Opponent adjustment: weekly ridge regression of EPA/play on offense and defense identity,
  fit only on earlier games, so a team that faced weak defenses is not overrated.
- Red-zone touchdown rate is shrunk toward the prior season's league median.
- Context: rest, travel miles, time-zone shift, home field, wind and cold.
No points-per-game, raw yardage or raw turnover totals are used anywhere.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

from geo import travel

EMA_HALFLIFE = 7          # games
CARRYOVER = 0.5           # share of last season's form kept at a new season
FULL_GAME_PLAYS = 45      # neutral plays that count as a full game in the EMA

ELO_K = 20.0
HFA_POINTS = 1.5          # home field, within the 1.2 to 1.8 point range
ELO_PER_POINT = 25.0
ELO_HFA = HFA_POINTS * ELO_PER_POINT
ELO_REGRESS = 1 / 3

OPP_ADJ_HALFLIFE = 7      # weeks
OPP_ADJ_ALPHA = 150.0     # ridge shrinkage, in weighted plays
RZ_PRIOR_TRIPS = 20       # red-zone trips of prior weight toward the league median

# Per-game stats from data.py that get an EMA (offense and what the defense allowed)
SIDE_STATS = ["epa", "sr", "pass_epa", "rush_epa", "early_epa", "explosive", "cpoe", "adot",
              "yac_tgt", "press_rate", "press_epa", "clean_epa", "twp_rate", "fum_rate", "ffum_rate"]
EMA_COLS = [f"off_{s}" for s in SIDE_STATS] + [f"def_{s}" for s in SIDE_STATS] + ["plays_total"]


# ---------------------------------------------------------------- Elo
def add_elo(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(["gameday", "game_id"]).copy()
    elo, last_season = {}, {}
    home_pre, away_pre = [], []
    for row in g.itertuples():
        for t in (row.home_team, row.away_team):
            r = elo.get(t, 1505.0)
            if last_season.get(t) is not None and last_season[t] != row.season:
                r = r * (1 - ELO_REGRESS) + 1505.0 * ELO_REGRESS
            elo[t] = r
            last_season[t] = row.season
        h, a = elo[row.home_team], elo[row.away_team]
        home_pre.append(h)
        away_pre.append(a)
        if pd.isna(row.result):
            continue
        hfa = 0.0 if row.location == "Neutral" else ELO_HFA
        exp_home = 1 / (1 + 10 ** (-(h + hfa - a) / 400))
        actual = 1.0 if row.result > 0 else 0.0 if row.result < 0 else 0.5
        diff = (h + hfa - a) if row.result > 0 else (a - h - hfa)
        mov = np.log(abs(row.result) + 1) * 2.2 / (diff * 0.001 + 2.2)
        shift = ELO_K * mov * (actual - exp_home)
        elo[row.home_team] = h + shift
        elo[row.away_team] = a - shift
    g["home_elo"], g["away_elo"] = home_pre, away_pre
    return g


# ---------------------------------------------------------------- long table
def long_table(g: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with the opponent's red-zone offense as this team's defense."""
    sides = []
    for me, opp in (("home", "away"), ("away", "home")):
        sides.append(g[["game_id", "season", "week", "gameday", f"{me}_team", f"{opp}_team"]]
                     .set_axis(["game_id", "season", "week", "gameday", "team", "opp"], axis=1))
    long = pd.concat(sides).merge(tg.drop(columns="season"), on=["game_id", "team"], how="left")
    rz_opp = tg[["game_id", "team", "rz_trips", "rz_tds"]].rename(
        columns={"team": "opp", "rz_trips": "rz_trips_allowed", "rz_tds": "rz_tds_allowed"})
    long = long.merge(rz_opp, on=["game_id", "opp"], how="left")
    return long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def rz_median_by_season(tg: pd.DataFrame) -> dict:
    """League red-zone TD rate from the previous season (known before any game is played)."""
    by = tg.groupby("season")[["rz_tds", "rz_trips"]].sum()
    rate = (by.rz_tds / by.rz_trips).to_dict()
    seasons = sorted(rate)
    return {s: rate.get(s - 1, rate[seasons[0]]) for s in range(seasons[0], seasons[-1] + 2)}


def team_form(long: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    league_mean = tg[EMA_COLS].mean().to_numpy()
    rz_med = rz_median_by_season(tg)
    alpha = 1 - 0.5 ** (1 / EMA_HALFLIFE)
    keep = []
    for team, d in long.groupby("team", sort=False):
        d = d.copy()
        vals = d[EMA_COLS].to_numpy(dtype=float)
        plays = d.off_n_plays.fillna(0).to_numpy()
        rz = d[["rz_tds", "rz_trips", "rz_tds_allowed", "rz_trips_allowed"]].to_numpy(dtype=float)
        seasons = d.season.to_numpy()
        state = league_mean.copy()
        rz_sum = np.zeros(4)
        pre = np.empty_like(vals)
        rz_pre = np.empty((len(d), 2))
        n_season, played = 0, np.zeros(len(d))
        for i in range(len(d)):
            if i > 0 and seasons[i] != seasons[i - 1]:
                state = CARRYOVER * state + (1 - CARRYOVER) * league_mean
                rz_sum *= CARRYOVER
                n_season = 0
            pre[i] = state
            med = rz_med.get(seasons[i], 0.55)
            rz_pre[i, 0] = (rz_sum[0] + RZ_PRIOR_TRIPS * med) / (rz_sum[1] + RZ_PRIOR_TRIPS)
            rz_pre[i, 1] = (rz_sum[2] + RZ_PRIOR_TRIPS * med) / (rz_sum[3] + RZ_PRIOR_TRIPS)
            played[i] = n_season
            row = vals[i]
            if not np.isnan(row).all():
                w = alpha * min(plays[i] / FULL_GAME_PLAYS, 1.0)
                row = np.where(np.isnan(row), state, row)
                state = (1 - w) * state + w * row
                rz_sum = rz_sum * (1 - alpha) + np.nan_to_num(rz[i])
                n_season += 1
        d[[f"f_{c}" for c in EMA_COLS]] = pre
        d["f_rz_off"], d["f_rz_def"] = rz_pre[:, 0], rz_pre[:, 1]
        d["games_this_season"] = played
        keep.append(d)
    cols = ["game_id", "team", "games_this_season", "f_rz_off", "f_rz_def"] + [f"f_{c}" for c in EMA_COLS]
    return pd.concat(keep)[cols]


# ---------------------------------------------------------------- opponent adjustment
def opponent_adjusted(long: pd.DataFrame) -> pd.DataFrame:
    """For each week, ridge-regress EPA/play on offense and defense team identity using only
    earlier games (decayed by age). Offense coefficient = adjusted offense, defense coefficient
    = adjusted EPA allowed. Returns one row per (team, game)."""
    rows = long[["game_id", "team", "opp", "season", "week", "off_epa", "off_pass_epa",
                 "off_rush_epa", "off_n_plays"]].copy()
    rows["t"] = rows.season * 30 + rows.week          # offseason counts as ~8 quiet weeks
    teams = sorted(set(rows.team) | set(rows.opp))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    played = rows.dropna(subset=["off_epa"]).reset_index(drop=True)
    X_all = np.zeros((len(played), 2 * n))
    X_all[np.arange(len(played)), played.team.map(ix).to_numpy()] = 1
    X_all[np.arange(len(played)), n + played.opp.map(ix).to_numpy()] = 1
    Y_all = played[["off_epa", "off_pass_epa", "off_rush_epa"]].fillna(0).to_numpy()
    out = []
    for t in sorted(rows.t.unique()):
        m = ((played.t < t) & (played.t >= t - 90)).to_numpy()
        if m.sum() < 200:
            continue
        age = t - played.t.to_numpy()[m]
        w = played.off_n_plays.fillna(0).to_numpy()[m] * 0.5 ** (age / OPP_ADJ_HALFLIFE)
        coef = Ridge(alpha=OPP_ADJ_ALPHA).fit(X_all[m], Y_all[m], sample_weight=w).coef_
        for r in rows[rows.t == t].itertuples():
            out.append((r.game_id, r.team, *coef[:, ix[r.team]], *coef[:, n + ix[r.team]]))
    cols = ["game_id", "team", "adj_off_epa", "adj_off_pass", "adj_off_rush",
            "adj_def_epa", "adj_def_pass", "adj_def_rush"]
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------- quarterbacks
QB_PRIOR = -0.05
QB_SHRINK = 200
QB_HALFLIFE = 600
QB_CARRYOVER = 0.75


def add_qb_ratings(g: pd.DataFrame, qg: pd.DataFrame) -> pd.DataFrame:
    """Pregame rating for each listed starter: shrunk, decayed neutral dropback EPA."""
    g = g.sort_values(["gameday", "game_id"]).copy()
    by_game = {k: d for k, d in qg.groupby("game_id")}
    S, N, last = {}, {}, {}
    home_r, away_r = [], []
    decay = 0.5 ** (1 / QB_HALFLIFE)
    for row in g.itertuples():
        for side, qb in (("h", row.home_qb_id), ("a", row.away_qb_id)):
            if isinstance(qb, str) and last.get(qb) not in (None, row.season):
                S[qb] *= QB_CARRYOVER
                N[qb] *= QB_CARRYOVER
                last[qb] = row.season
            s, n = S.get(qb, 0.0), N.get(qb, 0.0)
            r = (s + QB_SHRINK * QB_PRIOR) / (n + QB_SHRINK) if isinstance(qb, str) else QB_PRIOR
            (home_r if side == "h" else away_r).append(r)
        d = by_game.get(row.game_id)
        if d is None:
            continue
        for q in d.itertuples():
            f = decay ** q.dropbacks
            S[q.qb_id] = S.get(q.qb_id, 0.0) * f + q.epa_sum
            N[q.qb_id] = N.get(q.qb_id, 0.0) * f + q.dropbacks
            last[q.qb_id] = row.season
    g["home_qb_rating"], g["away_qb_rating"] = home_r, away_r
    return g


# ---------------------------------------------------------------- context
def add_context(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    venue = g.stadium_id.fillna("").str[:3]
    when = g.gameday.dt.to_pydatetime()
    for side in ("home", "away"):
        tr = [travel(t, v, w) for t, v, w in zip(g[f"{side}_team"], venue, when)]
        g[f"{side}_miles"] = [x[0] for x in tr]
        g[f"{side}_tz_shift"] = [x[1] for x in tr]
    g["home"] = (g.location != "Neutral").astype(int)
    g["rest_diff"] = (g.home_rest - g.away_rest).clip(-7, 7)
    g["travel_diff"] = (g.away_miles - g.home_miles) / 1000
    g["tz_shift_diff"] = g.away_tz_shift.abs() - g.home_tz_shift.abs()
    # traveling east means the body clock runs behind local time
    g["east_travel_diff"] = (g.away_tz_shift > 0).astype(int) - (g.home_tz_shift > 0).astype(int)
    outdoor = ~g.roof.isin(["dome", "closed"])
    g["indoor"] = (~outdoor).astype(int)
    g["wind_mph"] = np.where(outdoor, g.wind.fillna(0), 0)
    g["high_wind"] = (g.wind_mph > 15).astype(int)
    g["extreme_cold"] = (outdoor & (g.temp < 25)).astype(int)
    g["div_game"] = g.div_game.fillna(0)
    g["playoff"] = (g.game_type != "REG").astype(int)
    return g


# ---------------------------------------------------------------- assemble
def build_features(g: pd.DataFrame, tg: pd.DataFrame, qg: pd.DataFrame) -> pd.DataFrame:
    g = add_elo(g)
    g = add_qb_ratings(g, qg)
    g = add_context(g)
    long = long_table(g, tg)
    form = team_form(long, tg).merge(opponent_adjusted(long), on=["game_id", "team"], how="left")
    for side in ("home", "away"):
        f = form.add_prefix(f"{side}_").rename(columns={f"{side}_game_id": "game_id"})
        g = g.merge(f, on=["game_id", f"{side}_team"], how="left")
    for c in ["adj_off_epa", "adj_off_pass", "adj_off_rush", "adj_def_epa", "adj_def_pass", "adj_def_rush"]:
        g[[f"home_{c}", f"away_{c}"]] = g[[f"home_{c}", f"away_{c}"]].fillna(0)
    g = g.copy()

    # Neutral-site rating gap. Home field is added separately (fixed 1.2 to 1.8 points).
    g["elo_diff"] = g.home_elo - g.away_elo
    # Opponent-adjusted matchups: my adjusted offense vs your adjusted defense
    for k in ("epa", "pass", "rush"):
        home_net = g[f"home_adj_off_{k}"] + g[f"away_adj_def_{k}"]
        away_net = g[f"away_adj_off_{k}"] + g[f"home_adj_def_{k}"]
        g[f"adj_{k}_diff"] = home_net - away_net
        g[f"adj_{k}_sum"] = home_net + away_net
    for s in SIDE_STATS:
        g[f"d_off_{s}"] = g[f"home_f_off_{s}"] - g[f"away_f_off_{s}"]
        g[f"d_def_{s}"] = g[f"home_f_def_{s}"] - g[f"away_f_def_{s}"]
    g["d_rz_off"] = g.home_f_rz_off - g.away_f_rz_off
    g["d_rz_def"] = g.home_f_rz_def - g.away_f_rz_def
    g["qb_diff"] = g.home_qb_rating - g.away_qb_rating
    # QB value isolation: starter's EPA/dropback relative to his team's passing baseline
    g["home_qb_iso"] = g.home_qb_rating - g.home_f_off_pass_epa
    g["away_qb_iso"] = g.away_qb_rating - g.away_f_off_pass_epa
    g["qb_iso_diff"] = g.home_qb_iso - g.away_qb_iso
    g = g.copy()
    g["early_season"] = (np.minimum(g.home_games_this_season, g.away_games_this_season) < 3).astype(int)
    g["home_win"] = np.where(g.result > 0, 1, np.where(g.result < 0, 0, np.nan))

    # Totals: sums describe the scoring environment (no points-per-game inputs)
    g["sum_explosive"] = (g.home_f_off_explosive + g.away_f_off_explosive
                          + g.home_f_def_explosive + g.away_f_def_explosive)
    g["sum_sr"] = g.home_f_off_sr + g.away_f_off_sr + g.home_f_def_sr + g.away_f_def_sr
    g["sum_pace"] = g.home_f_plays_total + g.away_f_plays_total
    g["sum_qb"] = g.home_qb_rating + g.away_qb_rating
    g["sum_rz"] = g.home_f_rz_off + g.away_f_rz_off + g.home_f_rz_def + g.away_f_rz_def
    g["sum_twp"] = g.home_f_off_twp_rate + g.away_f_off_twp_rate
    g["season_trend"] = g.season - 2015
    return g


FEATURES = [
    "elo_diff", "adj_epa_diff", "adj_pass_diff", "adj_rush_diff",
    "d_off_sr", "d_def_sr", "d_off_early_epa", "d_def_early_epa", "d_off_explosive", "d_def_explosive",
    "d_off_cpoe", "d_def_cpoe", "d_off_adot", "d_off_yac_tgt", "d_def_yac_tgt",
    "d_off_press_rate", "d_def_press_rate", "d_off_press_epa", "d_off_clean_epa", "d_def_clean_epa",
    "d_off_twp_rate", "d_def_twp_rate", "d_off_fum_rate", "d_def_ffum_rate",
    "d_rz_off", "d_rz_def", "qb_diff", "qb_iso_diff",
    "rest_diff", "travel_diff", "tz_shift_diff", "east_travel_diff",
    "high_wind", "extreme_cold", "div_game", "playoff", "early_season",
]
# Features that describe the game, not one side. They do not flip when home and away swap.
SYMMETRIC = {"high_wind", "extreme_cold", "div_game", "playoff", "early_season"}
# Compact set for the linear models, which overfit the full list
CORE_FEATURES = ["elo_diff", "adj_epa_diff", "adj_pass_diff", "adj_rush_diff", "qb_diff",
                 "d_off_sr", "d_def_sr", "d_off_early_epa", "d_def_early_epa",
                 "d_off_explosive", "d_def_explosive", "rest_diff", "travel_diff", "east_travel_diff"]

TOTAL_FEATURES = [
    "adj_epa_sum", "adj_pass_sum", "adj_rush_sum", "sum_explosive", "sum_sr", "sum_pace",
    "sum_qb", "sum_rz", "sum_twp", "indoor", "wind_mph", "high_wind", "extreme_cold",
    "season_trend", "playoff", "div_game", "early_season",
]
