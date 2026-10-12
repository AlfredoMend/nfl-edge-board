"""Pregame features. Every value for a game uses only games played before it."""
import numpy as np
import pandas as pd

STATS = ["off_epa", "off_sr", "off_pass_epa", "off_rush_epa",
         "def_epa", "def_sr", "def_pass_epa", "def_rush_epa"]
FORM = STATS + ["pts_for", "pts_against"]

# Elo settings (FiveThirtyEight-style NFL Elo)
ELO_K = 20.0
ELO_HFA = 48.0          # home field in Elo points (~1.7 pts of spread)
ELO_REGRESS = 1 / 3     # pull toward 1505 between seasons
EWM_HALFLIFE = 8        # games; recent games count more
CARRYOVER = 0.5         # weight kept on last season's form at a new season


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
            continue  # future game: no update
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


def team_form(g: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """Long table (team, game) with decayed averages of prior games only."""
    long = pd.concat([
        g[["game_id", "season", "gameday", "home_team", "home_qb_id"]]
        .rename(columns={"home_team": "team", "home_qb_id": "qb_id"}),
        g[["game_id", "season", "gameday", "away_team", "away_qb_id"]]
        .rename(columns={"away_team": "team", "away_qb_id": "qb_id"}),
    ]).merge(tg.drop(columns="season"), on=["game_id", "team"], how="left")
    # Points scored and allowed, for projecting each team's score
    pts = pd.concat([
        g[["game_id", "home_team", "home_score", "away_score"]].set_axis(["game_id", "team", "pts_for", "pts_against"], axis=1),
        g[["game_id", "away_team", "away_score", "home_score"]].set_axis(["game_id", "team", "pts_for", "pts_against"], axis=1),
    ])
    long = long.merge(pts, on=["game_id", "team"], how="left")
    long = long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    pts_mean = float(pd.concat([g.home_score, g.away_score]).mean())
    league_mean = np.r_[tg[STATS].mean().to_numpy(), pts_mean, pts_mean]
    alpha = 1 - 0.5 ** (1 / EWM_HALFLIFE)
    out = []
    for team, d in long.groupby("team", sort=False):
        d = d.copy()
        vals = d[FORM].to_numpy(dtype=float)
        seasons = d.season.to_numpy()
        state = league_mean.copy()
        pre = np.empty_like(vals)
        played_season = np.zeros(len(d))
        n_season = 0
        for i in range(len(d)):
            if i > 0 and seasons[i] != seasons[i - 1]:
                # New season: regress form halfway back to league average.
                state = CARRYOVER * state + (1 - CARRYOVER) * league_mean
                n_season = 0
            pre[i] = state
            played_season[i] = n_season
            row = vals[i]
            if not np.isnan(row).all():
                row = np.where(np.isnan(row), state, row)
                state = (1 - alpha) * state + alpha * row
                n_season += 1
        d[[f"f_{s}" for s in FORM]] = pre
        d["games_this_season"] = played_season
        # QB change: starter differs from the team's most recent starter.
        prev_qb = d.qb_id.shift(1)
        d["qb_change"] = ((d.qb_id != prev_qb) & prev_qb.notna() & d.qb_id.notna()).astype(int)
        out.append(d)
    keep = ["game_id", "team", "games_this_season", "qb_change"] + [f"f_{s}" for s in FORM]
    return pd.concat(out)[keep]


QB_PRIOR = -0.05        # roughly replacement-level dropback EPA
QB_SHRINK = 200         # dropbacks of prior weight for an unknown QB
QB_HALFLIFE = 600       # dropbacks; older play fades out
QB_CARRYOVER = 0.75


def add_qb_ratings(g: pd.DataFrame, qg: pd.DataFrame) -> pd.DataFrame:
    """Pregame rating for each listed starter: shrunk, decayed dropback EPA."""
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
    g["qb_diff"] = g.home_qb_rating - g.away_qb_rating
    return g


def build_features(g: pd.DataFrame, tg: pd.DataFrame, qg: pd.DataFrame) -> pd.DataFrame:
    g = add_elo(g)
    g = add_qb_ratings(g, qg)
    form = team_form(g, tg)
    g = g.merge(form.add_prefix("home_").rename(columns={"home_game_id": "game_id", "home_team": "home_team"}),
                on=["game_id", "home_team"], how="left")
    g = g.merge(form.add_prefix("away_").rename(columns={"away_game_id": "game_id", "away_team": "away_team"}),
                on=["game_id", "away_team"], how="left")

    g["elo_diff"] = g.home_elo - g.away_elo + np.where(g.location == "Neutral", 0, ELO_HFA)
    for s in STATS:
        g[f"d_{s}"] = g[f"home_f_{s}"] - g[f"away_f_{s}"]
    # Matchups: my offense vs your defense
    g["home_off_vs_def"] = g.home_f_off_epa + g.away_f_def_epa
    g["away_off_vs_def"] = g.away_f_off_epa + g.home_f_def_epa
    g["matchup_diff"] = g.home_off_vs_def - g.away_off_vs_def
    g["rest_diff"] = (g.home_rest - g.away_rest).clip(-7, 7)
    g["neutral"] = (g.location == "Neutral").astype(int)
    g["div_game"] = g.div_game.fillna(0)
    g["playoff"] = (g.game_type != "REG").astype(int)
    g["qb_change_diff"] = g.home_qb_change - g.away_qb_change
    g["early_season"] = (np.minimum(g.home_games_this_season, g.away_games_this_season) < 3).astype(int)
    g["home_win"] = np.where(g.result > 0, 1, np.where(g.result < 0, 0, np.nan))
    # Scoring-environment features for the total-points model
    g["home_pts_exp"] = (g.home_f_pts_for + g.away_f_pts_against) / 2
    g["away_pts_exp"] = (g.away_f_pts_for + g.home_f_pts_against) / 2
    g["sum_pts_exp"] = g.home_pts_exp + g.away_pts_exp
    g["sum_off_epa"] = g.home_f_off_epa + g.away_f_off_epa
    g["sum_def_epa"] = g.home_f_def_epa + g.away_f_def_epa
    g["sum_off_sr"] = g.home_f_off_sr + g.away_f_off_sr
    g["sum_def_sr"] = g.home_f_def_sr + g.away_f_def_sr
    g["sum_qb"] = g.home_qb_rating + g.away_qb_rating
    g["indoor"] = g.roof.isin(["dome", "closed"]).astype(int)
    g["season_trend"] = g.season - 2015
    return g


FEATURES = ["elo_diff", "d_off_epa", "d_def_epa", "d_off_sr", "d_def_sr",
            "d_off_pass_epa", "d_def_pass_epa", "d_off_rush_epa", "d_def_rush_epa",
            "matchup_diff", "rest_diff", "neutral", "div_game", "playoff",
            "qb_change_diff", "early_season", "qb_diff"]

TOTAL_FEATURES = ["sum_pts_exp", "sum_off_epa", "sum_def_epa", "sum_off_sr", "sum_def_sr",
                  "sum_qb", "indoor", "season_trend", "playoff", "div_game", "early_season"]
