"""Player prop projections by Monte Carlo simulation, reported as medians.

Each simulated game:
  1. Draw the final margin and total from the game model (same mean as the board's projection),
     which sets each team's points and game script.
  2. Team pass attempts and carries come from the team's recent volume, shifted by the simulated
     margin (trailing teams throw more, leading teams run more).
  3. Targets and carries are split among active players by their recent usage shares, with share
     noise so one player can have a big or quiet day. Out and Doubtful players are removed and
     their share is redistributed.
  4. Catches follow each receiver's catch rate. Yards per catch and per carry are drawn from real
     NFL yardage distributions (heavily right-skewed), scaled to the player's efficiency and the
     opponent's defense.
  5. Touchdowns follow the team's simulated points and are handed out by red-zone usage.
  6. QB passing = the sum of his receivers' catches and yards, so the QB and receivers agree.
Every stat is summarized by its median across 10,000 simulations, plus a 25th to 75th percentile
range and touchdown probabilities. All inputs use only games played before kickoff.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_SIMS = 10_000
SEED = 7

SHARE_HALFLIFE = 4          # appearances
EFF_HALFLIFE = 24           # appearances (efficiency is noisier, so a longer memory)
TEAM_HALFLIFE = 6           # team games
MARGIN_SD = 13.0
TOTAL_SD = 13.0
SHARE_CONCENTRATION = 40    # Dirichlet; higher = steadier week-to-week usage
OTHER_TARGET, OTHER_CARRY = 0.02, 0.01   # usage left for players the model does not list
REDISTRIBUTE = 0.90         # share of a missing player's usage that goes to listed teammates
LINEUP_GAMES = 12           # team games checked for lineup similarity
LINEUP_POWER = 6.0          # how hard to down-weight games whose lineup differs from this one
SHARE_GAMMA_TGT = 1.15      # >1 concentrates targets toward established starters (set on the 2024 backtest)
SHARE_GAMMA_CAR = 1.10      # same for carries
QB_ATT_SHARE = 0.959        # listed starters throw this share of team attempts (2022-2025); backups and early exits take the rest
TD_SCALE = 1.12             # touchdown calibration, set on the 2024 backtest (actual/predicted 1.16) and checked on 2025
CARRY_CONCENTRATION = 20    # carries swing more week to week than targets (game script, committees)
RUSH_GAME_SD = 0.8          # game-level yards-per-carry noise (blocking, defense, weather on the day)
REC_GAME_SD = 0.10          # game-level noise on yards per catch, as a fraction
VOLUME_SEASONS = 3          # team volume model uses the most recent seasons (league passing has trended down)

POS_PRIOR = {  # target share, carry share, catch rate, (shrink counts below)
    "WR": (0.10, 0.005, 0.63), "TE": (0.08, 0.002, 0.70), "RB": (0.06, 0.15, 0.78),
    "QB": (0.0, 0.08, 0.0), "FB": (0.02, 0.03, 0.75),
}
CR_K, YPR_K, YPC_K, INT_K = 25, 30, 50, 200
QB_CR_K, QB_YPC_K = 150, 120   # QB completion-rate and yards-per-completion shrinkage
QB_WEIGHT = 0.5                # receivers' history already reflects their QB, so half credit
LEAGUE_INT = 0.023
DIST = json.loads((ROOT / "data" / "yardage_dists.json").read_text())


# ---------------------------------------------------------------- history helpers
def _decay_weights(n: int, halflife: float) -> np.ndarray:
    return 0.5 ** (np.arange(n)[::-1] / halflife)


def _ema(values: np.ndarray, halflife: float, prior: float = None, prior_w: float = 0.0, extra=None) -> float:
    if len(values) == 0:
        return prior if prior is not None else 0.0
    w = _decay_weights(len(values), halflife)
    if extra is not None:
        w = w * np.asarray(extra, dtype=float)
    num = (w * values).sum() + (prior_w * prior if prior is not None else 0.0)
    return float(num / (w.sum() + prior_w))


NAMES: dict = {}


def prepare(pg: pd.DataFrame, tm: pd.DataFrame, sched: pd.DataFrame, info: pd.DataFrame):
    """Attach dates and opponents; compute per-appearance usage shares."""
    NAMES.update(dict(zip(info.player_id, info.name)))
    games = sched[["game_id", "gameday", "season", "week", "home_team", "away_team", "home_score", "away_score"]]
    tm = tm.drop(columns=["season"], errors="ignore").merge(games, on="game_id")
    tm["opp"] = np.where(tm.team == tm.home_team, tm.away_team, tm.home_team)
    tm["points"] = np.where(tm.team == tm.home_team, tm.home_score, tm.away_score)
    tm["margin"] = np.where(tm.team == tm.home_team, tm.home_score - tm.away_score, tm.away_score - tm.home_score)
    tm = tm.sort_values(["gameday", "game_id"])
    tm = _add_volume_features(tm)

    pg = pg.drop(columns=["season"], errors="ignore").merge(games[["game_id", "gameday", "season", "week"]], on="game_id")
    vol = tm[["game_id", "team", "targets", "carries", "rz_targets", "rz_carries", "pass_td", "rush_td"]].add_prefix("t_").rename(
        columns={"t_game_id": "game_id", "t_team": "team"})
    pg = pg.merge(vol, on=["game_id", "team"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        pg["tgt_sh"] = (pg.targets / pg.t_targets).fillna(0)
        pg["car_sh"] = (pg.carries / pg.t_carries).fillna(0)
        pg["rzt_sh"] = (pg.rz_targets / pg.t_rz_targets).fillna(0)
        pg["rzc_sh"] = (pg.rz_carries / pg.t_rz_carries).fillna(0)
        pg["ptd_sh"] = (pg.rec_td / pg.t_pass_td).fillna(0)
        pg["rtd_sh"] = (pg.rush_td / pg.t_rush_td).fillna(0)
        pg["has_ptd"] = (pg.t_pass_td > 0).astype(float)
        pg["has_rtd"] = (pg.t_rush_td > 0).astype(float)
    pg = pg.merge(info, on="player_id", how="left")
    pg["position"] = pg.position.fillna("WR")
    pg["appeared"] = (pg.off_pct > 0) | (pg.targets + pg.carries + pg.pass_att > 0)
    return pg.sort_values(["gameday", "game_id"]), tm


VOL_HALFLIFE = 6.0
OWN_FEATS = ["pass_att", "carries", "proe", "sec_per_play"]
ALLOWED_FEATS = ["pa_allowed", "car_allowed", "proe_allowed"]
PASS_X = ["e_pass_att", "margin", "e_proe", "e_sec_per_play", "o_sec_per_play", "o_pa_allowed", "o_proe_allowed"]
RUSH_X = ["e_carries", "margin", "e_proe", "e_sec_per_play", "o_sec_per_play", "o_car_allowed", "o_proe_allowed"]


def _rec_ema(values: np.ndarray, halflife: float = VOL_HALFLIFE):
    """Prior-only exponential average for each position, and the value after the last game."""
    w = 0.5 ** (1 / halflife)
    s, prior = np.nan, np.empty(len(values))
    for i, x in enumerate(values):
        prior[i] = s
        if not np.isnan(x):
            s = x if np.isnan(s) else w * s + (1 - w) * x
    return prior, s


def _add_volume_features(tm: pd.DataFrame) -> pd.DataFrame:
    """Team pace (pass rate over expected, seconds per play) and what each defense allows,
    as exponential averages of earlier games only, plus the opponent's values for each row."""
    pace_path = ROOT / "data" / "team_pace.csv"
    if pace_path.exists():
        pace = pd.read_csv(pace_path).drop(columns=["season"], errors="ignore")[["game_id", "team", "proe", "sec_per_play"]]
        tm = tm.drop(columns=["proe", "sec_per_play"], errors="ignore").merge(pace, on=["game_id", "team"], how="left")
    else:
        tm = tm.assign(proe=np.nan, sec_per_play=np.nan)
    allowed = tm[["game_id", "opp", "pass_att", "carries", "proe"]].rename(
        columns={"opp": "team", "pass_att": "pa_allowed", "carries": "car_allowed", "proe": "proe_allowed"})
    tm = tm.merge(allowed, on=["game_id", "team"], how="left").sort_values(["gameday", "game_id"])
    for c in OWN_FEATS + ALLOWED_FEATS:
        tm[f"e_{c}"] = np.nan
    for team, idx in tm.groupby("team").groups.items():
        for c in OWN_FEATS + ALLOWED_FEATS:
            tm.loc[idx, f"e_{c}"] = _rec_ema(tm.loc[idx, c].to_numpy(float))[0]
    opp = tm[["game_id", "team", "e_sec_per_play", "e_pa_allowed", "e_car_allowed", "e_proe_allowed"]].rename(
        columns={"team": "opp", "e_sec_per_play": "o_sec_per_play", "e_pa_allowed": "o_pa_allowed",
                 "e_car_allowed": "o_car_allowed", "e_proe_allowed": "o_proe_allowed"})
    return tm.merge(opp, on=["game_id", "opp"], how="left").sort_values(["gameday", "game_id"])


def league_rates(tm: pd.DataFrame, pg: pd.DataFrame) -> dict:
    tds = tm.pass_td.sum() + tm.rush_td.sum()
    return {
        "td_per_point": float(tds / tm.points.sum()),
        "pass_td_frac": float(tm.pass_td.sum() / tds),
        "target_rate": float(tm.targets.sum() / tm.pass_att.sum()),
        "ypt": float(tm.pass_yds.sum() / tm.targets.sum()),
        "ypc": float(tm.rush_yds.sum() / tm.carries.sum()),
        "int_rate": float(pg.ints.sum() / max(pg.pass_att.sum(), 1)),
        "comp_rate": float(pg.comp.sum() / max(pg.pass_att.sum(), 1)),
        "yds_per_comp": float(pg.pass_yds.sum() / max(pg.comp.sum(), 1)),
    }


# ---------------------------------------------------------------- volume model
def fit_volume(tm: pd.DataFrame):
    """Team pass attempts and carries from recent volume, pass rate over expected, tempo, the
    opponent's tempo and what the opponent's defense allows, plus the final margin (game script).
    Least squares on games where every input comes from earlier games only."""
    out = {}
    for name, xs, y in (("pass", PASS_X, "pass_att"), ("rush", RUSH_X, "carries")):
        d = tm.dropna(subset=xs + [y])
        means = {x: float(d[x].mean()) for x in xs}
        A = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
        coef, *_ = np.linalg.lstsq(A, d[y].to_numpy(float), rcond=None)
        resid = d[y].to_numpy(float) - A @ coef
        out[name] = {"coef": coef.tolist(), "x": xs, "means": means, "sd": float(resid.std())}
    return out


def _volume(vol: dict, name: str, inputs: dict, margin: np.ndarray) -> np.ndarray:
    v = vol[name]
    base = v["coef"][0]
    for c, x in zip(v["coef"][1:], v["x"]):
        if x == "margin":
            continue
        val = inputs.get(x)
        base += c * (v["means"][x] if val is None or (isinstance(val, float) and np.isnan(val)) else val)
    return base + v["coef"][1 + v["x"].index("margin")] * margin


# ---------------------------------------------------------------- team and player profiles
def team_profile(tm_hist: pd.DataFrame, team: str, lg: dict) -> dict:
    d = tm_hist[tm_hist.team == team].tail(16)
    if d.empty:
        return {"pass_att": 33.0, "carries": 26.0, "pass_td_frac": lg["pass_td_frac"]}
    tds = d.pass_td + d.rush_td
    w = _decay_weights(len(d), TEAM_HALFLIFE)
    ptd = ((w * d.pass_td).sum() + 6 * lg["pass_td_frac"]) / ((w * tds).sum() + 6)
    full = tm_hist[tm_hist.team == team]
    post = {f"e_{c}": float(_rec_ema(full[c].to_numpy(float))[1]) if c in full else np.nan
            for c in ("pass_att", "carries", "proe", "sec_per_play")}
    return {"pass_att": _ema(d.pass_att.to_numpy(float), TEAM_HALFLIFE),
            "carries": _ema(d.carries.to_numpy(float), TEAM_HALFLIFE), "pass_td_frac": float(ptd), **post}


def defense_profile(tm_hist: pd.DataFrame, team: str, lg: dict) -> dict:
    """What this defense allowed per target and per carry, relative to league, shrunk and damped."""
    d = tm_hist[tm_hist.opp == team].tail(16)
    if d.empty:
        return {"pass_mult": 1.0, "rush_mult": 1.0}
    w = _decay_weights(len(d), 8)
    ypt = ((w * d.pass_yds).sum() + 120 * lg["ypt"]) / ((w * d.targets).sum() + 120)
    ypc = ((w * d.rush_yds).sum() + 100 * lg["ypc"]) / ((w * d.carries).sum() + 100)
    full = tm_hist[tm_hist.team == team]
    allowed = {f"o_{c}": float(_rec_ema(full[c].to_numpy(float))[1]) if c in full else np.nan
               for c in ("pa_allowed", "car_allowed", "proe_allowed")}
    return {"pass_mult": float(1 + 0.6 * (ypt / lg["ypt"] - 1)), "rush_mult": float(1 + 0.6 * (ypc / lg["ypc"] - 1)),
            **allowed}


ROSTER_BLOCK = {"RES", "CUT", "RET", "DEV", "TRD", "TRC", "EXE", "TRT", "PUP", "NFI", "SUS"}


def load_roster(season: int, week: int, refresh: bool = False):
    """nflverse weekly roster for one week: {team: {gsis_id: status}}. None if unavailable."""
    from injuries import BASE, _get
    try:
        path = _get(f"roster_weekly_{season}.parquet", f"{BASE}/weekly_rosters/roster_weekly_{season}.parquet", refresh)
        r = pd.read_parquet(path, columns=["team", "week", "gsis_id", "status"])
    except Exception:
        return None
    r = r[(r.week == week) & r.gsis_id.notna()]
    if r.empty:
        return None
    out = {}
    for t in r.itertuples():
        out.setdefault(t.team, {})[t.gsis_id] = t.status
    return out


def load_depth_starters(season: int, refresh: bool = False) -> dict:
    """Latest depth chart: {team: {gsis_id: position}} for first-string QB, RB, WR and TE."""
    from injuries import BASE, _get
    try:
        path = _get(f"depth_charts_{season}.parquet", f"{BASE}/depth_charts/depth_charts_{season}.parquet", refresh)
        d = pd.read_parquet(path)
    except Exception:
        return {}
    if "dt" not in d.columns or "pos_rank" not in d.columns:
        return {}
    d = d[d.dt == d.dt.max()]
    d = d[(d.pos_rank == 1) & d.pos_abb.isin(["QB", "RB", "WR", "TE"]) & d.gsis_id.notna()]
    out = {}
    for r in d.itertuples():
        out.setdefault(r.team, {})[r.gsis_id] = r.pos_abb
    return out


DEPTH_STARTERS: dict = {}   # filled for live runs; new starters with no NFL history get position priors


def player_profiles(pg_hist: pd.DataFrame, tm_hist: pd.DataFrame, team: str, qb_id: str,
                    injuries: dict, lg: dict, pos_eff: dict, roster: dict = None, use_inactives: bool = False) -> list:
    """Active players for this team with usage shares and efficiency, before redistribution.

    Without a roster: anyone who played in the team's last two games, plus Questionable players.
    With the weekly roster: also anyone on the active roster with offensive history (players back
    from injured reserve, trades, returning starters), and nobody on injured reserve, the practice
    squad, cut or traded. Game-day inactives are only used when use_inactives is set (live runs;
    they post about 90 minutes before kickoff, so the backtest does not use them)."""
    team_games = tm_hist[tm_hist.team == team].game_id.tail(2).tolist()
    recent = pg_hist[(pg_hist.team == team) & pg_hist.game_id.isin(team_games) & pg_hist.appeared]
    status = {r["id"]: r["status"] for r in injuries.get(team, [])}
    q_names = [r["id"] for r in injuries.get(team, []) if r["status"] == "Questionable"]
    ids = set(recent.player_id) | set(pg_hist[(pg_hist.team == team) & pg_hist.player_id.isin(q_names)].player_id)
    rs = (roster or {}).get(team)
    if rs:
        block = ROSTER_BLOCK | ({"INA"} if use_inactives else set())
        active = {pid for pid, st in rs.items() if st not in block}
        recent_ids = set(pg_hist[pg_hist.appeared].tail(200000).player_id)   # anyone with any history
        hist_games = tm_hist[tm_hist.team == team].game_id.tail(16).tolist()
        with_history = set(pg_hist[pg_hist.appeared & ((pg_hist.team == team) & pg_hist.game_id.isin(hist_games)
                                                       | pg_hist.player_id.isin(active))].player_id)
        ids = (ids & set(rs)) - {pid for pid, st in rs.items() if st in block}
        ids |= (active & with_history & recent_ids)
        ids |= {pid for pid in DEPTH_STARTERS.get(team, {}) if pid in active}
    out = []
    for pid in ids:
        h = pg_hist[(pg_hist.player_id == pid) & pg_hist.appeared].tail(40)
        if h.empty:
            dpos = DEPTH_STARTERS.get(team, {}).get(pid)
            if dpos and dpos != "QB" and status.get(pid) not in ("Out", "Doubtful"):
                out.append(_profile(pid, h, dpos, lg, pos_eff, questionable=status.get(pid) == "Questionable"))
            continue
        pos = h.position.iloc[-1]
        if pos not in POS_PRIOR:
            continue
        if pos == "QB" and pid != qb_id:
            continue
        if status.get(pid) in ("Out", "Doubtful"):
            continue
        out.append(_profile(pid, h, pos, lg, pos_eff, questionable=status.get(pid) == "Questionable"))
    if qb_id and not any(p["id"] == qb_id for p in out):
        h = pg_hist[(pg_hist.player_id == qb_id) & pg_hist.appeared].tail(40)
        out.append(_profile(qb_id, h, "QB", lg, pos_eff, questionable=False))
    out = _lineup_reweight(out, pg_hist, tm_hist, team, lg, pos_eff)
    return [p for p in out if p["tgt"] > 0.005 or p["car"] > 0.005 or p["pos"] == "QB"]


def _lineup_reweight(profiles, pg_hist, tm_hist, team, lg, pos_eff):
    """Recompute usage shares, weighting each recent team game by how much of tonight's
    active usage was on the field that day. Games where a starter sat out (and teammates
    soaked up his targets) count less, so a returning starter gets his normal share back
    and fill-in players lose the inflated share they had while he was out."""
    games = tm_hist[tm_hist.team == team].game_id.tail(LINEUP_GAMES).tolist()
    if not games:
        return profiles
    rows = pg_hist[(pg_hist.team == team) & pg_hist.game_id.isin(games) & pg_hist.appeared]
    present = rows.groupby("game_id").player_id.apply(set).to_dict()
    known = [p for p in profiles if p["pos"] != "QB" and p["id"] in set(rows.player_id)]
    imp = {p["id"]: p["tgt"] + p["car"] for p in known}
    total = sum(imp.values())
    if total <= 0:
        return profiles
    sim = {g: sum(v for k, v in imp.items() if k in present.get(g, set())) / total for g in games}
    out = []
    for p in profiles:
        h = pg_hist[(pg_hist.player_id == p["id"]) & pg_hist.appeared].tail(40)
        if h.empty:
            out.append(p)
            continue
        lw = np.where((h.team == team) & h.game_id.isin(games),
                      h.game_id.map(sim).fillna(1.0).to_numpy(float) ** LINEUP_POWER, 1.0)
        lw = np.maximum(lw, 0.05)
        out.append(_profile(p["id"], h, p["pos"], lg, pos_eff, p["questionable"], lw=lw))
    return out


def _profile(pid, h, pos, lg, pos_eff, questionable, lw=None):
    pt, pc, pcr = POS_PRIOR[pos]
    sh = h.tail(12)
    sw = None if lw is None else np.asarray(lw)[-len(sh):]
    tgt = _ema(sh.tgt_sh.to_numpy(float), SHARE_HALFLIFE, pt, 0.5, sw)
    car = _ema(sh.car_sh.to_numpy(float), SHARE_HALFLIFE, pc, 0.5, sw)
    rzt = _ema(sh.rzt_sh.to_numpy(float), SHARE_HALFLIFE * 2, pt, 1.0, sw)
    rzc = _ema(sh.rzc_sh.to_numpy(float), SHARE_HALFLIFE * 2, pc, 1.0, sw)
    # Touchdown shares over a longer window, only from games where the team scored that way
    hp, hr = h[h.has_ptd > 0].tail(24), h[h.has_rtd > 0].tail(24)
    ptd = _ema(hp.ptd_sh.to_numpy(float), 12, pt, 3.0)
    rtd = _ema(hr.rtd_sh.to_numpy(float), 12, pc, 3.0)
    w = _decay_weights(len(h), EFF_HALFLIFE)
    grp_rec = "RB" if pos in ("RB", "FB") else ("TE" if pos == "TE" else "WR")
    grp_rush = "QB" if pos == "QB" else ("RB" if pos in ("RB", "FB") else "WR")
    cr = ((w * h.rec).sum() + CR_K * pcr) / ((w * h.targets).sum() + CR_K) if pos != "QB" else 0.0
    ypr_lg = DIST["rec"][grp_rec]["mean"]
    ypr = ((w * h.rec_yds).sum() + YPR_K * ypr_lg) / ((w * h.rec).sum() + YPR_K)
    ypc_lg = DIST["rush"][grp_rush]["mean"]
    ypc = ((w * h.rush_yds).sum() + YPC_K * ypc_lg) / ((w * h.carries).sum() + YPC_K)
    intr = ((w * h.ints).sum() + INT_K * lg["int_rate"]) / ((w * h.pass_att).sum() + INT_K)
    qb_cr = ((w * h.comp).sum() + QB_CR_K * lg["comp_rate"]) / ((w * h.pass_att).sum() + QB_CR_K)
    qb_ypc = ((w * h.pass_yds).sum() + QB_YPC_K * lg["yds_per_comp"]) / ((w * h.comp).sum() + QB_YPC_K)
    q = 0.85 if questionable else 1.0
    nm = NAMES.get(pid)
    return {"id": pid, "name": nm if isinstance(nm, str) else pid,
            "pos": pos, "tgt": tgt * q, "car": car * q, "rzt": rzt * q, "rzc": rzc * q,
            "ptd": ptd * q, "rtd": rtd * q,
            "cr": float(cr), "ypr_mult": float(ypr / ypr_lg), "ypc_shift": float(ypc - ypc_lg),
            "rec_grp": grp_rec, "rush_grp": grp_rush, "int_rate": float(intr), "questionable": questionable,
            "qb_cr_mult": float((qb_cr / lg["comp_rate"]) ** QB_WEIGHT) if pos == "QB" else 1.0,
            "qb_ypc_mult": float((qb_ypc / lg["yds_per_comp"]) ** QB_WEIGHT) if pos == "QB" else 1.0}


def _normalize(players: list, key: str, other: float) -> np.ndarray:
    raw = np.array([p[key] for p in players], dtype=float)
    gamma = SHARE_GAMMA_TGT if key == "tgt" else SHARE_GAMMA_CAR
    if gamma != 1.0 and raw.sum() > 0:
        raw = raw ** gamma * raw.sum() / (raw ** gamma).sum()
    s = raw.sum()
    target = 1 - other
    if s <= 0:
        return np.r_[np.zeros(len(raw)), 1.0]
    if s < target:   # usage freed up by absences: most of it goes to listed teammates
        new_total = s + REDISTRIBUTE * (target - s)
    else:
        new_total = target
    sh = raw / s * new_total
    return np.r_[sh, 1 - sh.sum()]


def injury_status(season: int, week: int) -> dict:
    """{team: [{'id', 'status'}]} from that week's official injury report."""
    path = ROOT / "data" / "raw" / f"injuries_{season}.parquet"
    if not path.exists():
        return {}
    inj = pd.read_parquet(path, columns=["week", "team", "gsis_id", "report_status"])
    inj = inj[(inj.week == week) & inj.gsis_id.notna()]
    out = {}
    for r in inj.itertuples():
        out.setdefault(r.team, []).append({"id": r.gsis_id, "status": r.report_status})
    return out


def build_side(pg_hist, tm_hist, team, opp, qb_id, inj, lg, roster=None, use_inactives=False):
    return {"prof": team_profile(tm_hist, team, lg), "defense": defense_profile(tm_hist, team, lg),
            "players": player_profiles(pg_hist, tm_hist, team, qb_id, inj, lg, None, roster, use_inactives)}


# ---------------------------------------------------------------- simulation
def _draw_yards(rng, counts: np.ndarray, dist: dict, mult: float = 1.0, shift: float = 0.0) -> np.ndarray:
    """Sum of `counts[s]` per-play yardage draws for each sim s."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(len(counts))
    q = np.asarray(dist["q"])
    draws = np.interp(rng.random(total), np.linspace(0.0025, 0.9975, len(q)), q)
    draws = draws * mult + shift
    idx = np.repeat(np.arange(len(counts)), counts)
    return np.bincount(idx, weights=draws, minlength=len(counts))


def simulate_team(rng, team_pts, team_margin, prof, defense, players, vol, lg, opp_prof=None):
    n = len(team_pts)
    inputs = {"e_pass_att": prof.get("e_pass_att", prof["pass_att"]), "e_carries": prof.get("e_carries", prof["carries"]),
              "e_proe": prof.get("e_proe"), "e_sec_per_play": prof.get("e_sec_per_play"),
              "o_sec_per_play": (opp_prof or {}).get("e_sec_per_play"),
              "o_pa_allowed": defense.get("o_pa_allowed"), "o_car_allowed": defense.get("o_car_allowed"),
              "o_proe_allowed": defense.get("o_proe_allowed")}
    pass_att = np.maximum(np.rint(_volume(vol, "pass", inputs, team_margin) + rng.normal(0, vol["pass"]["sd"], n)), 12).astype(int)
    carries = np.maximum(np.rint(_volume(vol, "rush", inputs, team_margin) + rng.normal(0, vol["rush"]["sd"], n)), 8).astype(int)
    targets_total = rng.binomial(pass_att, lg["target_rate"])

    k = len(players)
    t_sh = _normalize(players, "tgt", OTHER_TARGET)
    c_sh = _normalize(players, "car", OTHER_CARRY)
    tgt = rng.multinomial(targets_total, rng.dirichlet(t_sh * SHARE_CONCENTRATION + 1e-3, n))
    car = rng.multinomial(carries, rng.dirichlet(c_sh * CARRY_CONCENTRATION + 1e-3, n))

    qbp = next((p for p in players if p["pos"] == "QB"), None)
    cr_m = qbp["qb_cr_mult"] if qbp else 1.0
    yd_m = qbp["qb_ypc_mult"] if qbp else 1.0
    res = {}
    rec_all = np.zeros(n, int)
    rec_yds_all = np.zeros(n)
    for j, p in enumerate(players + [{"id": "_other", "pos": "WR", "cr": 0.62, "ypr_mult": 0.95, "ypc_shift": 0.0,
                                       "rec_grp": "WR", "rush_grp": "RB"}]):
        rec = rng.binomial(tgt[:, j], min(p["cr"] * cr_m, 0.95)) if p["pos"] != "QB" else np.zeros(n, int)
        ry = _draw_yards(rng, rec, DIST["rec"][p["rec_grp"]], p["ypr_mult"] * yd_m * defense["pass_mult"])
        ry = ry * np.exp(rng.normal(0, REC_GAME_SD, n))
        uy = _draw_yards(rng, car[:, j], DIST["rush"][p["rush_grp"]],
                         defense["rush_mult"], p["ypc_shift"]) + car[:, j] * rng.normal(0, RUSH_GAME_SD, n)
        rec_all += rec
        rec_yds_all += ry
        res[p["id"]] = {"targets": tgt[:, j], "rec": rec, "rec_yds": ry, "carries": car[:, j], "rush_yds": uy,
                        "rec_td": np.zeros(n, int), "rush_td": np.zeros(n, int)}

    # touchdowns follow the simulated score
    n_td = rng.poisson(np.maximum(team_pts, 0) * lg["td_per_point"] * TD_SCALE)
    n_ptd = rng.binomial(n_td, prof["pass_td_frac"])
    n_rtd = n_td - n_ptd
    wt = np.r_[[0.4 * p["rzt"] + 0.3 * p["tgt"] + 0.3 * p["ptd"] for p in players], 0.03]
    wc = np.r_[[0.4 * p["rzc"] + 0.2 * p["car"] + 0.4 * p["rtd"] for p in players], 0.02]
    wt, wc = wt / wt.sum(), wc / wc.sum()
    ptd = rng.multinomial(n_ptd, wt)
    rtd = rng.multinomial(n_rtd, wc)
    keys = list(res)
    for j, key in enumerate(keys):
        res[key]["rec_td"] = ptd[:, j]
        res[key]["rush_td"] = rtd[:, j]

    qb = next((p for p in players if p["pos"] == "QB"), None)
    if qb:
        r = res[qb["id"]]
        qb_att = rng.binomial(pass_att, QB_ATT_SHARE)
        frac = qb_att / np.maximum(pass_att, 1)
        r.update({"pass_att": qb_att, "comp": rng.binomial(rec_all, frac), "pass_yds": rec_yds_all * frac,
                  "pass_td": rng.binomial(n_ptd, frac), "ints": rng.binomial(qb_att, qb["int_rate"])})
    team = {"pass_att": pass_att, "carries": carries, "pass_yds": rec_yds_all,
            "rush_yds": sum(v["rush_yds"] for v in res.values()), "points": team_pts}
    return res, team


DIST_Q = np.linspace(0, 100, 101)   # percentiles kept for pricing threshold markets ("70+ yards")


def _quant(x):
    return [round(float(v), 1) for v in np.percentile(x, DIST_Q)]


def _summ(x, lo=25, hi=75):
    return {"median": float(np.median(x)), "p25": float(np.percentile(x, lo)), "p75": float(np.percentile(x, hi))}


def simulate_game(game: dict, home: dict, away: dict, vol: dict, lg: dict, n: int = N_SIMS, seed: int = SEED):
    """game: pred_margin (home), pred_total. home/away: {'prof','defense','players'}."""
    rng = np.random.default_rng(seed)
    margin = rng.normal(game["pred_margin"], MARGIN_SD, n)
    total = np.maximum(rng.normal(game["pred_total"], TOTAL_SD, n), 6)
    hp, ap = np.maximum((total + margin) / 2, 0), np.maximum((total - margin) / 2, 0)
    out = {}
    for side, pts, mg, me, opp in (("home", hp, margin, home, away), ("away", ap, -margin, away, home)):
        res, team = simulate_team(rng, pts, mg, me["prof"], opp["defense"], me["players"], vol, lg, opp["prof"])
        rows = []
        for p in me["players"]:
            r = res[p["id"]]
            any_td = (r["rec_td"] + r["rush_td"]) > 0
            row = {"id": p["id"], "name": p["name"], "pos": p["pos"], "questionable": p["questionable"],
                   "anytime_td": float(any_td.mean()),
                   "rec": _summ(r["rec"]), "rec_yds": _summ(r["rec_yds"]), "targets": _summ(r["targets"]),
                   "rush_att": _summ(r["carries"]), "rush_yds": _summ(r["rush_yds"]),
                   "rush_rec_yds": _summ(r["rush_yds"] + r["rec_yds"])}
            row["dist"] = {"rec": _quant(r["rec"]), "rec_yds": _quant(r["rec_yds"]),
                           "rush_att": _quant(r["carries"]), "rush_yds": _quant(r["rush_yds"]),
                           "rush_rec_yds": _quant(r["rush_yds"] + r["rec_yds"])}
            tds = r["rec_td"] + r["rush_td"]
            row["dist"]["tds"] = _quant(tds)
            row["td_2plus"] = float((tds >= 2).mean())
            row["td_3plus"] = float((tds >= 3).mean())
            row["first_td_share"] = None
            if "pass_yds" in r:
                row["dist"].update({"pass_yds": _quant(r["pass_yds"]), "pass_td": _quant(r["pass_td"]),
                                    "comp": _quant(r["comp"]), "pass_att": _quant(r["pass_att"]),
                                    "pass_rush_yds": _quant(r["pass_yds"] + r["rush_yds"]), "ints": _quant(r["ints"])})
                row.update({"pass_att": _summ(r["pass_att"]), "comp": _summ(r["comp"]), "pass_yds": _summ(r["pass_yds"]),
                            "pass_td": _summ(r["pass_td"]), "ints": _summ(r["ints"]),
                            "pass_td_2plus": float((r["pass_td"] >= 2).mean()),
                            "pass_rush_yds": _summ(r["pass_yds"] + r["rush_yds"])})
            rows.append(row)
        out[side] = {"players": rows, "team": {k: _summ(v) for k, v in team.items()}}
    return out


def raw_samples(game, home, away, vol, lg, n=N_SIMS, seed=SEED):
    """Same simulation, returning raw per-sim arrays (used by the backtest)."""
    rng = np.random.default_rng(seed)
    margin = rng.normal(game["pred_margin"], MARGIN_SD, n)
    total = np.maximum(rng.normal(game["pred_total"], TOTAL_SD, n), 6)
    hp, ap = np.maximum((total + margin) / 2, 0), np.maximum((total - margin) / 2, 0)
    out = {}
    for side, pts, mg, me, opp in (("home", hp, margin, home, away), ("away", ap, -margin, away, home)):
        res, _ = simulate_team(rng, pts, mg, me["prof"], opp["defense"], me["players"], vol, lg, opp["prof"])
        out[side] = res
    return out


# ---------------------------------------------------------------- weekly run
def project_week(sched: pd.DataFrame, week_games: pd.DataFrame, refresh: bool = False) -> dict:
    """Simulate every upcoming game this week. week_games needs game_id, pred_margin, pred_total."""
    from players import build_player_tables, build_team_pace, player_info
    season = int(sched.season.max())
    pg, tm = build_player_tables(season, refresh=refresh)
    build_team_pace(season)     # pass rate over expected and tempo, current season refreshed
    DEPTH_STARTERS.clear()
    DEPTH_STARTERS.update(load_depth_starters(season, refresh=refresh))
    pg, tm = prepare(pg, tm, sched, player_info())
    lg = league_rates(tm[tm.season >= season - 3], pg[pg.season >= season - 3])
    vol = fit_volume(tm[(tm.season >= season - VOLUME_SEASONS) & (tm.season < season)])
    out = {}
    for g in week_games.itertuples():
        row = sched[sched.game_id == g.game_id].iloc[0]
        cut = row.gameday
        pgh, tmh = pg[pg.gameday < cut], tm[tm.gameday < cut]
        inj = injury_status(int(row.season), int(row.week))
        roster = load_roster(int(row.season), int(row.week), refresh=refresh)
        home = build_side(pgh, tmh, row.home_team, row.away_team, row.home_qb_id, inj, lg, roster, True)
        away = build_side(pgh, tmh, row.away_team, row.home_team, row.away_qb_id, inj, lg, roster, True)
        out[g.game_id] = simulate_game({"pred_margin": float(g.pred_margin), "pred_total": float(g.pred_total)},
                                       home, away, vol, lg)
        out[g.game_id]["n_sims"] = N_SIMS
    return out
