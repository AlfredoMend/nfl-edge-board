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
OTHER_TARGET, OTHER_CARRY = 0.04, 0.02   # usage left for players the model does not list
REDISTRIBUTE = 0.75         # share of a missing player's usage that goes to listed teammates
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


def _ema(values: np.ndarray, halflife: float, prior: float = None, prior_w: float = 0.0) -> float:
    if len(values) == 0:
        return prior if prior is not None else 0.0
    w = _decay_weights(len(values), halflife)
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
    """pass_att = a + b * recent_pass_att + c * margin (and the same for carries), fit by least
    squares on team games where the recent average is built only from earlier games."""
    rows = []
    for team, d in tm.groupby("team"):
        d = d.sort_values("gameday")
        for i in range(4, len(d)):
            h = d.iloc[max(0, i - 16):i]
            rows.append((_ema(h.pass_att.to_numpy(float), TEAM_HALFLIFE), _ema(h.carries.to_numpy(float), TEAM_HALFLIFE),
                         d.iloc[i].pass_att, d.iloc[i].carries, d.iloc[i].margin))
    X = np.array(rows)
    out = {}
    for name, col, y in (("pass", 0, 2), ("rush", 1, 3)):
        A = np.column_stack([np.ones(len(X)), X[:, col], X[:, 4]])
        coef, *_ = np.linalg.lstsq(A, X[:, y], rcond=None)
        resid = X[:, y] - A @ coef
        out[name] = {"coef": coef.tolist(), "sd": float(resid.std())}
    return out


# ---------------------------------------------------------------- team and player profiles
def team_profile(tm_hist: pd.DataFrame, team: str, lg: dict) -> dict:
    d = tm_hist[tm_hist.team == team].tail(16)
    if d.empty:
        return {"pass_att": 33.0, "carries": 26.0, "pass_td_frac": lg["pass_td_frac"]}
    tds = d.pass_td + d.rush_td
    w = _decay_weights(len(d), TEAM_HALFLIFE)
    ptd = ((w * d.pass_td).sum() + 6 * lg["pass_td_frac"]) / ((w * tds).sum() + 6)
    return {"pass_att": _ema(d.pass_att.to_numpy(float), TEAM_HALFLIFE),
            "carries": _ema(d.carries.to_numpy(float), TEAM_HALFLIFE), "pass_td_frac": float(ptd)}


def defense_profile(tm_hist: pd.DataFrame, team: str, lg: dict) -> dict:
    """What this defense allowed per target and per carry, relative to league, shrunk and damped."""
    d = tm_hist[tm_hist.opp == team].tail(16)
    if d.empty:
        return {"pass_mult": 1.0, "rush_mult": 1.0}
    w = _decay_weights(len(d), 8)
    ypt = ((w * d.pass_yds).sum() + 120 * lg["ypt"]) / ((w * d.targets).sum() + 120)
    ypc = ((w * d.rush_yds).sum() + 100 * lg["ypc"]) / ((w * d.carries).sum() + 100)
    return {"pass_mult": float(1 + 0.6 * (ypt / lg["ypt"] - 1)), "rush_mult": float(1 + 0.6 * (ypc / lg["ypc"] - 1))}


def player_profiles(pg_hist: pd.DataFrame, tm_hist: pd.DataFrame, team: str, qb_id: str,
                    injuries: dict, lg: dict, pos_eff: dict) -> list:
    """Active players for this team with usage shares and efficiency, before redistribution."""
    team_games = tm_hist[tm_hist.team == team].game_id.tail(2).tolist()
    recent = pg_hist[(pg_hist.team == team) & pg_hist.game_id.isin(team_games) & pg_hist.appeared]
    status = {r["id"]: r["status"] for r in injuries.get(team, [])}
    q_names = [r["id"] for r in injuries.get(team, []) if r["status"] == "Questionable"]
    ids = set(recent.player_id) | set(pg_hist[(pg_hist.team == team) & pg_hist.player_id.isin(q_names)].player_id)
    out = []
    for pid in ids:
        h = pg_hist[(pg_hist.player_id == pid) & pg_hist.appeared].tail(40)
        if h.empty:
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
    return [p for p in out if p["tgt"] > 0.005 or p["car"] > 0.005 or p["pos"] == "QB"]


def _profile(pid, h, pos, lg, pos_eff, questionable):
    pt, pc, pcr = POS_PRIOR[pos]
    sh = h.tail(12)
    tgt = _ema(sh.tgt_sh.to_numpy(float), SHARE_HALFLIFE, pt, 0.5)
    car = _ema(sh.car_sh.to_numpy(float), SHARE_HALFLIFE, pc, 0.5)
    rzt = _ema(sh.rzt_sh.to_numpy(float), SHARE_HALFLIFE * 2, pt, 1.0)
    rzc = _ema(sh.rzc_sh.to_numpy(float), SHARE_HALFLIFE * 2, pc, 1.0)
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


def build_side(pg_hist, tm_hist, team, opp, qb_id, inj, lg):
    return {"prof": team_profile(tm_hist, team, lg), "defense": defense_profile(tm_hist, team, lg),
            "players": player_profiles(pg_hist, tm_hist, team, qb_id, inj, lg, None)}


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


def simulate_team(rng, team_pts, team_margin, prof, defense, players, vol, lg):
    n = len(team_pts)
    a, b, c = vol["pass"]["coef"]
    pass_att = np.maximum(np.rint(a + b * prof["pass_att"] + c * team_margin + rng.normal(0, vol["pass"]["sd"], n)), 12).astype(int)
    a, b, c = vol["rush"]["coef"]
    carries = np.maximum(np.rint(a + b * prof["carries"] + c * team_margin + rng.normal(0, vol["rush"]["sd"], n)), 8).astype(int)
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
        res, team = simulate_team(rng, pts, mg, me["prof"], opp["defense"], me["players"], vol, lg)
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
        res, _ = simulate_team(rng, pts, mg, me["prof"], opp["defense"], me["players"], vol, lg)
        out[side] = res
    return out


# ---------------------------------------------------------------- weekly run
def project_week(sched: pd.DataFrame, week_games: pd.DataFrame, refresh: bool = False) -> dict:
    """Simulate every upcoming game this week. week_games needs game_id, pred_margin, pred_total."""
    from players import build_player_tables, player_info
    season = int(sched.season.max())
    pg, tm = build_player_tables(season, refresh=refresh)
    pg, tm = prepare(pg, tm, sched, player_info())
    lg = league_rates(tm[tm.season >= season - 3], pg[pg.season >= season - 3])
    vol = fit_volume(tm[(tm.season >= season - VOLUME_SEASONS) & (tm.season < season)])
    out = {}
    for g in week_games.itertuples():
        row = sched[sched.game_id == g.game_id].iloc[0]
        cut = row.gameday
        pgh, tmh = pg[pg.gameday < cut], tm[tm.gameday < cut]
        inj = injury_status(int(row.season), int(row.week))
        home = build_side(pgh, tmh, row.home_team, row.away_team, row.home_qb_id, inj, lg)
        away = build_side(pgh, tmh, row.away_team, row.home_team, row.away_qb_id, inj, lg)
        out[g.game_id] = simulate_game({"pred_margin": float(g.pred_margin), "pred_total": float(g.pred_total)},
                                       home, away, vol, lg)
        out[g.game_id]["n_sims"] = N_SIMS
    return out
