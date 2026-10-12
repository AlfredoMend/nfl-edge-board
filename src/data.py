"""Fetch nflfastR play-by-play (via nflverse releases) and build per-team-game tables.

Everything here is situation-neutral: efficiency stats only use plays where the game
is still competitive (win probability 20-80%, or quarters 1-3 within 14 points), which
strips out garbage time and prevent-defense game script.

No raw yardage, points-per-game or raw turnover totals are produced. Turnovers are
represented by turnover-worthy proxies (interceptions plus pass breakups, and all
fumbles regardless of who recovered), and red-zone results are kept as counts so
they can be regressed toward the league median later.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FIRST_SEASON = 2006
CACHE_VERSION = 2  # bump when the aggregation changes so caches rebuild

PBP_COLS = ["game_id", "posteam", "defteam", "epa", "success", "pass", "rush", "play_type",
            "wp", "qtr", "down", "score_differential", "id", "qb_dropback", "cpoe", "air_yards",
            "yards_after_catch", "pass_attempt", "sack", "qb_hit", "interception", "fumble",
            "fumble_forced", "pass_defense_1_player_id", "receiver_player_id", "fixed_drive",
            "fixed_drive_result", "yardline_100"]


def fetch(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def load_schedule() -> pd.DataFrame:
    raw = DATA / "raw" / "games.csv"
    if "--refresh" in sys.argv or not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(fetch(f"{BASE}/schedules/games.csv"))
    g = pd.read_csv(raw)
    g = g[g.season >= FIRST_SEASON].copy()
    g["gameday"] = pd.to_datetime(g.gameday)
    return g


def load_pbp(season: int, refresh: bool = False) -> pd.DataFrame:
    path = DATA / "raw" / f"pbp_{season}.parquet"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(f"{BASE}/pbp/play_by_play_{season}.parquet"))
    return pd.read_parquet(path, columns=PBP_COLS)


def neutral(p: pd.DataFrame) -> pd.DataFrame:
    """Competitive-situation plays only."""
    wp_ok = p.wp.between(0.20, 0.80)
    close_early = (p.qtr <= 3) & (p.score_differential.abs() <= 14)
    return p[wp_ok | close_early]


def _side_stats(p: pd.DataFrame, side: str) -> pd.DataFrame:
    """Efficiency for the offense (side='posteam') or what a defense allowed (side='defteam')."""
    is_pass, is_rush = p["pass"] == 1, p["rush"] == 1
    dropback = p.qb_dropback == 1
    target = (p.pass_attempt == 1) & p.receiver_player_id.notna()
    pressured = dropback & ((p.sack == 1) | (p.qb_hit == 1))
    q = p.assign(
        pass_epa=p.epa.where(is_pass), rush_epa=p.epa.where(is_rush),
        early_epa=p.epa.where(p.down.isin([1, 2])),
        explosive=(p.epa > 1.5).astype(float),
        cpoe_v=p.cpoe.where(p.pass_attempt == 1),
        adot_v=p.air_yards.where(target),
        yac_v=p.yards_after_catch.fillna(0).where(target),
        press=pressured.astype(float).where(dropback),
        press_epa=p.epa.where(pressured), clean_epa=p.epa.where(dropback & ~pressured),
        # turnover-worthy proxy: passes intercepted or broken up
        twp=((p.interception == 1) | p.pass_defense_1_player_id.notna()).astype(float).where(p.pass_attempt == 1),
        fum=(p.fumble == 1).astype(float),
        ffum=(p.fumble_forced == 1).astype(float),
    )
    return q.groupby(["game_id", side]).agg(
        epa=("epa", "mean"), sr=("success", "mean"),
        pass_epa=("pass_epa", "mean"), rush_epa=("rush_epa", "mean"),
        early_epa=("early_epa", "mean"), explosive=("explosive", "mean"),
        cpoe=("cpoe_v", "mean"), adot=("adot_v", "mean"), yac_tgt=("yac_v", "mean"),
        press_rate=("press", "mean"), press_epa=("press_epa", "mean"), clean_epa=("clean_epa", "mean"),
        twp_rate=("twp", "mean"), fum_rate=("fum", "mean"), ffum_rate=("ffum", "mean"),
        n_plays=("epa", "size"),
    ).reset_index().rename(columns={side: "team"})


def _red_zone(p: pd.DataFrame) -> pd.DataFrame:
    """Red-zone trips and touchdowns per offense per game (all situations)."""
    d = p[p.posteam.notna() & p.fixed_drive.notna()]
    drives = d.groupby(["game_id", "posteam", "fixed_drive"]).agg(
        rz=("yardline_100", lambda s: bool((s <= 20).any())),
        td=("fixed_drive_result", lambda s: s.iloc[-1] == "Touchdown"))
    drives = drives[drives.rz]
    out = drives.groupby(["game_id", "posteam"]).agg(rz_trips=("rz", "size"), rz_tds=("td", "sum"))
    return out.reset_index().rename(columns={"posteam": "team"})


def aggregate_season(season: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_pbp(season, refresh)
    plays = raw[raw.play_type.isin(["pass", "run"]) & raw.epa.notna() & raw.posteam.notna()]
    n = neutral(plays)
    off = _side_stats(n, "posteam").add_prefix("off_").rename(columns={"off_game_id": "game_id", "off_team": "team"})
    dfn = _side_stats(n, "defteam").add_prefix("def_").rename(columns={"def_game_id": "game_id", "def_team": "team"})
    pace = plays.groupby(["game_id", "posteam"]).size().rename("plays_total").reset_index().rename(columns={"posteam": "team"})
    tg = (off.merge(dfn, on=["game_id", "team"], how="outer")
             .merge(pace, on=["game_id", "team"], how="left")
             .merge(_red_zone(raw), on=["game_id", "team"], how="left"))
    tg[["rz_trips", "rz_tds"]] = tg[["rz_trips", "rz_tds"]].fillna(0)
    tg["season"] = season

    db = n[(n.qb_dropback == 1) & n.id.notna()]
    qg = db.groupby(["game_id", "posteam", "id"]).agg(epa_sum=("epa", "sum"), dropbacks=("epa", "size"))
    qg = qg.reset_index().rename(columns={"posteam": "team", "id": "qb_id"})
    qg["season"] = season
    return tg, qg


def build_tables(current_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse cached seasons and refresh only the current one."""
    tcache, qcache = DATA / "team_games.csv", DATA / "qb_games.csv"
    vfile = DATA / "cache_version.txt"
    fresh = not vfile.exists() or vfile.read_text().strip() != str(CACHE_VERSION)
    th = pd.read_csv(tcache) if tcache.exists() and not fresh else pd.DataFrame()
    qh = pd.read_csv(qcache) if qcache.exists() and not fresh else pd.DataFrame()
    have = set(th.season.unique()) if len(th) else set()
    tf = [th[th.season != current_season]] if len(th) else []
    qf = [qh[qh.season != current_season]] if len(qh) else []
    for s in range(FIRST_SEASON, current_season + 1):
        if s == current_season or s not in have:
            print(f"aggregating pbp {s}", flush=True)
            t, q = aggregate_season(s, refresh=(s == current_season))
            tf.append(t)
            qf.append(q)
    tg = pd.concat(tf, ignore_index=True).sort_values(["season", "game_id", "team"])
    qg = pd.concat(qf, ignore_index=True).sort_values(["season", "game_id"])
    tg.to_csv(tcache, index=False)
    qg.to_csv(qcache, index=False)
    vfile.write_text(str(CACHE_VERSION))
    return tg, qg


if __name__ == "__main__":
    sched = load_schedule()
    tg, qg = build_tables(int(sched.season.max()))
    print(tg.shape, qg.shape)
    print(tg.describe().T.round(3).to_string())
