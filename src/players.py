"""Player box scores built from nflfastR play-by-play, plus team volume per game.

Counting rules match official stats:
- pass attempts = completions + incompletions + interceptions (no sacks, spikes or two-point tries)
- rushing includes QB scrambles and excludes kneels
- red-zone touches are targets or carries inside the opponent's 10
Appearances (did the player suit up and play offense) come from snap counts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from injuries import BASE, _get, _player_map

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "player_games.csv"
TEAM_CACHE = ROOT / "data" / "team_volume.csv"
FIRST = 2016
CACHE_VERSION = "1"

COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
        "complete_pass", "incomplete_pass", "interception", "sack", "qb_spike", "qb_kneel",
        "two_point_attempt", "passer_player_id", "rusher_player_id", "receiver_player_id",
        "passing_yards", "rushing_yards", "receiving_yards", "pass_touchdown", "rush_touchdown",
        "yardline_100", "touchdown", "td_team", "fumble_lost"]


def _season_tables(season: int):
    p = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{season}.parquet", columns=COLS)
    p = p[(p.two_point_attempt != 1) & p.posteam.notna()]
    rz = p.yardline_100 <= 10

    att = p[(p.complete_pass == 1) | (p.incomplete_pass == 1) | (p.interception == 1)]
    att = att[att.qb_spike != 1]
    passing = att.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        pass_att=("game_id", "size"), comp=("complete_pass", "sum"), pass_yds=("passing_yards", "sum"),
        pass_td=("pass_touchdown", "sum"), ints=("interception", "sum")).reset_index().rename(
        columns={"passer_player_id": "player_id"})

    tg = att[att.receiver_player_id.notna()].assign(rz_t=rz[att.index].astype(int))
    receiving = tg.groupby(["game_id", "posteam", "receiver_player_id"]).agg(
        targets=("game_id", "size"), rec=("complete_pass", "sum"), rec_yds=("receiving_yards", "sum"),
        rec_td=("pass_touchdown", "sum"), rz_targets=("rz_t", "sum")).reset_index().rename(
        columns={"receiver_player_id": "player_id"})

    ru = p[(p.play_type == "run") & p.rusher_player_id.notna() & (p.qb_kneel != 1)]
    ru = ru.assign(rz_c=rz[ru.index].astype(int))
    rushing = ru.groupby(["game_id", "posteam", "rusher_player_id"]).agg(
        carries=("game_id", "size"), rush_yds=("rushing_yards", "sum"), rush_td=("rush_touchdown", "sum"),
        rz_carries=("rz_c", "sum")).reset_index().rename(columns={"rusher_player_id": "player_id"})

    pg = passing.merge(receiving, on=["game_id", "posteam", "player_id"], how="outer") \
                .merge(rushing, on=["game_id", "posteam", "player_id"], how="outer").fillna(0)
    pg = pg.rename(columns={"posteam": "team"})
    pg["season"] = season

    # team volume
    team = pd.DataFrame({
        "pass_att": att.groupby(["game_id", "posteam"]).size(),
        "targets": tg.groupby(["game_id", "posteam"]).size(),
        "rz_targets": tg.groupby(["game_id", "posteam"]).rz_t.sum(),
        "carries": ru.groupby(["game_id", "posteam"]).size(),
        "rz_carries": ru.groupby(["game_id", "posteam"]).rz_c.sum(),
        "pass_yds": att.groupby(["game_id", "posteam"]).passing_yards.sum(),
        "rush_yds": ru.groupby(["game_id", "posteam"]).rushing_yards.sum(),
        "pass_td": att.groupby(["game_id", "posteam"]).pass_touchdown.sum(),
        "rush_td": ru.groupby(["game_id", "posteam"]).rush_touchdown.sum(),
    }).fillna(0).reset_index().rename(columns={"posteam": "team"})
    team["season"] = season
    return pg, team


def _appearances(seasons) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = ROOT / "data" / "raw" / f"snap_counts_{s}.parquet"
        if not path.exists():
            continue
        sn = pd.read_parquet(path, columns=["game_id", "pfr_player_id", "team", "offense_snaps", "offense_pct"])
        frames.append(sn[sn.offense_snaps > 0])
    if not frames:
        return pd.DataFrame(columns=["game_id", "team", "player_id", "off_pct"])
    sn = pd.concat(frames).merge(_player_map(False), left_on="pfr_player_id", right_on="pfr_id")
    return sn.rename(columns={"gsis_id": "player_id", "offense_pct": "off_pct"})[["game_id", "team", "player_id", "off_pct"]]


def build_player_tables(current: int, refresh: bool = False):
    """Cached player box scores and team volume; the current season is rebuilt each run."""
    vfile = ROOT / "data" / "player_cache_version.txt"
    ok = vfile.exists() and vfile.read_text().strip() == CACHE_VERSION and CACHE.exists() and TEAM_CACHE.exists()
    pg_old = pd.read_csv(CACHE) if ok else pd.DataFrame()
    tm_old = pd.read_csv(TEAM_CACHE) if ok else pd.DataFrame()
    if len(pg_old):
        pg_old, tm_old = pg_old[pg_old.season < current], tm_old[tm_old.season < current]
    have = set(pg_old.season.unique()) if len(pg_old) else set()
    pgs, tms = [pg_old], [tm_old]
    for s in range(FIRST, current + 1):
        if s in have:
            continue
        _get(f"snap_counts_{s}.parquet", f"{BASE}/snap_counts/snap_counts_{s}.parquet", refresh and s == current)
        pg, tm = _season_tables(s)
        app = _appearances([s])
        pg = pg.merge(app, on=["game_id", "team", "player_id"], how="outer")
        pg["season"] = s
        pg = pg.fillna({c: 0 for c in pg.columns if c not in ("game_id", "team", "player_id")})
        pgs.append(pg)
        tms.append(tm)
    pg = pd.concat(pgs, ignore_index=True)
    tm = pd.concat(tms, ignore_index=True)
    pg.to_csv(CACHE, index=False)
    tm.to_csv(TEAM_CACHE, index=False)
    vfile.write_text(CACHE_VERSION)
    return pg, tm


def player_info(refresh: bool = False) -> pd.DataFrame:
    p = pd.read_parquet(_get("players.parquet", f"{BASE}/players/players.parquet", refresh),
                        columns=["gsis_id", "display_name", "position"])
    return p.rename(columns={"gsis_id": "player_id", "display_name": "name"}).drop_duplicates("player_id")


if __name__ == "__main__":
    import time
    t = time.time()
    pg, tm = build_player_tables(2026)
    print(pg.shape, tm.shape, round(time.time() - t, 1), "s")
    info = player_info()
    x = pg[pg.season == 2025].groupby("player_id")[["pass_yds", "rush_yds", "rec_yds", "rec", "targets"]].sum()
    x = x.join(info.set_index("player_id").name)
    for c in ["pass_yds", "rush_yds", "rec_yds"]:
        print(x.sort_values(c).tail(3)[["name", c]].to_string())
