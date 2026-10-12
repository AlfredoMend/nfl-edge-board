"""Download nflverse data and build a compact team-game stats table.

Play-by-play files are large, so historical seasons are aggregated once into
data/team_games.csv and only the current season is re-downloaded each week.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FIRST_SEASON = 2006


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
    g = g[(g.season >= FIRST_SEASON)].copy()
    g["gameday"] = pd.to_datetime(g.gameday)
    return g


def _pbp(season: int, refresh: bool) -> pd.DataFrame:
    path = DATA / "raw" / f"pbp_{season}.parquet"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(f"{BASE}/pbp/play_by_play_{season}.parquet"))
    cols = ["game_id", "posteam", "defteam", "epa", "success", "pass", "rush",
            "play_type", "season_type", "wp", "qtr", "id", "qb_dropback"]
    return pd.read_parquet(path, columns=cols)


def aggregate_season(season: int, refresh: bool = False) -> pd.DataFrame:
    """One row per team per game with offensive and defensive efficiency."""
    p = _pbp(season, refresh)
    p = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.posteam.notna()]
    # Garbage time filter: drop plays where the game is basically decided.
    p = p[(p.wp.between(0.05, 0.95)) | (p.qtr <= 2)]
    p = p.assign(pass_epa=p.epa.where(p["pass"] == 1), rush_epa=p.epa.where(p["rush"] == 1))
    off = p.groupby(["game_id", "posteam"]).agg(
        off_epa=("epa", "mean"),
        off_sr=("success", "mean"),
        off_pass_epa=("pass_epa", "mean"),
        off_rush_epa=("rush_epa", "mean"),
        off_plays=("epa", "size"),
    ).reset_index().rename(columns={"posteam": "team"})
    dfn = p.groupby(["game_id", "defteam"]).agg(
        def_epa=("epa", "mean"),
        def_sr=("success", "mean"),
        def_pass_epa=("pass_epa", "mean"),
        def_rush_epa=("rush_epa", "mean"),
    ).reset_index().rename(columns={"defteam": "team"})
    out = off.merge(dfn, on=["game_id", "team"], how="outer")
    out["season"] = season
    return out


def aggregate_qbs(season: int) -> pd.DataFrame:
    """Dropback EPA per quarterback per game (no garbage-time filter needed)."""
    p = _pbp(season, refresh=False)
    p = p[(p.qb_dropback == 1) & p.epa.notna() & p.id.notna()]
    q = p.groupby(["game_id", "posteam", "id"]).agg(epa_sum=("epa", "sum"), dropbacks=("epa", "size"))
    q = q.reset_index().rename(columns={"posteam": "team", "id": "qb_id"})
    q["season"] = season
    return q


def build_team_games(current_season: int) -> pd.DataFrame:
    """Reuse cached history and refresh only the current season."""
    cache = DATA / "team_games.csv"
    hist = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    have = set(hist.season.unique()) if len(hist) else set()
    frames = [hist[hist.season != current_season]] if len(hist) else []
    for s in range(FIRST_SEASON, current_season + 1):
        if s == current_season or s not in have:
            print(f"aggregating pbp {s}")
            frames.append(aggregate_season(s, refresh=(s == current_season)))
    tg = pd.concat(frames, ignore_index=True).sort_values(["season", "game_id", "team"])
    tg.to_csv(cache, index=False)

    qcache = DATA / "qb_games.csv"
    qh = pd.read_csv(qcache) if qcache.exists() else pd.DataFrame()
    qhave = set(qh.season.unique()) if len(qh) else set()
    qframes = [qh[qh.season != current_season]] if len(qh) else []
    for s in range(FIRST_SEASON, current_season + 1):
        if s == current_season or s not in qhave:
            qframes.append(aggregate_qbs(s))
    qg = pd.concat(qframes, ignore_index=True).sort_values(["season", "game_id"])
    qg.to_csv(qcache, index=False)
    return tg


if __name__ == "__main__":
    sched = load_schedule()
    season = int(sched.season.max())
    tg = build_team_games(season)
    print(tg.shape, tg.season.min(), tg.season.max())
