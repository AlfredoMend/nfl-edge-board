"""Injury features: how much of each team's regular playing time is missing for this game.

For every player listed Out or Doubtful on the official injury report, take his share of
snaps from his recent appearances (offense snaps for offensive players, defense snaps for
defensive players), fade it the longer he has already been out (team form already reflects a
long absence), and add it up by position group. Questionable players are counted separately
at their snap share, since most of them play.

Everything uses information available before kickoff: the injury report for that week and
snap counts from earlier games only. QBs are skipped because the starting-QB rating covers them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FIRST_INJ, FIRST_SNAP = 2012, 2013
CACHE = ROOT / "data" / "injury_games.csv"
CACHE_VERSION = "2"

GROUPS = {
    "ol": {"T", "G", "C", "OL", "OT", "OG"},
    "rec": {"WR", "TE"},
    "rb": {"RB", "FB", "HB"},
    "dl": {"DE", "DT", "DL", "NT"},
    "lb": {"LB", "ILB", "OLB", "MLB"},
    "db": {"CB", "S", "DB", "FS", "SS", "SAF"},
}
OFFENSE = {"ol", "rec", "rb"}
RECENT_APPEARANCES = 3     # average snap share over a player's last few games
ABSENCE_HALFLIFE = 2.0     # weeks; how fast an ongoing absence stops counting as news
STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.9}
INJ_COLS = [f"inj_{g}" for g in GROUPS] + ["inj_q_off", "inj_q_def"]


def _get(name: str, url: str, refresh: bool) -> Path:
    path = RAW / name
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def _group(pos: str):
    for g, s in GROUPS.items():
        if pos in s:
            return g
    return None


def _player_map(refresh: bool) -> pd.DataFrame:
    p = pd.read_parquet(_get("players.parquet", f"{BASE}/players/players.parquet", refresh),
                        columns=["gsis_id", "pfr_id"])
    return p.dropna().drop_duplicates("pfr_id")


def _snap_history(seasons, sched: pd.DataFrame, current: int, refresh: bool) -> pd.DataFrame:
    """One row per player appearance with his snap share and a global week index."""
    frames = []
    for s in seasons:
        if s < FIRST_SNAP:
            continue
        f = _get(f"snap_counts_{s}.parquet", f"{BASE}/snap_counts/snap_counts_{s}.parquet", refresh and s == current)
        frames.append(pd.read_parquet(f, columns=["game_id", "season", "week", "pfr_player_id", "position",
                                                  "team", "offense_pct", "defense_pct"]))
    if not frames:
        return pd.DataFrame()
    sn = pd.concat(frames, ignore_index=True)
    sn["group"] = sn.position.map(_group)
    sn = sn[sn.group.notna()]
    sn["share"] = np.where(sn.group.isin(OFFENSE), sn.offense_pct, sn.defense_pct).astype(float)
    sn = sn[sn.share > 0]
    sn = sn.merge(_player_map(refresh), left_on="pfr_player_id", right_on="pfr_id", how="inner")
    sn["t"] = sn.season * 25 + sn.week
    return sn[["gsis_id", "t", "share", "group"]].sort_values(["gsis_id", "t"])


def _recent_share(snaps: pd.DataFrame, asks: pd.DataFrame) -> np.ndarray:
    """For each (gsis_id, t) in asks: mean snap share over the player's last few appearances
    strictly before t, faded by weeks since his last appearance."""
    out = np.zeros(len(asks))
    if snaps.empty:
        return out
    by_player = {k: (d.t.to_numpy(), d.share.to_numpy()) for k, d in snaps.groupby("gsis_id")}
    for i, (pid, t) in enumerate(zip(asks.gsis_id.to_numpy(), asks.t.to_numpy())):
        rec = by_player.get(pid)
        if rec is None:
            continue
        ts, sh = rec
        k = np.searchsorted(ts, t)          # appearances before this week only
        if k == 0:
            continue
        lo = max(0, k - RECENT_APPEARANCES)
        gap = max(t - ts[k - 1] - 1, 0)      # weeks missed since last appearance
        out[i] = sh[lo:k].mean() * 0.5 ** (gap / ABSENCE_HALFLIFE)
    return out


def build_injury_table(sched: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Per (game_id, team): missing snap share by position group. Cached; current season refreshed."""
    current = int(sched.season.max())
    tag = f"{CACHE_VERSION}|{RECENT_APPEARANCES}|{ABSENCE_HALFLIFE}"
    old = pd.DataFrame()
    if CACHE.exists() and (ROOT / "data" / "injury_cache_version.txt").exists() \
            and (ROOT / "data" / "injury_cache_version.txt").read_text().strip() == tag:
        old = pd.read_csv(CACHE)
        old = old[old.season < current]
    have = set(old.season.unique()) if len(old) else set()
    need = [s for s in range(FIRST_INJ, current + 1) if s not in have]

    # Snap history must include the season before each needed season (for week-1 shares)
    snap_seasons = sorted({x for s in need for x in (s - 1, s)})
    snaps = _snap_history(snap_seasons, sched, current, refresh)

    rows = []
    for s in need:
        try:
            f = _get(f"injuries_{s}.parquet", f"{BASE}/injuries/injuries_{s}.parquet", refresh and s == current)
        except requests.HTTPError:
            continue
        inj = pd.read_parquet(f, columns=["season", "week", "team", "gsis_id", "position", "report_status"])
        inj = inj[inj.report_status.isin(["Out", "Doubtful", "Questionable"]) & inj.gsis_id.notna()]
        inj["group"] = inj.position.map(_group)
        inj = inj[inj.group.notna()].drop_duplicates(["season", "week", "team", "gsis_id"])
        inj["t"] = inj.season * 25 + inj.week
        inj["share"] = _recent_share(snaps, inj)
        w = inj.report_status.map(STATUS_WEIGHT).fillna(0.0)
        inj["missing"] = inj.share * w
        inj["q"] = np.where(inj.report_status == "Questionable", inj.share, 0.0)
        agg = inj.pivot_table(index=["season", "week", "team"], columns="group", values="missing",
                              aggfunc="sum", fill_value=0.0)
        agg = agg.reindex(columns=list(GROUPS), fill_value=0.0).add_prefix("inj_")
        q = inj.assign(off=inj.group.isin(OFFENSE)).groupby(["season", "week", "team", "off"]).q.sum().unstack(fill_value=0.0)
        q = q.reindex(columns=[True, False], fill_value=0.0).rename(columns={True: "inj_q_off", False: "inj_q_def"})
        rows.append(agg.join(q, how="outer").fillna(0.0).reset_index())

    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["season", "week", "team"] + INJ_COLS)
    tab = pd.concat([old, new], ignore_index=True)
    tab.to_csv(CACHE, index=False)
    (ROOT / "data" / "injury_cache_version.txt").write_text(tag)
    return tab


def add_injuries(g: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    """Attach home and away injury values. Seasons without reports get zeros plus a flag."""
    g = g.copy()
    for side in ("home", "away"):
        t = tab.rename(columns={"team": f"{side}_team", **{c: f"{side}_{c}" for c in INJ_COLS}})
        g = g.merge(t, on=["season", "week", f"{side}_team"], how="left")
    g["inj_known"] = (g.season >= FIRST_SNAP).astype(int)
    for c in INJ_COLS:
        g[[f"home_{c}", f"away_{c}"]] = g[[f"home_{c}", f"away_{c}"]].fillna(0.0)
        g[f"d_{c}"] = g[f"home_{c}"] - g[f"away_{c}"]
    for side in ("home", "away"):
        g[f"{side}_inj_off_total"] = g[f"{side}_inj_ol"] + g[f"{side}_inj_rec"] + g[f"{side}_inj_rb"]
        g[f"{side}_inj_def_total"] = g[f"{side}_inj_dl"] + g[f"{side}_inj_lb"] + g[f"{side}_inj_db"]
    g["d_inj_off"] = g.d_inj_ol + g.d_inj_rec + g.d_inj_rb
    g["d_inj_def"] = g.d_inj_dl + g.d_inj_lb + g.d_inj_db
    g["sum_inj_off"] = sum(g[f"{s}_inj_{x}"] for s in ("home", "away") for x in ("ol", "rec", "rb"))
    g["sum_inj_def"] = sum(g[f"{s}_inj_{x}"] for s in ("home", "away") for x in ("dl", "lb", "db"))
    return g


def injury_notes(sched: pd.DataFrame, season: int, week: int, min_share: float = 0.4) -> dict:
    """Regular players listed Out, Doubtful or Questionable this week, for display on the site."""
    path = RAW / f"injuries_{season}.parquet"
    if not path.exists():
        return {}
    inj = pd.read_parquet(path, columns=["season", "week", "team", "gsis_id", "full_name", "position", "report_status"])
    inj = inj[(inj.week == week) & inj.report_status.isin(["Out", "Doubtful", "Questionable"]) & inj.gsis_id.notna()]
    if inj.empty:
        return {}
    inj["t"] = inj.season * 25 + inj.week
    snaps = _snap_history([season - 1, season], sched, season, False)
    inj["share"] = _recent_share(snaps, inj)
    qb = inj.position == "QB"
    inj = inj[(inj.share >= min_share) | qb].sort_values("share", ascending=False)
    out = {}
    for team, d in inj.groupby("team"):
        out[team] = [{"name": r.full_name, "pos": r.position, "status": r.report_status,
                      "share": round(float(r.share), 2)} for r in d.itertuples()]
    return out


INJURY_FEATURES = ["d_inj_ol", "d_inj_rec", "d_inj_rb", "d_inj_dl", "d_inj_lb", "d_inj_db",
                   "d_inj_q_off", "d_inj_q_def", "d_inj_off", "d_inj_def"]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data import load_schedule
    s = load_schedule()
    tab = build_injury_table(s)
    print(tab.shape)
    print(tab.groupby("season")[INJ_COLS].mean().round(3).to_string())
