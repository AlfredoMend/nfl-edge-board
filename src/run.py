"""Full weekly pipeline: data -> features -> backtest -> predictions -> site data."""
import json

import pandas as pd

from data import build_team_games, load_schedule
from features import build_features
from model import ROOT, run


def main():
    sched = load_schedule()
    season = int(sched.season.max())
    tg = build_team_games(season)
    qg = pd.read_csv(ROOT / "data" / "qb_games.csv")
    feats = build_features(sched, tg, qg)
    summary, preds = run(feats, season)
    print(json.dumps(summary["overall"], indent=1))
    print(json.dumps(summary["betting"], indent=1))
    print(preds[["game_id", "p_model", "p_market"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
