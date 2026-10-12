# NFL Edge Board

An unbiased NFL prediction model (win probability, point spread, total) built on situation-neutral efficiency metrics, compared against sportsbook closing lines and live Kalshi prices, with a website that updates itself.

Live site: https://alfredomend.github.io/nfl-edge-board/

## Layout

| File | Job |
|---|---|
| `src/data.py` | Downloads nflfastR play-by-play and schedules (nflverse releases), applies the situation-neutral filter, aggregates per team per game |
| `src/features.py` | Exponential moving averages, opponent adjustment, red-zone regression, QB isolation, rest, travel, weather |
| `src/geo.py` | Stadium coordinates and time zones for travel and body-clock features |
| `src/train.py` | Base models (logistic, XGBoost, LightGBM, ridge, extra trees, neural net, Elo), neutral-site stacking, home field, totals |
| `src/backtest.py` | Walk-forward time-series evaluation against closing lines, closing line value on Kalshi |
| `src/run.py` | Runs everything and writes `site/data/*.json` |
| `src/kalshi.py` | Snapshots Kalshi moneyline and spread prices and prices them with the model |
| `.github/workflows/update.yml` | Runs hourly, every 30 min during games, and on every code push |

## Modeling rules

- **Prohibited inputs:** no passing or rushing yards per game, no points per game or points allowed per game, no raw turnover totals.
- **No leakage:** every feature for a game is built only from earlier games. The backtest predicts each season with models, stacking weights and home-field values fit only on earlier seasons.
- **Situation-neutral:** efficiency stats use only plays with win probability between 20% and 80%, or quarters 1 to 3 with the score within 14.
- **Volatile outcomes regressed:** turnovers are modeled through turnover-worthy proxies (passes intercepted or broken up per attempt, all fumbles per play, forced fumbles per play). Red-zone touchdown rate is shrunk toward the prior season's league median with 20 trips of prior weight.

## Features

- **Efficiency:** EPA per pass and per rush, success rate, early-down (1st and 2nd) EPA, explosive play rate (EPA above 1.5), offense and defense
- **Passing:** CPOE, average depth of target, yards after catch per target, pressure rate and EPA under pressure vs clean pocket (pressure = sack or QB hit, the best signal available in nflfastR), starting QB EPA per dropback and its gap to the team's passing baseline
- **Transformations:** 7-game exponential moving average weighted by neutral plays, regressed halfway to league average each offseason. Weekly ridge regression of EPA on offense and defense identity for opponent-adjusted ratings.
- **Context:** rest difference, travel miles, time-zone shift and eastward travel, wind above 15 mph, temperature below 25°F, division and playoff games
- **Home field:** models are trained on each game in both directions (home and away swapped), so they learn neutral-site strength. Home field is then added as a single term chosen from 1.2 to 1.8 points by out-of-sample log loss.

## Backtest (2012 to now, 3,893 games, all out of sample)

| | Model | Closing line |
|---|---|---|
| Win probability Brier | 0.2175 | 0.2112 |
| Win probability log loss | 0.6239 | 0.6099 |
| Straight-up picks correct | 64.8% | 66.5% |
| Spread RMSE (points) | 13.19 | 12.94 |
| Total RMSE (points) | 13.50 | 13.26 |
| Against the spread | 49.6% (52.4% needed) | |
| Over/under | 50.6% (52.4% needed) | |

Closing lines remain sharper, as expected. Closing line value against Kalshi is tracked from `data/clv_log.csv` as games are logged.

## Hybrid choices (tested walk-forward, 2012 to now)

The strict spec version scored a Brier of 0.2189. Three changes, each tested out of sample, brought it to 0.2175:

- **Broad-window EPA added next to the neutral stats.** EPA per play, per pass, per rush and success rate over every play outside clear garbage time (win probability 5% to 95%, or the first half). The neutral 20% to 80% set is still used.
- **QB rating from every dropback.** The quarterback signal is noisy, and more dropbacks per game outweigh the small game-script bias.
- **Compact inputs for the linear models.** Logistic regression and the neural net use a 10-feature core (Elo, QB, opponent-adjusted EPA, broad EPA and success rate, rest). The tree models use every feature.

Tested and rejected: a wider 10% to 90% neutral window, recency-weighted training, a looser QB prior, and a wider home-field range (no gain).

## Run locally

```bash
pip install -r requirements.txt
python src/run.py --refresh
python src/kalshi.py
python -m http.server -d site
```
