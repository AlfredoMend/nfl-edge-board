# NFL Edge Board

An unbiased NFL prediction model (win probability, point spread, total) built on situation-neutral efficiency metrics, compared against sportsbook closing lines and live Kalshi prices, with a website that updates itself.

Live site: https://alfredomend.github.io/nfl-edge-board/

## Layout

| File | Job |
|---|---|
| `src/data.py` | Downloads nflfastR play-by-play and schedules (nflverse releases), applies the situation-neutral filter, aggregates per team per game |
| `src/features.py` | Exponential moving averages, opponent adjustment, red-zone regression, QB isolation, rest, travel, weather |
| `src/injuries.py` | Official injury reports weighted by each player's recent snap share, by position group |
| `src/weather.py` | Kickoff forecasts for upcoming outdoor games (Open-Meteo) |
| `src/players.py` | Player box scores from play-by-play (official counting rules) and team volume |
| `src/props.py` | Player prop projections: usage x efficiency, 10,000-game Monte Carlo per game, medians |
| `src/props_backtest.py` | Walk-forward calibration check of the prop medians on a past season |
| `src/record.py` | Results tracker: logs pregame picks every run, grades picks and Kalshi paper bets (P&L after fees, closing line value) once games are final |
| `src/geo.py` | Stadium coordinates and time zones for travel and body-clock features |
| `src/train.py` | Base models (logistic, XGBoost, LightGBM, ridge, extra trees, neural net, Elo), neutral-site stacking, home field, totals |
| `src/backtest.py` | Walk-forward time-series evaluation against closing lines, closing line value on Kalshi |
| `src/run.py` | Runs everything and writes `site/data/*.json` |
| `src/kalshi.py` | Snapshots Kalshi moneyline and spread prices and prices them with the model |
| `.github/workflows/update.yml` | Runs every 30 minutes (external timer via cron-job.org, GitHub's schedule as backup) and on every code push |

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
| Win probability Brier | 0.2168 | 0.2112 |
| Win probability log loss | 0.6224 | 0.6099 |
| Straight-up picks correct | 65.2% | 66.5% |
| Spread RMSE (points) | 13.19 | 12.94 |
| Total RMSE (points) | 13.48 | 13.26 |
| Against the spread | 49.2% (52.4% needed) | |
| Over/under | 51.3% (52.4% needed) | |

Closing lines remain sharper, as expected. Closing line value against Kalshi is tracked from `data/clv_log.csv` as games are logged.

## Hybrid choices (tested walk-forward, 2012 to now)

The strict spec version scored a Brier of 0.2189. Three changes, each tested out of sample, brought it to 0.2175:

- **Broad-window EPA added next to the neutral stats.** EPA per play, per pass, per rush and success rate over every play outside clear garbage time (win probability 5% to 95%, or the first half). The neutral 20% to 80% set is still used.
- **QB rating from every dropback.** The quarterback signal is noisy, and more dropbacks per game outweigh the small game-script bias.
- **Compact inputs for the linear models.** Logistic regression and the neural net use a 10-feature core (Elo, QB, opponent-adjusted EPA, broad EPA and success rate, rest). The tree models use every feature.

Tested and rejected: a wider 10% to 90% neutral window, recency-weighted training, a looser QB prior, and a wider home-field range (no gain).

## Accuracy upgrades (tested walk-forward, kept only if they helped)

| Change | Brier |
|---|---|
| Hybrid model before these upgrades | 0.2175 |
| + Injuries: Out/Doubtful players weighted by recent snap share (tree models get each position group, linear models get offense and defense totals) | 0.2172 |
| + Backup and rookie QB priors from draft slot (estimated from 2006-2011 rookies only) | 0.2171 |
| + Tuned model settings: depth-1 XGBoost, simpler LightGBM, lighter logistic regularization, stronger ridge | **0.2168** |

Also added for live games, which the backtest cannot measure: kickoff weather forecasts (nflverse only records weather after a game) and correct handling of retractable roofs, which have no roof value before kickoff.

Tested and not kept: 10 mph wind threshold, injury position groups in the linear models, different offseason carryover for Elo, team form and QB ratings, recency-weighted training.

## Player props

Each upcoming game is simulated 10,000 times. A simulation draws the final margin and total from the game model, sets each team's pass attempts and carries from its recent volume and the simulated game script, splits targets and carries among active players by recent usage (Out and Doubtful players removed and their usage redistributed), draws catches from catch rates and yards from real per-catch and per-carry NFL yardage distributions scaled to the player and opponent, and hands out touchdowns from the simulated score by red-zone usage and touchdown history. QB passing is the sum of his receivers, so the numbers agree. Every stat is reported as the **median** of the simulations (the fair line for a half-point prop), with the middle 50% range and anytime-TD probability.

Walk-forward check on the 2025 regular season (weeks 3 to 18, every projection made only from earlier games, that week's injury report and the out-of-sample game model). Touchdown and spread calibration was set on 2024 and not re-tuned on 2025.

| Market | Projections | Actual over median | Actual under median | In middle 50% (target 50%) | Error, model | Error, last-4 avg |
|---|---|---|---|---|---|---|
| Passing yards | 471 | 50.5% | 49.5% | 53% | 58.5 | 68.6 |
| Completions | 471 | 45.6% | 47.6% | 50% | 4.7 | 5.7 |
| Pass attempts | 471 | 45.4% | 50.1% | 46% | 6.7 | 8.1 |
| Passing TDs | 471 | 36.1% | 33.3% | 58% | 0.9 | 1.0 |
| Rushing yards | 1,485 | 56.3% | 40.9% | 48% | 19.2 | 20.4 |
| Rush attempts | 1,485 | 52.7% | 33.9% | 48% | 3.2 | 3.2 |
| Receptions | 3,218 | 42.3% | 35.1% | 49% | 1.4 | 1.5 |
| Receiving yards | 3,218 | 52.7% | 45.2% | 48% | 18.7 | 20.7 |

Count stats land exactly on the median often, so over and under do not add to 100%. Anytime TD Brier 0.1256 vs 0.1410 for a flat rate.

**Usage shares.** Each player's share of targets and carries is weighted by how closely past games' lineups match this week's, and shares are concentrated slightly toward established starters (targets 1.15, carries 1.10) with 2% and 1% left for unlisted players. These were set on the 2024 backtest after it showed starters being under-projected (top receivers went over their median 60% of the time) and checked on 2025, where receiving catches and yards now average within 1% of actual and top receivers go over 48% of the time. Top running backs now lean slightly high on rushing yards (45% over, 55% under).

**Who plays.** Active players come from nflverse weekly rosters: anyone on the active roster with offensive history is included (players back from injured reserve, trades, returning starters), and anyone on injured reserve, the practice squad, cut or traded is excluded. Brand-new first-string players on the depth chart get position priors. Game-day inactives are used live (they post about 90 minutes before kickoff) but not in the backtest. On 2024 and 2025 this lowered the error on every market and the anytime-TD Brier (0.1306 to 0.1280 and 0.1297 to 0.1257).

**Team volume.** Pass attempts and carries come from each team's recent volume, its pass rate over expected (nflfastR `xpass`, competitive situations), its tempo (seconds between snaps), the opponent's tempo and what the opponent's defense allows, plus the simulated margin. Pass attempt error fell from 6.95 to 6.90 (2024) and 6.76 to 6.67 (2025). Top running backs still lean slightly under (actual goes over their median about 54 to 57% of the time).

**Kalshi props.** `src/kalshi_props.py` prices every open Kalshi NFL player prop (passing, rushing, receiving, rush + rec yards, attempts, completions, interceptions and the 1+/2+/3+ TD ladders) with the simulated distributions and computes the value of YES and NO after Kalshi's taker fee. The site lists contracts with 5 cents or more of value in a tight market and marks gaps of 15 cents or more "check news". Prices are logged every run to `data/kalshi_props_history.csv` so the edges can be graded once games are played.

## Run locally

```bash
pip install -r requirements.txt
python src/run.py --refresh
python src/kalshi.py
python -m http.server -d site
```
