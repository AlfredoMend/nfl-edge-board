# NFL Edge Board

An NFL win-probability model built from scratch, compared against Vegas closing lines and live Kalshi prices, with a website that updates itself.

## What runs where

| Piece | File | When it runs |
|---|---|---|
| Data pull (nflverse schedules + play-by-play) | `src/data.py` | Tue, Thu, Sat, Sun mornings |
| Pregame features | `src/features.py` | same |
| Model, backtest, predictions | `src/model.py` | same |
| Kalshi price snapshot | `src/kalshi.py` | every 3 hours |
| Website | `site/index.html` + `site/data/*.json` | redeployed after every run |
| Scheduler | `.github/workflows/update.yml` | GitHub Actions |

Nothing runs on your laptop. GitHub Actions does the work and GitHub Pages hosts the site, both free for a public repo.

## Deploy (about 5 minutes, one time)

1. Create an empty **public** repo on GitHub, for example `nfl-edge-board`.
2. Push this folder to it:
   ```bash
   cd nfl-model
   git remote add origin https://github.com/<your-username>/nfl-edge-board.git
   git push -u origin main
   ```
3. In the repo go to **Settings → Pages** and set **Source** to **GitHub Actions**.
4. Go to **Actions → Update model, Kalshi prices, and website → Run workflow**.

When it finishes, the site is live at `https://<your-username>.github.io/nfl-edge-board/`. After that it updates on its own schedule.

No Kalshi API key is needed. The script only reads public market prices.

## Run locally

```bash
pip install -r requirements.txt
python src/run.py --refresh   # data, features, backtest, predictions
python src/kalshi.py          # Kalshi prices (needs normal internet access)
python -m http.server -d site # open http://localhost:8000
```

## The model

**Features** (all computed from games played before kickoff):
- Elo ratings with margin-of-victory scaling and offseason regression
- Exponentially decayed offensive and defensive EPA per play, success rate, pass EPA, rush EPA (garbage time removed)
- Matchup term: each offense against the opposing defense
- Starting QB rating: shrunk, decayed dropback EPA for the listed starter, so backups and rookies start near replacement level
- QB change flag, rest difference, neutral site, division game, playoffs, early season

**Models:** a stacked ensemble of six models (logistic regression, LightGBM, point-margin ridge regression, extra trees, a small neural net, Elo). Weights are non-negative, sum to 1, and are fit only on earlier seasons' out-of-sample predictions.

**Backtest:** walk-forward by season. Every season from 2012 on is predicted by a model trained only on earlier seasons, so all results are out of sample.

## First results (3,893 games, 2012 to Week 5 of 2026)

| | Brier ↓ | Log loss ↓ | Picks correct |
|---|---|---|---|
| Model (ensemble) | 0.2180 | 0.625 | 64.8% |
| Logistic only | 0.2176 | 0.624 | 64.9% |
| LightGBM only | 0.2201 | 0.630 | 65.0% |
| Elo only | 0.2204 | 0.631 | 64.5% |
| Vegas closing line (no-vig) | 0.2112 | 0.610 | 66.5% |

Paper betting every game where the model beat the no-vig line by 3+ points of probability: 2,699 bets, about **−3.8% ROI** at real moneyline odds.

The model is well calibrated and beats plain Elo, but closing lines are still sharper. That is the normal starting point. The Kalshi comparison is where it gets interesting, because Kalshi prices earlier in the week are often softer than Vegas closing lines. `data/kalshi_history.csv` logs every snapshot so that can be measured properly after a few weeks.

## Ideas for v2

- Injury report features (nflverse `injuries` data)
- Weather and wind for outdoor games
- Use Kalshi price history to test entry timing (early week vs. game day)
- Tune the model on Brier score with a validation season instead of fixed settings
- Spread and totals models using the same features
