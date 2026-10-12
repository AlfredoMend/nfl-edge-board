"""Model definitions and fitting.

Neutral-site design: every base model is trained on each game twice, once as played and once
with home and away swapped (features flipped, outcome flipped). That forces the models to
learn team strength only, with no built-in home edge. Home field is then added back as one
explicit term, chosen inside the 1.2 to 1.8 point range.

Win probability: stacked ensemble of seven base models (logistic regression, XGBoost,
LightGBM, point-margin ridge, extra trees, a small neural net, Elo). The stacker combines
them in point-margin space:
    margin = SIGMA * a * sum_i w_i * probit(p_i) + HFA * home
    P(home win) = Phi(margin / SIGMA)
with w_i >= 0, sum w_i = 1, fit only on earlier seasons' out-of-sample predictions.

Totals: ridge and XGBoost regressors on scoring-environment features, averaged.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import CORE_FEATURES, FEATURES, SYMMETRIC, TOTAL_FEATURES

BASES = ["logit", "xgb", "lgbm", "margin", "trees", "mlp", "elo"]
BASE_LABELS = {"logit": "Logistic regression", "xgb": "XGBoost", "lgbm": "LightGBM",
               "margin": "Margin ridge regression", "trees": "Extra trees", "mlp": "Neural net",
               "elo": "Elo"}
RECENCY_HALFLIFE = None  # seasons; None = all training seasons count equally
HFA_GRID = np.round(np.arange(1.2, 1.81, 0.1), 2)   # home field, points
SIGMA = 13.0            # SD of NFL margins around a good prediction
COVER_SD = SIGMA


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _probit(p):
    return norm.ppf(np.clip(np.asarray(p, dtype=float), 0.005, 0.995))


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Append each game with home and away swapped."""
    flip = df.copy()
    anti = [c for c in FEATURES if c not in SYMMETRIC]
    flip[anti] = -flip[anti]
    flip["home_win"] = 1 - flip.home_win
    flip["result"] = -flip.result
    return pd.concat([df, flip], ignore_index=True)


# ---------------------------------------------------------------- base models
def fit_bases(train: pd.DataFrame) -> dict:
    t = mirror(train)
    X, Xc, y = t[FEATURES], t[CORE_FEATURES], t.home_win
    if RECENCY_HALFLIFE:
        sw = 0.5 ** ((t.season.max() - t.season).to_numpy() / RECENCY_HALFLIFE)
    else:
        sw = np.ones(len(t))
    m = {}
    m["logit"] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(
        Xc, y, logisticregression__sample_weight=sw)
    m["xgb"] = xgb.XGBClassifier(n_estimators=400, learning_rate=0.02, max_depth=3, min_child_weight=40,
                                 subsample=0.8, colsample_bytree=0.6, reg_lambda=10.0,
                                 eval_metric="logloss", n_jobs=4).fit(X, y, sample_weight=sw)
    m["lgbm"] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=120,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                                   verbose=-1).fit(X, y, sample_weight=sw)
    r = make_pipeline(StandardScaler(), Ridge(alpha=300, fit_intercept=False)).fit(X, t.result, ridge__sample_weight=sw)
    m["margin"] = (r, float((t.result - r.predict(X)).std()))
    m["trees"] = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=80, max_features=0.4,
                                      n_jobs=-1, random_state=0).fit(X, y, sample_weight=sw)
    m["mlp"] = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=3.0, max_iter=600,
                                                             early_stopping=True, random_state=0)).fit(
        Xc, y, mlpclassifier__sample_weight=sw)
    return m


def predict_bases(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Neutral-site home win probability from each base model."""
    X, Xc = df[FEATURES], df[CORE_FEATURES]
    out = pd.DataFrame(index=df.index)
    for k in ("xgb", "lgbm", "trees"):
        out[f"b_{k}"] = models[k].predict_proba(X)[:, 1]
    for k in ("logit", "mlp"):
        out[f"b_{k}"] = models[k].predict_proba(Xc)[:, 1]
    r, sd = models["margin"]
    out["b_margin"] = norm.cdf(r.predict(X) / sd)
    out["b_elo"] = 1 / (1 + 10 ** (-df.elo_diff.to_numpy() / 400))
    return out


# ---------------------------------------------------------------- stacking + home field
def _neutral_margin(df, w, a):
    Z = np.column_stack([_probit(df[f"b_{b}"]) for b in BASES])
    return SIGMA * a * (Z @ w)


def fit_stacker(oof: pd.DataFrame):
    """Returns (weights, a, hfa). HFA is picked from 1.2-1.8 points by out-of-sample log loss."""
    y = oof.home_win.to_numpy()
    home = oof.home.to_numpy()
    best = None
    for hfa in HFA_GRID:
        def loss(th):
            w = np.exp(th[:-1]); w /= w.sum()
            m = _neutral_margin(oof, w, th[-1]) + hfa * home
            return logloss(norm.cdf(m / SIGMA), y)

        th = minimize(loss, np.r_[np.zeros(len(BASES)), 0.8], method="L-BFGS-B").x
        val = loss(th)
        if best is None or val < best[0]:
            w = np.exp(th[:-1]); w /= w.sum()
            best = (val, w, float(th[-1]), float(hfa))
    return best[1], best[2], best[3]


def finalize(df: pd.DataFrame, stack) -> pd.DataFrame:
    w, a, hfa = stack
    df = df.copy()
    df["neutral_margin"] = _neutral_margin(df, w, a)
    df["pred_margin"] = df.neutral_margin + hfa * df.home.to_numpy()
    df["p_model"] = norm.cdf(df.pred_margin / SIGMA)
    df["p_home_cover"] = norm.cdf((df.pred_margin - df.spread_line) / COVER_SD)
    return df


# ---------------------------------------------------------------- totals
def fit_totals(train: pd.DataFrame):
    X, y = train[TOTAL_FEATURES], train.home_score + train.away_score
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=30)).fit(X, y)
    boost = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=30,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=4).fit(X, y)
    return ridge, boost


def predict_totals(models, df: pd.DataFrame) -> np.ndarray:
    ridge, boost = models
    X = df[TOTAL_FEATURES]
    return 0.5 * ridge.predict(X) + 0.5 * boost.predict(X)


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pred_home_pts"] = (df.pred_total + df.pred_margin) / 2
    df["pred_away_pts"] = (df.pred_total - df.pred_margin) / 2
    return df
