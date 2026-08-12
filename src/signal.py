"""
Residual reversal signal — the predictive extension.

Idea: the attribution model says what the market "should" have done given the
macro tape. The gap between actual and model-implied return — the residual —
is a mispricing candidate. If residuals mean-revert, today's unexplained
underperformance predicts a positive return tomorrow (and vice versa).

Honesty is built in:
  * Residuals are computed WALK-FORWARD: the model that scores day t was
    trained only on days before t (refit every `refit_every` days).
  * The lead-lag regression (tomorrow's return ~ today's residual) reports its
    t-statistic. If |t| < 2, the dashboard says "no significant signal" —
    a null result is a legitimate finding, not a failure.
  * The residual uses the MACRO-ONLY factor set: the full set includes VIX,
    which mechanically absorbs the move and leaves a meaningless residual.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from src.features import MECHANICAL_FEATURES


@dataclass
class SignalResult:
    date: pd.Timestamp
    residual_today: float          # pp, actual minus model-implied
    residual_z: float              # vs trailing 60-day residual distribution
    beta: float                    # slope of tomorrow_return ~ today_residual
    t_stat: float
    n_obs: int
    expected_edge: float           # beta * residual_today (pp)
    direction: str                 # "Bullish" / "Bearish" / "None"
    significant: bool              # |t| >= 2
    residuals: pd.Series           # full walk-forward residual history


def walk_forward_residuals(
    features: pd.DataFrame,
    target: str = "RET_SPX",
    window: int = 500,
    refit_every: int = 21,
    exclude: list[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Out-of-sample residual for each day: actual - model prediction,
    where the model was trained strictly on days before that day."""
    exclude = MECHANICAL_FEATURES if exclude is None else exclude
    feats = features.drop(columns=[c for c in exclude if c in features.columns])
    y = feats[target]
    X = feats[[c for c in feats.columns if not c.startswith("RET_")]]
    data = pd.concat([y, X], axis=1).dropna(subset=[target]).fillna(0.0)
    y, X = data[target], data.drop(columns=[target])

    residuals = pd.Series(index=y.index, dtype=float)
    start = max(120, window // 2)  # need some history before first fit
    model = scaler = cols = None
    for i in range(start, len(y)):
        if model is None or (i - start) % refit_every == 0:
            lo = max(0, i - window)
            scaler = StandardScaler().fit(X.iloc[lo:i])
            cols = X.columns
            model = LassoCV(cv=5, max_iter=20000).fit(
                scaler.transform(X.iloc[lo:i]), y.iloc[lo:i])
        pred = float(model.predict(scaler.transform(X.iloc[[i]][cols]))[0])
        residuals.iloc[i] = y.iloc[i] - pred
    residuals = residuals.dropna()
    return residuals, y


def build_signal(
    features: pd.DataFrame,
    target: str = "RET_SPX",
    window: int = 500,
    refit_every: int = 21,
    exclude: list[str] | None = None,
) -> SignalResult:
    residuals, y = walk_forward_residuals(features, target, window, refit_every, exclude)
    if len(residuals) < 120:
        raise ValueError("Not enough history to evaluate the residual signal.")

    # Lead-lag: tomorrow's return ~ today's residual (both known without look-ahead)
    tomorrow = y.shift(-1).reindex(residuals.index)
    pair = pd.concat([residuals.rename("resid"), tomorrow.rename("fwd")], axis=1).dropna()
    x, fwd = pair["resid"].values, pair["fwd"].values
    n = len(pair)

    x_c = x - x.mean()
    beta = float((x_c * (fwd - fwd.mean())).sum() / (x_c ** 2).sum())
    alpha = float(fwd.mean() - beta * x.mean())
    eps = fwd - (alpha + beta * x)
    se = float(np.sqrt((eps ** 2).sum() / (n - 2) / (x_c ** 2).sum()))
    t_stat = beta / se if se > 0 else 0.0

    resid_today = float(residuals.iloc[-1])
    trail = residuals.iloc[-61:-1]
    z = float((resid_today - trail.mean()) / trail.std()) if trail.std() > 0 else 0.0
    expected_edge = beta * resid_today
    significant = abs(t_stat) >= 2

    if not significant or abs(z) < 1.0:
        direction = "None"
    else:
        direction = "Bullish" if expected_edge > 0 else "Bearish"

    return SignalResult(
        date=residuals.index[-1], residual_today=resid_today, residual_z=z,
        beta=beta, t_stat=float(t_stat), n_obs=n, expected_edge=expected_edge,
        direction=direction, significant=significant, residuals=residuals,
    )
