"""
Layer 4 — Research Engine

Explains a given day's market move by decomposing it into factor contributions.

Method (MVP, deliberately transparent):
  1. Take a rolling training window (default 500 business days) ending the day
     before the day being explained (no look-ahead).
  2. Standardize features; fit LassoCV of the target return on the features.
     LASSO performs the "which drivers matter" selection your write-up describes.
  3. Contribution of factor j today = beta_j * standardized feature value today.
     Contributions sum (plus intercept + residual) to the actual return, so
     the decomposition is exact and honest about what's unexplained.
  4. Find historical analogues: past days whose feature vectors are closest
     (Euclidean distance in standardized space) to today.

Random forest / gradient boosting / SHAP can be slotted in later behind the
same `explain_day` interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


@dataclass
class Explanation:
    date: pd.Timestamp
    target: str
    actual_return: float
    predicted_return: float
    contributions: pd.Series          # signed, in return units (%), sorted by |size|
    residual: float
    r2_train: float
    analogues: pd.DataFrame = field(default_factory=pd.DataFrame)


def _clean_xy(features: pd.DataFrame, target: str):
    y = features[target]
    X = features[[c for c in features.columns if not c.startswith("RET_")]]
    data = pd.concat([y, X], axis=1)
    # surprises are 0-filled already; market features need real observations
    data = data.dropna(subset=[target]).fillna(0.0)
    return data[target], data.drop(columns=[target])


def explain_day(
    features: pd.DataFrame,
    target: str = "RET_SPX",
    date: str | pd.Timestamp | None = None,
    window: int = 500,
    n_analogues: int = 5,
    exclude: list[str] | None = None,
) -> Explanation:
    if exclude:
        features = features.drop(columns=[c for c in exclude if c in features.columns])
    y, X = _clean_xy(features, target)
    date = pd.Timestamp(date) if date is not None else y.index[-1]
    if date not in y.index:
        # snap to the most recent day on or before the requested date
        prior = y.index[y.index <= date]
        if len(prior) == 0:
            raise ValueError(f"No data on or before {date.date()}")
        date = prior[-1]

    loc = y.index.get_loc(date)
    train_slice = slice(max(0, loc - window), loc)  # strictly before `date`
    X_train, y_train = X.iloc[train_slice], y.iloc[train_slice]
    if len(y_train) < 60:
        raise ValueError("Not enough history to fit the model (need ~60+ days).")

    scaler = StandardScaler().fit(X_train)
    Xs_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X.columns)
    xs_today = pd.Series(
        scaler.transform(X.loc[[date]])[0], index=X.columns, name=date
    )

    model = LassoCV(cv=5, max_iter=20000).fit(Xs_train, y_train)
    betas = pd.Series(model.coef_, index=X.columns)

    contributions = (betas * xs_today).replace(0.0, np.nan).dropna()
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

    predicted = float(model.intercept_ + contributions.sum())
    actual = float(y.loc[date])

    # Historical analogues: nearest neighbours in standardized feature space,
    # using only the features the model kept (non-zero betas).
    kept = betas[betas != 0].index
    analogues = pd.DataFrame()
    if len(kept) > 0:
        dists = np.linalg.norm(Xs_train[kept].values - xs_today[kept].values, axis=1)
        order = np.argsort(dists)[:n_analogues]
        analogues = pd.DataFrame({
            "date": Xs_train.index[order],
            "distance": dists[order],
            "return_that_day": y_train.iloc[order].values,
        })

    return Explanation(
        date=date,
        target=target,
        actual_return=actual,
        predicted_return=predicted,
        contributions=contributions,
        residual=actual - predicted,
        r2_train=float(model.score(Xs_train, y_train)),
        analogues=analogues,
    )
