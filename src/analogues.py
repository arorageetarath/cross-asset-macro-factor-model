"""
Analogue enrichment — turns "2024-10-29, distance 0.65" into a story:
what the macro tape looked like that day, what the market did over the
following week and month, and which sector led afterwards.

All "subsequent performance" numbers are simple historical facts about those
specific past dates — context, not a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import db
from src.attribution import Explanation
from src.features import FEATURE_LABELS
from src.outlook import SECTORS


def _context_bullets(X: pd.DataFrame, date: pd.Timestamp, k: int = 3) -> list[str]:
    """Top-k most unusual features that day, phrased with direction arrows."""
    hist = X.loc[:date].tail(250)
    mu, sd = hist.mean(), hist.std().replace(0, np.nan)
    z = ((X.loc[date] - mu) / sd).dropna()
    bullets = []
    for feat in z.abs().nlargest(k).index:
        arrow = "↑" if z[feat] > 0 else "↓"
        label = FEATURE_LABELS.get(feat, feat)
        # strip verbose suffixes for card display
        label = (label.replace(" change", "").replace(" move", "")
                      .replace(" widening", "").replace(" (5d)", ""))
        bullets.append(f"{arrow} {label}")
    return bullets


def _forward_return(y: pd.Series, date: pd.Timestamp, days: int) -> float | None:
    """Cumulative % return of the target over the `days` trading days after `date`."""
    if date not in y.index:
        return None
    pos = y.index.get_loc(date)
    fwd = y.iloc[pos + 1: pos + 1 + days]
    if len(fwd) < days:
        return None
    return float(((1 + fwd / 100).prod() - 1) * 100)


def _sector_leader(conn, date: pd.Timestamp, days: int = 21) -> str | None:
    """Which sector beat the S&P by the most over the following month."""
    prices = db.load_wide(list(SECTORS) + ["SPX"], conn)
    if prices.empty or "SPX" not in prices:
        return None
    idx = prices.index
    start = idx.searchsorted(date)
    end = start + days
    if start >= len(idx) or end >= len(idx):
        return None
    p0, p1 = prices.iloc[start], prices.iloc[end]
    rel = {}
    spx_ret = p1["SPX"] / p0["SPX"] - 1
    for s in SECTORS:
        if s in prices and pd.notna(p0.get(s)) and pd.notna(p1.get(s)):
            rel[s] = (p1[s] / p0[s] - 1) - spx_ret
    if not rel:
        return None
    return SECTORS[max(rel, key=rel.get)]


def enrich_analogues(exp: Explanation, features: pd.DataFrame,
                     conn=None, top: int = 3) -> list[dict]:
    """List of dicts: date, distance, context (bullets), ret_1w, ret_1m, sector_leader."""
    conn = conn or db.get_conn()
    y = features[exp.target].dropna()
    X = features[[c for c in features.columns if not c.startswith("RET_")]].fillna(0.0)

    cards = []
    for _, row in exp.analogues.head(top).iterrows():
        d = pd.Timestamp(row["date"])
        cards.append({
            "date": d,
            "distance": float(row["distance"]),
            "same_day": float(row["return_that_day"]),
            "context": _context_bullets(X, d),
            "ret_1w": _forward_return(y, d, 5),
            "ret_1m": _forward_return(y, d, 21),
            "sector_leader": _sector_leader(conn, d),
        })
    return cards
