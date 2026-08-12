"""
Regime engine — classifies the macro backdrop into the classic 2x2:

                     Inflation rising      Inflation falling
    Growth rising    REFLATION             GOLDILOCKS
    Growth falling   STAGFLATION           SLOWDOWN

Growth and inflation are composite z-scores of their components' 3-month
momentum, z-scored against a rolling 36-month window ending at each date
(so classification at time t uses only information available at t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import db

GROWTH_COMPONENTS = {  # variable -> sign (+1 = higher is stronger growth)
    "PAYROLLS": +1,          # stored as MoM change already
    "UNEMPLOYMENT": -1,
    "INDPRO": +1,            # stored as YoY %
    "CONS_CONF": +1,
    "RETAIL_SALES": +1,      # YoY %
}
INFLATION_COMPONENTS = {
    "CPI_YOY": +1,
    "CORE_CPI_YOY": +1,
    "BREAKEVEN_10Y": +1,
}

REGIME_NAMES = {
    (True, True): "Reflation",
    (True, False): "Goldilocks",
    (False, True): "Stagflation",
    (False, False): "Slowdown",
}
REGIME_COLORS = {
    "Reflation": "#e8b44c", "Goldilocks": "#3ddc84",
    "Stagflation": "#ff5c5c", "Slowdown": "#6ea8ff",
}
REGIME_DESCRIPTIONS = {
    "Reflation": "growth firming while inflation pressure builds — cyclicals and commodities historically lead",
    "Goldilocks": "growth firm, inflation cooling — the friendliest backdrop for risk assets, especially long-duration equities",
    "Stagflation": "growth rolling over while inflation stays hot — historically the toughest quadrant; defensives and energy hold up best",
    "Slowdown": "growth and inflation both falling — bond-proxies and defensives historically outperform",
}


def _rolling_z(s: pd.Series, window: int = 36, min_periods: int = 12) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-3, 3)


def _composite(monthly: pd.DataFrame, components: dict[str, int]) -> pd.Series:
    zs = []
    for var, sign in components.items():
        if var in monthly and monthly[var].notna().sum() >= 15:
            mom = monthly[var].diff(3)          # 3-month momentum
            zs.append(sign * _rolling_z(mom))
    if not zs:
        return pd.Series(dtype=float)
    return pd.concat(zs, axis=1).mean(axis=1)


def build_regimes(conn=None) -> pd.DataFrame:
    """Monthly DataFrame: growth_z, inflation_z, regime (label)."""
    conn = conn or db.get_conn()
    macro = db.load_wide(
        list(GROWTH_COMPONENTS) + list(INFLATION_COMPONENTS), conn
    )
    if macro.empty:
        raise RuntimeError("No macro data — regimes need the FRED collector to have run.")
    monthly = macro.resample("ME").last()

    out = pd.DataFrame(index=monthly.index)
    out["growth_z"] = _composite(monthly, GROWTH_COMPONENTS)
    out["inflation_z"] = _composite(monthly, INFLATION_COMPONENTS)
    out = out.dropna()
    out["regime"] = [
        REGIME_NAMES[(g >= 0, i >= 0)]
        for g, i in zip(out["growth_z"], out["inflation_z"])
    ]
    return out


def current_regime(regimes: pd.DataFrame) -> dict:
    last = regimes.iloc[-1]
    return {
        "date": regimes.index[-1],
        "regime": last["regime"],
        "growth_z": float(last["growth_z"]),
        "inflation_z": float(last["inflation_z"]),
        "description": REGIME_DESCRIPTIONS[last["regime"]],
    }
