"""
Layer 3 — Feature Engineering

Turns raw stored series into the variables strategists actually talk about:
yield changes, curve steepening, real-yield moves, dollar/oil momentum,
credit-spread widening, risk sentiment, and macro *surprises*.

On surprises: true consensus-expectations data (Bloomberg/Refinitiv) is paywalled,
so we proxy the surprise statistically — the release's deviation from a simple
time-series expectation (rolling mean), scaled by rolling volatility (a z-score).
This captures "hotter/cooler than the recent trend", which is a reasonable
first-order stand-in and is fully reproducible from free data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import db

TRADING_DAYS = "B"  # business-day index


# ---------- helpers ----------

def daily_return(s: pd.Series) -> pd.Series:
    return s.pct_change() * 100  # percent


def bp_change(s: pd.Series) -> pd.Series:
    return s.diff() * 100  # yields quoted in %, change in basis points


def momentum(s: pd.Series, window: int = 5) -> pd.Series:
    return s.pct_change(window) * 100


def surprise(s: pd.Series, window: int = 12) -> pd.Series:
    """
    Statistical surprise for a (typically monthly) macro series:
    z-score of the newest value vs its rolling mean/std.
    Non-release days get 0 after reindexing to business days.
    """
    mu = s.rolling(window, min_periods=4).mean().shift(1)
    sd = s.rolling(window, min_periods=4).std().shift(1)
    z = (s - mu) / sd.replace(0, np.nan)
    return z.clip(-4, 4)


def to_daily(s: pd.Series, index: pd.DatetimeIndex, how: str = "ffill") -> pd.Series:
    """Align a lower-frequency series onto the daily market index."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if how == "ffill":
        return s.reindex(index, method="ffill")
    if how == "impulse":  # value only on release day, 0 elsewhere
        out = pd.Series(0.0, index=index)
        # snap each release date to the nearest following business day on the index
        for dt, val in s.dropna().items():
            pos = index.searchsorted(dt)
            if pos < len(index):
                out.iloc[pos] = val
        return out
    raise ValueError(how)


# ---------- main builder ----------

def build_features(conn=None) -> pd.DataFrame:
    """
    Returns a daily DataFrame with:
      targets:  RET_SPX, RET_NASDAQ, RET_NIFTY, ...
      features: D_US10Y_BP, D_CURVE_BP, D_REAL10Y_BP, DXY_MOM5, OIL_MOM5,
                D_VIX, D_HY_OAS_BP, D_NFCI, CPI_SURPRISE, PAYROLLS_SURPRISE, ...
    """
    conn = conn or db.get_conn()

    mkt = db.load_wide(
        ["SPX", "NASDAQ", "STOXX50", "NIFTY50", "MSCI_EM",
         "US10Y", "US2Y", "GOLD", "BRENT", "WTI", "COPPER",
         "DXY", "EURUSD", "USDJPY", "USDINR", "VIX"],
        conn,
    )
    macro = db.load_wide(
        ["US10Y_FRED", "US2Y_FRED", "BREAKEVEN_10Y", "HY_OAS", "IG_OAS",
         "NFCI", "FED_ASSETS", "FED_FUNDS",
         "CPI_YOY", "CORE_CPI_YOY", "PAYROLLS", "UNEMPLOYMENT",
         "RETAIL_SALES", "INDPRO", "CONS_CONF"],
        conn,
    )
    if mkt.empty:
        raise RuntimeError("No market data in warehouse — run the collectors first.")

    idx = mkt.index
    f = pd.DataFrame(index=idx)

    # --- targets: daily returns ---
    for eq in ["SPX", "NASDAQ", "STOXX50", "NIFTY50", "MSCI_EM"]:
        if eq in mkt:
            f[f"RET_{eq}"] = daily_return(mkt[eq])

    # --- rates ---
    def pick(mkt_col: str, macro_col: str, prefer_macro: bool = False) -> pd.Series | None:
        market = mkt[mkt_col] if (mkt_col in mkt and mkt[mkt_col].notna().any()) else None
        fred = (to_daily(macro[macro_col].dropna(), idx)
                if (macro_col in macro and macro[macro_col].notna().any()) else None)
        if prefer_macro:
            return fred if fred is not None else market
        return market if market is not None else fred

    # Prefer FRED's official daily Treasury yields: single consistent source,
    # avoids noise from mixing Yahoo's ^TNX with 2Y futures quotes.
    us10 = pick("US10Y", "US10Y_FRED", prefer_macro=True)
    us2 = pick("US2Y", "US2Y_FRED", prefer_macro=True)
    if us10 is not None:
        f["D_US10Y_BP"] = bp_change(us10)
        if us2 is not None:
            f["D_CURVE_BP"] = bp_change(us10 - us2)  # steepening (+) / flattening (-)
    if "BREAKEVEN_10Y" in macro and macro["BREAKEVEN_10Y"].notna().any():
        be = to_daily(macro["BREAKEVEN_10Y"].dropna(), idx)
        f["D_BREAKEVEN_BP"] = bp_change(be)
        if us10 is not None:
            f["D_REAL10Y_BP"] = bp_change(us10 - be)  # real yield change

    # --- dollar, commodities ---
    if "DXY" in mkt:
        f["D_DXY"] = daily_return(mkt["DXY"])
        f["DXY_MOM5"] = momentum(mkt["DXY"])
    oil = mkt.get("BRENT", mkt.get("WTI"))
    if oil is not None:
        f["D_OIL"] = daily_return(oil)
        f["OIL_MOM5"] = momentum(oil)
    if "GOLD" in mkt:
        f["D_GOLD"] = daily_return(mkt["GOLD"])
    if "COPPER" in mkt:
        f["D_COPPER"] = daily_return(mkt["COPPER"])

    # --- risk sentiment / conditions ---
    def has_macro(col: str) -> bool:
        return col in macro and macro[col].notna().any()

    if "VIX" in mkt:
        f["D_VIX"] = daily_return(mkt["VIX"])
    if has_macro("HY_OAS"):
        f["D_HY_OAS_BP"] = bp_change(to_daily(macro["HY_OAS"].dropna(), idx))
    if has_macro("NFCI"):
        f["D_NFCI"] = to_daily(macro["NFCI"].dropna(), idx).diff()
    if has_macro("FED_ASSETS"):
        f["LIQUIDITY_MOM"] = to_daily(macro["FED_ASSETS"].dropna(), idx).pct_change(20) * 100

    # --- macro surprises (impulse on release day, 0 otherwise) ---
    for var in ["CPI_YOY", "CORE_CPI_YOY", "PAYROLLS", "UNEMPLOYMENT",
                "RETAIL_SALES", "INDPRO", "CONS_CONF"]:
        if has_macro(var):
            f[f"{var}_SURPRISE"] = to_daily(surprise(macro[var].dropna()), idx, how="impulse")

    return f


# Features that are near-mechanically tied to equity prices themselves
# (VIX is computed from S&P options; credit spreads and the NFCI embed equity
# volatility). Excluding them gives a purer "macro-only" attribution.
MECHANICAL_FEATURES = ["D_VIX", "D_HY_OAS_BP", "D_NFCI"]

FEATURE_LABELS = {
    "D_US10Y_BP": "10Y Treasury yield change",
    "D_CURVE_BP": "Yield curve steepening (2s10s)",
    "D_REAL10Y_BP": "Real yield change",
    "D_BREAKEVEN_BP": "Inflation expectations change",
    "D_DXY": "Dollar move",
    "DXY_MOM5": "Dollar momentum (5d)",
    "D_OIL": "Oil price move",
    "OIL_MOM5": "Oil momentum (5d)",
    "D_GOLD": "Gold move",
    "D_COPPER": "Copper move",
    "D_VIX": "Implied volatility (VIX) move",
    "D_HY_OAS_BP": "High-yield credit spread widening",
    "D_NFCI": "Financial conditions tightening",
    "LIQUIDITY_MOM": "Fed balance-sheet (liquidity) momentum",
    "CPI_YOY_SURPRISE": "CPI surprise",
    "CORE_CPI_YOY_SURPRISE": "Core CPI surprise",
    "PAYROLLS_SURPRISE": "Payrolls surprise",
    "UNEMPLOYMENT_SURPRISE": "Unemployment surprise",
    "RETAIL_SALES_SURPRISE": "Retail sales surprise",
    "INDPRO_SURPRISE": "Industrial production surprise",
    "CONS_CONF_SURPRISE": "Consumer confidence surprise",
}
