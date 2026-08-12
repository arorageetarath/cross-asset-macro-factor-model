"""
Layer 1 — Market data collection (daily prices/yields) via yfinance.
Run: python -m src.collect.markets  (from repo root)
"""
from __future__ import annotations

import pandas as pd

from src import db

# variable_name -> (yahoo_ticker, country)
TICKERS: dict[str, tuple[str, str]] = {
    # Equities
    "SPX":        ("^GSPC",      "US"),
    "NASDAQ":     ("^IXIC",      "US"),
    "STOXX50":    ("^STOXX50E",  "EZ"),
    "NIFTY50":    ("^NSEI",      "IN"),
    "MSCI_EM":    ("EEM",        "GLOBAL"),   # ETF proxy
    # Rates (Yahoo quotes these as yield * 10 for ^TNX etc. — handled below)
    "US10Y":      ("^TNX",       "US"),
    "US2Y":       ("2YY=F",      "US"),       # 2Y yield future; fallback: ^IRX (13w bill)
    # Commodities
    "GOLD":       ("GC=F",       "GLOBAL"),
    "BRENT":      ("BZ=F",       "GLOBAL"),
    "WTI":        ("CL=F",       "GLOBAL"),
    "COPPER":     ("HG=F",       "GLOBAL"),
    # FX
    "DXY":        ("DX-Y.NYB",   "US"),
    "EURUSD":     ("EURUSD=X",   "GLOBAL"),
    "USDJPY":     ("USDJPY=X",   "GLOBAL"),
    "USDINR":     ("USDINR=X",   "GLOBAL"),
    # Volatility / credit
    "VIX":        ("^VIX",       "US"),
    "HYG":        ("HYG",        "US"),       # high-yield credit ETF (spread proxy)
    "LQD":        ("LQD",        "US"),       # investment-grade credit ETF
    # US sector ETFs (SPDR) — used by the rotation model
    "SEC_XLK":    ("XLK",        "US"),       # Technology
    "SEC_XLF":    ("XLF",        "US"),       # Financials
    "SEC_XLE":    ("XLE",        "US"),       # Energy
    "SEC_XLU":    ("XLU",        "US"),       # Utilities
    "SEC_XLV":    ("XLV",        "US"),       # Health Care
    "SEC_XLI":    ("XLI",        "US"),       # Industrials
    "SEC_XLP":    ("XLP",        "US"),       # Consumer Staples
    "SEC_XLY":    ("XLY",        "US"),       # Consumer Discretionary
    "SEC_XLB":    ("XLB",        "US"),       # Materials
    "SEC_XLRE":   ("XLRE",       "US"),       # Real Estate (starts 2015)
    "SEC_XLC":    ("XLC",        "US"),       # Communication Services (starts 2018)
}

# Yahoo's ^TNX is yield x 10? No — ^TNX quotes the yield directly (e.g. 4.31).
# No scaling needed, but keep hook here in case a ticker needs adjustment.
SCALE = {}


def fetch_market_data(period: str = "10y") -> pd.DataFrame:
    """Download daily closes for all tickers. Returns tall-format DataFrame."""
    import yfinance as yf  # imported here so the rest of the repo works offline

    frames = []
    for variable, (ticker, country) in TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        except Exception as e:  # noqa: BLE001 - log and continue
            print(f"[markets] FAILED {variable} ({ticker}): {e}")
            continue
        if hist.empty:
            print(f"[markets] EMPTY  {variable} ({ticker})")
            continue
        close = hist["Close"] * SCALE.get(variable, 1.0)
        frames.append(
            pd.DataFrame({
                "date": close.index.tz_localize(None).normalize(),
                "country": country,
                "variable": variable,
                "value": close.values,
                "source": "yfinance",
            })
        )
        print(f"[markets] OK     {variable}: {len(close)} rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    conn = db.get_conn()
    data = fetch_market_data()
    n = db.upsert(data, conn)
    print(f"[markets] wrote {n} rows -> {db.DB_PATH}")


if __name__ == "__main__":
    main()
