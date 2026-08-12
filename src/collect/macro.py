"""
Layer 1 — Macroeconomic data via FRED (free API key: https://fred.stlouisfed.org/docs/api/api_key.html)

Set the key once:  export FRED_API_KEY=your_key_here
Run:               python -m src.collect.macro

FRED covers US comprehensively and has decent international series (OECD/IMF mirrors).
India/China releases beyond what FRED mirrors can be added later via other sources
(e.g. dbnomics, or manual CSV drops through src/collect/manual.py).
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from src import db

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# variable_name -> (fred_series_id, country)
SERIES: dict[str, tuple[str, str]] = {
    # --- United States ---
    "CPI_YOY":        ("CPIAUCSL",        "US"),   # index -> converted to YoY below
    "CORE_CPI_YOY":   ("CPILFESL",        "US"),
    "PPI_YOY":        ("PPIACO",          "US"),
    "GDP_QOQ":        ("A191RL1Q225SBEA", "US"),   # real GDP SAAR %
    "ISM_PMI":        ("NAPM",            "US"),   # discontinued on FRED; falls back gracefully
    "PAYROLLS":       ("PAYEMS",          "US"),   # level -> converted to MoM change below
    "UNEMPLOYMENT":   ("UNRATE",          "US"),
    "RETAIL_SALES":   ("RSAFS",           "US"),
    "INDPRO":         ("INDPRO",          "US"),
    "CONS_CONF":      ("UMCSENT",         "US"),
    "FED_FUNDS":      ("DFF",             "US"),
    "US10Y_FRED":     ("DGS10",           "US"),
    "US2Y_FRED":      ("DGS2",            "US"),
    "BREAKEVEN_10Y":  ("T10YIE",          "US"),   # inflation expectations
    "HY_OAS":         ("BAMLH0A0HYM2",    "US"),   # true high-yield spread
    "IG_OAS":         ("BAMLC0A0CM",      "US"),   # true IG spread
    "FED_ASSETS":     ("WALCL",           "US"),   # balance sheet (liquidity)
    "NFCI":           ("NFCI",            "US"),   # Chicago Fed financial conditions
    # --- Eurozone ---
    "EZ_CPI_YOY":     ("CP0000EZ19M086NEST", "EZ"),
    "ECB_RATE":       ("ECBDFR",          "EZ"),
    "BUND10Y":        ("IRLTLT01DEM156N", "EZ"),   # monthly German 10Y (daily needs another source)
    # --- UK ---
    "UK_CPI_YOY":     ("GBRCPIALLMINMEI", "UK"),
    "BOE_RATE":       ("IUDSOIA",         "UK"),   # SONIA overnight rate proxy
    "GILT10Y":        ("IRLTLT01GBM156N", "UK"),
    # --- Japan ---
    "JP_CPI_YOY":     ("JPNCPIALLMINMEI", "JP"),
    "JGB10Y":         ("IRLTLT01JPM156N", "JP"),
    # --- India ---
    "IN_CPI_YOY":     ("INDCPIALLMINMEI", "IN"),
    # --- China ---
    "CN_CPI_YOY":     ("CHNCPIALLMINMEI", "CN"),
}

# Series stored as index levels that we convert to YoY % change
CONVERT_YOY = {"CPI_YOY", "CORE_CPI_YOY", "PPI_YOY", "EZ_CPI_YOY", "UK_CPI_YOY",
               "JP_CPI_YOY", "IN_CPI_YOY", "CN_CPI_YOY", "RETAIL_SALES", "INDPRO"}
# Series stored as levels that we convert to 1-period change (e.g. payrolls, in thousands)
CONVERT_DIFF = {"PAYROLLS"}


def fetch_series(series_id: str, api_key: str, start: str = "2000-01-01") -> pd.Series:
    r = requests.get(
        FRED_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
        },
        timeout=30,
    )
    r.raise_for_status()
    obs = r.json().get("observations", [])
    s = pd.Series(
        {o["date"]: o["value"] for o in obs if o["value"] != "."},
        dtype="float64",
    )
    s.index = pd.to_datetime(s.index)
    return s


def fetch_macro_data(api_key: str | None = None) -> pd.DataFrame:
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit("Set FRED_API_KEY environment variable (free at fred.stlouisfed.org).")

    frames = []
    for variable, (series_id, country) in SERIES.items():
        try:
            s = fetch_series(series_id, api_key)
        except Exception as e:  # noqa: BLE001
            print(f"[macro] FAILED {variable} ({series_id}): {e}")
            continue
        if s.empty:
            print(f"[macro] EMPTY  {variable} ({series_id})")
            continue
        if variable in CONVERT_YOY:
            s = s.pct_change(12) * 100  # monthly series -> YoY %
        elif variable in CONVERT_DIFF:
            s = s.diff()
        s = s.dropna()
        frames.append(
            pd.DataFrame({
                "date": s.index, "country": country,
                "variable": variable, "value": s.values, "source": "FRED",
            })
        )
        print(f"[macro] OK     {variable}: {len(s)} rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    conn = db.get_conn()
    data = fetch_macro_data()
    n = db.upsert(data, conn)
    print(f"[macro] wrote {n} rows -> {db.DB_PATH}")


if __name__ == "__main__":
    main()
