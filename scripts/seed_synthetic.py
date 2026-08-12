"""
Seed the warehouse with SYNTHETIC data (for testing the pipeline without APIs).
Generates ~4 years of correlated daily market data plus monthly macro releases,
with planted relationships (equities fall when yields/VIX/dollar rise) so the
attribution engine has real structure to recover.

Run from repo root:  python scripts/seed_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import db

rng = np.random.default_rng(7)
idx = pd.bdate_range("2022-07-01", "2026-07-03")
n = len(idx)

# --- latent daily shocks ---
yield_shock = rng.normal(0, 0.05, n)            # ~5bp daily vol on 10Y
vix_shock = rng.normal(0, 4.5, n)               # % moves in VIX
dxy_shock = rng.normal(0, 0.35, n)
oil_shock = rng.normal(0, 1.6, n)

# occasional payroll-day jumps (monthly): yields spike on strong prints
payroll_days = idx.to_series().groupby([idx.year, idx.month]).head(1).index  # 1st bday of month
payroll_surprise = pd.Series(0.0, index=idx)
payroll_surprise.loc[payroll_days] = rng.normal(0, 1.0, len(payroll_days))
yield_shock += 0.10 * payroll_surprise.values   # strong payrolls -> yields up
vix_shock += 2.0 * np.abs(payroll_surprise.values)

# --- planted equity relationship ---
spx_ret = (0.03
           - 3.5 * yield_shock                  # -3.5% per 1pt yield rise (~ -3.5bp per bp... i.e. -0.035%/bp)
           - 0.12 * vix_shock
           - 0.55 * dxy_shock
           + 0.06 * oil_shock
           + rng.normal(0, 0.55, n))            # idiosyncratic noise

def cum(level0, ret_pct):
    return level0 * np.cumprod(1 + np.asarray(ret_pct) / 100)

series = {
    ("SPX", "US"): cum(4500, spx_ret),
    ("NASDAQ", "US"): cum(14000, 1.3 * spx_ret + rng.normal(0, 0.3, n)),
    ("NIFTY50", "IN"): cum(19500, 0.7 * spx_ret + rng.normal(0, 0.6, n)),
    ("MSCI_EM", "GLOBAL"): cum(40, 0.8 * spx_ret - 0.4 * dxy_shock + rng.normal(0, 0.5, n)),
    ("STOXX50", "EZ"): cum(4300, 0.9 * spx_ret + rng.normal(0, 0.5, n)),
    ("US10Y", "US"): 4.0 + np.cumsum(yield_shock) - np.linspace(0, np.cumsum(yield_shock)[-1], n),  # mean-reverting-ish
    ("US2Y", "US"): 4.4 + np.cumsum(yield_shock * 1.2) - np.linspace(0, np.cumsum(yield_shock * 1.2)[-1], n),
    ("VIX", "US"): np.clip(cum(16, vix_shock), 9, 80),
    ("DXY", "US"): cum(103, dxy_shock),
    ("EURUSD", "GLOBAL"): cum(1.08, -0.8 * dxy_shock + rng.normal(0, 0.15, n)),
    ("USDJPY", "GLOBAL"): cum(148, 0.6 * dxy_shock + 2.0 * yield_shock + rng.normal(0, 0.3, n)),
    ("USDINR", "GLOBAL"): cum(83, 0.2 * dxy_shock + rng.normal(0, 0.1, n)),
    ("BRENT", "GLOBAL"): cum(82, oil_shock),
    ("WTI", "GLOBAL"): cum(78, oil_shock + rng.normal(0, 0.3, n)),
    ("GOLD", "GLOBAL"): cum(1950, -0.3 * dxy_shock - 1.5 * yield_shock + rng.normal(0, 0.7, n)),
    ("COPPER", "GLOBAL"): cum(3.8, 0.4 * spx_ret * 0 + rng.normal(0, 1.1, n)),
}

# --- sector ETFs with PLANTED macro sensitivities (for the rotation model) ---
sector_specs = {  # name: (beta_spx, yield_sens, oil_sens, dxy_sens)
    "SEC_XLK":  (1.25, -2.5, 0.0, -0.2),   # tech hates rising yields
    "SEC_XLF":  (1.00, +2.0, 0.0,  0.0),   # financials like rising yields
    "SEC_XLE":  (0.90, +0.5, 0.45, 0.0),   # energy tracks oil
    "SEC_XLU":  (0.55, -3.0, 0.0,  0.0),   # utilities = bond proxy
    "SEC_XLV":  (0.70, -0.5, 0.0,  0.0),
    "SEC_XLI":  (1.00, +0.5, 0.05, -0.1),
    "SEC_XLP":  (0.55, -1.0, 0.0,  0.1),
    "SEC_XLY":  (1.15, -1.5, -0.05, -0.2),
    "SEC_XLB":  (1.00, +0.3, 0.10, -0.3),
    "SEC_XLRE": (0.75, -2.8, 0.0,  0.0),   # real estate = bond proxy
    "SEC_XLC":  (1.10, -1.2, 0.0,  0.0),
}
for name, (beta, ys, osens, ds) in sector_specs.items():
    sec_ret = (beta * spx_ret + ys * yield_shock + osens * oil_shock
               + ds * dxy_shock + rng.normal(0, 0.35, n))
    series[(name, "US")] = cum(100, sec_ret)

frames = [
    pd.DataFrame({"date": idx, "country": c, "variable": v, "value": vals, "source": "synthetic"})
    for (v, c), vals in series.items()
]

# --- monthly macro (levels; features.py converts) ---
mdates = pd.date_range("2022-07-01", "2026-07-01", freq="MS")
m = len(mdates)
cpi_index = 300 * np.cumprod(1 + rng.normal(0.0025, 0.0012, m))
payroll_level = 156000 + np.cumsum(rng.normal(180, 90, m))

macro_series = {
    ("CPI_YOY", "US"): pd.Series(cpi_index, index=mdates),               # converted to YoY downstream? no —
    ("PAYROLLS", "US"): pd.Series(payroll_level, index=mdates),
    ("UNEMPLOYMENT", "US"): pd.Series(np.clip(3.8 + np.cumsum(rng.normal(0, 0.07, m)), 3.2, 6.5), index=mdates),
    ("CONS_CONF", "US"): pd.Series(np.clip(68 + np.cumsum(rng.normal(0, 1.5, m)), 50, 110), index=mdates),
    ("HY_OAS", "US"): pd.Series(np.clip(3.5 + np.cumsum(rng.normal(0, 0.08, m)), 2.5, 8.0), index=mdates),
    ("NFCI", "US"): pd.Series(np.cumsum(rng.normal(0, 0.03, m)) - 0.4, index=mdates),
    ("BREAKEVEN_10Y", "US"): pd.Series(np.clip(2.3 + np.cumsum(rng.normal(0, 0.02, m)), 1.5, 3.2), index=mdates),
    ("FED_ASSETS", "US"): pd.Series(7.5e6 - np.cumsum(rng.normal(1.5e4, 5e3, m)), index=mdates),
}
# NOTE: features.build_features treats CPI_YOY as already-YoY when loaded from FRED
# (macro.py converts). For synthetic data we store YoY directly:
macro_series[("CPI_YOY", "US")] = pd.Series(
    np.clip(3.0 + np.cumsum(rng.normal(0, 0.12, m)), 0.5, 9.0), index=mdates
)
# PAYROLLS in the real pipeline is stored as MoM change; do the same here:
macro_series[("PAYROLLS", "US")] = pd.Series(rng.normal(190, 95, m), index=mdates)
# plant the payroll surprises used in the market shocks onto release months
ps_monthly = payroll_surprise[payroll_surprise != 0]
k = min(m, len(ps_monthly))
macro_series[("PAYROLLS", "US")].iloc[:k] = (190 + 95 * ps_monthly.values[:k])

for (v, c), s in macro_series.items():
    frames.append(pd.DataFrame({
        "date": s.index, "country": c, "variable": v, "value": s.values, "source": "synthetic",
    }))

conn = db.get_conn()
total = sum(db.upsert(f, conn) for f in frames)
print(f"Seeded {total:,} synthetic rows -> {db.DB_PATH}")
