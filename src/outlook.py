"""
Model Outlook — decision-support layer (deliberately simple and explainable).

No ML here: classify today's macro regime (growth x inflation quadrant), then
rank sectors by their HISTORICAL average monthly return relative to the S&P 500
within that regime. "What has worked in backdrops like this one" — the classic
strategist playbook, fully transparent.

Confidence combines two things:
  * sample size — how many months of history the current regime has, and
  * signal clarity — how far today sits from the quadrant boundaries
    (a regime call with growth_z = +0.05 is a coin toss; +1.2 is conviction).
"""
from __future__ import annotations

import pandas as pd

from src import db
from src.regimes import build_regimes, current_regime

SECTORS = {
    "SEC_XLK": "Technology", "SEC_XLF": "Financials", "SEC_XLE": "Energy",
    "SEC_XLU": "Utilities", "SEC_XLV": "Health Care", "SEC_XLI": "Industrials",
    "SEC_XLP": "Cons. Staples", "SEC_XLY": "Cons. Discretionary",
    "SEC_XLB": "Materials", "SEC_XLRE": "Real Estate", "SEC_XLC": "Communications",
}


def _regime_sector_table(conn):
    prices = db.load_wide(list(SECTORS) + ["SPX"], conn)
    me = prices.resample("ME").last()
    sec = [c for c in SECTORS if c in me and me[c].notna().sum() >= 24]
    if len(sec) < 5 or "SPX" not in me:
        raise RuntimeError("Sector data missing — run `python scripts/daily_update.py` "
                           "once to download the sector ETFs.")
    rets = me[sec + ["SPX"]].pct_change() * 100
    rel = rets[sec].sub(rets["SPX"], axis=0)

    regimes = build_regimes(conn)
    labels = regimes["regime"].reindex(rel.index, method="ffill")
    joined = rel[labels.notna()]
    tbl = joined.groupby(labels.dropna()).mean().T  # sectors x regimes
    counts = labels.value_counts()
    return tbl, counts, regimes


def build_outlook(conn=None, threshold: float = 0.15) -> dict:
    """
    Returns:
      regime (dict from current_regime), favours / neutral / less_favourable
      (lists of sector names), confidence ('High'/'Moderate'/'Low'),
      n_months (history in this regime), table (per-sector avg rel return, pp/mo).
    """
    conn = conn or db.get_conn()
    tbl, counts, regimes = _regime_sector_table(conn)
    cur = current_regime(regimes)
    if cur["regime"] not in tbl.columns:
        raise RuntimeError("No sector history for the current regime yet.")

    col = tbl[cur["regime"]].sort_values(ascending=False)
    named = pd.Series(col.values, index=[SECTORS[s] for s in col.index])

    favours = [s for s, v in named.items() if v >= threshold][:4]
    less = [s for s, v in named.items() if v <= -threshold][-4:]
    neutral = [s for s in named.index if s not in favours and s not in less]

    n = int(counts.get(cur["regime"], 0))
    clarity = min(abs(cur["growth_z"]), abs(cur["inflation_z"]))
    if n >= 15 and clarity >= 0.5:
        confidence = "High"
    elif n < 8 or clarity < 0.25:
        confidence = "Low"
    else:
        confidence = "Moderate"

    return {
        "regime": cur, "favours": favours, "neutral": neutral,
        "less_favourable": less, "confidence": confidence,
        "n_months": n, "table": named,
    }
