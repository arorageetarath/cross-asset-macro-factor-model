"""
Sector rotation model — the predictive layer.

Question answered: given today's macro state (rates, curve, dollar, oil,
growth/inflation regime), which US equity sectors are positioned to outperform
the S&P 500 over the NEXT month?

Design choices (all made for honesty, not fit):
  * Target = next-month sector return RELATIVE to the S&P 500 (relative bets
    carry more macro signal than absolute direction).
  * Monthly horizon (macro matters at 1-3 months; daily is noise).
  * One Ridge regression per sector on standardized macro features. Ridge, not
    OLS: features are correlated and monthly samples are few.
  * Walk-forward out-of-sample: the prediction for month t+1 uses a model
    trained ONLY on months <= t. The backtest contains zero look-ahead.
  * Backtest: long top-3 / short bottom-3 sectors, rebalanced monthly, minus
    10bp per changed position — a realistic friction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src import db
from src.regimes import build_regimes

SECTORS = {
    "SEC_XLK": "Technology", "SEC_XLF": "Financials", "SEC_XLE": "Energy",
    "SEC_XLU": "Utilities", "SEC_XLV": "Health Care", "SEC_XLI": "Industrials",
    "SEC_XLP": "Cons. Staples", "SEC_XLY": "Cons. Discretionary",
    "SEC_XLB": "Materials", "SEC_XLRE": "Real Estate", "SEC_XLC": "Communications",
}

ROTATION_FEATURE_LABELS = {
    "D_US10Y_1M": "10Y yield change (1m, bp)",
    "D_CURVE_1M": "Curve steepening (1m, bp)",
    "DXY_RET_1M": "Dollar return (1m)",
    "OIL_RET_1M": "Oil return (1m)",
    "GOLD_RET_1M": "Gold return (1m)",
    "GROWTH_Z": "Growth composite (z)",
    "INFLATION_Z": "Inflation composite (z)",
}


@dataclass
class RotationResult:
    latest_pred: pd.Series        # predicted next-month relative return per sector (pp)
    latest_asof: pd.Timestamp
    oos_preds: pd.DataFrame       # month x sector OOS predictions
    oos_actual: pd.DataFrame      # month x sector realized relative returns
    backtest: pd.DataFrame        # monthly long-short returns + cumulative curve
    stats: dict
    regime_table: pd.DataFrame    # avg monthly relative return per sector per regime


def _monthly_dataset(conn):
    prices = db.load_wide(list(SECTORS) + ["SPX", "US10Y", "US2Y", "DXY",
                                           "BRENT", "WTI", "GOLD"], conn)
    macro = db.load_wide(["US10Y_FRED", "US2Y_FRED"], conn)
    me = prices.resample("ME").last()

    sec_cols = [c for c in SECTORS if c in me and me[c].notna().sum() >= 24]
    if len(sec_cols) < 5 or "SPX" not in me:
        raise RuntimeError("Not enough sector data — re-run the market collector "
                           "(sector ETFs were added to it).")

    rets = me[sec_cols + ["SPX"]].pct_change() * 100
    rel = rets[sec_cols].sub(rets["SPX"], axis=0)  # monthly relative returns (pp)

    def yield_series(mkt_col, fred_col):
        if mkt_col in me and me[mkt_col].notna().sum() >= 24:
            return me[mkt_col]
        if fred_col in macro and macro[fred_col].notna().any():
            return macro[fred_col].resample("ME").last()
        return None

    y10, y2 = yield_series("US10Y", "US10Y_FRED"), yield_series("US2Y", "US2Y_FRED")
    oil = me.get("BRENT") if "BRENT" in me else me.get("WTI")

    X = pd.DataFrame(index=me.index)
    if y10 is not None:
        X["D_US10Y_1M"] = y10.diff() * 100
        if y2 is not None:
            X["D_CURVE_1M"] = (y10 - y2).diff() * 100
    if "DXY" in me:
        X["DXY_RET_1M"] = me["DXY"].pct_change() * 100
    if oil is not None:
        X["OIL_RET_1M"] = oil.pct_change() * 100
    if "GOLD" in me:
        X["GOLD_RET_1M"] = me["GOLD"].pct_change() * 100

    try:
        regimes = build_regimes(conn)
        X["GROWTH_Z"] = regimes["growth_z"].reindex(X.index, method="ffill")
        X["INFLATION_Z"] = regimes["inflation_z"].reindex(X.index, method="ffill")
        regime_labels = regimes["regime"].reindex(X.index, method="ffill")
    except RuntimeError:
        regime_labels = pd.Series(index=X.index, dtype=object)

    return X, rel, regime_labels, sec_cols


def run_rotation(conn=None, min_train: int = 24, top_n: int = 3,
                 cost_bp_per_change: float = 10.0) -> RotationResult:
    conn = conn or db.get_conn()
    X, rel, regime_labels, sec_cols = _monthly_dataset(conn)

    # Target: NEXT month's relative return. Features at month t predict t+1.
    y_fwd = rel.shift(-1)
    data = pd.concat([X, y_fwd], axis=1).dropna(subset=X.columns.tolist())
    Xc = data[X.columns]

    months = Xc.index
    oos_preds, oos_actual = {}, {}
    for t in range(min_train, len(months) - 1):
        train_idx, pred_month = months[:t], months[t]
        scaler = StandardScaler().fit(Xc.loc[train_idx])
        Xtr = scaler.transform(Xc.loc[train_idx])
        xp = scaler.transform(Xc.loc[[pred_month]])
        row_p, row_a = {}, {}
        for sec in sec_cols:
            ytr = y_fwd[sec].loc[train_idx]
            mask = ytr.notna()
            if mask.sum() < min_train // 2:
                continue
            model = Ridge(alpha=5.0).fit(Xtr[mask.values], ytr[mask])
            row_p[sec] = float(model.predict(xp)[0])
            actual = y_fwd[sec].get(pred_month, np.nan)
            row_a[sec] = actual
        oos_preds[pred_month] = row_p
        oos_actual[pred_month] = row_a

    oos_preds = pd.DataFrame(oos_preds).T
    oos_actual = pd.DataFrame(oos_actual).T

    # ---- backtest: long top-N / short bottom-N, monthly, with costs ----
    bt_rows, prev_long, prev_short = [], set(), set()
    for month in oos_preds.index:
        p = oos_preds.loc[month].dropna()
        a = oos_actual.loc[month]
        if len(p) < top_n * 2:
            continue
        longs = set(p.nlargest(top_n).index)
        shorts = set(p.nsmallest(top_n).index)
        gross = a[list(longs)].mean() - a[list(shorts)].mean()
        n_changed = len(longs ^ prev_long) + len(shorts ^ prev_short)
        cost = n_changed * cost_bp_per_change / 100 / (2 * top_n)
        bt_rows.append({"month": month, "gross": gross, "cost": cost,
                        "net": gross - cost,
                        "longs": ", ".join(SECTORS[s] for s in sorted(longs)),
                        "shorts": ", ".join(SECTORS[s] for s in sorted(shorts))})
        prev_long, prev_short = longs, shorts
    backtest = pd.DataFrame(bt_rows).set_index("month") if bt_rows else pd.DataFrame()
    if not backtest.empty:
        backtest["cumulative"] = (1 + backtest["net"] / 100).cumprod()

    stats = {}
    if not backtest.empty and backtest["net"].std() > 0:
        net = backtest["net"].dropna()
        stats = {
            "months": len(net),
            "ann_return_pct": float(net.mean() * 12),
            "ann_vol_pct": float(net.std() * np.sqrt(12)),
            "sharpe": float(net.mean() * 12 / (net.std() * np.sqrt(12))),
            "hit_rate": float((net > 0).mean()),
            "worst_month_pct": float(net.min()),
        }

    # ---- latest live prediction (train on everything, predict from last row) ----
    scaler = StandardScaler().fit(Xc)
    Xall = scaler.transform(Xc)
    x_now = scaler.transform(X.dropna().iloc[[-1]][Xc.columns])
    latest = {}
    for sec in sec_cols:
        ytr = y_fwd[sec].loc[Xc.index]
        mask = ytr.notna()
        if mask.sum() < min_train // 2:
            continue
        model = Ridge(alpha=5.0).fit(Xall[mask.values], ytr[mask])
        latest[sec] = float(model.predict(x_now)[0])
    latest_pred = pd.Series(latest).sort_values(ascending=False)
    latest_asof = X.dropna().index[-1]

    # ---- regime-conditional historical table ----
    regime_table = pd.DataFrame()
    if regime_labels.notna().any():
        joined = rel.copy()
        joined["regime"] = regime_labels
        regime_table = (joined.dropna(subset=["regime"])
                        .groupby("regime")[sec_cols].mean().T)
        regime_table.index = [SECTORS[s] for s in regime_table.index]

    return RotationResult(latest_pred, latest_asof, oos_preds, oos_actual,
                          backtest, stats, regime_table)
