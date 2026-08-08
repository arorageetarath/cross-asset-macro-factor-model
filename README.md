# What Moves Markets?

A system that *explains* daily market moves — rather than predicting them — using
macro data, monetary policy, and factor attribution. Modeled on how macro
strategy desks actually work.

## Architecture

| Layer | Module | What it does |
|---|---|---|
| 1. Collection | `src/collect/markets.py`, `src/collect/macro.py` | Daily prices via yfinance; macro releases via FRED |
| 2. Warehouse | `src/db.py` | SQLite, tall schema `(date, country, variable, value, source)` |
| 3. Features | `src/features.py` | Yield/curve/real-yield changes, dollar & oil momentum, credit spreads, VIX, statistical macro *surprises* |
| 4. Research engine | `src/attribution.py` | Rolling standardized LassoCV; exact contribution decomposition + nearest historical analogues |
| 5. Commentary | `src/commentary.py` | Deterministic strategist-style note from the attribution |
| 6. Prediction | `src/signal.py` | Residual-reversal overnight signal: walk-forward residuals, lead-lag t-stat, shown only when significant |
| Dashboard | `app/dashboard.py` | Streamlit: the tape, driver chart, overnight signal, analogues, correlation matrix |

Optional extension modules (not wired into the dashboard): `src/regimes.py`
(growth×inflation regime classification) and `src/rotation.py` (walk-forward
sector rotation with OOS backtest) — a heavier predictive layer kept in the
repo for future work.

## Quick start

```bash
pip install -r requirements.txt

# Option A — real data (needs a free FRED key: fred.stlouisfed.org)
export FRED_API_KEY=your_key
python scripts/daily_update.py            # collect + explain today

# Option B — no APIs, just test the pipeline
python scripts/seed_synthetic.py
python scripts/daily_update.py --skip-collect

# Dashboard
streamlit run app/dashboard.py
```

Explain a specific day or index:

```bash
python scripts/daily_update.py --skip-collect --target RET_NIFTY50 --date 2026-03-10
```

## Methodology notes (be honest about these in any write-up)

- **Surprises are proxied.** True consensus expectations (Bloomberg/Refinitiv) are
  paywalled, so `features.surprise()` z-scores each release against its own rolling
  history. Document this limitation; upgrading to real consensus data is a clean
  extension.
- **Attribution is statistical, not causal.** Contributions are `beta * feature`,
  which sum exactly to the model's prediction; the residual is reported, never hidden.
- **No look-ahead.** The model for day *t* is trained strictly on days before *t*.
- **Endogeneity caveat.** Same-day regressions of equities on yields/VIX capture
  co-movement, not direction of causality. VIX in particular is nearly mechanical.
  Consider a "macro-only" feature set as a robustness check.

## Roadmap

1. **News layer** — pull central-bank RSS feeds and Reuters headlines; tag release
   days; optional NLP sentiment.
2. **Policy tracker** — FOMC/ECB/RBI meeting calendar, statement diffs.
3. **Nonlinear models** — RandomForest/GradientBoosting behind the same
   `explain_day` interface, with SHAP values replacing linear contributions.
4. **Real consensus surprises** — scrape or license expectations data.
5. **Automation** — GitHub Actions cron job committing daily notes to the repo.
6. **More data sources** — dbnomics for India/China releases FRED doesn't mirror;
   ECB SDW for daily Bund yields.
