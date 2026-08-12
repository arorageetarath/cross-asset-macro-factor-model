# What Moves Markets — User Manual

## What this system is

A daily market-attribution engine. It does **not** forecast prices. Given a day's
market move, it estimates *which macro forces explain it* — the same question a
macro strategist answers in a morning note.

## How it works (the five layers)

1. **Collection** — daily prices from Yahoo Finance (equities, rates, FX,
   commodities, VIX) and macro releases plus policy/credit/liquidity series from
   FRED, the St. Louis Fed's public database.
2. **Warehouse** — everything lands in one SQLite table:
   `(date, country, variable, value, source)`.
3. **Features** — raw series become the variables strategists discuss:
   basis-point changes in the 10Y yield, curve steepening (2s10s), real-yield and
   breakeven changes, dollar and oil moves and momentum, credit-spread widening,
   VIX moves, Fed balance-sheet momentum, and **macro surprises** (see below).
4. **Research engine** — for the day being explained, a LASSO regression is
   trained on the ~500 trading days *before* it (never after — no look-ahead).
   LASSO's key property: it pushes irrelevant factors' coefficients to exactly
   zero, so it selects the drivers that matter. Each factor's contribution =
   its learned sensitivity × its move that day. Contributions plus the residual
   sum exactly to the actual return.
5. **Commentary** — the attribution is rendered as a short, deterministic
   strategist note (templated, fully auditable — no black box).

## Reading the dashboard

**The tape** — latest available level for each instrument and its change vs the
prior close, each stamped with its own "as of" date. Instruments trade on
different calendars (US holidays ≠ Indian holidays), so dates can differ across
the row — that is expected, not a bug.

**Mode (sidebar)** —
- *Full factor set*: every feature, including VIX and credit spreads. Best fit,
  but VIX is computed **from** S&P 500 option prices, so "VIX up, stocks down"
  is near-mechanical: two views of the same event, not an explanation.
- *Macro-only*: drops VIX, credit spreads, and the financial-conditions index.
  Lower R², but the attribution now runs through genuine macro channels —
  rates, inflation expectations, the dollar, oil, data surprises. **Use this
  mode when you want the economic story; use full mode when you want fit.**

**Largest drivers (bar chart)** — signed contributions in percentage points of
the day's return. Green pushed the market up, red pushed it down, sorted by
size. A bar of −0.6 on "10Y Treasury yield change" reads: rising yields alone
account for an estimated 0.6pp of the decline.

**The note** — the same numbers in prose. Watch two figures:
- *Predicted vs actual*: the gap is the **residual** — the part of the move the
  macro factors can't explain (earnings, geopolitics, flows). A large residual
  is honest information, not failure.
- *R²*: how much of daily return variance the model explains historically.
  20–40% is normal for macro-only; higher with VIX included (partly for the
  mechanical reason above).

**Historical analogues** — the past days whose factor fingerprint most resembles
the chosen day (nearest neighbours in standardized feature space, using only the
factors LASSO kept). "Return that day" shows how the market behaved on those
lookalike days — context, not prophecy.

**Correlation matrix** — how the factors co-move with *each other* (last 250
days). Red = move together, blue = move opposite. When two factors are deep
red, the model cannot cleanly split credit between them; read their bars as a
combined contribution rather than trusting the split.

## Interpretation rules of thumb

- Attribution is **statistical, not causal**. Same-day regressions capture
  co-movement.
- Surprises are **proxied**: each release is z-scored against its own recent
  history (consensus-forecast data is paywalled). A CPI "surprise" here means
  "hotter/cooler than its recent trend," not "vs economist consensus."
- Small bars (< 0.05pp) are noise; don't narrate them.
- Cross-check any bold claim against the correlation matrix and the residual.
- If the model attributes little and the residual is large, the honest note is:
  "today was driven by something outside the macro factor set."

## The overnight signal (predictive layer)

Under the daily note sits a small predictive element built on a classic quant
concept: **residual reversal**. The attribution model says what the market
"should" have returned given the macro tape; the gap between actual and
model-implied return is the residual. If residuals mean-revert, a market that
underperformed its macro drivers today tends to catch up tomorrow.

How it is computed, honestly:
- Residuals are **walk-forward**: the model scoring day t was trained only on
  days before t. No look-ahead anywhere.
- The residual uses the **macro-only** factor set — the full set includes VIX,
  which mechanically absorbs the move and leaves nothing meaningful behind.
- A lead-lag regression (tomorrow's return on today's residual) is evaluated
  over the full out-of-sample history, and its **t-statistic is displayed**.

Reading it:
- 🟢/🔴 appears only when BOTH the historical relationship is statistically
  significant (|t| ≥ 2) AND today's residual is unusually large (|z| ≥ 1).
- ⚪ "No significant signal" is the expected state much of the time. Markets
  rarely leave easy overnight edges lying around; showing a null result
  honestly is the point of the design, and saying so is a strength in any
  write-up or interview.
- "Expected edge" is the regression's point estimate for tomorrow — a
  statistical tendency with wide error bands, **not investment advice**.

## Model outlook (regime playbook)

A decision-support card, deliberately ML-free: today is classified into a
growth × inflation regime (Reflation / Goldilocks / Stagflation / Slowdown),
and sectors are ranked by their historical average monthly return **relative to
the S&P 500 within that regime**. Sectors beating the index by ≥ 0.15pp/month
historically are "favoured"; trailing by the same margin, "less favourable."

**Confidence** reflects two things: how many months of history the current
regime has (sample size), and how far today's composites sit from the quadrant
boundaries (a regime call at growth +0.05σ is a coin toss; at +1.2σ it's a
conviction call). Low confidence means: don't lean on the tilts.

## Reading the analogue cards

Each card is a past day whose macro fingerprint most resembles the selected
day. "Market context" lists that day's most unusual factor moves. "Subsequent
performance" shows what the index actually did over the following week and
month, and which sector led. These are historical facts about specific dates —
useful context for judgment, **not a forecast**: three analogues is a tiny
sample, and regimes differ in ways the distance metric can't capture.

## Known limitations / roadmap

- No news layer yet (headlines, central-bank statements).
- International coverage is thinner than US (FRED mirrors are mostly monthly).
- Consensus surprises, SHAP-based nonlinear models, and an FOMC/ECB/RBI meeting
  tracker are planned extensions.
