"""
Dashboard — run from repo root:  streamlit run app/dashboard.py
(or on Windows without PATH:     py -m streamlit run app/dashboard.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import db
from src.attribution import explain_day
from src.commentary import write_note
from src.features import FEATURE_LABELS, MECHANICAL_FEATURES, build_features
from src.signal import build_signal
from src.outlook import build_outlook
from src.analogues import enrich_analogues
from src.regimes import REGIME_COLORS

st.set_page_config(page_title="What Moves Markets?", page_icon="📈", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1250px; }
  h1 { font-weight: 800; letter-spacing: -0.02em; }
  [data-testid="stMetric"] {
      background: #161B26; border: 1px solid #262d3d; border-radius: 12px;
      padding: 12px 12px 8px 12px;
  }
  [data-testid="stMetricLabel"] { opacity: 0.75; font-size: 0.78rem; }
  [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; }
  [data-testid="stMetricDelta"] { font-size: 0.82rem; }
  .asof { color: #8a93a6; font-size: 0.72rem; margin-top: -6px; }
  .note-card {
      background: #161B26; border: 1px solid #262d3d; border-left: 4px solid #E8B44C;
      border-radius: 12px; padding: 18px 22px; line-height: 1.65; font-size: 0.98rem;
  }
  hr { border-color: #262d3d; }
</style>
""", unsafe_allow_html=True)

GREEN, RED, MUTED = "#3ddc84", "#ff5c5c", "#8a93a6"

TAPE_SPEC = [  # (variable, display name, kind)
    ("SPX", "S&P 500", "index"), ("NASDAQ", "Nasdaq", "index"),
    ("NIFTY50", "Nifty 50", "index"), ("US10Y", "US 10Y", "yield"),
    ("GOLD", "Gold", "index"), ("BRENT", "Brent", "index"),
    ("DXY", "Dollar Idx", "index"), ("VIX", "VIX", "index"),
]


@st.cache_data(ttl=3600)
def load():
    conn = db.get_conn()
    feats = build_features(conn)
    tape = db.load_wide([v for v, _, _ in TAPE_SPEC] + ["US10Y_FRED"], conn)
    return feats, tape


try:
    features, tape = load()
except Exception as e:  # noqa: BLE001
    st.error(f"No data yet — run `python scripts/daily_update.py` first. ({e})")
    st.stop()

# ---------------- sidebar ----------------
st.sidebar.header("Controls")
targets = [c for c in features.columns if c.startswith("RET_")]
nice = {"RET_SPX": "S&P 500", "RET_NASDAQ": "Nasdaq", "RET_STOXX50": "Euro Stoxx 50",
        "RET_NIFTY50": "Nifty 50", "RET_MSCI_EM": "MSCI EM"}
target = st.sidebar.selectbox("Explain", targets, index=0,
                              format_func=lambda t: nice.get(t, t))

dates = features[target].dropna().index
date = st.sidebar.date_input("Date", value=dates[-1].date(),
                             min_value=dates[60].date(), max_value=dates[-1].date())

mode = st.sidebar.radio(
    "Factor set", ["Full", "Macro-only"], index=0,
    help=("Full: all factors, best fit — but VIX/credit are near-mechanically tied "
          "to equities. Macro-only drops them for a purer economic story."),
)
exclude = MECHANICAL_FEATURES if mode == "Macro-only" else None
st.sidebar.caption("Macro-only removes: VIX, credit spreads, financial-conditions index.")

# ---------------- header ----------------
st.title("What Moved Markets Today?")

# ---------------- the tape (uses each instrument's own latest values) ----------------
cols = st.columns(len(TAPE_SPEC))
for col, (var, name, kind) in zip(cols, TAPE_SPEC):
    s = tape[var].dropna() if var in tape else pd.Series(dtype=float)
    if var == "US10Y" and len(s) < 2 and "US10Y_FRED" in tape:
        s = tape["US10Y_FRED"].dropna()
    if len(s) < 2:
        col.metric(name, "—", "no data")
        continue
    last, prev = s.iloc[-1], s.iloc[-2]
    if kind == "yield":
        col.metric(name, f"{last:.2f}%", f"{(last - prev) * 100:+.0f}bp")
    else:
        fmt = f"{last:,.0f}" if last >= 100 else f"{last:,.2f}"
        col.metric(name, fmt, f"{(last / prev - 1) * 100:+.2f}%")
    col.markdown(f"<div class='asof'>as of {s.index[-1].date()}</div>",
                 unsafe_allow_html=True)

st.markdown("---")

tab_today, tab_guide = st.tabs(["📊  Today's attribution", "📖  Guide"])

# ================= TAB 1: attribution =================
with tab_today:
    try:
        exp = explain_day(features, target=target, date=str(date), exclude=exclude)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    left, right = st.columns([5, 6], gap="large")

    with left:
        st.subheader(f"{nice.get(target, target)} — {exp.date.date()}")
        a, b, c = st.columns(3)
        a.metric("Actual", f"{exp.actual_return:+.2f}%")
        b.metric("Model", f"{exp.predicted_return:+.2f}%")
        c.metric("Unexplained", f"{exp.residual:+.2f}pp")
        st.markdown(f"<div class='note-card'>{write_note(exp).replace(chr(10), '<br><br>')}</div>",
                    unsafe_allow_html=True)

        # ---- overnight residual signal ----
        @st.cache_data(ttl=3600, show_spinner="Evaluating residual signal…")
        def load_signal(tgt: str):
            return build_signal(features, target=tgt)

        st.markdown("##### Overnight signal (residual reversal)")
        try:
            sig = load_signal(target)
            if sig.direction == "Bullish":
                icon, color, label = "🟢", GREEN, "Bullish overnight bias"
            elif sig.direction == "Bearish":
                icon, color, label = "🔴", RED, "Bearish overnight bias"
            else:
                icon, color, label = "⚪", MUTED, "No significant signal"
            sig_note = (
                f"{icon} <b style='color:{color}'>{label}</b><br>"
                f"Today's residual: <b>{sig.residual_today:+.2f}pp</b> "
                f"(z = {sig.residual_z:+.1f} vs last 60 days)<br>"
                f"Expected next-day edge: <b>{sig.expected_edge:+.2f}pp</b><br>"
                f"<span style='color:#8a93a6'>Lead-lag t-stat: {sig.t_stat:+.2f} over "
                f"{sig.n_obs} out-of-sample days "
                f"{'— statistically significant' if sig.significant else '— NOT significant; treat the edge as noise'}."
                f" Macro-only residual; not investment advice.</span>"
            )
            st.markdown(f"<div class='note-card' style='border-left-color:{color}'>{sig_note}</div>",
                        unsafe_allow_html=True)
        except ValueError as e:
            st.caption(f"Signal unavailable: {e}")

    with right:
        st.subheader("Largest drivers")
        contrib = exp.contributions.head(10)
        contrib = contrib[contrib.abs() > 1e-4]  # hide pure-noise slivers
        if contrib.empty:
            st.info("No factor moved enough to register — likely an idiosyncratic day.")
        else:
            plot = contrib.iloc[::-1]
            labels = [FEATURE_LABELS.get(i, i) for i in plot.index]
            fig = go.Figure(go.Bar(
                x=plot.values, y=labels, orientation="h",
                marker_color=[GREEN if v > 0 else RED for v in plot.values],
                text=[f"{v:+.2f}" for v in plot.values],
                textposition="auto", insidetextanchor="middle",
                insidetextfont=dict(color="#0E1117", size=12),
                outsidetextfont=dict(color="#E6E6E6", size=12),
                cliponaxis=False,
                hovertemplate="%{y}: %{x:+.2f}pp<extra></extra>",
            ))
            fig.update_layout(
                height=max(300, 42 * len(plot)),
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis_title="contribution (pp of return)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6E6E6"),
                xaxis=dict(zerolinecolor=MUTED, gridcolor="#262d3d"),
            )
            st.plotly_chart(fig, width='stretch')

    # ---------------- model outlook (regime playbook) ----------------
    st.markdown("---")
    st.subheader("Model outlook")

    @st.cache_data(ttl=3600)
    def load_outlook():
        return build_outlook(db.get_conn())

    try:
        out = load_outlook()
        reg = out["regime"]
        rcol = REGIME_COLORS.get(reg["regime"], MUTED)
        oc1, oc2, oc3, oc4 = st.columns([4, 3, 3, 3], gap="medium")
        with oc1:
            st.markdown(
                f"<div class='note-card' style='border-left-color:{rcol}'>"
                f"Current macro regime<br>"
                f"<span style='font-size:1.4rem;font-weight:800;color:{rcol}'>{reg['regime']}</span><br>"
                f"<span style='color:#8a93a6'>growth {reg['growth_z']:+.2f}σ · "
                f"inflation {reg['inflation_z']:+.2f}σ · as of {reg['date'].date()}</span><br><br>"
                f"Confidence: <b>{out['confidence']}</b> "
                f"<span style='color:#8a93a6'>({out['n_months']} months of history in this regime)</span>"
                f"</div>", unsafe_allow_html=True)
        def sector_list(title, names, color):
            items = "<br>".join(f"{'↑' if color==GREEN else ('↓' if color==RED else '·')} {n}"
                                for n in names) or "—"
            return (f"<div class='note-card' style='border-left-color:{color}'>"
                    f"<b>{title}</b><br><br>{items}</div>")
        oc2.markdown(sector_list("Historically favoured", out["favours"], GREEN),
                     unsafe_allow_html=True)
        oc3.markdown(sector_list("Neutral", out["neutral"], MUTED), unsafe_allow_html=True)
        oc4.markdown(sector_list("Less favourable", out["less_favourable"], RED),
                     unsafe_allow_html=True)
        st.caption("Playbook = average monthly sector return vs S&P 500 within the current "
                   "regime, over the full sample. Historical tendency, not investment advice.")
    except RuntimeError as e:
        st.info(str(e))

    # ---------------- rich analogue cards ----------------
    if not exp.analogues.empty:
        st.markdown("---")
        st.subheader("Closest historical analogues")
        cards = enrich_analogues(exp, features, db.get_conn())
        acols = st.columns(len(cards), gap="medium")
        for col, c in zip(acols, cards):
            ctx = "<br>".join(f"• {b}" for b in c["context"])
            def fr(v):
                if v is None:
                    return "<span style='color:#8a93a6'>n/a</span>"
                color = GREEN if v > 0 else RED
                return f"<b style='color:{color}'>{v:+.1f}%</b>"
            leader = (f"<br><span style='color:#8a93a6'>Sector leader next month:</span> "
                      f"<b>{c['sector_leader']}</b>" if c["sector_leader"] else "")
            col.markdown(
                f"<div class='note-card'>"
                f"<b style='font-size:1.05rem'>{c['date'].strftime('%d %b %Y')}</b> "
                f"<span style='color:#8a93a6'>· same-day {c['same_day']:+.1f}% · "
                f"similarity {c['distance']:.2f}</span><br><br>"
                f"<span style='color:#8a93a6'>Market context that day</span><br>{ctx}<br><br>"
                f"<span style='color:#8a93a6'>Subsequent performance</span><br>"
                f"1 week: {fr(c['ret_1w'])} &nbsp;·&nbsp; 1 month: {fr(c['ret_1m'])}"
                f"{leader}</div>", unsafe_allow_html=True)
        st.caption("Subsequent-performance figures are historical facts about those specific "
                   "dates — context for judgment, not a forecast.")

    with st.expander("Feature correlation matrix (last 250 days)"):
        feat_cols = [c for c in features.columns
                     if not c.startswith("RET_") and (not exclude or c not in exclude)]
        corr = features[feat_cols].tail(250).corr()
        corr.index = corr.columns = [FEATURE_LABELS.get(c, c) for c in corr.columns]
        heat = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                         aspect="auto", height=620)
        heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E6E6E6"),
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(heat, width='stretch')
        st.caption("Deep red pairs move together — the model cannot cleanly split "
                   "credit between them; read their bars as a combined contribution.")

# ================= TAB 2: guide =================
with tab_guide:
    manual = ROOT / "USER_MANUAL.md"
    if manual.exists():
        st.markdown(manual.read_text(encoding="utf-8"))
    else:
        st.info("USER_MANUAL.md not found in the project folder.")
