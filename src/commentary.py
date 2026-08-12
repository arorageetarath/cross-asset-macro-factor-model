"""
Layer 5 — Automatic Commentary

Turns an Explanation object into a short macro-strategist-style note.
Pure templating (no LLM required), so output is deterministic and auditable.
"""
from __future__ import annotations

import pandas as pd

from src.attribution import Explanation
from src.features import FEATURE_LABELS

INDEX_NAMES = {
    "RET_SPX": "US equities (S&P 500)",
    "RET_NASDAQ": "the Nasdaq",
    "RET_STOXX50": "Eurozone equities (Euro Stoxx 50)",
    "RET_NIFTY50": "Indian equities (Nifty 50)",
    "RET_MSCI_EM": "emerging-market equities",
}


def _direction(x: float) -> str:
    return "declined" if x < 0 else "rose"


def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").lower())


def _describe_driver(feature: str, contribution: float, move_sign: float) -> str:
    """Human phrasing: whether this factor pushed with or against the day's move."""
    lbl = _label(feature)
    if contribution * move_sign > 0:
        return lbl
    return f"{lbl} (partially offsetting)"


def write_note(exp: Explanation, top_n: int = 3) -> str:
    name = INDEX_NAMES.get(exp.target, exp.target)
    move = exp.actual_return
    date_str = exp.date.strftime("%d %b %Y")

    lines = [f"{name[0].upper() + name[1:]} {_direction(move)} {abs(move):.1f}% on {date_str}."]

    top = exp.contributions.head(top_n)
    if len(top) == 0:
        lines.append(
            "The model attributes little of today's move to the tracked macro factors, "
            "suggesting idiosyncratic or news-driven flows."
        )
    else:
        drivers = [_describe_driver(f, c, move) for f, c in top.items()]
        if len(drivers) == 1:
            driver_txt = drivers[0]
        else:
            driver_txt = ", ".join(drivers[:-1]) + f" and {drivers[-1]}"
        lines.append(f"The largest estimated contributors were {driver_txt}.")

    explained = exp.predicted_return
    unexplained = abs(exp.residual)
    if unexplained > abs(move) * 0.5 and abs(move) > 0.3:
        lines.append(
            f"Roughly {unexplained:.1f}pp of the move is unexplained by the factor model, "
            "pointing to drivers outside the tracked macro set (earnings, positioning, headlines)."
        )
    else:
        lines.append(
            f"The factor model accounts for most of the move "
            f"(predicted {explained:+.1f}% vs actual {move:+.1f}%)."
        )

    if not exp.analogues.empty:
        avg = exp.analogues["return_that_day"].mean()
        dates = ", ".join(d.strftime("%b %Y") for d in exp.analogues["date"].head(3))
        lines.append(
            f"The closest historical analogues ({dates}) saw an average same-day "
            f"return of {avg:+.1f}%, "
            + ("consistent with today's risk-off tone."
               if avg < 0 else "a more mixed historical signal.")
        )

    lines.append(
        f"(In-sample R² of the rolling factor model: {exp.r2_train:.0%}. "
        "Attribution is statistical, not causal.)"
    )
    return "\n".join(lines)


def contributions_table(exp: Explanation) -> pd.DataFrame:
    df = exp.contributions.rename("contribution_pct").to_frame()
    df["factor"] = [_label(f) for f in df.index]
    return df[["factor", "contribution_pct"]].reset_index(drop=True)
