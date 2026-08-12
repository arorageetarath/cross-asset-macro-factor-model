"""
Layer 2 — Data Warehouse
A single SQLite table in "tall" format: (date, country, variable, value, source).
Everything — macro releases, market prices, policy rates — lives here.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "macro.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    date     TEXT NOT NULL,          -- ISO date 'YYYY-MM-DD'
    country  TEXT NOT NULL,          -- 'US', 'EZ', 'UK', 'JP', 'IN', 'CN', 'GLOBAL'
    variable TEXT NOT NULL,          -- 'CPI_YOY', 'SPX', 'US10Y', ...
    value    REAL,
    source   TEXT DEFAULT '',        -- 'FRED', 'yfinance', 'manual', ...
    PRIMARY KEY (date, country, variable)
);
CREATE INDEX IF NOT EXISTS idx_obs_var  ON observations (variable, date);
CREATE INDEX IF NOT EXISTS idx_obs_date ON observations (date);
"""


def get_conn(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def upsert(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    """
    Insert-or-replace rows. `df` must have columns:
    date, country, variable, value (source optional).
    Returns number of rows written.
    """
    if df.empty:
        return 0
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "source" not in df.columns:
        df["source"] = ""
    rows = df[["date", "country", "variable", "value", "source"]].itertuples(index=False)
    conn.executemany(
        "INSERT OR REPLACE INTO observations (date, country, variable, value, source) "
        "VALUES (?, ?, ?, ?, ?)",
        list(rows),
    )
    conn.commit()
    return len(df)


def load_series(variable: str, conn: sqlite3.Connection, country: str | None = None) -> pd.Series:
    """Return one variable as a date-indexed Series."""
    q = "SELECT date, value FROM observations WHERE variable = ?"
    params: list = [variable]
    if country:
        q += " AND country = ?"
        params.append(country)
    q += " ORDER BY date"
    df = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    return df.set_index("date")["value"].rename(variable)


def load_wide(variables: list[str], conn: sqlite3.Connection) -> pd.DataFrame:
    """Return several variables as a wide date-indexed DataFrame."""
    placeholders = ",".join("?" * len(variables))
    df = pd.read_sql(
        f"SELECT date, variable, value FROM observations "
        f"WHERE variable IN ({placeholders}) ORDER BY date",
        conn,
        params=variables,
        parse_dates=["date"],
    )
    if df.empty:
        return pd.DataFrame(columns=variables)
    return df.pivot_table(index="date", columns="variable", values="value")


def latest_date(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM observations").fetchone()
    return row[0]
