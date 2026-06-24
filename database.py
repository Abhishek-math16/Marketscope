"""
database.py
-----------
A small SQLite layer. SQLite is a single self-contained file (data/app.db),
needs no separate server, and is perfect for a project like this.

It stores three things:
  searches      – every ticker someone looks up (powers "Recently viewed")
  predictions   – the accuracy metrics from each run (a history you can show)
  model_cache   – when each ticker's model was last trained, so we know
                  whether we can reuse a saved model instead of retraining
"""

import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join("data", "app.db")


def _connect():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    conn = _connect()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS searches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            searched_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker        TEXT NOT NULL,
            rmse          REAL,
            mae           REAL,
            mape          REAL,
            baseline_rmse REAL,
            beats_baseline INTEGER,
            last_price    REAL,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_cache (
            ticker     TEXT PRIMARY KEY,
            trained_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def log_search(ticker):
    conn = _connect()
    conn.execute(
        "INSERT INTO searches (ticker, searched_at) VALUES (?, ?)",
        (ticker, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def save_prediction(ticker, metrics, last_price):
    conn = _connect()
    conn.execute(
        """INSERT INTO predictions
           (ticker, rmse, mae, mape, baseline_rmse, beats_baseline, last_price, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker,
            metrics.get("rmse"),
            metrics.get("mae"),
            metrics.get("mape"),
            metrics.get("baseline_rmse"),
            1 if metrics.get("beats_baseline") else 0,
            last_price,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def recent_searches(limit=8):
    """Distinct most-recent tickers, newest first."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT ticker, MAX(searched_at) AS last_seen
        FROM searches
        GROUP BY ticker
        ORDER BY last_seen DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [r["ticker"] for r in rows]


def mark_model_trained(ticker):
    conn = _connect()
    conn.execute(
        """INSERT INTO model_cache (ticker, trained_at) VALUES (?, ?)
           ON CONFLICT(ticker) DO UPDATE SET trained_at=excluded.trained_at""",
        (ticker, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def model_is_fresh(ticker, max_age_hours=24):
    """True if we trained this ticker's model recently enough to reuse it."""
    conn = _connect()
    row = conn.execute(
        "SELECT trained_at FROM model_cache WHERE ticker = ?", (ticker,)
    ).fetchone()
    conn.close()
    if not row:
        return False
    trained_at = datetime.fromisoformat(row["trained_at"])
    return datetime.utcnow() - trained_at < timedelta(hours=max_age_hours)
