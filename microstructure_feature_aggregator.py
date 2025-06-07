"""
Microstructure Feature Aggregator
================================
Computes rolling microstructure statistics from depth_features table and stores
into microstructure_stats table for consumption by the analytics pipeline.

Design:
- Runs as a lightweight cron / systemd timer every minute (or ad-hoc).
- For each symbol:
    • Fetch depth_features rows within *WINDOW_SECONDS* (default 60).
    • Compute average spread and average imbalance.
    • Persist snapshot so the analyzer can join easily.

Feature flag controlled: enable by running script; otherwise no side-effects.

This keeps heavy calculations out of the real-time streamer while still
providing fresh microstructure-derived features for scoring/monitoring.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

WINDOW_SECONDS = 60  # rolling window size
DB_PATH = Path(__file__).resolve().parent / "market_data.db"
TOKENS: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "AVAXUSDT",
    # ... keep in sync with main config
]

def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS microstructure_stats (
            symbol TEXT,
            snapshot_ts INTEGER,
            avg_spread REAL,
            avg_imbalance REAL,
            max_abs_imbalance REAL,
            PRIMARY KEY (symbol, snapshot_ts)
        )
        """
    )
    conn.commit()


def aggregate_symbol(conn: sqlite3.Connection, symbol: str, now_ts: int) -> None:
    """Aggregate stats over recent depth_features rows."""
    cur = conn.cursor()
    window_start = now_ts - WINDOW_SECONDS * 1000  # depth_features uses ms
    cur.execute(
        """
        SELECT spread, imbalance FROM depth_features
        WHERE symbol = ? AND event_ts >= ?
        """,
        (symbol, window_start),
    )
    rows = cur.fetchall()
    if not rows:
        return  # nothing to aggregate yet

    spreads = [r[0] for r in rows]
    imbs = [r[1] for r in rows]
    avg_spread = sum(spreads) / len(spreads)
    avg_imbalance = sum(imbs) / len(imbs)
    max_abs_imbalance = max(abs(x) for x in imbs)

    # Persist
    cur.execute(
        """
        INSERT OR REPLACE INTO microstructure_stats
        (symbol, snapshot_ts, avg_spread, avg_imbalance, max_abs_imbalance)
        VALUES (?, ?, ?, ?, ?)
        """,
        (symbol, now_ts, avg_spread, avg_imbalance, max_abs_imbalance),
    )
    conn.commit()


def main() -> None:
    now_ts = int(time.time() * 1000)
    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    for sym in TOKENS:
        aggregate_symbol(conn, sym, now_ts)

    conn.close()
    print(f"✅ Microstructure stats updated at {datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()
