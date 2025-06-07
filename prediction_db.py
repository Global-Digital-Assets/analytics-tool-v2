"""Utility functions for prediction outcome logging and performance tracking.

This small helper isolates all SQLite DDL / DML so the rest of the
code-base can stay clean.  It will automatically create the tables the
first time it is imported.

Tables
------
    prediction_outcomes
        id INTEGER PRIMARY KEY
        prediction_id TEXT UNIQUE
        symbol TEXT
        timestamp REAL         -- unix epoch seconds (float)
        predicted_direction TEXT
        predicted_magnitude REAL
        model_version TEXT
        feature_snapshot BLOB  -- json‐encoded string of feature vector / metadata
        actual_direction TEXT  -- NULL until filled by outcome_collector
        actual_magnitude REAL  -- NULL until filled
        outcome_score REAL     -- user-defined metric

    model_performance
        model_version TEXT
        timestamp REAL
        window TEXT            -- e.g. ":7d", "full"
        auc REAL
        precision REAL
        recall REAL
        sharpe REAL
        sample_count INTEGER

All write helpers open a short-lived connection to avoid keeping file
handles around.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

_DB_PATH = os.getenv("PREDICTION_DB_PATH", os.path.join(Path(__file__).parent, "prediction_outcomes.db"))

_SCHEMA_CREATED = False


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH, timeout=20, check_same_thread=False)


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they do not yet exist."""
    cur = conn.cursor()
    # prediction_outcomes
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            id                INTEGER PRIMARY KEY,
            prediction_id     TEXT UNIQUE,
            symbol            TEXT,
            timestamp         REAL,
            predicted_direction TEXT,
            predicted_magnitude REAL,
            model_version     TEXT,
            feature_snapshot  BLOB,
            actual_direction  TEXT,
            actual_magnitude  REAL,
            outcome_score     REAL
        );
        """
    )

    # model_performance
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_performance (
            model_version  TEXT,
            timestamp      REAL,
            window         TEXT,
            auc            REAL,
            precision      REAL,
            recall         REAL,
            sharpe         REAL,
            sample_count   INTEGER,
            UNIQUE(model_version, timestamp, window)
        );
        """
    )
    conn.commit()


def _ensure_schema():
    global _SCHEMA_CREATED
    if _SCHEMA_CREATED:
        return
    with _get_conn() as conn:
        _create_schema(conn)
    _SCHEMA_CREATED = True


# Public helpers ----------------------------------------------------------------

def insert_prediction(row: Dict[str, Any]) -> None:
    """Insert a single prediction row. Missing columns are ignored."""
    _ensure_schema()
    cols = [
        "prediction_id",
        "symbol",
        "timestamp",
        "predicted_direction",
        "predicted_magnitude",
        "model_version",
        "feature_snapshot",
    ]
    values = [row.get(c) for c in cols]
    placeholders = ",".join(["?"] * len(cols))
    with _get_conn() as conn:
        conn.execute(
            f"INSERT OR IGNORE INTO prediction_outcomes ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()


def fetch_pending(limit: int = 200) -> List[Dict[str, Any]]:
    """Return rows where actual_direction is NULL."""
    _ensure_schema()
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT id, prediction_id, symbol, timestamp, predicted_direction, predicted_magnitude FROM prediction_outcomes WHERE actual_direction IS NULL LIMIT ?",
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def update_outcome(row_id: int, actual_direction: str, actual_magnitude: float, outcome_score: Optional[float]) -> None:
    _ensure_schema()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE prediction_outcomes SET actual_direction = ?, actual_magnitude = ?, outcome_score = ? WHERE id = ?",
            (actual_direction, actual_magnitude, outcome_score, row_id),
        )
        conn.commit()


def log_model_performance(record: Dict[str, Any]) -> None:
    """Insert or update model_performance row."""
    _ensure_schema()
    cols = list(record.keys())
    placeholders = ",".join(["?"] * len(cols))
    with _get_conn() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO model_performance ({','.join(cols)}) VALUES ({placeholders})",
            [record[c] for c in cols],
        )
        conn.commit()


# Convenience for CLI -----------------------------------------------------------------
if __name__ == "__main__":
    _ensure_schema()
    print(f"Prediction DB ready at {_DB_PATH}")
