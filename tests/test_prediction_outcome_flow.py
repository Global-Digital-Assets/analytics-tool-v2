"""Integration test: prediction insert ➜ outcome collector scoring.

Runs the full flow in-memory using a temporary SQLite DB. Ensures that
`outcome_collector.evaluate_batch()` correctly fills `actual_direction`
and `outcome_score` based on mocked price data.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sqlite3
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_tmp_db(tmp_path: Path) -> Path:
    """Configure PREDICTION_DB_PATH to point at a temp file and reload the DB."""
    db_path = tmp_path / "prediction_outcomes.db"
    os.environ["PREDICTION_DB_PATH"] = str(db_path)

    # Reload prediction_db so the new env var takes effect
    import prediction_db  # local import to avoid circular issues

    importlib.reload(prediction_db)
    return db_path


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------


def test_prediction_to_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Insert dummy prediction, run collector, expect BUY_LONG outcome == 1.0."""
    db_path = _setup_tmp_db(tmp_path)

    # Now that DB path is fixed, (re-)import outcome_collector
    import outcome_collector  # noqa: WPS433 (allow import inside test)

    importlib.reload(outcome_collector)

    import prediction_db  # re-import after reload to get updated module

    # Insert dummy prediction older than the evaluation delay (6 h default)
    prediction_db.insert_prediction(
        {
            "prediction_id": "dummy_flow_1",
            "symbol": "TESTUSDT",
            "timestamp": time.time() - 8 * 3600,  # 8 h ago
            "predicted_direction": "BUY_LONG",
            "predicted_magnitude": 75.0,
            "model_version": "integration_test",
            "feature_snapshot": "{}",
        }
    )

    # Monkey-patch price helpers to deterministic values
    monkeypatch.setattr(outcome_collector, "_get_price_at", lambda symbol, ts: 100.0)
    monkeypatch.setattr(outcome_collector, "_get_price_now", lambda symbol: 110.0)

    # Run evaluation coroutine once
    asyncio.run(outcome_collector.evaluate_batch())

    # Verify DB updated
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT actual_direction, outcome_score FROM prediction_outcomes WHERE prediction_id = ?",
            ("dummy_flow_1",),
        ).fetchone()

    assert row is not None, "Row should exist"
    actual_dir, outcome_score = row
    assert actual_dir == "BUY_LONG"
    assert outcome_score == 1.0
