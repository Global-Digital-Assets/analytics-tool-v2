#!/usr/bin/env python3
"""Outcome Collector – Phase-1 of adaptive learning.

Runs as a lightweight daemon every 5 minutes:
1. Pull predictions that have no outcome yet (prediction_db.fetch_pending).
2. For rows older than EVAL_DELAY (default 6 h) evaluate:
   • actual_direction  – direction of realised move.
   • actual_magnitude  – % move from entry price to now.
   • outcome_score     – 1 if predicted direction correct else 0.
3. Write back via prediction_db.update_outcome.

A simple REST price query keeps the dependency footprint tiny – no ccxt
or python-binance required.  If rate-limits become an issue, switch to a
shared price cache or local market_data.db query.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

import sqlite3
import requests

import prediction_db

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- config --------------------------------------------------------------------
LOG_LEVEL      = os.getenv("COLLECTOR_LOG_LEVEL", "INFO").upper()
EVAL_DELAY_H   = float(os.getenv("COLLECTOR_EVAL_DELAY_H", 6))       # wait 6 h before scoring
BATCH_LIMIT    = int(os.getenv("COLLECTOR_BATCH_LIMIT", 200))
MARKET_DB_PATH = os.getenv("MARKET_DB_PATH", os.path.join(SCRIPT_DIR, "market_data.db"))
BINANCE_API    = os.getenv("BINANCE_REST", "https://api.binance.com/api/v3/ticker/price?symbol={symbol}")

logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s [collector] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("collector")

# --- helpers -------------------------------------------------------------------

def _get_price_now(symbol: str) -> Optional[float]:
    url = BINANCE_API.format(symbol=symbol)
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception as exc:
        logger.warning(f"Could not fetch spot price for {symbol}: {exc}")
        return None


def _get_price_at(symbol: str, ts_seconds: float) -> Optional[float]:
    """Return closest candle close ≤ timestamp."""
    try:
        with sqlite3.connect(MARKET_DB_PATH, timeout=15) as conn:
            row = conn.execute(
                "SELECT close FROM candles WHERE symbol = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
                (symbol, int(ts_seconds * 1000)),
            ).fetchone()
            return row[0] if row else None
    except Exception as exc:
        logger.warning(f"DB lookup failed for {symbol}: {exc}")
        return None


async def evaluate_batch():
    pending = prediction_db.fetch_pending(limit=BATCH_LIMIT)
    if not pending:
        return

    eval_cutoff = time.time() - EVAL_DELAY_H * 3600
    updated = 0

    for row in pending:
        if row["timestamp"] > eval_cutoff:
            # still too fresh – skip until next cycle
            continue

        symbol = row["symbol"]
        entry_ts = row["timestamp"]

        entry_price = _get_price_at(symbol, entry_ts)
        current_price = _get_price_now(symbol)

        if entry_price is None or current_price is None:
            continue  # try again next cycle

        pct_move = (current_price - entry_price) / entry_price * 100
        actual_dir = "BUY_LONG" if pct_move > 0 else "SHORT" if pct_move < 0 else "FLAT"

        pred_dir_raw = row["predicted_direction"]
        pred_dir = "BUY_LONG" if pred_dir_raw in {"BUY_LONG", "LONG"} else "SHORT"

        outcome_score = 1.0 if (
            (pct_move > 0 and pred_dir == "BUY_LONG") or (pct_move < 0 and pred_dir == "SHORT")
        ) else 0.0

        prediction_db.update_outcome(
            row_id=row["id"],
            actual_direction=actual_dir,
            actual_magnitude=pct_move,
            outcome_score=outcome_score,
        )
        updated += 1

    if updated:
        logger.info(f"Evaluated {updated} predictions → DB updated")


async def main_loop():
    while True:
        try:
            await evaluate_batch()
        except Exception as e:
            logger.exception("Collector error")
        await asyncio.sleep(300)  # 5 min


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("Outcome collector stopped")
