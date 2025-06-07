#!/usr/bin/env python3
"""Bulk historical candle back-fill for missing symbols.

Fetches 1-minute klines from Binance REST API and inserts them into
`candles` table of `market_data.db` so the live analyzer has enough
history.  Designed to be idempotent: existing rows are skipped via
PRIMARY KEY (symbol, timestamp).

Example:
    python historical_backfill.py \
        --symbols MATICUSDT,ALGOUSDT,BCHUSDT \
        --days 180
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import List
import json
import urllib.request
import urllib.parse
import ssl

API_URL = "https://api.binance.com/api/v3/klines"
DB_PATH = "market_data.db"  # relative to script working directory on VPS
MAX_LIMIT = 1000  # Binance max rows per request
REQUEST_SLEEP = 0.3  # seconds between consecutive requests to be gentle

ssl_ctx = ssl.create_default_context()

def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> List[list]:
    """Fetch klines [openTime, open, high, low, close, volume, ...]."""
    qs = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
    )
    url = f"{API_URL}?{qs}"
    with urllib.request.urlopen(url, context=ssl_ctx, timeout=15) as resp:
        return json.load(resp)


def backfill_symbol(symbol: str, days: int):
    print(f"▶ Back-filling {symbol} for {days} days (~{days*1440} candles)…")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT,
            timestamp INTEGER,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, timestamp)
        )"""
    )
    conn.commit()

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    fetched = 0
    while start_ms < end_ms:
        batch_end = min(start_ms + MAX_LIMIT * 60_000, end_ms)
        try:
            klines = fetch_klines(symbol, start_ms, batch_end - 1)
        except Exception as exc:
            print(f"  ⚠️ Request error {exc} – sleeping 5s and retrying…")
            time.sleep(5)
            continue

        if not klines:
            break

        rows = [
            (
                symbol,
                int(k[0]),
                float(k[1]),
                float(k[2]),
                float(k[3]),
                float(k[4]),
                float(k[5]),
            )
            for k in klines
        ]
        cur.executemany(
            """INSERT OR IGNORE INTO candles (symbol, timestamp, open, high, low, close, volume)
               VALUES (?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        fetched += len(rows)

        # Move start pointer
        start_ms = klines[-1][0] + 60_000  # next minute
        print(f"    …{fetched} rows")
        time.sleep(REQUEST_SLEEP)
    conn.close()
    print(f"✅ {symbol} done – inserted ≈{fetched} rows")


def main():
    parser = argparse.ArgumentParser(description="Historical candle back-fill")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbol list (e.g. BTCUSDT,ETHUSDT)")
    parser.add_argument("--days", type=int, default=180, help="Lookback window in days (default 180)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        backfill_symbol(sym, args.days)


if __name__ == "__main__":
    main()
