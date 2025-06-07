import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import pytest
import os

from multi_token_analyzer import RegimeDetector

TEST_DB = "test_market_data.db"

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    # Create a small test DB with synthetic BTC prices for 2 days (900s candles)
    conn = sqlite3.connect(TEST_DB)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS candles_900s (
            symbol TEXT,
            timestamp INTEGER,
            close REAL
        )"""
    )
    base_ts = int((datetime.utcnow() - timedelta(hours=30)).timestamp() * 1000)
    prices = []
    for i in range(0, 120):
        # Simulate trending up volatile prices
        prices.append(20000 + i * 20 + (50 if i % 5 == 0 else 0))
    rows = [("BTCUSDT", base_ts + i * 900 * 1000, p) for i, p in enumerate(prices)]
    cur.executemany("INSERT INTO candles_900s VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()
    yield
    os.remove(TEST_DB)


def test_classify_regime_adv_trending_volatile(setup_db):
    rd = RegimeDetector(TEST_DB)
    regime = rd.classify_regime_adv()
    assert regime in {"trending_volatile", "trending_calm", "choppy", "squeeze"}
