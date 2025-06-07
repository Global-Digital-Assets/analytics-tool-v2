"""
Microstructure Streamer
======================
Collects real-time tick-by-tick (aggTrades) data from Binance and persists to
SQLite.  Designed to run **alongside** existing candle streamer without
interference.  Additional feeds (depth, funding, OI) will be added incrementally.

Safety:
• Feature-flag `ENABLE_MICROSTRUCTURE = False` (default) so prod isn’t affected
  until fully verified.
• Each token stream runs in its own asyncio Task; exceptions are logged and the
  task is restarted after a cooldown.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict

from binance import AsyncClient
from binance.streams import BinanceSocketManager  # type: ignore

# ---------------------------------------------------------------------------
# Configuration & Globals
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "market_data.db"
TOKENS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "AVAXUSDT",
    # (trimmed – same list as simple_streamer)
]

# Read feature flags from environment for easy staging toggle
def _env_flag(name: str, default: str = "False") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}

ENABLE_MICROSTRUCTURE = _env_flag("ENABLE_MICROSTRUCTURE", "False")
ENABLE_FUNDING_OI = _env_flag("ENABLE_FUNDING_OI", "False")
RECONNECT_COOLDOWN = int(os.getenv("RECONNECT_COOLDOWN", "10"))  # seconds before restarting failed stream task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("microstructure")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create agg_trades table if missing."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agg_trades (
            symbol TEXT,
            trade_id INTEGER,
            price REAL,
            qty REAL,
            trade_ts INTEGER,
            is_buyer_maker INTEGER,
            PRIMARY KEY (symbol, trade_id)
        )
        """
    )
    conn.commit()

    # New: order book depth snapshots (best bid/ask)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS order_book_depth (
            symbol TEXT,
            update_id INTEGER,
            bid_price REAL,
            bid_qty REAL,
            ask_price REAL,
            ask_qty REAL,
            event_ts INTEGER,
            PRIMARY KEY (symbol, update_id)
        )
        """
    )
    conn.commit()

    # New: derived depth features
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS depth_features (
            symbol TEXT,
            event_ts INTEGER,
            spread REAL,
            imbalance REAL,
            PRIMARY KEY (symbol, event_ts)
        )
        """
    )
    conn.commit()

    # New: funding rates
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS funding_rates (
            symbol TEXT,
            event_ts INTEGER,
            funding_rate REAL,
            PRIMARY KEY (symbol, event_ts)
        )
        """
    )
    conn.commit()

    # New: open interest
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS open_interest (
            symbol TEXT,
            event_ts INTEGER,
            oi REAL,
            PRIMARY KEY (symbol, event_ts)
        )
        """
    )
    conn.commit()
    conn.close()


def save_trade(symbol: str, data: Dict) -> None:
    """Insert a single aggTrade row (ignore duplicates)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT OR IGNORE INTO agg_trades
            (symbol, trade_id, price, qty, trade_ts, is_buyer_maker)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                data["a"],  # aggregate trade id
                float(data["p"]),
                float(data["q"]),
                int(data["T"]),
                int(data["m"]),
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.exception("DB insert error for %s: %s", symbol, exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Depth snapshot helpers
# ---------------------------------------------------------------------------
def save_depth(symbol: str, data: Dict) -> None:
    """Persist best bid/ask snapshot from depth stream."""
    try:
        bids = data.get("bids") or data.get("b")  # handle WS payload variants
        asks = data.get("asks") or data.get("a")
        if not bids or not asks:
            return
        best_bid_price, best_bid_qty = map(float, bids[0])
        best_ask_price, best_ask_qty = map(float, asks[0])
        update_id = data.get("lastUpdateId") or data.get("u") or int(data["E"])  # fallback
        event_ts = int(data["E"])
    except Exception:
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT OR IGNORE INTO order_book_depth
            (symbol, update_id, bid_price, bid_qty, ask_price, ask_qty, event_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                update_id,
                best_bid_price,
                best_bid_qty,
                best_ask_price,
                best_ask_qty,
                event_ts,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.exception("DB depth insert error for %s: %s", symbol, exc)
    finally:
        conn.close()

    # --- derived features ---
    try:
        spread = best_ask_price - best_bid_price
        imbalance = (
            (best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)
            if (best_bid_qty + best_ask_qty) != 0
            else 0.0
        )
        conn_feat = sqlite3.connect(DB_PATH)
        cur_feat = conn_feat.cursor()
        cur_feat.execute(
            """
            INSERT OR IGNORE INTO depth_features
            (symbol, event_ts, spread, imbalance)
            VALUES (?, ?, ?, ?)
            """,
            (symbol, event_ts, spread, imbalance),
        )
        conn_feat.commit()
    except Exception as exc:
        logger.exception("DB depth feature insert error for %s: %s", symbol, exc)
    finally:
        conn_feat.close()


# ---------------------------------------------------------------------------
# Funding & Open Interest helpers
# ---------------------------------------------------------------------------
def save_funding(symbol: str, data: Dict) -> None:
    """Persist funding rate update."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT OR IGNORE INTO funding_rates (symbol, event_ts, funding_rate)
            VALUES (?, ?, ?)
            """,
            (
                symbol,
                int(data.get("E", 0)),
                float(data.get("r", 0.0)),
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.debug("Funding insert error %s: %s", symbol, exc)
    finally:
        conn.close()


def save_oi(symbol: str, data: Dict) -> None:
    """Persist open interest update."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT OR IGNORE INTO open_interest (symbol, event_ts, oi)
            VALUES (?, ?, ?)
            """,
            (
                symbol,
                int(data.get("E", 0)),
                float(data.get("oi", 0.0)),
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.debug("OI insert error %s: %s", symbol, exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stream tasks
# ---------------------------------------------------------------------------
async def trade_stream_task(symbol: str, client: AsyncClient) -> None:
    """Listen to aggTrades stream for *symbol* and write to DB."""
    bsm = BinanceSocketManager(client)
    stream_name = f"{symbol.lower()}@aggTrade"
    while True:
        try:
            logger.info("📡 Starting aggTrade stream %s", stream_name)
            async with bsm.trade_socket(symbol.lower()) as stream:  # type: ignore
                async for msg in stream:
                    if msg and "a" in msg:  # ensure aggTrade payload
                        save_trade(symbol, msg)
        except Exception as exc:
            logger.exception("Stream error %s: %s", stream_name, exc)
        logger.info("🔄 Restarting %s in %s s", stream_name, RECONNECT_COOLDOWN)
        await asyncio.sleep(RECONNECT_COOLDOWN)


async def depth_stream_task(symbol: str, client: AsyncClient) -> None:
    """Listen to partial order book depth stream and store best bid/ask."""
    bsm = BinanceSocketManager(client)
    stream_name = f"{symbol.lower()}@depth5@100ms"  # top 5 levels, 100ms updates
    while True:
        try:
            logger.info("📡 Starting depth stream %s", stream_name)
            async with bsm.depth_socket(symbol.lower()) as stream:  # type: ignore
                async for msg in stream:
                    if msg and ("bids" in msg or "b" in msg):
                        save_depth(symbol, msg)
        except Exception as exc:
            logger.exception("Depth stream error %s: %s", stream_name, exc)
        logger.info("🔄 Restarting depth %s in %s s", stream_name, RECONNECT_COOLDOWN)
        await asyncio.sleep(RECONNECT_COOLDOWN)


async def funding_stream_task(symbol: str, client: AsyncClient) -> None:
    """Listen to funding rate stream."""
    bsm = BinanceSocketManager(client)
    stream_name = f"{symbol.lower()}@fundingRate"
    while True:
        try:
            logger.info("📡 Starting funding stream %s", stream_name)
            async with bsm._get_socket(stream_name):  # type: ignore
                async for msg in bsm.recv():
                    if msg and "r" in msg:
                        save_funding(symbol, msg)
        except Exception as exc:
            logger.exception("Funding stream error %s: %s", stream_name, exc)
        logger.info("🔄 Restarting funding %s in %s s", stream_name, RECONNECT_COOLDOWN)
        await asyncio.sleep(RECONNECT_COOLDOWN)


async def oi_stream_task(symbol: str, client: AsyncClient) -> None:
    """Listen to open interest stream."""
    bsm = BinanceSocketManager(client)
    stream_name = f"{symbol.lower()}@openInterest"
    while True:
        try:
            logger.info("📡 Starting OI stream %s", stream_name)
            async with bsm._get_socket(stream_name):  # type: ignore
                async for msg in bsm.recv():
                    if msg and "oi" in msg:
                        save_oi(symbol, msg)
        except Exception as exc:
            logger.exception("OI stream error %s: %s", stream_name, exc)
        logger.info("🔄 Restarting OI %s in %s s", stream_name, RECONNECT_COOLDOWN)
        await asyncio.sleep(RECONNECT_COOLDOWN)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
async def main() -> None:
    if not ENABLE_MICROSTRUCTURE:
        logger.warning(
            "⚠️  Microstructure streamer disabled (ENABLE_MICROSTRUCTURE = False). Exit."
        )
        return

    init_db()
    client = await AsyncClient.create()

    tasks = [trade_stream_task(sym, client) for sym in TOKENS]
    tasks.extend(depth_stream_task(sym, client) for sym in TOKENS)
    if ENABLE_FUNDING_OI:
        tasks.extend(funding_stream_task(sym, client) for sym in TOKENS)
        tasks.extend(oi_stream_task(sym, client) for sym in TOKENS)
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Microstructure streamer stopped by user.")
