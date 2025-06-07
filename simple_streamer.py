#!/usr/bin/env python3
"""
Simple Binance Data Streamer
Collects 1-minute candle data for configured tokens and stores in SQLite
"""
import asyncio
import sqlite3
from datetime import datetime
import json
import os
from binance import AsyncClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import tokens from config
from config import DEFAULT_TOKENS
TOKENS = DEFAULT_TOKENS

class DataStreamer:
    def __init__(self):
        self.client = None
        self.db_path = '/root/analytics-tool-v2/market_data.db'
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized")
        
    async def get_latest_candles(self):
        """Get latest 1-minute candles for all tokens"""
        self.client = await AsyncClient.create()
        
        try:
            data = {}
            for symbol in TOKENS:
                try:
                    klines = await self.client.get_klines(
                        symbol=symbol,
                        interval='1m',
                        limit=1
                    )
                    
                    if klines:
                        kline = klines[0]
                        candle_data = {
                            'timestamp': int(kline[0]),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        }
                        data[symbol] = candle_data
                        
                        # Store in database
                        self.store_candle(symbol, candle_data)
                        
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
                    
            logger.info(f"Fetched data for {len(data)} symbols")
            return data
            
        except Exception as e:
            logger.error(f"Error in get_latest_candles: {e}")
            return {}
        finally:
            if self.client:
                await self.client.close_connection()
                
    def store_candle(self, symbol, candle_data):
        """Store candle data in SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO candles 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                candle_data['timestamp'],
                candle_data['open'],
                candle_data['high'],
                candle_data['low'],
                candle_data['close'],
                candle_data['volume']
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing candle for {symbol}: {e}")

async def main():
    """Main streaming loop"""
    streamer = DataStreamer()
    logger.info("Starting data streamer...")
    
    while True:
        try:
            await streamer.get_latest_candles()
            await asyncio.sleep(60)  # Wait 1 minute
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
