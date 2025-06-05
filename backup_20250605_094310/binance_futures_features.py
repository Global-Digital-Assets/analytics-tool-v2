#!/usr/bin/env python3
"""
Enhanced Binance Futures Microstructure Data Collector
Collects funding rates, open interest, long/short ratios, liquidations
For institutional-grade ML feature enhancement
Version: v1.0_futures_microstructure
"""

import asyncio
import aiohttp
import sqlite3
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceFuturesFeatures:
    def __init__(self, db_path: str = 'market_data.db'):
        self.db_path = db_path
        self.session = None
        self.base_url = "https://fapi.binance.com"
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Target symbols for futures data
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
            "LTCUSDT", "ETCUSDT", "TRXUSDT", "NEARUSDT", "INJUSDT",
            "SUIUSDT", "AAVEUSDT", "SHIBUSDT", "DOGEUSDT", "PEPEUSDT",
            "OPUSDT", "APTUSDT", "ARBUSDT", "WLDUSDT", "FETUSDT",
            "SUSHIUSDT", "WIFUSDT", "TRUMPUSDT", "TAOUSDT", "TIAUSDT"
        ]

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def init_futures_microstructure_table(self):
        """Initialize futures microstructure data table"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS futures_microstructure (
                        symbol TEXT,
                        timestamp INTEGER,
                        funding_rate REAL,
                        funding_8h_sum REAL,
                        funding_24h_sum REAL,
                        funding_zscore REAL,
                        oi_usd REAL,
                        oi_1h_change_pct REAL,
                        oi_4h_change_pct REAL,
                        oi_24h_change_pct REAL,
                        long_short_ratio REAL,
                        top_trader_long_ratio REAL,
                        top_trader_short_ratio REAL,
                        liquidation_long_count_1h INTEGER,
                        liquidation_short_count_1h INTEGER,
                        liquidation_long_usd_1h REAL,
                        liquidation_short_usd_1h REAL,
                        large_trade_count_1h INTEGER,
                        orderbook_imbalance REAL,
                        price_1h_change_pct REAL,
                        price_4h_change_pct REAL,
                        price_24h_change_pct REAL,
                        volume_1h_ratio REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, timestamp)
                    )
                """)
                conn.commit()
                logger.info("✅ Futures microstructure table initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing futures microstructure table: {e}")

    async def fetch_funding_rate_data(self, symbol: str) -> Dict:
        """Fetch funding rate and premium index data"""
        try:
            # Get funding rate history
            funding_url = f"{self.base_url}/fapi/v1/fundingRate"
            funding_params = {"symbol": symbol, "limit": 8}  # Last 8 funding periods (24h)
            
            await asyncio.sleep(self.rate_limit_delay)
            async with self.session.get(funding_url, params=funding_params) as response:
                if response.status == 200:
                    funding_data = await response.json()
                    
                    # Get premium index (includes next funding time and current rate)
                    premium_url = f"{self.base_url}/fapi/v1/premiumIndex"
                    premium_params = {"symbol": symbol}
                    
                    await asyncio.sleep(self.rate_limit_delay)
                    async with self.session.get(premium_url, params=premium_params) as premium_response:
                        if premium_response.status == 200:
                            premium_data = await premium_response.json()
                            
                            # Calculate funding metrics
                            funding_rates = [float(item["fundingRate"]) for item in funding_data]
                            current_funding = float(premium_data["lastFundingRate"])
                            
                            # Calculate rolling sums
                            funding_8h_sum = sum(funding_rates[:3]) if len(funding_rates) >= 3 else 0
                            funding_24h_sum = sum(funding_rates) if len(funding_rates) >= 8 else 0
                            
                            # Calculate z-score vs historical
                            funding_zscore = 0.0
                            if len(funding_rates) > 1:
                                mean_funding = np.mean(funding_rates)
                                std_funding = np.std(funding_rates)
                                if std_funding > 0:
                                    funding_zscore = (current_funding - mean_funding) / std_funding
                            
                            return {
                                "funding_rate": current_funding,
                                "funding_8h_sum": funding_8h_sum,
                                "funding_24h_sum": funding_24h_sum,
                                "funding_zscore": funding_zscore
                            }
        except Exception as e:
            logger.warning(f"⚠️ Error fetching funding data for {symbol}: {e}")
            return {}

    async def fetch_open_interest_data(self, symbol: str) -> Dict:
        """Fetch open interest data and calculate changes"""
        try:
            url = f"{self.base_url}/futures/data/openInterestHist"
            params = {"symbol": symbol, "period": "5m", "limit": 300}  # 25 hours of 5m data
            
            await asyncio.sleep(self.rate_limit_delay)
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if len(data) > 0:
                        current_oi = float(data[-1]["sumOpenInterest"])
                        
                        # Calculate percentage changes
                        oi_1h_change_pct = 0.0
                        oi_4h_change_pct = 0.0
                        oi_24h_change_pct = 0.0
                        
                        if len(data) >= 12:  # 1 hour ago (12 * 5min)
                            oi_1h_ago = float(data[-12]["sumOpenInterest"])
                            if oi_1h_ago > 0:
                                oi_1h_change_pct = ((current_oi - oi_1h_ago) / oi_1h_ago) * 100
                        
                        if len(data) >= 48:  # 4 hours ago (48 * 5min)
                            oi_4h_ago = float(data[-48]["sumOpenInterest"])
                            if oi_4h_ago > 0:
                                oi_4h_change_pct = ((current_oi - oi_4h_ago) / oi_4h_ago) * 100
                        
                        if len(data) >= 288:  # 24 hours ago (288 * 5min)
                            oi_24h_ago = float(data[-288]["sumOpenInterest"])
                            if oi_24h_ago > 0:
                                oi_24h_change_pct = ((current_oi - oi_24h_ago) / oi_24h_ago) * 100
                        
                        return {
                            "oi_usd": current_oi,
                            "oi_1h_change_pct": oi_1h_change_pct,
                            "oi_4h_change_pct": oi_4h_change_pct,
                            "oi_24h_change_pct": oi_24h_change_pct
                        }
        except Exception as e:
            logger.warning(f"⚠️ Error fetching OI data for {symbol}: {e}")
            return {}

    async def fetch_long_short_ratios(self, symbol: str) -> Dict:
        """Fetch global and top trader long/short ratios"""
        try:
            results = {}
            
            # Global long/short account ratio
            global_url = f"{self.base_url}/futures/data/globalLongShortAccountRatio"
            global_params = {"symbol": symbol, "period": "5m", "limit": 1}
            
            await asyncio.sleep(self.rate_limit_delay)
            async with self.session.get(global_url, params=global_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 0:
                        results["long_short_ratio"] = float(data[0]["longShortRatio"])
            
            # Top trader position ratio
            top_url = f"{self.base_url}/futures/data/topLongShortPositionRatio"
            top_params = {"symbol": symbol, "period": "5m", "limit": 1}
            
            await asyncio.sleep(self.rate_limit_delay)
            async with self.session.get(top_url, params=top_params) as response:
                if response.status == 200:
                    data = await response.json()
                    if len(data) > 0:
                        results["top_trader_long_ratio"] = float(data[0]["longAccount"])
                        results["top_trader_short_ratio"] = float(data[0]["shortAccount"])
            
            return results
        except Exception as e:
            logger.warning(f"⚠️ Error fetching long/short ratios for {symbol}: {e}")
            return {}

    async def fetch_liquidation_data(self, symbol: str) -> Dict:
        """Fetch recent liquidation data"""
        try:
            url = f"{self.base_url}/futures/data/allForceOrders"
            params = {"symbol": symbol, "limit": 1000}
            
            await asyncio.sleep(self.rate_limit_delay)
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter last hour
                    one_hour_ago = int((datetime.now() - timedelta(hours=1)).timestamp() * 1000)
                    recent_liquidations = [liq for liq in data if int(liq["time"]) > one_hour_ago]
                    
                    # Count and sum by side
                    long_count = len([liq for liq in recent_liquidations if liq["side"] == "BUY"])
                    short_count = len([liq for liq in recent_liquidations if liq["side"] == "SELL"])
                    
                    long_usd = sum([float(liq["quantity"]) * float(liq["price"]) for liq in recent_liquidations if liq["side"] == "BUY"])
                    short_usd = sum([float(liq["quantity"]) * float(liq["price"]) for liq in recent_liquidations if liq["side"] == "SELL"])
                    
                    return {
                        "liquidation_long_count_1h": long_count,
                        "liquidation_short_count_1h": short_count,
                        "liquidation_long_usd_1h": long_usd,
                        "liquidation_short_usd_1h": short_usd
                    }
        except Exception as e:
            logger.warning(f"⚠️ Error fetching liquidation data for {symbol}: {e}")
            return {}

    async def fetch_price_momentum_data(self, symbol: str) -> Dict:
        """Fetch price data for momentum calculations"""
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {"symbol": symbol, "interval": "1h", "limit": 25}  # 25 hours
            
            await asyncio.sleep(self.rate_limit_delay)
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if len(data) > 0:
                        current_price = float(data[-1][4])  # Close price
                        
                        # Calculate price changes
                        price_1h_change_pct = 0.0
                        price_4h_change_pct = 0.0
                        price_24h_change_pct = 0.0
                        
                        if len(data) >= 2:
                            price_1h_ago = float(data[-2][4])
                            price_1h_change_pct = ((current_price - price_1h_ago) / price_1h_ago) * 100
                        
                        if len(data) >= 5:
                            price_4h_ago = float(data[-5][4])
                            price_4h_change_pct = ((current_price - price_4h_ago) / price_4h_ago) * 100
                        
                        if len(data) >= 25:
                            price_24h_ago = float(data[-25][4])
                            price_24h_change_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100
                        
                        # Volume ratio (current vs 24h average)
                        current_volume = float(data[-1][5])
                        avg_volume_24h = np.mean([float(candle[5]) for candle in data])
                        volume_1h_ratio = current_volume / avg_volume_24h if avg_volume_24h > 0 else 1.0
                        
                        return {
                            "price_1h_change_pct": price_1h_change_pct,
                            "price_4h_change_pct": price_4h_change_pct,
                            "price_24h_change_pct": price_24h_change_pct,
                            "volume_1h_ratio": volume_1h_ratio
                        }
        except Exception as e:
            logger.warning(f"⚠️ Error fetching price momentum for {symbol}: {e}")
            return {}

    async def collect_symbol_microstructure(self, symbol: str) -> Optional[Dict]:
        """Collect all microstructure data for a symbol"""
        try:
            logger.info(f"🔄 Collecting microstructure data for {symbol}")
            
            # Fetch all data sources concurrently
            tasks = [
                self.fetch_funding_rate_data(symbol),
                self.fetch_open_interest_data(symbol),
                self.fetch_long_short_ratios(symbol),
                self.fetch_liquidation_data(symbol),
                self.fetch_price_momentum_data(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine all results
            microstructure_data = {
                "symbol": symbol,
                "timestamp": int(datetime.now().timestamp() * 1000),
                "funding_rate": 0.0,
                "funding_8h_sum": 0.0,
                "funding_24h_sum": 0.0,
                "funding_zscore": 0.0,
                "oi_usd": 0.0,
                "oi_1h_change_pct": 0.0,
                "oi_4h_change_pct": 0.0,
                "oi_24h_change_pct": 0.0,
                "long_short_ratio": 1.0,
                "top_trader_long_ratio": 0.5,
                "top_trader_short_ratio": 0.5,
                "liquidation_long_count_1h": 0,
                "liquidation_short_count_1h": 0,
                "liquidation_long_usd_1h": 0.0,
                "liquidation_short_usd_1h": 0.0,
                "large_trade_count_1h": 0,
                "orderbook_imbalance": 0.0,
                "price_1h_change_pct": 0.0,
                "price_4h_change_pct": 0.0,
                "price_24h_change_pct": 0.0,
                "volume_1h_ratio": 1.0
            }
            
            # Update with actual data
            for result in results:
                if isinstance(result, dict):
                    microstructure_data.update(result)
            
            return microstructure_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting microstructure for {symbol}: {e}")
            return None

    def insert_microstructure_data(self, data: Dict) -> bool:
        """Insert microstructure data into database"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO futures_microstructure (
                        symbol, timestamp, funding_rate, funding_8h_sum, funding_24h_sum, funding_zscore,
                        oi_usd, oi_1h_change_pct, oi_4h_change_pct, oi_24h_change_pct,
                        long_short_ratio, top_trader_long_ratio, top_trader_short_ratio,
                        liquidation_long_count_1h, liquidation_short_count_1h, 
                        liquidation_long_usd_1h, liquidation_short_usd_1h,
                        large_trade_count_1h, orderbook_imbalance,
                        price_1h_change_pct, price_4h_change_pct, price_24h_change_pct, volume_1h_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["symbol"], data["timestamp"], data["funding_rate"], data["funding_8h_sum"],
                    data["funding_24h_sum"], data["funding_zscore"], data["oi_usd"], data["oi_1h_change_pct"],
                    data["oi_4h_change_pct"], data["oi_24h_change_pct"], data["long_short_ratio"],
                    data["top_trader_long_ratio"], data["top_trader_short_ratio"], data["liquidation_long_count_1h"],
                    data["liquidation_short_count_1h"], data["liquidation_long_usd_1h"], data["liquidation_short_usd_1h"],
                    data["large_trade_count_1h"], data["orderbook_imbalance"], data["price_1h_change_pct"],
                    data["price_4h_change_pct"], data["price_24h_change_pct"], data["volume_1h_ratio"]
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"❌ Error inserting microstructure data: {e}")
            return False

    async def collect_all_microstructure(self) -> Dict:
        """Collect microstructure data for all symbols"""
        logger.info("🚀 Starting futures microstructure data collection")
        
        successful = 0
        failed = 0
        
        for symbol in self.symbols:
            try:
                data = await self.collect_symbol_microstructure(symbol)
                if data and self.insert_microstructure_data(data):
                    successful += 1
                    logger.info(f"✅ {symbol}: Microstructure data collected")
                else:
                    failed += 1
                    logger.warning(f"⚠️ {symbol}: Failed to collect/store data")
                
                # Brief pause between symbols
                await asyncio.sleep(0.2)
                
            except Exception as e:
                failed += 1
                logger.error(f"❌ {symbol}: Error - {e}")
        
        results = {
            "successful": successful,
            "failed": failed,
            "total": len(self.symbols),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"📊 Collection complete: {successful}/{len(self.symbols)} successful")
        return results

async def main():
    """Main execution function"""
    print("🏦 ENHANCED BINANCE FUTURES MICROSTRUCTURE COLLECTOR")
    print("=" * 60)
    
    try:
        async with BinanceFuturesFeatures() as collector:
            # Initialize database
            collector.init_futures_microstructure_table()
            
            # Collect data
            results = await collector.collect_all_microstructure()
            
            # Save results
            with open('futures_microstructure_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Collection complete!")
            print(f"📊 Successful: {results['successful']}/{results['total']}")
            print(f"💾 Results saved to futures_microstructure_results.json")
            
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
