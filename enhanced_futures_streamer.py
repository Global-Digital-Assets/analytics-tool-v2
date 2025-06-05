#!/usr/bin/env python3
"""
Enhanced Futures Microstructure Data Streamer
Continuous collection of funding rates, OI, liquidations, long/short ratios
Integrates with existing candle streamer for institutional-grade ML features
Version: v1.0_continuous_microstructure
"""

import asyncio
import logging
import json
import signal
import sys
from datetime import datetime, timedelta
from binance_futures_features import BinanceFuturesFeatures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_futures_streamer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedFuturesStreamer:
    def __init__(self):
        self.collector = None
        self.running = False
        self.collection_interval = 900  # 15 minutes (900 seconds)
        self.stats = {
            "collections": 0,
            "successful": 0,
            "failed": 0,
            "last_collection": None,
            "start_time": None
        }

    async def start_streaming(self):
        """Start the continuous microstructure data streaming"""
        logger.info("🚀 Starting Enhanced Futures Microstructure Streamer")
        logger.info(f"📊 Collection interval: {self.collection_interval} seconds (15 minutes)")
        
        self.running = True
        self.stats["start_time"] = datetime.now()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        async with BinanceFuturesFeatures() as collector:
            self.collector = collector
            
            # Initialize database table
            collector.init_futures_microstructure_table()
            
            logger.info("✅ Database initialized, starting collection loop")
            
            while self.running:
                try:
                    collection_start = datetime.now()
                    logger.info(f"🔄 Starting collection cycle #{self.stats['collections'] + 1}")
                    
                    # Collect microstructure data
                    results = await collector.collect_all_microstructure()
                    
                    # Update statistics
                    self.stats["collections"] += 1
                    self.stats["successful"] += results["successful"]
                    self.stats["failed"] += results["failed"]
                    self.stats["last_collection"] = collection_start.isoformat()
                    
                    collection_duration = (datetime.now() - collection_start).total_seconds()
                    
                    logger.info(f"✅ Collection cycle completed in {collection_duration:.1f}s")
                    logger.info(f"📊 Results: {results['successful']}/{results['total']} successful")
                    logger.info(f"📈 Total stats: {self.stats['successful']} successful collections, {self.stats['failed']} failed")
                    
                    # Save updated stats
                    await self.save_stats()
                    
                    # Wait for next collection interval
                    if self.running:
                        logger.info(f"⏱️ Waiting {self.collection_interval} seconds until next collection...")
                        await asyncio.sleep(self.collection_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Collection cycle failed: {e}")
                    self.stats["failed"] += 1
                    
                    # Wait shorter interval on error before retry
                    if self.running:
                        logger.info("⏱️ Waiting 60 seconds before retry...")
                        await asyncio.sleep(60)
        
        logger.info("🛑 Enhanced Futures Streamer stopped")

    async def save_stats(self):
        """Save streaming statistics"""
        try:
            stats_data = {
                **self.stats,
                "uptime_hours": (datetime.now() - datetime.fromisoformat(self.stats["start_time"].isoformat())).total_seconds() / 3600 if self.stats["start_time"] else 0
            }
            
            with open('enhanced_futures_streamer_stats.json', 'w') as f:
                json.dump(stats_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to save stats: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"📡 Received signal {signum}, shutting down gracefully...")
        self.running = False

    async def get_database_stats(self):
        """Get current database statistics"""
        try:
            if self.collector:
                with self.collector.get_db_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Get total microstructure records
                    cursor.execute("SELECT COUNT(*) FROM futures_microstructure")
                    total_records = cursor.fetchone()[0]
                    
                    # Get unique symbols
                    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM futures_microstructure")
                    unique_symbols = cursor.fetchone()[0]
                    
                    # Get latest timestamp
                    cursor.execute("SELECT MAX(timestamp) FROM futures_microstructure")
                    latest_timestamp = cursor.fetchone()[0]
                    
                    # Get records in last 24 hours
                    day_ago = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
                    cursor.execute("SELECT COUNT(*) FROM futures_microstructure WHERE timestamp > ?", (day_ago,))
                    records_24h = cursor.fetchone()[0]
                    
                    return {
                        "total_records": total_records,
                        "unique_symbols": unique_symbols,
                        "latest_timestamp": latest_timestamp,
                        "records_last_24h": records_24h,
                        "last_update": datetime.fromtimestamp(latest_timestamp / 1000).isoformat() if latest_timestamp else None
                    }
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}

async def main():
    """Main execution function"""
    print("🏦 ENHANCED FUTURES MICROSTRUCTURE STREAMER")
    print("=" * 60)
    print("📊 Continuous collection of funding rates, OI, liquidations")
    print("🔄 15-minute collection cycles for institutional-grade ML features")
    print("=" * 60)
    
    streamer = EnhancedFuturesStreamer()
    
    try:
        await streamer.start_streaming()
    except KeyboardInterrupt:
        logger.info("👋 Streamer stopped by user")
    except Exception as e:
        logger.error(f"❌ Streamer crashed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
