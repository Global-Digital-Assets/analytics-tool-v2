#!/usr/bin/env python3
"""
Real-time Backfill Progress Monitor
Monitors backfill.log and database for progress updates
"""

import sqlite3
import time
import json
import os
from datetime import datetime

def get_database_stats(db_path='market_data.db'):
    """Get current database statistics"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_candles,
                    COUNT(DISTINCT symbol) as symbols,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM candles
            """)
            overall = cursor.fetchone()
            
            # Per symbol counts
            cursor.execute("""
                SELECT symbol, COUNT(*) as count
                FROM candles 
                GROUP BY symbol 
                ORDER BY count DESC
                LIMIT 10
            """)
            top_symbols = cursor.fetchall()
            
            return {
                'total_candles': overall[0],
                'symbols': overall[1],
                'earliest': datetime.fromtimestamp(overall[2]/1000).strftime('%Y-%m-%d %H:%M') if overall[2] else None,
                'latest': datetime.fromtimestamp(overall[3]/1000).strftime('%Y-%m-%d %H:%M') if overall[3] else None,
                'top_symbols': top_symbols
            }
    except Exception as e:
        return {'error': str(e)}

def monitor_progress():
    """Monitor backfill progress"""
    print("🔍 Backfill Progress Monitor")
    print("=" * 50)
    
    start_time = datetime.now()
    last_candle_count = 0
    
    while True:
        try:
            # Get current stats
            stats = get_database_stats()
            
            if 'error' not in stats:
                current_candles = stats['total_candles']
                candles_added = current_candles - last_candle_count
                
                # Calculate rate
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = current_candles / elapsed if elapsed > 0 else 0
                
                # Display progress
                print(f"\n📊 {datetime.now().strftime('%H:%M:%S')} | Progress Update:")
                print(f"   📈 Total Candles: {current_candles:,}")
                print(f"   🎯 Symbols: {stats['symbols']}")
                print(f"   📅 Range: {stats['earliest']} → {stats['latest']}")
                print(f"   🚄 Rate: {rate:.1f} candles/sec")
                print(f"   ⬆️  Added (last 30s): {candles_added:,}")
                
                # Top symbols
                if stats['top_symbols']:
                    print(f"   🏆 Top Symbols:")
                    for symbol, count in stats['top_symbols'][:5]:
                        print(f"      {symbol}: {count:,}")
                
                last_candle_count = current_candles
                
                # Check if we have enough for ML
                if current_candles > 500000:
                    print(f"\n🎉 MILESTONE: {current_candles:,} candles - Ready for ML training!")
                
            else:
                print(f"❌ Database error: {stats['error']}")
            
            # Wait 30 seconds
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_progress()
