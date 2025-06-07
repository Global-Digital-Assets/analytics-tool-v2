#!/usr/bin/env python3
import json
import sqlite3
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# All 30 tokens from database
ALL_TOKENS = [
    "AAVEUSDT", "ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT", 
    "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETCUSDT", "ETHUSDT", 
    "FETUSDT", "INJUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "OPUSDT", 
    "PEPEUSDT", "SHIBUSDT", "SOLUSDT", "SUIUSDT", "SUSHIUSDT", "TAOUSDT", 
    "TIAUSDT", "TRUMPUSDT", "TRXUSDT", "WIFUSDT", "WLDUSDT", "XRPUSDT"
]

def calculate_rsi(prices, period=14):
    """Calculate RSI for price series"""
    if len(prices) < period:
        return 50
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_token(symbol, conn):
    """Analyze a single token"""
    try:
        # Get recent candles (last 100 for analysis)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT close, volume, timestamp FROM candles 
            WHERE symbol = ? ORDER BY timestamp DESC LIMIT 100
        """, (symbol,))
        
        candles = cursor.fetchall()
        if len(candles) < 20:
            return None
            
        prices = [float(c[0]) for c in reversed(candles)]
        volumes = [float(c[1]) for c in reversed(candles)]
        current_price = prices[-1]
        
        # Calculate indicators
        rsi = calculate_rsi(prices)
        
        # Price changes
        price_1h = ((current_price - prices[-5]) / prices[-5]) * 100 if len(prices) >= 5 else 0
        price_4h = ((current_price - prices[-20]) / prices[-20]) * 100 if len(prices) >= 20 else 0
        price_24h = ((current_price - prices[-96]) / prices[-96]) * 100 if len(prices) >= 96 else 0
        
        # Volume analysis
        avg_volume = sum(volumes[-24:]) / len(volumes[-24:]) if len(volumes) >= 24 else 0
        volume_spike = volumes[-1] > avg_volume * 1.5 if avg_volume > 0 else False
        
        # Simple scoring
        long_score = 0
        short_score = 0
        
        # RSI scoring
        if rsi < 30:
            long_score += 30
        elif rsi > 70:
            short_score += 30
            
        # Price momentum scoring
        if price_1h > 0.5:
            long_score += 20
        elif price_1h < -0.5:
            short_score += 20
            
        if price_4h > 1:
            long_score += 15
        elif price_4h < -1:
            short_score += 15
            
        # Volume boost
        if volume_spike:
            long_score += 10
            short_score += 10
            
        # Cap scores
        long_score = min(long_score, 100)
        short_score = min(short_score, 100)
        
        # Binary actions
        simple_long = "BUY_LONG" if long_score >= 60 else "DON'T_BUY_LONG"
        simple_short = SHORT if short_score >= 60 else DONT_SHORT"
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "long_score": long_score,
            "short_score": short_score,
            "simple_long_action": simple_long,
            "simple_short_action": simple_short,
            "rsi": rsi,
            "price_changes": {
                "1h": price_1h,
                "4h": price_4h,
                "24h": price_24h
            },
            "volume_spike": volume_spike
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

def main():
    """Run 30-token analysis"""
    start_time = time.time()
    logger.info("Starting 30-token analysis...")
    
    # Connect to database
    conn = sqlite3.connect("market_data.db")
    
    opportunities = []
    for symbol in ALL_TOKENS:
        logger.info(f"Analyzing {symbol}...")
        result = analyze_token(symbol, conn)
        if result:
            opportunities.append(result)
    
    conn.close()
    
    # Create analysis result
    analysis = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metadata": {
            "tokens_analyzed": len(opportunities),
            "analysis_version": "simple_30token_v1.0",
            "market_regime": "ranging",  # Simple default
            "analysis_period": "30_minutes"
        },
        "opportunities": opportunities
    }
    
    # Save to file
    with open("multi_token_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Analysis complete! {len(opportunities)} tokens analyzed in {elapsed:.1f}s")
    
    # Quick summary
    buy_signals = [o for o in opportunities if o["simple_long_action"] == "BUY_LONG"]
    short_signals = [o for o in opportunities if o["simple_short_action"] == "SHORT"]
    
    print(f"📊 30-TOKEN ANALYSIS COMPLETE")
    print(f"🎯 BUY signals: {len(buy_signals)}")
    print(f"🎯 SHORT signals: {len(short_signals)}")
    
    return analysis

if __name__ == "__main__":
    main()
