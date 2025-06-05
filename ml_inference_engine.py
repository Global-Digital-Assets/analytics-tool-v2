#!/usr/bin/env python3
"""
⚡ ML INFERENCE ENGINE
Real-time scoring integration for analytics pipeline
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class MLInferenceEngine:
    """High-speed ML inference for real-time crypto scoring"""
    
    def __init__(self, models_dir: str = "models", db_path: str = "market_data.db"):
        self.models_dir = Path(models_dir)
        self.db_path = db_path
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.cached_features = {}  # Cache recent features for speed
        
    def load_latest_model(self) -> bool:
        """Load the latest trained model"""
        try:
            model_path = self.models_dir / "latest_model.txt"
            metadata_path = self.models_dir / "latest_metadata.json"
            
            if not model_path.exists() or not metadata_path.exists():
                logger.error("❌ No trained model found! Run production pipeline first.")
                return False
            
            # Load model
            self.model = lgb.Booster(model_file=str(model_path))
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['feature_names']
            
            logger.info(f"✅ Loaded model: {self.metadata['model_info']['filename']}")
            logger.info(f"📊 Model accuracy: {self.metadata['performance']['accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def get_latest_features(self, symbol: str, lookback_hours: int = 48) -> Optional[np.ndarray]:
        """Get latest engineered features for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_timestamp = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp())
            
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles 
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp
            """
            
            df_pd = pd.read_sql_query(query, conn, params=[symbol, cutoff_timestamp])
            df = pl.from_pandas(df_pd)
            conn.close()
            
            if len(df) < 100:  # Need enough data for features
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} rows")
                return None
            
            # Create 5-minute bars
            df_5m = self._create_5min_bars(df)
            
            # Engineer features
            df_features = self._engineer_features(df_5m)
            
            # Get latest complete row (not NaN)
            clean_df = df_features.drop_nulls()
            if len(clean_df) == 0:
                return None
                
            latest_features = clean_df.select(self.feature_names).tail(1).to_numpy()
            
            return latest_features[0] if len(latest_features) > 0 else None
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for {symbol}: {e}")
            return None
    
    def predict_symbol(self, symbol: str) -> Optional[Dict]:
        """Generate ML prediction for a symbol"""
        if self.model is None:
            logger.error("❌ No model loaded! Call load_latest_model() first.")
            return None
            
        features = self.get_latest_features(symbol)
        if features is None:
            return None
            
        try:
            # Get prediction
            prediction_proba = self.model.predict([features])[0]
            prediction_binary = int(prediction_proba > 0.5)
            confidence = max(prediction_proba, 1 - prediction_proba)
            
            # Get feature importance for this prediction
            feature_importance = dict(zip(self.feature_names, features.tolist()))
            
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "ml_prediction": {
                    "probability": float(prediction_proba),
                    "binary_prediction": prediction_binary,
                    "confidence": float(confidence),
                    "signal": "BUY_LONG" if prediction_binary == 1 else "NEUTRAL"
                },
                "features": feature_importance,
                "model_version": self.metadata['model_info']['timestamp']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {symbol}: {e}")
            return None
    
    def batch_predict(self, symbols: List[str]) -> Dict[str, Dict]:
        """Generate predictions for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            prediction = self.predict_symbol(symbol)
            if prediction:
                results[symbol] = prediction
                
        logger.info(f"📊 Generated predictions for {len(results)}/{len(symbols)} symbols")
        return results
    
    def update_cached_5min_bars(self):
        """Update cached 5-minute bars for faster inference"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if 5min cache table exists
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles_5m_cache (
                    symbol TEXT,
                    ts5m INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, ts5m)
                )
            """)
            
            # Get latest timestamp from cache
            cursor.execute("SELECT MAX(ts5m) FROM candles_5m_cache")
            last_cached = cursor.fetchone()[0] or 0
            
            # Update cache with new data
            query = """
            INSERT OR REPLACE INTO candles_5m_cache (symbol, ts5m, open, high, low, close, volume)
            SELECT 
                symbol,
                (timestamp / 300) * 300 AS ts5m,
                (SELECT open FROM candles c2 
                 WHERE c2.symbol = c1.symbol 
                 AND (c2.timestamp / 300) * 300 = (c1.timestamp / 300) * 300
                 ORDER BY c2.timestamp ASC LIMIT 1) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                (SELECT close FROM candles c3 
                 WHERE c3.symbol = c1.symbol 
                 AND (c3.timestamp / 300) * 300 = (c1.timestamp / 300) * 300
                 ORDER BY c3.timestamp DESC LIMIT 1) AS close,
                SUM(volume) AS volume
            FROM candles c1
            WHERE timestamp > ?
            GROUP BY symbol, (timestamp / 300) * 300
            """
            
            cursor.execute(query, (last_cached,))
            conn.commit()
            conn.close()
            
            logger.info("✅ Updated 5-minute bar cache")
            
        except Exception as e:
            logger.error(f"❌ Cache update failed: {e}")
    
    def integrate_with_analytics(self, analytics_results: Dict) -> Dict:
        """Integrate ML predictions with existing analytics"""
        if 'opportunities' not in analytics_results:
            return analytics_results
            
        symbols = [opp['symbol'] for opp in analytics_results['opportunities']]
        ml_predictions = self.batch_predict(symbols)
        
        # Enhance each opportunity with ML prediction
        for opportunity in analytics_results['opportunities']:
            symbol = opportunity['symbol']
            if symbol in ml_predictions:
                ml_pred = ml_predictions[symbol]['ml_prediction']
                
                # Add ML fields
                opportunity['ml_probability'] = ml_pred['probability']
                opportunity['ml_confidence'] = ml_pred['confidence']
                opportunity['ml_signal'] = ml_pred['signal']
                
                # Boost score if ML agrees
                if ml_pred['binary_prediction'] == 1 and opportunity.get('action') == 'BUY_LONG':
                    opportunity['score'] = min(100, opportunity['score'] * 1.1)  # 10% boost
                    opportunity['confidence_boost'] = True
                else:
                    opportunity['confidence_boost'] = False
        
        # Add ML metadata to results
        analytics_results['ml_metadata'] = {
            "model_version": self.metadata['model_info']['timestamp'] if self.metadata else None,
            "predictions_generated": len(ml_predictions),
            "inference_timestamp": datetime.now().isoformat()
        }
        
        return analytics_results
    
    def _create_5min_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create 5-minute OHLCV bars"""
        return (
            df
            .with_columns([(pl.col("timestamp") // 300 * 300).alias("ts5m")])
            .group_by(["ts5m"])
            .agg([
                pl.col("open").first(),
                pl.col("high").max(), 
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum()
            ])
            .sort(["ts5m"])
        )
    
    def _engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Engineer ML features for single symbol"""
        return (
            df
            .sort(["ts5m"])
            .with_columns([
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_5m"),
            ])
            .with_columns([
                (pl.col("close") / pl.col("close").shift(12) - 1).alias("ret_1h"), 
                (pl.col("close") / pl.col("close").shift(48) - 1).alias("ret_4h"),
                (pl.col("close") / pl.col("close").shift(288) - 1).alias("ret_24h"),
                pl.col("volume").rolling_mean(12).alias("vol_ma_1h"),
                (pl.col("volume") / pl.col("volume").rolling_mean(12)).alias("vol_spike"),
                pl.col("ret_5m").rolling_std(12).alias("volatility_1h"),
                pl.col("ret_5m").rolling_std(48).alias("volatility_4h"),
                ((pl.col("high") + pl.col("low")) / 2 / pl.col("close") - 1).alias("hl_spread"),
                (pl.col("close") / pl.col("open") - 1).alias("bar_return"),
                pl.col("close").rolling_mean(14).alias("sma_14"),
                (pl.col("close") / pl.col("close").rolling_mean(14) - 1).alias("sma_distance"),
                pl.when(pl.col("ret_5m") > 0).then(pl.col("ret_5m")).otherwise(0).rolling_mean(14).alias("avg_gains"),
                pl.when(pl.col("ret_5m") < 0).then(-pl.col("ret_5m")).otherwise(0).rolling_mean(14).alias("avg_losses")
            ])
            .with_columns([
                (100 - 100 / (1 + pl.col("avg_gains") / (pl.col("avg_losses") + 1e-10))).alias("rsi14")
            ])
            .drop(["avg_gains", "avg_losses", "vol_ma_1h", "sma_14"])
        )

# Integration helper function
def enhance_analytics_with_ml(analytics_results: Dict, models_dir: str = "models") -> Dict:
    """Easy integration function for existing analytics pipeline"""
    try:
        ml_engine = MLInferenceEngine(models_dir=models_dir)
        
        if not ml_engine.load_latest_model():
            logger.warning("⚠️ ML model not available, returning original results")
            return analytics_results
            
        enhanced_results = ml_engine.integrate_with_analytics(analytics_results)
        logger.info("✅ Analytics enhanced with ML predictions")
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"❌ ML enhancement failed: {e}")
        return analytics_results  # Return original results on failure
