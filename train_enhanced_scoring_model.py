#!/usr/bin/env python3
"""
Enhanced ML Training with Futures Microstructure Features
Combines candle data + funding rates + OI + liquidations + long/short ratios
For institutional-grade crypto scoring with maximum alpha
Version: v2.0_enhanced_microstructure_ml
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    def __init__(self, db_path: str = 'market_data.db'):
        self.db_path = db_path
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
            "LTCUSDT", "ETCUSDT", "TRXUSDT", "NEARUSDT", "INJUSDT",
            "SUIUSDT", "AAVEUSDT", "SHIBUSDT", "DOGEUSDT", "PEPEUSDT",
            "OPUSDT", "APTUSDT", "ARBUSDT", "WLDUSDT", "FETUSDT",
            "SUSHIUSDT", "WIFUSDT", "TRUMPUSDT", "TAOUSDT", "TIAUSDT"
        ]
        
        self.min_training_samples = 5000
        self.model = None
        self.scaler = None
        self.feature_names = []

    def get_db_connection(self):
        """Get thread-safe database connection"""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def load_candle_data(self, symbol: str) -> pd.DataFrame:
        """Load candle data for a symbol"""
        try:
            with self.get_db_connection() as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume 
                    FROM candles 
                    WHERE symbol = ? 
                    ORDER BY timestamp
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if len(df) > 0:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = df.astype(float)
                
                return df
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading candle data for {symbol}: {e}")
            return pd.DataFrame()

    def load_microstructure_data(self, symbol: str) -> pd.DataFrame:
        """Load futures microstructure data for a symbol"""
        try:
            with self.get_db_connection() as conn:
                query = """
                    SELECT * FROM futures_microstructure 
                    WHERE symbol = ? 
                    ORDER BY timestamp
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if len(df) > 0:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                
                return df
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading microstructure data for {symbol}: {e}")
            return pd.DataFrame()

    def extract_enhanced_features_and_labels(self, candle_df: pd.DataFrame, micro_df: pd.DataFrame):
        """Extract enhanced features combining candles + microstructure"""
        features = []
        labels = []
        
        if len(candle_df) < 200 or len(micro_df) < 1:
            return features, labels
        
        # Calculate technical indicators on candles
        candle_df['rsi'] = self.calculate_rsi(candle_df['close'])
        candle_df['volume_spike'] = candle_df['volume'] / candle_df['volume'].rolling(20).mean()
        candle_df['volatility'] = candle_df['close'].pct_change().rolling(20).std() * 100
        candle_df['price_change_1h'] = candle_df['close'].pct_change(60) * 100
        candle_df['price_change_4h'] = candle_df['close'].pct_change(240) * 100
        candle_df['price_change_24h'] = candle_df['close'].pct_change(1440) * 100
        
        # Future returns for labels
        candle_df['future_return_1h'] = candle_df['close'].shift(-60) / candle_df['close'] - 1
        candle_df['future_return_4h'] = candle_df['close'].shift(-240) / candle_df['close'] - 1
        
        # Merge microstructure data with candles (nearest time match)
        combined_df = pd.merge_asof(
            candle_df.reset_index().sort_values('timestamp'),
            micro_df.reset_index().sort_values('timestamp'),
            on='timestamp',
            direction='backward',
            suffixes=('_candle', '_micro')
        ).set_index('timestamp')
        
        # Extract features from combined dataset
        for i in range(1440, len(combined_df) - 240):  # Leave buffer for look-ahead
            try:
                row = combined_df.iloc[i]
                
                # Base candle features
                candle_features = [
                    row['rsi'] if not pd.isna(row['rsi']) else 50.0,
                    row['price_change_1h'] if not pd.isna(row['price_change_1h']) else 0.0,
                    row['price_change_4h'] if not pd.isna(row['price_change_4h']) else 0.0,
                    row['price_change_24h'] if not pd.isna(row['price_change_24h']) else 0.0,
                    row['volume_spike'] if not pd.isna(row['volume_spike']) else 1.0,
                    row['volatility'] if not pd.isna(row['volatility']) else 0.0,
                ]
                
                # Enhanced microstructure features
                micro_features = [
                    row.get('funding_rate', 0.0) * 10000,  # Convert to basis points
                    row.get('funding_8h_sum', 0.0) * 10000,
                    row.get('funding_24h_sum', 0.0) * 10000,
                    row.get('funding_zscore', 0.0),
                    row.get('oi_1h_change_pct', 0.0),
                    row.get('oi_4h_change_pct', 0.0),
                    row.get('oi_24h_change_pct', 0.0),
                    row.get('long_short_ratio', 1.0),
                    row.get('top_trader_long_ratio', 0.5),
                    row.get('top_trader_short_ratio', 0.5),
                    row.get('liquidation_long_count_1h', 0),
                    row.get('liquidation_short_count_1h', 0),
                    np.log1p(row.get('liquidation_long_usd_1h', 0)),  # Log transform large values
                    np.log1p(row.get('liquidation_short_usd_1h', 0)),
                    row.get('volume_1h_ratio', 1.0),
                ]
                
                # Derived features
                derived_features = [
                    # Funding rate signals
                    1.0 if row.get('funding_rate', 0.0) > 0.0001 else -1.0,  # High funding (crowded long)
                    1.0 if row.get('funding_zscore', 0.0) > 2.0 else 0.0,    # Extreme funding
                    
                    # OI momentum signals
                    1.0 if row.get('oi_1h_change_pct', 0.0) > 5.0 else 0.0,  # Strong OI increase
                    1.0 if row.get('oi_4h_change_pct', 0.0) < -5.0 else 0.0, # Strong OI decrease
                    
                    # Crowd sentiment signals
                    1.0 if row.get('long_short_ratio', 1.0) > 2.0 else 0.0,   # Crowded long
                    1.0 if row.get('long_short_ratio', 1.0) < 0.5 else 0.0,   # Crowded short
                    
                    # Elite trader divergence
                    abs(row.get('top_trader_long_ratio', 0.5) - row.get('long_short_ratio', 1.0) / (1 + row.get('long_short_ratio', 1.0))),
                    
                    # Liquidation pressure
                    row.get('liquidation_long_count_1h', 0) - row.get('liquidation_short_count_1h', 0),
                    
                    # Multi-feature combinations
                    row.get('funding_rate', 0.0) * 10000 * row.get('oi_1h_change_pct', 0.0),  # Funding * OI momentum
                    row['volume_spike'] * (1 if row.get('long_short_ratio', 1.0) > 1.5 else -1),  # Volume with sentiment
                ]
                
                # Combine all features
                feature_vector = candle_features + micro_features + derived_features
                
                # Create labels based on future returns
                future_1h = row['future_return_1h'] if not pd.isna(row['future_return_1h']) else 0.0
                future_4h = row['future_return_4h'] if not pd.isna(row['future_return_4h']) else 0.0
                
                # Multi-class labels: 0=down, 1=neutral, 2=up
                if future_1h > 0.02 or future_4h > 0.04:  # Strong up
                    label = 2
                elif future_1h < -0.02 or future_4h < -0.04:  # Strong down
                    label = 0
                else:  # Neutral
                    label = 1
                
                features.append(feature_vector)
                labels.append(label)
                
            except Exception as e:
                continue  # Skip problematic rows
        
        return features, labels

    def process_symbol_data(self, symbol: str):
        """Process a single symbol's data (thread-safe)"""
        try:
            logger.info(f"🔄 Processing {symbol}...")
            
            # Load both candle and microstructure data
            candle_df = self.load_candle_data(symbol)
            micro_df = self.load_microstructure_data(symbol)
            
            if len(candle_df) < 200:
                logger.warning(f"⚠️ {symbol}: Insufficient candle data ({len(candle_df)} rows)")
                return None
            
            if len(micro_df) < 1:
                logger.warning(f"⚠️ {symbol}: No microstructure data, using candle-only features")
                micro_df = pd.DataFrame()  # Will use default values
            
            # Extract enhanced features
            features, labels = self.extract_enhanced_features_and_labels(candle_df, micro_df)
            
            if len(features) > 0:
                logger.info(f"✅ {symbol}: Added {len(features)} samples")
                return list(zip(features, labels))
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            return None

    def prepare_enhanced_training_data(self):
        """Prepare enhanced training dataset combining candles + microstructure (PARALLEL)"""
        logger.info("🧠 Preparing enhanced training data (candles + microstructure)")
        logger.info("🚀 Using 8-core parallel data loading for maximum speed!")
        
        training_data = []
        processed_symbols = 0
        
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all symbol processing tasks
            future_to_symbol = {
                executor.submit(self.process_symbol_data, symbol): symbol 
                for symbol in self.symbols
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result is not None:
                        training_data.extend(result)
                        processed_symbols += 1
                except Exception as e:
                    logger.error(f"❌ Failed to process {symbol}: {e}")
        
        logger.info(f"🎯 Parallel processing completed: {processed_symbols}/{len(self.symbols)} symbols processed")
        
        if len(training_data) < self.min_training_samples:
            logger.error(f"❌ Insufficient training samples: {len(training_data)} < {self.min_training_samples}")
            return None, None
        
        # Convert to arrays
        X = np.array([item[0] for item in training_data])
        y = np.array([item[1] for item in training_data])
        
        # Define feature names for interpretability
        self.feature_names = [
            'rsi', 'price_1h_change', 'price_4h_change', 'price_24h_change', 'volume_spike', 'volatility',
            'funding_rate_bp', 'funding_8h_sum_bp', 'funding_24h_sum_bp', 'funding_zscore',
            'oi_1h_change', 'oi_4h_change', 'oi_24h_change', 'long_short_ratio',
            'top_trader_long', 'top_trader_short', 'liq_long_count', 'liq_short_count',
            'liq_long_usd_log', 'liq_short_usd_log', 'volume_ratio',
            'high_funding_flag', 'extreme_funding_flag', 'oi_surge_flag', 'oi_drop_flag',
            'crowded_long_flag', 'crowded_short_flag', 'elite_divergence', 'liq_pressure',
            'funding_oi_combo', 'volume_sentiment_combo'
        ]
        
        logger.info(f"✅ Enhanced training data prepared: {len(training_data)} samples, {X.shape[1]} features")
        logger.info(f"📊 Processed symbols: {processed_symbols}/{len(self.symbols)}")
        logger.info(f"🎯 Label distribution: Up={np.sum(y==2)}, Neutral={np.sum(y==1)}, Down={np.sum(y==0)}")
        
        return X, y

    def train_enhanced_model(self):
        """Train enhanced ML model with microstructure features"""
        logger.info("🚀 Starting enhanced ML model training")
        
        # Prepare data
        X, y = self.prepare_enhanced_training_data()
        if X is None:
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info("🧠 Training RandomForestClassifier with enhanced features...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            verbose=1,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, n_jobs=-1)
        
        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test_scaled)
        
        logger.info("📊 ENHANCED MODEL PERFORMANCE:")
        logger.info(f"✅ Training accuracy: {train_score:.4f}")
        logger.info(f"✅ Test accuracy: {test_score:.4f}")
        logger.info(f"✅ CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = sorted(zip(self.feature_names, self.model.feature_importances_), 
                                   key=lambda x: x[1], reverse=True)
        
        logger.info("🔍 TOP 10 FEATURE IMPORTANCE:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            logger.info(f"   {i+1:2d}. {feature:<25}: {importance:.4f}")
        
        # Classification report
        logger.info("📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Neutral', 'Up']))
        
        return True

    def save_enhanced_model(self):
        """Save enhanced model and scaler"""
        try:
            if self.model and self.scaler:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'training_timestamp': datetime.now().isoformat(),
                    'model_type': 'enhanced_microstructure_ml_v2'
                }
                
                joblib.dump(model_data, 'enhanced_scoring_model.pkl')
                logger.info("✅ Enhanced model saved to enhanced_scoring_model.pkl")
                return True
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False

def main():
    """Main training execution"""
    print("🏦 ENHANCED INSTITUTIONAL ML TRAINING")
    print("=" * 60)
    print("🧠 Training with candles + funding + OI + liquidations")
    print("🎯 Target: +10-15pp accuracy boost over baseline")
    print("=" * 60)
    
    trainer = EnhancedMLTrainer()
    
    try:
        success = trainer.train_enhanced_model()
        if success:
            trainer.save_enhanced_model()
            logger.info("🎉 Enhanced ML training completed successfully!")
            return True
        else:
            logger.error("❌ Enhanced ML training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training crashed: {e}")
        return False

if __name__ == "__main__":
    success = main()
