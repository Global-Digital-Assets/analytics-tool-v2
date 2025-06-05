#!/usr/bin/env python3
"""
Training Script for Enhanced Scoring Model
Version: enhanced_v2.5_institutional_cascade
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from datetime import datetime, timedelta

# Configuration
DB_PATH = 'market_data.db'
ML_MODEL_PATH = 'scoring_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
MIN_TRAINING_SAMPLES = 1000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_training_data():
    """Prepare training data from historical candle data"""
    logger.info("🔄 Preparing training data from historical candles...")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get symbols with sufficient data
        symbols_query = """
            SELECT symbol, COUNT(*) as count 
            FROM candles 
            GROUP BY symbol 
            HAVING count >= 500
            ORDER BY count DESC
        """
        symbols_df = pd.read_sql_query(symbols_query, conn)
        
        if len(symbols_df) < 5:
            logger.error("❌ Insufficient symbols with enough data for training")
            return None, None
        
        training_data = []
        
        for symbol in symbols_df['symbol'].head(20):  # Use top 20 symbols with most data
            logger.info(f"Processing {symbol}...")
            
            # Get historical data
            query = """
                SELECT open, high, low, close, volume, timestamp
                FROM candles 
                WHERE symbol = ? 
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
            
            if len(df) < 200:
                continue
            
            # Calculate features and labels
            features, labels = extract_features_and_labels(df)
            
            if len(features) > 0:
                training_data.extend(list(zip(features, labels)))
        
        conn.close()
        
        if len(training_data) < MIN_TRAINING_SAMPLES:
            logger.error(f"❌ Insufficient training samples: {len(training_data)} < {MIN_TRAINING_SAMPLES}")
            return None, None
        
        # Convert to arrays
        X = np.array([item[0] for item in training_data])
        y = np.array([item[1] for item in training_data])
        
        logger.info(f"✅ Prepared {len(training_data)} training samples with {X.shape[1]} features")
        return X, y
        
    except Exception as e:
        logger.error(f"❌ Error preparing training data: {e}")
        return None, None

def extract_features_and_labels(df):
    """Extract features and create labels from historical data"""
    features = []
    labels = []
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['volume_spike'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
    df['price_change_1h'] = df['close'].pct_change(60) * 100
    df['price_change_4h'] = df['close'].pct_change(240) * 100
    df['price_change_24h'] = df['close'].pct_change(1440) * 100
    
    # Look ahead for labels (future returns)
    df['future_return_1h'] = df['close'].shift(-60) / df['close'] - 1
    df['future_return_4h'] = df['close'].shift(-240) / df['close'] - 1
    
    for i in range(1440, len(df) - 240):  # Leave buffer for look-ahead
        try:
            # Extract features
            feature_vector = [
                df['rsi'].iloc[i] if not pd.isna(df['rsi'].iloc[i]) else 50.0,
                df['price_change_1h'].iloc[i] if not pd.isna(df['price_change_1h'].iloc[i]) else 0.0,
                df['price_change_4h'].iloc[i] if not pd.isna(df['price_change_4h'].iloc[i]) else 0.0,
                df['price_change_24h'].iloc[i] if not pd.isna(df['price_change_24h'].iloc[i]) else 0.0,
                df['volume_spike'].iloc[i] if not pd.isna(df['volume_spike'].iloc[i]) else 1.0,
                0.0,  # support_distance placeholder
                0.0,  # resistance_distance placeholder
                0,    # timeframe_bullish_count placeholder
                0,    # timeframe_bearish_count placeholder
                df['volatility'].iloc[i] if not pd.isna(df['volatility'].iloc[i]) else 0.0,
                0.0,  # momentum_3d placeholder
                0.0   # momentum_7d placeholder
            ]
            
            # Create label based on future returns
            future_1h = df['future_return_1h'].iloc[i]
            future_4h = df['future_return_4h'].iloc[i]
            
            if pd.isna(future_1h) or pd.isna(future_4h):
                continue
            
            # Simple labeling: 1 for profitable (>1% gain), 0 otherwise
            label = 1 if (future_1h > 0.01 or future_4h > 0.02) else 0
            
            features.append(feature_vector)
            labels.append(label)
            
        except Exception as e:
            continue
    
    return features, labels

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model():
    """Train the enhanced scoring model"""
    logger.info("🚀 Starting ML model training...")
    
    # Prepare data
    X, y = prepare_training_data()
    if X is None or y is None:
        logger.error("❌ Failed to prepare training data")
        return False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("🔄 Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = accuracy_score(y_train, model.predict(X_train_scaled))
    test_score = accuracy_score(y_test, model.predict(X_test_scaled))
    
    logger.info(f"📊 Model Performance:")
    logger.info(f"   Training Accuracy: {train_score:.3f}")
    logger.info(f"   Testing Accuracy: {test_score:.3f}")
    
    # Save model and scaler
    joblib.dump(model, ML_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    logger.info(f"✅ Model saved to {ML_MODEL_PATH}")
    logger.info(f"✅ Scaler saved to {SCALER_PATH}")
    
    # Detailed classification report
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return True

if __name__ == "__main__":
    success = train_model()
    if success:
        logger.info("🎉 Model training completed successfully!")
    else:
        logger.error("❌ Model training failed!")
