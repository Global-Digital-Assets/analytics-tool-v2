#!/usr/bin/env python3
"""
Test Script for Enhanced Institutional-Grade Scoring System
Version: enhanced_v2.5_institutional_cascade
"""

import asyncio
import aiohttp
import json
import logging
from multi_token_analyzer import EnhancedTokenAnalyzer, EnhancedScoringEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_features():
    """Test the enhanced institutional-grade features"""
    logger.info("🧪 Testing Enhanced Institutional-Grade Scoring System...")
    
    # Initialize analyzer
    analyzer = EnhancedTokenAnalyzer('market_data.db')
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    async with aiohttp.ClientSession() as session:
        # Validate symbols
        valid_symbols = await analyzer.validate_symbols(session)
        logger.info(f"✅ Validated symbols: {len(valid_symbols)}")
        
        # Test enhanced analysis on a few tokens
        results = []
        for symbol in test_symbols[:3]:  # Test first 3
            logger.info(f"🔄 Testing enhanced analysis for {symbol}...")
            result = await analyzer.analyze_token_async(symbol, session)
            if result:
                results.append(result)
                logger.info(f"📊 {symbol} Results:")
                logger.info(f"   Scoring Method: {result.scoring_method}")
                logger.info(f"   Raw Scores: L={result.raw_long_score:.1f}, S={result.raw_short_score:.1f}")
                logger.info(f"   Final Scores: L={result.long_score:.1f}, S={result.short_score:.1f}")
                logger.info(f"   Correlation: BTC={result.correlation_btc:.3f}, ETH={result.correlation_eth:.3f}")
                logger.info(f"   Penalty: {result.correlation_penalty:.3f}")
                logger.info(f"   Actions: {result.simple_long_action} / {result.simple_short_action}")
                logger.info(f"   ML Confidence: {result.ml_confidence:.3f}")
                print()
        
        # Test percentile ranking
        if results:
            logger.info("🔄 Testing cross-sectional percentile ranking...")
            ranked_results = analyzer.add_percentile_ranking(results)
            
            for result in ranked_results:
                logger.info(f"📊 {result.symbol}: Long Percentile={result.percentile_long:.1f}%, Short Percentile={result.percentile_short:.1f}%")
        
        # Test regime detection
        logger.info("🔄 Testing market regime detection...")
        regime_info = analyzer.regime_detector.detect_regime()
        logger.info(f"📊 Market Regime: {regime_info}")
        
        # Test scoring engine features
        logger.info("🔄 Testing EnhancedScoringEngine features...")
        scoring_engine = EnhancedScoringEngine('market_data.db')
        logger.info(f"✅ ML Available: {scoring_engine.is_trained}")
        logger.info(f"✅ Feature Names: {scoring_engine.feature_names}")
        
    logger.info("🎉 Enhanced system test completed!")

def test_json_serialization():
    """Test that enhanced TokenOpportunity can be serialized to JSON"""
    logger.info("🧪 Testing JSON serialization of enhanced TokenOpportunity...")
    
    from multi_token_analyzer import TokenOpportunity
    
    # Create a test opportunity with all enhanced fields
    test_opp = TokenOpportunity(
        symbol="TESTUSDT",
        current_price=100.0,
        price_change_1h=1.5,
        price_change_4h=2.0,
        price_change_24h=-0.5,
        rsi=65.5,
        volume_spike=True,
        support_bounce=False,
        resistance_rejection=True,
        multi_timeframe_alignment=True,
        composite_score=75.5,
        entry_recommendation="BUY_LONG",
        volatility=2.5,
        returns_vec=[0.01, -0.005, 0.02],
        long_score=78.0,
        short_score=45.0,
        raw_long_score=82.0,
        raw_short_score=50.0,
        percentile_long=85.5,
        percentile_short=23.2,
        correlation_btc=0.75,
        correlation_eth=0.68,
        correlation_penalty=0.15,
        regime_multiplier=0.9,
        direction="LONG",
        simple_long_action="BUY_LONG",
        simple_short_action="DON'T_SHORT",
        feature_vector=[65.5, 1.5, 2.0, -0.5, 1.2, 0.01, 0.02, 2, 0, 2.5, 0.5, -1.0],
        ml_confidence=0.85,
        scoring_method="ml_model"
    )
    
    try:
        # Test serialization
        from dataclasses import asdict
        opp_dict = asdict(test_opp)
        json_str = json.dumps(opp_dict, indent=2)
        
        # Test deserialization
        parsed_dict = json.loads(json_str)
        
        logger.info("✅ JSON serialization test passed!")
        logger.info(f"📄 Sample JSON structure: {json_str[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ JSON serialization test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ENHANCED INSTITUTIONAL-GRADE SCORING SYSTEM TEST")
    print("=" * 60)
    
    # Test JSON serialization first
    json_test_passed = test_json_serialization()
    print()
    
    if json_test_passed:
        # Test enhanced features
        asyncio.run(test_enhanced_features())
    else:
        logger.error("❌ Skipping async tests due to JSON serialization failure")
