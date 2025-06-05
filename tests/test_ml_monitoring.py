#!/usr/bin/env python3
"""
Unit tests for ml_monitoring.py
Target: ≥85% line coverage for production ML monitoring system
"""

import pytest
import json
import sqlite3
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the modules we're testing
import sys
sys.path.append('..')
from ml_monitoring import MLMonitoringSystem


class TestMLMonitoring:
    """Test suite for MLMonitoring class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        
        # Create test schema
        conn = sqlite3.connect(db_path)
        conn.execute('''
            CREATE TABLE candles (
                timestamp INTEGER,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        
        # Insert test data
        test_data = [
            (int(datetime.now().timestamp()) - i*3600, 'BTCUSDT', 50000+i*100, 51000+i*100, 49000+i*100, 50500+i*100, 1000+i*10)
            for i in range(100)
        ]
        conn.executemany(
            'INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?)',
            test_data
        )
        conn.commit()
        conn.close()
        
        yield db_path
        Path(db_path).unlink()
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory with mock metadata"""
        import tempfile
        models_dir = Path(tempfile.mkdtemp())
        
        # Create mock metadata
        metadata = {
            "model_info": {
                "filename": "test_model.txt",
                "timestamp": "20250605_120000",
                "model_type": "LightGBM"
            },
            "performance": {
                "accuracy": 0.943,
                "auc": 0.821
            },
            "cross_validation": {
                "mean_auc": 0.785,
                "std_auc": 0.023,
                "mean_accuracy": 0.920
            }
        }
        
        with open(models_dir / "latest_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create dummy model file
        (models_dir / "test_model.txt").touch()
        
        yield models_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(models_dir)
    
    @pytest.fixture
    def ml_monitor(self, temp_db, temp_models_dir):
        """Create MLMonitoring instance with test data"""
        return MLMonitoringSystem(db_path=temp_db, models_dir=str(temp_models_dir))
    
    def test_initialization(self, temp_db, temp_models_dir):
        """Test MLMonitoring initialization"""
        monitor = MLMonitoringSystem(db_path=temp_db, models_dir=str(temp_models_dir))
        
        assert monitor.db_path == temp_db
        assert monitor.models_dir == Path(temp_models_dir)
        assert len(monitor.symbols) == 30  # All symbols loaded
    
    def test_load_recent_data(self, ml_monitor):
        """Test loading recent market data"""
        data = ml_monitor.load_recent_data(days=7)
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check data structure
        sample_row = data[0]
        expected_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        assert len(sample_row) == len(expected_columns)
    
    @patch('ml_monitoring.lgb.Booster')
    def test_evaluate_current_model_success(self, mock_booster, ml_monitor):
        """Test successful model evaluation"""
        # Mock model predictions
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.7, 0.8, 0.6, 0.9, 0.5] * 20)  # 100 predictions
        mock_booster.return_value = mock_model
        
        # Mock data loading
        with patch.object(ml_monitor, 'load_recent_data') as mock_load:
            mock_load.return_value = [(i, 'BTCUSDT', 50000, 51000, 49000, 50500, 1000) for i in range(100)]
            
            with patch.object(ml_monitor, 'engineer_features') as mock_features:
                mock_features.return_value = (np.random.rand(100, 11), np.random.randint(0, 2, 100))
                
                results = ml_monitor.evaluate_current_model()
        
        assert results is not None
        assert 'current_performance' in results
        assert 'drift' in results
        assert 'baseline_auc' in results
        
        # Check performance metrics
        perf = results['current_performance']
        assert 'accuracy' in perf
        assert 'auc' in perf
        assert 0 <= perf['accuracy'] <= 1
        assert 0 <= perf['auc'] <= 1
    
    def test_check_retrain_conditions_no_retrain(self, ml_monitor):
        """Test retrain conditions when no retrain needed"""
        evaluation_results = {
            'current_performance': {'accuracy': 0.92, 'auc': 0.80},
            'drift': {'accuracy_drift': 0.02, 'auc_drift': 0.02}
        }
        
        decision = ml_monitor.check_retrain_conditions(evaluation_results)
        
        assert not decision['should_retrain']
        assert decision['retrain_type'] == 'none'
        assert decision['urgency'] == 'normal'
    
    def test_check_retrain_conditions_drift_trigger(self, ml_monitor):
        """Test retrain conditions when drift detected"""
        evaluation_results = {
            'current_performance': {'accuracy': 0.88, 'auc': 0.72},  # 0.785 - 0.72 = 0.065 > 0.05
            'drift': {'accuracy_drift': 0.05, 'auc_drift': 0.065}
        }
        
        decision = ml_monitor.check_retrain_conditions(evaluation_results)
        
        assert decision['should_retrain']
        assert decision['retrain_type'] == 'drift_hotfix'
        assert decision['urgency'] == 'high'
        assert 'degraded by' in decision['reason']
    
    def test_check_retrain_conditions_emergency(self, ml_monitor):
        """Test retrain conditions for emergency scenarios"""
        evaluation_results = {
            'current_performance': {'accuracy': 0.80, 'auc': 0.70},  # Both below critical thresholds
            'drift': {'accuracy_drift': 0.10, 'auc_drift': 0.085}
        }
        
        decision = ml_monitor.check_retrain_conditions(evaluation_results)
        
        assert decision['should_retrain']
        assert decision['retrain_type'] == 'emergency'
        assert decision['urgency'] == 'critical'
    
    def test_check_retrain_conditions_empty_data(self, ml_monitor):
        """Test retrain conditions with empty evaluation data"""
        decision = ml_monitor.check_retrain_conditions({})
        
        assert not decision['should_retrain']
        assert decision['reason'] == "No evaluation data"
    
    @patch('ml_monitoring.EnhancedRetrainManager')
    def test_auto_retrain_pipeline_success(self, mock_retrain_manager, ml_monitor):
        """Test successful auto retrain pipeline"""
        # Mock evaluation results
        with patch.object(ml_monitor, 'evaluate_current_model') as mock_eval:
            mock_eval.return_value = {
                'current_performance': {'accuracy': 0.88, 'auc': 0.72},
                'drift': {'accuracy_drift': 0.05, 'auc_drift': 0.065}
            }
            
            # Mock retrain manager
            mock_manager = Mock()
            mock_manager.drift_hotfix_retrain.return_value = True
            mock_retrain_manager.return_value = mock_manager
            
            result = ml_monitor.auto_retrain_pipeline()
        
        assert result is True
        mock_manager.drift_hotfix_retrain.assert_called_once_with(days=180)
    
    @patch('ml_monitoring.EnhancedRetrainManager')
    def test_auto_retrain_pipeline_emergency(self, mock_retrain_manager, ml_monitor):
        """Test auto retrain pipeline for emergency scenario"""
        # Mock evaluation results for emergency
        with patch.object(ml_monitor, 'evaluate_current_model') as mock_eval:
            mock_eval.return_value = {
                'current_performance': {'accuracy': 0.80, 'auc': 0.70},
                'drift': {'accuracy_drift': 0.10, 'auc_drift': 0.085}
            }
            
            # Mock retrain manager
            mock_manager = Mock()
            mock_manager.full_retrain.return_value = True
            mock_retrain_manager.return_value = mock_manager
            
            result = ml_monitor.auto_retrain_pipeline()
        
        assert result is True
        mock_manager.full_retrain.assert_called_once_with(days=90, tag="emergency")
    
    def test_engineer_features_structure(self, ml_monitor):
        """Test feature engineering output structure"""
        # Create mock data
        test_data = [(i, 'BTCUSDT', 50000+i, 51000+i, 49000+i, 50500+i, 1000+i) for i in range(100)]
        
        X, y = ml_monitor.engineer_features(test_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[1] == 11  # 11 features as defined in the system
        assert len(X) == len(y)
        assert len(X) > 0
    
    def test_environment_variable_threshold(self, ml_monitor):
        """Test drift threshold configuration via environment variable"""
        import os
        
        # Test default threshold
        evaluation_results = {
            'current_performance': {'accuracy': 0.88, 'auc': 0.73},  # 0.785 - 0.73 = 0.055 > 0.05
            'drift': {'accuracy_drift': 0.05, 'auc_drift': 0.055}
        }
        
        decision = ml_monitor.check_retrain_conditions(evaluation_results)
        assert decision['should_retrain']  # Should trigger with default 0.05 threshold
        
        # Test custom threshold
        os.environ['DRIFT_THRESHOLD'] = '0.07'
        try:
            decision = ml_monitor.check_retrain_conditions(evaluation_results)
            assert not decision['should_retrain']  # Should NOT trigger with 0.07 threshold
        finally:
            del os.environ['DRIFT_THRESHOLD']


class TestMLMonitoringEdgeCases:
    """Test edge cases and error handling"""
    
    def test_missing_model_file(self, temp_db):
        """Test behavior when model file is missing"""
        models_dir = Path(tempfile.mkdtemp())
        monitor = MLMonitoringSystem(db_path=temp_db, models_dir=str(models_dir))
        
        result = monitor.evaluate_current_model()
        assert result is None  # Should handle missing model gracefully
        
        # Cleanup
        import shutil
        shutil.rmtree(models_dir)
    
    def test_corrupt_metadata(self, temp_db):
        """Test behavior with corrupt metadata file"""
        models_dir = Path(tempfile.mkdtemp())
        
        # Create corrupt metadata
        with open(models_dir / "latest_metadata.json", 'w') as f:
            f.write("invalid json {")
        
        monitor = MLMonitoringSystem(db_path=temp_db, models_dir=str(models_dir))
        
        # Should use fallback baseline
        evaluation_results = {
            'current_performance': {'accuracy': 0.88, 'auc': 0.72},
            'drift': {'accuracy_drift': 0.05, 'auc_drift': 0.065}
        }
        
        decision = monitor.check_retrain_conditions(evaluation_results)
        assert decision['metrics']['baseline_auc'] == 0.785  # Fallback value
        
        # Cleanup
        import shutil
        shutil.rmtree(models_dir)


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_ml_monitoring.py -v --cov=ml_monitoring --cov-report=term-missing
    pytest.main([__file__, '-v'])
