#!/usr/bin/env python3
"""
🚀 ENHANCED ML RETRAINING SYSTEM
Institutional-grade model lifecycle management with warm-start capability
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedRetrainManager:
    def __init__(self, base_dir: str = "/root/analytics-tool-v2"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.venv_python = self.base_dir / "venv/bin/python"
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def get_model_metadata(self, model_path: Path) -> dict:
        """Get metadata from existing model"""
        metadata_path = model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def full_retrain(self, days: int = 365, tag: str = "monthly") -> bool:
        """Full model retrain from scratch"""
        logger.info(f"🔄 Starting FULL retrain ({tag}) with {days} days of data...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"{tag}_retrain_{timestamp}.log"
        
        try:
            # Run production ML pipeline
            cmd = [
                str(self.venv_python),
                str(self.base_dir / "production_ml_pipeline.py"),
                "--days", str(days),
                "--tag", tag
            ]
            
            logger.info(f"🚀 Executing: {' '.join(cmd)}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=self.base_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            if result.returncode == 0:
                logger.info(f"✅ Full retrain ({tag}) completed successfully")
                self._archive_old_models(tag)
                return True
            else:
                logger.error(f"❌ Full retrain ({tag}) failed with code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Full retrain error: {e}")
            return False
    
    def drift_hotfix_retrain(self, days: int = 180) -> bool:
        """Quick drift-triggered retrain with WARM-START (additive learning)"""
        logger.info(f"🚨 Starting DRIFT HOTFIX retrain with {days} days of data...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"drift_hotfix_{timestamp}.log"
        
        try:
            # Run warm-start pipeline - preserves existing model knowledge
            cmd = [
                str(self.venv_python),
                str(self.base_dir / "production_ml_pipeline.py"),
                "--days", str(days),
                "--tag", "drift_hotfix",
                "--warm"  # KEY: Additive learning, no reset to zero
            ]
            
            logger.info(f"🚀 Executing WARM-START: {' '.join(cmd)}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=self.base_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            if result.returncode == 0:
                logger.info(f"✅ Drift hotfix (warm-start) completed successfully")
                return True
            else:
                logger.error(f"❌ Drift hotfix failed with code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Drift hotfix error: {e}")
            return False
    
    def warm_start_update(self, days: int = 90) -> bool:
        """Warm-start model update (preserves existing trees)"""
        logger.info(f"🔧 Starting WARM-START update with {days} days of new data...")
        
        # This would require LightGBM warm-start implementation
        # For now, fall back to hotfix retrain
        logger.warning("⚠️ Warm-start not yet implemented, using hotfix retrain")
        return self.drift_hotfix_retrain(days)
    
    def _archive_old_models(self, tag: str, keep_count: int = 5):
        """Archive old models, keeping only recent ones"""
        try:
            # Find models with this tag
            pattern = f"lgbm_*_{tag}.txt"
            models = list(self.models_dir.glob(pattern))
            
            if len(models) > keep_count:
                # Sort by creation time and remove oldest
                models.sort(key=lambda x: x.stat().st_mtime)
                for old_model in models[:-keep_count]:
                    logger.info(f"🗂️ Archiving old model: {old_model.name}")
                    old_model.unlink()
                    # Also remove corresponding metadata
                    metadata_file = old_model.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                        
        except Exception as e:
            logger.warning(f"⚠️ Model archiving failed: {e}")
    
    def emergency_retrain(self) -> bool:
        """Emergency retrain for critical performance drops"""
        logger.warning("🆘 EMERGENCY RETRAIN triggered!")
        
        # Use aggressive short-term window for immediate adaptation
        return self.full_retrain(days=90, tag="emergency")

def main():
    parser = argparse.ArgumentParser(description="Enhanced ML Retraining System")
    parser.add_argument("--mode", choices=["full", "drift", "warm", "emergency"], 
                       default="full", help="Retraining mode")
    parser.add_argument("--days", type=int, default=365, 
                       help="Days of data to use")
    parser.add_argument("--tag", type=str, default="monthly",
                       help="Model tag for versioning")
    
    args = parser.parse_args()
    
    manager = EnhancedRetrainManager()
    
    success = False
    if args.mode == "full":
        success = manager.full_retrain(args.days, args.tag)
    elif args.mode == "drift":
        success = manager.drift_hotfix_retrain(args.days)
    elif args.mode == "warm":
        success = manager.warm_start_update(args.days)
    elif args.mode == "emergency":
        success = manager.emergency_retrain()
    
    if success:
        logger.info("🎉 Retrain operation completed successfully")
        sys.exit(0)
    else:
        logger.error("💥 Retrain operation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
