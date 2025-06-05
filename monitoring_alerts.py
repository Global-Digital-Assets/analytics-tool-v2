#!/usr/bin/env python3
"""
24/7 Monitoring and Alerting System
Checks system health and sends alerts when issues are detected
"""

import sqlite3
import os
import time
import json
import requests
import logging
from datetime import datetime, timedelta
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/root/analytics-tool-v2/logs/monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HealthMonitor:
    def __init__(self):
        self.db_path = "/root/analytics-tool-v2/market_data.db"
        self.analysis_path = "/root/analytics-tool-v2/multi_token_analysis.json"
        self.alert_file = "/root/analytics-tool-v2/logs/alerts.log"
        
    def check_database_health(self):
        """Check if database is receiving recent data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check recent candles (last 10 minutes)
            cursor.execute("""
                SELECT COUNT(*) FROM candles 
                WHERE timestamp > (strftime('%s', 'now') - 600) * 1000
            """)
            recent_candles = cursor.fetchone()[0]
            
            # Check total candles
            cursor.execute("SELECT COUNT(*) FROM candles")
            total_candles = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "status": "healthy" if recent_candles > 0 else "unhealthy",
                "recent_candles": recent_candles,
                "total_candles": total_candles,
                "issue": None if recent_candles > 0 else "No recent data ingestion"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "issue": f"Database error: {str(e)}"
            }
    
    def check_analysis_freshness(self):
        """Check if analysis files are being updated"""
        try:
            if not os.path.exists(self.analysis_path):
                return {
                    "status": "error",
                    "issue": "Analysis file missing"
                }
            
            age = time.time() - os.path.getmtime(self.analysis_path)
            is_fresh = age < 1800  # 30 minutes
            
            return {
                "status": "healthy" if is_fresh else "stale",
                "age_minutes": int(age / 60),
                "issue": None if is_fresh else f"Analysis file is {int(age/60)} minutes old"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "issue": f"Analysis check error: {str(e)}"
            }
    
    def check_services(self):
        """Check if critical services are running"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "analytics-streamer.service", "crypto-streamer.service"],
                capture_output=True, text=True
            )
            
            services = result.stdout.strip().split('\n')
            active_count = sum(1 for s in services if s == "active")
            
            return {
                "status": "healthy" if active_count == 2 else "degraded",
                "active_services": active_count,
                "total_services": 2,
                "issue": None if active_count == 2 else f"Only {active_count}/2 services active"
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "issue": f"Service check error: {str(e)}"
            }
    
    def check_disk_space(self):
        """Check available disk space"""
        try:
            result = subprocess.run(["df", "/"], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            usage_line = lines[1].split()
            used_percent = int(usage_line[4].rstrip('%'))
            
            return {
                "status": "healthy" if used_percent < 85 else "warning",
                "used_percent": used_percent,
                "issue": None if used_percent < 85 else f"Disk usage high: {used_percent}%"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "issue": f"Disk check error: {str(e)}"
            }
    
    def send_alert(self, message, level="WARNING"):
        """Log alerts and optionally send notifications"""
        alert_msg = f"[{level}] {datetime.utcnow().isoformat()}Z - {message}"
        
        # Log to file
        with open(self.alert_file, "a") as f:
            f.write(alert_msg + "\n")
        
        logger.warning(alert_msg)
        
        # TODO: Add webhook/email integration here
        # Example: send to Discord webhook, Slack, email, etc.
    
    def run_health_check(self):
        """Run complete health check"""
        logger.info("Starting health check...")
        
        checks = {
            "database": self.check_database_health(),
            "analysis": self.check_analysis_freshness(),
            "services": self.check_services(),
            "disk": self.check_disk_space()
        }
        
        issues = []
        for check_name, result in checks.items():
            if result["status"] in ["unhealthy", "error", "warning", "stale", "degraded"]:
                issues.append(f"{check_name}: {result.get('issue', 'Unknown issue')}")
        
        if issues:
            self.send_alert(f"System health issues detected: {'; '.join(issues)}")
        else:
            logger.info("✅ All health checks passed")
        
        # Save health report
        health_report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "overall_status": "healthy" if not issues else "issues_detected",
            "checks": checks,
            "issues": issues
        }
        
        with open("/root/analytics-tool-v2/logs/health_report.json", "w") as f:
            json.dump(health_report, f, indent=2)
        
        return health_report

if __name__ == "__main__":
    monitor = HealthMonitor()
    monitor.run_health_check()
