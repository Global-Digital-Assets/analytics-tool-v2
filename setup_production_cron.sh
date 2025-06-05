#!/bin/bash
"""
🕰️ PRODUCTION CRON SETUP
Automated monitoring and retraining schedule
"""

PROJECT_PATH="/root/analytics-tool-v2"
PYTHON_ENV="$PROJECT_PATH/venv/bin/python"
LOG_PATH="$PROJECT_PATH/logs"

# Create logs directory
mkdir -p "$LOG_PATH"

# Cron jobs to add:
cat << 'EOF' > /tmp/ml_cron_jobs
# ML Model Monitoring (every 4 hours)
0 */4 * * * cd /root/analytics-tool-v2 && /root/analytics-tool-v2/venv/bin/python ml_monitoring.py --action monitor >> logs/ml_monitor.log 2>&1

# Cache update for 5-minute bars (every 15 minutes)
*/15 * * * * cd /root/analytics-tool-v2 && /root/analytics-tool-v2/venv/bin/python -c "from ml_inference_engine import MLInferenceEngine; MLInferenceEngine().update_cached_5min_bars()" >> logs/cache_update.log 2>&1

# Weekly full retrain (Sunday 2 AM)
0 2 * * 0 cd /root/analytics-tool-v2 && /root/analytics-tool-v2/venv/bin/python production_ml_pipeline.py --days 180 >> logs/weekly_retrain.log 2>&1

# Performance evaluation (daily at 6 AM) 
0 6 * * * cd /root/analytics-tool-v2 && /root/analytics-tool-v2/venv/bin/python ml_monitoring.py --action evaluate --days 7 >> logs/daily_eval.log 2>&1

# Log cleanup (keep last 30 days, runs weekly)
0 3 * * 1 find /root/analytics-tool-v2/logs -name "*.log" -mtime +30 -delete
EOF

echo "📋 Cron jobs to install:"
cat /tmp/ml_cron_jobs

echo ""
echo "To install these cron jobs on your VPS, run:"
echo "scp setup_production_cron.sh root@78.47.150.122:/tmp/"
echo "ssh root@78.47.150.122 'bash /tmp/setup_production_cron.sh'"
