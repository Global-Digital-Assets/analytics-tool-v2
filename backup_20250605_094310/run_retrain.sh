#!/bin/bash
"""
🚀 INSTITUTIONAL ML RETRAIN WRAPPER
Smart retrain routing with enhanced logging
"""

set -e  # Exit on any error

# Configuration
ANALYTICS_DIR="/root/analytics-tool-v2"
VENV_PYTHON="$ANALYTICS_DIR/venv/bin/python"
ENHANCED_RETRAIN="$ANALYTICS_DIR/enhanced_retrain.py"
LOG_DIR="$ANALYTICS_DIR/logs"

# Ensure we're in the right directory
cd "$ANALYTICS_DIR"

# Parse arguments with defaults
MODE="full"
DAYS="365"
TAG="monthly"
WARM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --days)
            DAYS="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --warm)
            WARM="--warm"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Log the operation
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/retrain_${MODE}_${TIMESTAMP}.log"

echo "🚀 Enhanced Retrain Manager" | tee "$LOG_FILE"
echo "Mode: $MODE | Days: $DAYS | Tag: $TAG | Warm: ${WARM:-'false'}" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"

# Execute the enhanced retrain script
source "$ANALYTICS_DIR/venv/bin/activate"
$VENV_PYTHON "$ENHANCED_RETRAIN" --mode "$MODE" --days "$DAYS" --tag "$TAG" $WARM 2>&1 | tee -a "$LOG_FILE"

RETRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "======================================" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Exit Code: $RETRAIN_EXIT_CODE" | tee -a "$LOG_FILE"

# If successful, trigger a quick analytics test run
if [ $RETRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Retrain successful, testing inference..." | tee -a "$LOG_FILE"
    timeout 30s $VENV_PYTHON "$ANALYTICS_DIR/ml_inference_engine.py" --test >> "$LOG_FILE" 2>&1 || true
else
    echo "❌ Retrain failed!" | tee -a "$LOG_FILE"
fi

exit $RETRAIN_EXIT_CODE
