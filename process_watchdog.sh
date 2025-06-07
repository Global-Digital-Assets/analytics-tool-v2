#!/bin/bash
set -euo pipefail

# Process Watchdog - Ensures critical services stay running
LOG_FILE="/root/analytics-tool-v2/logs/watchdog.log"

log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

# Check and restart services if needed
check_service() {
    local service_name=$1
    if ! systemctl is-active --quiet "$service_name"; then
        log "⚠️ Service $service_name is not active, restarting..."
        systemctl restart "$service_name"
        if systemctl is-active --quiet "$service_name"; then
            log "✅ Service $service_name restarted successfully"
        else
            log "❌ Failed to restart service $service_name"
        fi
    fi
}

# Check API server process
check_api_server() {
    if ! pgrep -f "simple_api_server.py" > /dev/null; then
        log "⚠️ API server not running, restarting..."
        cd /root/analytics-tool-v2
        nohup python3 simple_api_server.py > api_server.log 2>&1 &
        sleep 2
        if pgrep -f "simple_api_server.py" > /dev/null; then
            log "✅ API server restarted successfully"
        else
            log "❌ Failed to restart API server"
        fi
    fi
}

# Main watchdog routine
log "Starting watchdog check..."
check_service "analytics-streamer.service"
check_service "crypto-streamer.service"
check_api_server
log "Watchdog check complete"
