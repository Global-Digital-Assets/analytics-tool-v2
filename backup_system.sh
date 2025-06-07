#!/bin/bash
set -euo pipefail

# Database Backup Script
BACKUP_DIR="/root/analytics-tool-v2/backups"
LOG_FILE="/root/analytics-tool-v2/logs/backup.log"
DB_FILE="/root/analytics-tool-v2/market_data.db"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup directory
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$LOG_FILE"
}

# Backup main database
log "Starting database backup..."
if [ -f "$DB_FILE" ]; then
    cp "$DB_FILE" "$BACKUP_DIR/market_data_${TIMESTAMP}.db"
    log "✅ Database backed up: market_data_${TIMESTAMP}.db ($(du -h "$BACKUP_DIR/market_data_${TIMESTAMP}.db" | cut -f1))"
else
    log "❌ Database file not found: $DB_FILE"
    exit 1
fi

# Backup analysis files
cp multi_token_analysis.json "$BACKUP_DIR/analysis_${TIMESTAMP}.json" 2>/dev/null || log "⚠️ No analysis file to backup"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "market_data_*.db" -mtime +7 -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "analysis_*.json" -mtime +7 -delete 2>/dev/null || true

log "✅ Backup complete. Total backups: $(ls -1 "$BACKUP_DIR"/market_data_*.db | wc -l)"
