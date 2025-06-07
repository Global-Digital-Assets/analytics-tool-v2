#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Daily Git backup script – VPS is the single source of truth.
# Commits and pushes any pending changes from /root/analytics-tool-v2 to origin.
# ---------------------------------------------------------------------------
# CRITICAL WORKFLOW:
#   VPS commits & pushes → remote; local/GitHub only pull.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK_FILE="/tmp/git_backup.lock"

# Safety: ensure running on production VPS path
if [[ "$REPO_DIR" != "/root/analytics-tool-v2" ]]; then
  echo "❌ git_backup.sh must run on /root/analytics-tool-v2 (current: $REPO_DIR)" >&2
  exit 1
fi

# Get lock to avoid overlapping executions
exec 200>"$LOCK_FILE"
flock -n 200 || { echo "🚧 git_backup already running"; exit 0; }

cd "$REPO_DIR"

# Verify git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not a git repository: $REPO_DIR" >&2
  exit 1
fi

# Optional override of remote URL via env var
if [[ -n "${GIT_BACKUP_REMOTE_URL:-}" ]]; then
  git remote set-url origin "$GIT_BACKUP_REMOTE_URL"
fi

# Stage any changes (additions, deletions, modifications)
git add -A

# If nothing staged, exit gracefully
if git diff --cached --quiet; then
  echo "✅ No changes to back up"
  exit 0
fi

TIMESTAMP="$(date '+%Y-%m-%d %H:%M')"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT_MSG="$TIMESTAMP – auto-backup"

git commit -m "$COMMIT_MSG"
git push origin "$BRANCH"

echo "🎉 Backup pushed to origin/$BRANCH at $TIMESTAMP"
