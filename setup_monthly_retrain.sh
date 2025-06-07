#!/bin/bash
set -e

echo "🔧 Setting up monthly retrain timer with correct paths..."

# Find the real project root
PROJECT_ROOT=$(find / -name run_retrain.sh 2>/dev/null | grep -v backup | head -1 | xargs dirname)
echo "📁 Found project at: $PROJECT_ROOT"

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "❌ ERROR: Could not find run_retrain.sh script"
    exit 1
fi

# Create the service file with correct paths
echo "📝 Creating monthly_retrain.service..."
sudo tee /etc/systemd/system/monthly_retrain.service >/dev/null <<EOF
[Unit]
Description=Monthly full ML retrain (cold start, 365-day window)
After=network.target

[Service]
Type=oneshot
Nice=10
LimitNOFILE=65535
WorkingDirectory=$PROJECT_ROOT
ExecStart=$PROJECT_ROOT/run_retrain.sh --mode full --days 365 --tag monthly
TimeoutStartSec=6h
EOF

# Create the timer file
echo "⏰ Creating monthly_retrain.timer..."
sudo tee /etc/systemd/system/monthly_retrain.timer >/dev/null <<EOF
[Unit]
Description=Run monthly_retrain.service on the 1st Sunday 02:00 UTC

[Timer]
OnCalendar=Sun *-*-01..07 02:00:00
RandomizedDelaySec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Reload systemd and enable timer
echo "🔄 Reloading systemd and enabling timer..."
sudo systemctl daemon-reload
sudo systemctl enable --now monthly_retrain.timer

# Show timer status
echo "📊 Timer status:"
systemctl status monthly_retrain.timer --no-pager -l

# Test the service
echo "🧪 Testing monthly retrain service..."
sudo systemctl start monthly_retrain.service

# Wait a moment for it to start
sleep 3

# Show service logs
echo "📋 Recent service logs:"
journalctl -u monthly_retrain.service -n 20 --no-pager

# Final status check
echo "✅ Final timer status:"
systemctl list-timers monthly_retrain.timer --no-pager

echo ""
echo "🎉 Monthly retrain timer setup complete!"
echo "🕐 Next run: $(systemctl list-timers monthly_retrain.timer --no-pager | grep monthly_retrain | awk '{print $1, $2}')"
echo "📁 Project path: $PROJECT_ROOT"
