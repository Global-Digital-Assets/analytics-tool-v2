# 📊 CRYPTO ANALYTICS PLATFORM - PRODUCTION DOCUMENTATION

## 🚀 Current Production Server
- **IP**: 78.47.150.122
- **Analytics Port**: 8080
- **Application Location**: `/root/analytics-tool-v2`
- **SSH Access**: `ssh -i ~/.ssh/binance_futures_tool root@78.47.150.122`
- **Source of Truth**: VPS server (Git serves as backup/version control)

## 📡 Live API Endpoints
- **http://78.47.150.122:8080/api/latest** - Retrieves the latest candle data for all tracked tokens
- **http://78.47.150.122:8080/api/status** - Provides system status, including active token count  
- **http://78.47.150.122:8080/api/analysis** - **[PRIMARY]** Serves detailed multi-token analysis with binary BUY_LONG/SHORT recommendations

## 🎯 Latest Features & Binary Recommendations

### Binary Trading Signals (Score Threshold ≥ 60)
- **BUY_LONG**: Recommended for long positions (score ≥ 60)
- **DON'T_BUY_LONG**: Not recommended for long positions (score < 60)
- **SHORT**: Recommended for short positions (score ≥ 60) 
- **DON'T_SHORT**: Not recommended for short positions (score < 60)

### Current Market Analysis (Latest Run)
- **📈 BUY_LONG**: ADAUSDT (70), SUSHIUSDT (70)
- **📉 SHORT**: BTCUSDT (75), TRXUSDT (75), NEARUSDT (75), SHIBUSDT (75), APTUSDT (75), ETCUSDT (75)
- **Total Tokens Analyzed**: 30

## 💎 Token Coverage
- **30 tokens** analyzed across major categories: MAJORS, DEFI, MEME, EMERGING, LAYER2/OTHER
- Token list defined in `config.py` with `DEFAULT_TOKENS` array
- Dynamic validation against Binance API for active trading pairs
- Configurable via environment variables or direct config modification

## 🏗️ Architecture & Running Services

### 1. WebSocket Streamer (`simple_streamer.py`)
- **Service**: `crypto-streamer.service` (systemd managed)
- **Function**: Real-time Binance WebSocket connection for 1-minute candle data
- **Storage**: Central SQLite database (`market_data.db`)
- **Features**: Auto-reconnect, robust error handling
- **Logs**: `/root/analytics-tool-v2/streamer.log`

### 2. API Server (`simple_api_server.py`)
- **Service**: `crypto-api-server.service` (systemd managed)
- **Framework**: aiohttp (async request handling)
- **Port**: 8080
- **Features**: Binary recommendation serving, system status, latest data
- **Dependencies**: Requires `crypto-streamer.service`
- **Logs**: `/root/analytics-tool-v2/api_server.log`

### 3. Multi-Token Analyzer (`multi_token_analyzer.py`)
- **Version**: enhanced_v2.4_cascade_final
- **Schedule**: Every 15 minutes (cron job)
- **Analysis**: RSI, volume, correlation, market regime detection
- **Output**: `multi_token_analysis.json` with binary recommendations
- **Features**: Async processing, score-based binary signals
- **Logs**: `/root/analytics-tool-v2/analyzer.log`

## 📁 File Structure on Server (`/root/analytics-tool-v2/`)

### Core Application Files
```
simple_streamer.py          # WebSocket streamer script
simple_api_server.py        # API server script  
multi_token_analyzer.py     # Analytics engine (enhanced_v2.4_cascade_final)
config.py                   # Configuration with 30-token DEFAULT_TOKENS list
requirements.txt            # Python dependencies (pandas, numpy, aiohttp, etc.)
show_recommendations.py     # Utility script for displaying analysis results
```

### Data & Logs
```
market_data.db              # SQLite database for candle data
multi_token_analysis.json   # Latest analysis results with binary recommendations
streamer.log               # WebSocket streamer logs
api_server.log             # API server logs
analyzer.log               # Multi-token analyzer logs
```

### Environment & Git
```
venv/                      # Python virtual environment (using virtualenv package)
.gitignore                 # Comprehensive exclusions for data/log files
.git/                      # Git repository (synced with GitHub)
```

## ⚡ Enhanced Features

### Binary Decision Engine
- **Simplified Logic**: Clear BUY_LONG/SHORT signals based on score thresholds
- **No Ambiguity**: Eliminates WAIT_LONG/NEUTRAL categories for automated trading
- **Score-Based**: Uses 60 as threshold for actionable signals
- **Real-Time**: Updated every 15 minutes via cron execution

### Robust Infrastructure  
- **Python Environment**: Uses `virtualenv` package (resolved `python3 -m venv` issues)
- **Git Synchronization**: VPS ↔ Local ↔ GitHub workflow established
- **Dependency Management**: Complete `requirements.txt` with version pins
- **Service Management**: systemd services with auto-restart on failure

### Development Workflow
- **Source of Truth**: VPS server contains live, operational code
- **Git Backup**: GitHub repository serves as version control and backup
- **Sync Process**: Changes made on VPS → committed → pushed to GitHub → synced locally

## 🔧 Quick Commands

### Server Access
```bash
# Connect to production server
ssh -i ~/.ssh/binance_futures_tool root@78.47.150.122
```

### Service Management (on server)
```bash
# Check service status
systemctl status crypto-streamer.service
systemctl status crypto-api-server.service

# Restart services
systemctl restart crypto-streamer.service  
systemctl restart crypto-api-server.service
```

### Log Monitoring (on server)
```bash
# Real-time log viewing
tail -f /root/analytics-tool-v2/streamer.log
tail -f /root/analytics-tool-v2/api_server.log  
tail -f /root/analytics-tool-v2/analyzer.log
```

### API Testing
```bash
# Test endpoints from any machine
curl http://78.47.150.122:8080/api/analysis    # Primary endpoint
curl http://78.47.150.122:8080/api/status      # System status
curl http://78.47.150.122:8080/api/latest      # Latest candle data
```

### Manual Analysis (on server)
```bash
# Run analyzer manually
cd /root/analytics-tool-v2
source venv/bin/activate
python multi_token_analyzer.py

# View recommendations
python show_recommendations.py
```

### Maintenance Tasks (on server)
```bash
# Check cron jobs
crontab -l

# Update Python environment
source venv/bin/activate
pip install -r requirements.txt

# Git status
git status
git log --oneline -5
```

## 📦 GitHub Repository
- **URL**: https://github.com/Global-Digital-Assets/analytics-tool-v2
- **Branch**: main
- **Sync Status**: ✅ Synchronized with VPS server
- **Key Files**: All core scripts, configuration, and documentation

## 🔧 Technical Specifications

### Python Environment
- **Python Version**: 3.8.10
- **Virtual Environment**: `virtualenv` package (not built-in `venv`)
- **Key Dependencies**: pandas~=1.5.0, numpy~=1.23.0, aiohttp==3.10.11, python-binance==1.0.29

### Database Schema
- **Type**: SQLite
- **File**: `market_data.db`
- **Table**: `candles` with columns: symbol, timestamp, open, high, low, close, volume

### API Response Format
```json
{
  "analysis_time": "2025-06-04T19:09:49.384Z",
  "market_context": {
    "regime": "ranging", 
    "confidence": 0.5
  },
  "opportunities": [
    {
      "symbol": "BTCUSDT",
      "long_score": 30,
      "short_score": 75,
      "simple_long_action": "DON'T_BUY_LONG",
      "simple_short_action": "SHORT"
    }
  ]
}
```

## 📊 Current Status: ✅ OPERATIONAL

- **✅ Data Streaming**: Live WebSocket connection active
- **✅ API Services**: All endpoints responding normally  
- **✅ Analysis Engine**: Regular 15-minute analysis execution
- **✅ Git Synchronization**: VPS ↔ GitHub ↔ Local all synchronized
- **✅ Binary Signals**: Clear BUY_LONG/SHORT recommendations active
- **✅ Python Environment**: Virtual environment operational with all dependencies

### Latest Analysis Results
- **Strong Long Signals**: 2 tokens (ADA, SUSHI with scores 70)
- **Strong Short Signals**: 6 tokens (BTC, TRX, NEAR, SHIB, APT, ETC with scores 75)
- **Market Regime**: Ranging market detected
- **System Performance**: 530ms analysis time for 30 tokens

## 🚨 Security Notes
- SSH private key content removed from documentation (stored securely)
- API endpoints currently have no authentication layer
- Database and log files excluded from Git via comprehensive `.gitignore`
- Environment variables should be used for sensitive configuration

---
**Last Updated**: 2025-06-04 23:13 UTC | **Version**: enhanced_v2.4_cascade_final
