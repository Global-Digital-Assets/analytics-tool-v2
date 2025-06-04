# /root/analytics-tool-v2/config.py

# Core Token List
DEFAULT_TOKENS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT",
    "TRXUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "NEARUSDT", "ATOMUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "PEPEUSDT", "SHIBUSDT", "WIFUSDT",
    "TRUMPUSDT", "FETUSDT", "SUSHIUSDT", "APTUSDT", "WLDUSDT", "TIAUSDT", "TAOUSDT",
    "ETCUSDT", "AAVEUSDT"
]

# Technical Analysis Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VOLUME_SPIKE_FACTOR = 2.5
ANALYSIS_CANDLE_LIMIT = 200  # Number of candles to fetch for analysis

# External Services / API
TOKEN_LIST_URL = None  # Or a specific URL if you use one, e.g., "https://api.example.com/tokens"
API_TIMEOUT = 10  # Seconds

# File Paths (set to None to use script-defined defaults, or specify absolute paths)
DB_PATH_OVERRIDE = None
OUTPUT_FILE_OVERRIDE = None
DEBUG_FILE_OVERRIDE = None
LOG_FILE_OVERRIDE = None

# Logging
LOG_LEVEL = "INFO"  # e.g., "DEBUG", "INFO", "WARNING", "ERROR"

