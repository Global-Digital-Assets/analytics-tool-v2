# /root/analytics-tool-v2/config.py

# Core Token List
DEFAULT_TOKENS = [
    "AAVEUSDT", "ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT",
    "EIGENUSDT", "ENAUSDT", "ETCUSDT", "ETHUSDT", "FETUSDT", "FILUSDT", "INJUSDT", "LDOUSDT", "LINKUSDT", "LPTUSDT",
    "LTCUSDT", "MASKUSDT", "NEARUSDT", "OPUSDT", "PEPEUSDT", "RVNUSDT", "SHIBUSDT", "SOLUSDT", "SUIUSDT", "SUSHIUSDT",
    "TAOUSDT", "TIAUSDT", "TRBUSDT", "TRUMPUSDT", "TRXUSDT", "UNIUSDT", "VIRTUALUSDT", "WIFUSDT", "WLDUSDT", "XRPUSDT",
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

# --- Advanced ML / Institutional Parameters (optional) ---
# Leave defaults if you are not using the ML scoring engine
MEME_TOKENS = []  # Additional meme-coin universe (optional)

# Paths for the scoring model / feature scaler – used by continuous learner & analyzer
ML_MODEL_PATH = None  # e.g. "models/latest_model.pkl"
SCALER_PATH = None  # e.g. "models/latest_scaler.pkl"

# Correlation & regime parameters
CORRELATION_WINDOW = 90  # Rolling window (candles) for BTC/ETH correlation assessment
CORRELATION_PENALTY_FACTOR = 0.3  # Weight to reduce score when highly correlated
REGIME_MULTIPLIER_BEAR = 0.8  # Global multiplier in bear regimes
REGIME_MULTIPLIER_BULL = 0.8  # Global multiplier in bull regimes

# Scoring & training behaviour
PERCENTILE_RANKING = True  # Use cross-sectional percentile instead of absolute score
DATA_DRIVEN_SCORING = False  # Flip to True if ML model is available
MIN_TRAINING_SAMPLES = 1000  # Guardrail for continuous learner
