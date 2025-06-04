# Analytics Tool v2

Enhanced crypto analytics tool with simplified binary recommendations for automated trading.

## Features
- Multi-token analysis (30 tokens)
- Simplified binary recommendations: BUY_LONG/DON'T_BUY_LONG, SHORT/DON'T_SHORT
- Real-time data streaming
- JSON API endpoint for bot consumption
- Production deployment ready

## Key Files
- `multi_token_analyzer.py` - Main analytics engine with binary recommendations
- `simple_api_server.py` - API server for serving analysis results
- `simple_streamer.py` - Data collection and streaming
- `config.py` - Configuration with 30-token list

## API Endpoint
`/api/analysis` - Returns JSON with binary trading recommendations

## Binary Logic
- BUY_LONG: Only when score >= 60
- SHORT: Only when score >= 60
- Everything else: DON'T_BUY_LONG / DON'T_SHORT

Deployed on VPS: 78.47.150.122:8080
