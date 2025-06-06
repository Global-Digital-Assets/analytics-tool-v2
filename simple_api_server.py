#!/usr/bin/env python3
"""
Simple Analytics API Server
Serves the latest market data collected by the streamer and analysis results.
"""
import sqlite3
import json
from datetime import datetime
from aiohttp import web
import asyncio
import logging
import os # Added import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalyticsAPI:
    def __init__(self):
        self.db_path = '/root/analytics-tool-v2/market_data.db'
        # Define path for the analysis JSON file
        self.analysis_file_path = '/root/analytics-tool-v2/multi_token_analysis.json'
        
    def get_latest_data(self):
        """Get latest candle data for all symbols"""
        try:
            conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
            cursor = conn.cursor()
            
            # Get latest timestamp for each symbol
            cursor.execute('''
                SELECT symbol, timestamp, open, high, low, close, volume
                FROM candles c1
                WHERE timestamp = (
                    SELECT MAX(timestamp) 
                    FROM candles c2 
                    WHERE c2.symbol = c1.symbol
                )
                ORDER BY symbol
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            data = {}
            for row in results:
                symbol, timestamp, open_price, high, low, close, volume = row
                data[symbol] = {
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                }
                
            return {
                'count': len(data),
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return {'error': str(e)}

    async def handle_latest(self, request):
        """Handle /api/latest endpoint"""
        try:
            data = self.get_latest_data()
            response = web.json_response(data)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        except Exception as e:
            logger.error(f"Error in handle_latest: {e}")
            return web.json_response({'error': str(e)}, status=500)
        
    async def handle_status(self, request):
        """Handle /api/status endpoint"""
        try:
            data = self.get_latest_data() # Keep this to check DB connection for status
            response_data = {
                'status': 'operational',
                'timestamp': datetime.now().isoformat(),
                'tokens': {
                    'active': data.get('count', 0), # Use count from DB check
                    'total': 31 # This might need to be dynamic from config later
                }
            }
            response = web.json_response(response_data)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        except Exception as e:
            logger.error(f"Error in handle_status: {e}")
            return web.json_response({'error': str(e)}, status=500)

    async def handle_analysis(self, request): # New handler
        """Handle /api/analysis endpoint"""
        try:
            if not os.path.exists(self.analysis_file_path):
                logger.warning(f"Analysis file not found: {self.analysis_file_path}")
                return web.json_response({'error': 'Analysis data not yet available. Please try again later.'}, status=404)

            with open(self.analysis_file_path, 'r') as f:
                analysis_data = json.load(f)
            
            response = web.json_response(analysis_data)
            response.headers['Access-Control-Allow-Origin'] = '*'
            return response
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding analysis JSON: {e}")
            return web.json_response({'error': 'Invalid analysis data format.'}, status=500)
        except Exception as e:
            logger.error(f"Error in handle_analysis: {e}")
            return web.json_response({'error': str(e)}, status=500)

async def create_app():
    """Create the web application"""
    api = AnalyticsAPI()
    app = web.Application()
    
    # Add routes
    app.router.add_get('/api/latest', api.handle_latest)
    app.router.add_get('/api/status', api.handle_status)
    app.router.add_get('/healthz', api.handle_status)
    app.router.add_get('/api/analysis', api.handle_analysis) # Added new route
    
    return app

async def main():
    """Start the API server"""
    app = await create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    
    logger.info("Analytics API server started on port 8080")
    logger.info("Endpoints:")
    logger.info("  GET /api/latest - Latest market data")
    logger.info("  GET /api/status - Service status")
    logger.info("  GET /healthz   - Health probe (alias)")
    logger.info("  GET /api/analysis - Token analysis results") # Added log for new endpoint
    
    # Keep server running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
