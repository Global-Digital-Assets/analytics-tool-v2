#!/usr/bin/env python3
"""
External Health Check Service - Runs on port 8081
"""
from flask import Flask, jsonify
import sqlite3
import os
import time
from datetime import datetime

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connectivity
        conn = sqlite3.connect("market_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM candles WHERE timestamp > (strftime(\"%s\", \"now\") - 300) * 1000")
        recent_candles = cursor.fetchone()[0]
        conn.close()
        
        # Check analysis file freshness
        analysis_age = time.time() - os.path.getmtime("multi_token_analysis.json") if os.path.exists("multi_token_analysis.json") else 9999
        
        status = {
            "status": "healthy" if recent_candles > 0 and analysis_age < 1800 else "degraded",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": {
                "database": "ok" if recent_candles > 0 else "no_recent_data",
                "recent_candles_5min": recent_candles,
                "analysis_freshness_seconds": int(analysis_age),
                "analysis_status": "fresh" if analysis_age < 1800 else "stale"
            }
        }
        
        return jsonify(status), 200 if status["status"] == "healthy" else 503
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)
