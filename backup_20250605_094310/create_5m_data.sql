-- Create optimized 5-minute OHLCV data (5x speedup)
DROP TABLE IF EXISTS candles_5m;

CREATE TABLE candles_5m AS
SELECT
    symbol,
    (timestamp / 300) * 300 AS ts5m,
    MIN(low) AS low,
    MAX(high) AS high,
    (SELECT open FROM candles c2 
     WHERE c2.symbol = c1.symbol 
     AND (c2.timestamp / 300) * 300 = (c1.timestamp / 300) * 300
     ORDER BY c2.timestamp ASC 
     LIMIT 1) AS open,
    (SELECT close FROM candles c3 
     WHERE c3.symbol = c1.symbol 
     AND (c3.timestamp / 300) * 300 = (c1.timestamp / 300) * 300
     ORDER BY c3.timestamp DESC 
     LIMIT 1) AS close,
    SUM(volume) AS volume,
    COUNT(*) AS samples_count
FROM candles c1
GROUP BY symbol, (timestamp / 300) * 300
ORDER BY symbol, ts5m;

-- Create index for fast lookups
CREATE INDEX idx_candles_5m_symbol_time ON candles_5m(symbol, ts5m);

-- Verify results
SELECT 
    'Original 1m data' AS dataset,
    COUNT(*) AS row_count,
    COUNT(DISTINCT symbol) AS symbols
FROM candles
UNION ALL
SELECT 
    '5m downsampled' AS dataset,
    COUNT(*) AS row_count,
    COUNT(DISTINCT symbol) AS symbols  
FROM candles_5m;
