import json

with open("multi_token_analysis.json", "r") as f:
    data = json.load(f)

print("\n=== LATEST TOKEN ANALYSIS REPORT ===")
analysis_time = data.get("analysis_time", "N/A")
print(f"Analysis Time: {analysis_time} UTC")
market_context = data.get("market_context", {})
regime = market_context.get("regime", "N/A")
confidence = market_context.get("confidence", "N/A")
print(f"Market Regime: {regime} (confidence: {confidence})")
opportunities = data.get("opportunities", [])
print(f"Tokens Analyzed: {len(opportunities)}")
print()

# Extract tokens with scores >= 60 for either long or short
buy_long_tokens = []
short_tokens = []

for token in opportunities:
    symbol = token.get("symbol", "")
    long_score = token.get("long_score", 0)
    short_score = token.get("short_score", 0)
    simple_long = token.get("simple_long_action", "")
    simple_short = token.get("simple_short_action", "")
    
    if simple_long == "BUY_LONG":
        buy_long_tokens.append((symbol, long_score))
    if simple_short == "SHORT":
        short_tokens.append((symbol, short_score))

# Sort by score (highest first)
buy_long_tokens.sort(key=lambda x: x[1], reverse=True)
short_tokens.sort(key=lambda x: x[1], reverse=True)

print("📈 BUY_LONG RECOMMENDATIONS (Score >= 60):")
if buy_long_tokens:
    for symbol, score in buy_long_tokens:
        print(f"  • {symbol}: {score}")
else:
    print("  None")

print()
print("📉 SHORT RECOMMENDATIONS (Score >= 60):")
if short_tokens:
    for symbol, score in short_tokens:
        print(f"  • {symbol}: {score}")
else:
    print("  None")

print()
print("🔍 ALL TOKENS SUMMARY:")
for token in opportunities:
    symbol = token.get("symbol", "")
    long_score = token.get("long_score", 0)
    short_score = token.get("short_score", 0)
    simple_long = token.get("simple_long_action", "")
    simple_short = token.get("simple_short_action", "")
    
    print(f"{symbol:10s} | Long: {long_score:5.1f} ({simple_long:13s}) | Short: {short_score:5.1f} ({simple_short:11s})")
