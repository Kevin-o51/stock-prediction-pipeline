import os

os.makedirs("/Users/aghakeivan/Documents/Stocks/data/sp500_daily", exist_ok=True)
print("Directory check passed: /Users/aghakeivan/Documents/Stocks/data/sp500_daily")

tickers_path = "/Users/aghakeivan/Documents/Stocks/sp500_tickers.csv"
with open(tickers_path, "r") as file:
    tickers = [line.strip() for line in file]
print(f"Tickers loaded: {tickers[:5]} (total: {len(tickers)})")
