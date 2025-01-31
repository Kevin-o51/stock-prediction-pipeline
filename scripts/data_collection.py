import yfinance as yf
import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Define stock symbols to track
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Fetch stock data
def fetch_stock_data(symbols):
    all_data = []
    for symbol in symbols:
        print(f"🔹 Fetching stock data for {symbol}...")
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")  # Get 1-year historical data
        hist["Symbol"] = symbol
        all_data.append(hist)
    return pd.concat(all_data)

# Fetch financial news
def fetch_news():
    url = f"https://newsapi.org/v2/everything?q=stock+market&apiKey={NEWS_API_KEY}"
    print(f"🔹 Fetching financial news from: {url}")
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"🚨 ERROR: News API request failed with status code {response.status_code}")
        print(response.json())  # Print response for debugging
        return pd.DataFrame(columns=["publishedAt", "title", "source"])
    
    data = response.json()
    
    if "articles" in data:
        articles = pd.DataFrame(data["articles"])[["publishedAt", "title", "source"]]
        print(f"✅ Retrieved {len(articles)} news articles.")
        return articles

    print("🚨 ERROR: No articles found in API response!")
    return pd.DataFrame(columns=["publishedAt", "title", "source"])

# Run data collection
if __name__ == "__main__":
    # Fetch and save stock data
    stock_data = fetch_stock_data(STOCK_SYMBOLS)
    os.makedirs("data", exist_ok=True)
    stock_data.to_csv("data/stock_data.csv")
    print("✅ Stock data saved to data/stock_data.csv")

    # Fetch and save news data
    news_data = fetch_news()
    news_data.to_csv("data/news_data.csv")
    print("✅ News data saved to data/news_data.csv")

