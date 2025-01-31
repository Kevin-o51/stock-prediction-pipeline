import requests
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(
    filename="/Users/aghakeivan/Documents/Stocks/logs/sentiment_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# API Key and URL
API_KEY = "d3fe3df10e674496b633d735f24a8548"
API_URL = f"https://newsapi.org/v2/everything?q=stock+market&apiKey={API_KEY}"

def fetch_news():
    """Fetch financial news with error handling and timeout."""
    try:
        logging.info("Fetching news from API...")
        response = requests.get(API_URL, timeout=10)  # 10-second timeout
        response.raise_for_status()  # Raise exception for 4xx/5xx responses
        data = response.json()
        logging.info(f"✅ Successfully fetched {len(data.get('articles', []))} articles.")
        return data.get("articles", [])
    
    except requests.exceptions.Timeout:
        logging.error("⏳ API request timed out.")
        return []
    
    except requests.exceptions.RequestException as e:
        logging.error(f"⚠️ API request failed: {e}")
        return []

def analyze_sentiment(articles):
    """Analyze sentiment (dummy example, replace with actual sentiment analysis)."""
    logging.info("Starting sentiment analysis...")
    results = []

    for i, article in enumerate(articles):  # Process ALL articles
  # TEMP: Process only first 5 articles
        title = article.get("title", "No Title")
        content = article.get("content", "No Content")

        # Simulated sentiment score (-1 to 1)
        sentiment_score = len(title) % 3 - 1  # Dummy logic (replace with real model)

        results.append({"title": title, "sentiment_score": sentiment_score})
        logging.info(f"🔹 Processed {i+1}/{len(articles)} articles - {title}")

        # TEMP: Disable sleep to speed up processing
        # time.sleep(1)

    return results

def save_results(results):
    """Save results to CSV."""
    if results:
        df = pd.DataFrame(results)
        df.to_csv("/Users/aghakeivan/Documents/Stocks/data/sentiment_analysis.csv", index=False)
        logging.info("✅ Sentiment analysis results saved successfully.")
    else:
        logging.warning("⚠️ No results to save.")

def main():
    """Main function to run sentiment analysis pipeline."""
    logging.info("🚀 Starting sentiment analysis pipeline...")
    
    articles = fetch_news()
    if articles:
        results = analyze_sentiment(articles)
        save_results(results)
    else:
        logging.warning("⚠️ No articles fetched. Exiting.")

    logging.info("✅ Sentiment analysis pipeline completed.")

if __name__ == "__main__":
    main()

