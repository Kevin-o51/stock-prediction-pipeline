import pandas as pd

# Load sentiment and stock data
sentiment_df = pd.read_csv('data/news_sentiment.csv')
stock_df = pd.read_csv('data/stock_data.csv')

# Convert `publishedAt` in sentiment data
sentiment_df['publishedAt'] = pd.to_datetime(sentiment_df['publishedAt'], errors='coerce', utc=True).dt.date

# Convert `Date` in stock data, ensuring all are in datetime format
stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce', utc=True).dt.date

# Drop any rows where dates failed to convert
sentiment_df.dropna(subset=['publishedAt'], inplace=True)
stock_df.dropna(subset=['Date'], inplace=True)

# Ensure symbol column exists in both
sentiment_df.rename(columns={'symbol': 'Symbol'}, inplace=True)

# Merge on Symbol and Date
merged_df = pd.merge(sentiment_df, stock_df, left_on=['Symbol', 'publishedAt'], right_on=['Symbol', 'Date'], how='inner')

# Normalize sentiment scores
merged_df['sentiment_score_normalized'] = merged_df['sentiment_score'] / 10

# Save merged data
merged_df.to_csv('data/merged_data.csv', index=False)

print("✅ Feature engineering completed successfully. Merged data saved to data/merged_data.csv")

