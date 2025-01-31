#!/bin/bash

echo "🚀 Starting Stock Prediction Pipeline..."

# Activate Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stock_analysis_env

# Navigate to scripts directory
cd /Users/aghakeivan/Documents/Stocks/scripts/

# Run data collection
echo "📥 Collecting stock data..."
python data_collection.py

# Run sentiment analysis
echo "📰 Running sentiment analysis..."
python sentiment_analysis.py

# Run feature engineering
#!/bin/bash

echo "🚀 Starting Stock Prediction Pipeline..."

# Activate Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stock_analysis_env

# Navigate to scripts directory
cd /Users/aghakeivan/Documents/Stocks/scripts/

# Run data collection
echo "📥 Collecting stock data..."
python data_collection.py

# Run sentiment analysis
echo "📰 Running sentiment analysis..."
python sentiment_analysis.py

# Run feature engineering
echo "🔄 Processing feature engineering..."
python feature_engineering.py

# Run model training
echo "🤖 Training stock prediction model..."
python model_training.py

# Confirm completion
echo "✅ Stock Prediction Pipeline Completed!"


