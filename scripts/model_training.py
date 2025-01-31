import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Configure logging
logging.basicConfig(
    filename="logs/model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("🚀 Starting Model Training Pipeline...")

# ✅ Step 1: Load Dataset (Relative Path)
DATA_PATH = "data/merged_data.csv"
MODEL_PATH = "models/stock_model.pkl"
PREDICTIONS_PATH = "data/predictions.csv"

try:
    logging.info(f"📂 Loading dataset from {DATA_PATH}...")
    data = pd.read_csv(DATA_PATH)
    logging.info(f"✅ Dataset loaded successfully. Columns: {list(data.columns)}")
except FileNotFoundError:
    logging.error(f"❌ Error: File not found at {DATA_PATH}")
    raise SystemExit("Dataset file not found. Exiting.")

# ✅ Step 2: Preprocess Data
logging.info("🔄 Starting preprocessing...")

# Convert Date column to timestamp (if exists)
if "Date" in data.columns:
    logging.info("📅 Converting 'Date' column to timestamp...")
    data["Date"] = pd.to_datetime(data["Date"]).astype(np.int64) // 10**9

# Drop unnecessary columns
drop_columns = ["Symbol", "Unnamed: 0", "publishedAt", "title", "source"]
data = data.drop(columns=[col for col in drop_columns if col in data.columns], errors="ignore")

# Ensure all features are numeric
logging.info("🔢 Ensuring all features are numeric...")
data = data.apply(pd.to_numeric, errors="coerce")
data = data.fillna(0)  # Replace NaN values with 0
logging.info(f"✅ Data shape after preprocessing: {data.shape}")

# Define features (X) and target (y)
if "Close" in data.columns:
    X = data.drop(columns=["Close"])
    y = data["Close"]
else:
    logging.error("❌ Target column 'Close' not found in dataset.")
    raise SystemExit("Target column 'Close' missing. Exiting.")

# ✅ Step 3: Split Data into Train and Test Sets
logging.info("📊 Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"✅ Training Set: {X_train.shape}, Test Set: {X_test.shape}")

# ✅ Step 4: Model Training & Hyperparameter Tuning
logging.info("⚙️ Starting Grid Search for best hyperparameters...")

param_grid = {"n_estimators": [50], "max_depth": [None], "min_samples_split": [2]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_absolute_error")

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

logging.info(f"✅ Best Hyperparameters: {grid_search.best_params_}")
logging.info("🤖 Training optimized model...")

# ✅ Step 5: Model Evaluation
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
logging.info(f"✅ Model Trained Successfully | MAE: {mae:.4f}")

# ✅ Step 6: Save the Trained Model
logging.info(f"💾 Saving optimized model to {MODEL_PATH}...")
joblib.dump(best_model, MODEL_PATH)
logging.info("✅ Model saved successfully.")

# ✅ Step 7: Save Predictions
predictions_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
predictions_df.to_csv(PREDICTIONS_PATH, index=False)
logging.info(f"✅ Predictions saved to {PREDICTIONS_PATH}")

logging.info("🎉 Model Training Pipeline Completed Successfully!")

