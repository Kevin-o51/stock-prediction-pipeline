import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/Users/aghakeivan/Documents/Stocks/logs/model_training.log')
    ]
)

def main():
    try:
        # Load the dataset
        data_path = '/Users/aghakeivan/Documents/Stocks/data/merged_data.csv'
        logging.info(f"Loading dataset from {data_path}...")
        data = pd.read_csv(data_path)

        # Display columns
        logging.info(f"Columns in the dataset: {data.columns.tolist()}")

        # Identify the target column
        target_column = 'Close'  # Change this if you want to predict a different column

        # Ensure the 'Close' column exists
        if target_column not in data.columns:
            logging.error(f"Error: Target column '{target_column}' not found in the dataset.")
            sys.exit(1)

        # Preprocessing
        logging.info("Starting preprocessing...")

        # Convert 'Date' to datetime and then to timestamp
        if 'Date' in data.columns:
            logging.info("Converting 'Date' column to timestamp...")
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['Date'] = data['Date'].astype('int64') // 10**9  # Convert to Unix timestamp
        else:
            logging.warning("'Date' column not found in the dataset.")

        # Drop non-numeric and irrelevant columns
        columns_to_drop = ['Symbol', 'Unnamed: 0', 'publishedAt', 'title', 'source']
        existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
        if existing_columns_to_drop:
            logging.info(f"Dropping columns: {existing_columns_to_drop}")
            data = data.drop(columns=existing_columns_to_drop)
        else:
            logging.warning("No irrelevant columns to drop.")

        # Handle missing values if any
        if data.isnull().values.any():
            logging.info("Handling missing values by dropping rows with any missing values...")
            data = data.dropna()
            logging.info(f"Dataset shape after dropping missing values: {data.shape}")

        # Features and target variable
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        logging.info(f"Features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")

        # Ensure all features are numeric
        logging.info("Ensuring all features are numeric...")
        non_numeric_columns = X.select_dtypes(include=['object', 'datetime']).columns.tolist()
        if non_numeric_columns:
            logging.error(f"Non-numeric columns found: {non_numeric_columns}. Please convert or drop them.")
            sys.exit(1)
        else:
            logging.info("All features are numeric.")

        # Split the data into train and test sets
        logging.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize RandomForestRegressor
        model = RandomForestRegressor()

        # Perform Grid Search for best hyperparameters
        logging.info("Starting Grid Search for hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50],  # Start with fewer estimators for quicker testing
            'max_depth': [None],
            'min_samples_split': [2]
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            verbose=2,
            n_jobs=-1,
            error_score='raise'  # To raise errors and stop GridSearchCV on failures
        )

        grid_search.fit(X_train, y_train)
        logging.info(f"✅ Best Hyperparameters: {grid_search.best_params_}")

        # Train the optimized model
        logging.info("Training the optimized model with best hyperparameters...")
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Calculate MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f"✅ Optimized Model Trained | MAE: {mae}")

        # Save the optimized model
        model_save_path = '/Users/aghakeivan/Documents/Stocks/models/stock_model.pkl'
        logging.info(f"Saving the optimized model to {model_save_path}...")
        joblib.dump(best_model, model_save_path)
        logging.info("✅ Optimized model saved successfully.")

        # Save predictions to CSV
        predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        predictions_save_path = '/Users/aghakeivan/Documents/Stocks/data/predictions.csv'
        logging.info(f"Saving predictions to {predictions_save_path}...")
        predictions_df.to_csv(predictions_save_path, index=False)
        logging.info("✅ Predictions saved successfully.")

    except Exception as e:
        logging.exception("An error occurred during model training.")
        sys.exit(1)

if __name__ == "__main__":
    main()

