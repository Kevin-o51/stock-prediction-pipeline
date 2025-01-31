from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/stock_model.pkl")

# Define features
FEATURES = ["Open", "High", "Low", "Volume", "RSI"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        # Ensure all required features are present
        for feature in FEATURES:
            if feature not in input_df.columns:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Predict
        prediction = model.predict(input_df[FEATURES])
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)

