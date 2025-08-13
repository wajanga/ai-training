# app.py

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 1. Initialize the Flask App
app = Flask(__name__)

# 2. Load our trained model and scaler
try:
    # Resolve absolute paths relative to this file so it works no matter where it's run from
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")

    model_path = os.path.join(MODELS_DIR, "aggregate_claims_forecaster.pkl")
    scaler_path = os.path.join(MODELS_DIR, "aggregate_data_scaler.pkl")
    data_path = os.path.join(DATA_DIR, "aggregate_annual_claims.csv")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    # Load historical data to get the latest available year for forecasting
    historical_data = pd.read_csv(data_path)
    print("Model, scaler, and historical data loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    model = None
    scaler = None
    historical_data = None


# 3. Define the API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    """
    This function handles prediction requests.
    It expects a POST request with JSON data for the latest year.
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    # Get the JSON data sent to the endpoint
    input_data = request.get_json()

    # --- Data Preparation for a single prediction ---
    # We need to construct the feature vector (the lagged data)
    # For a real app, you might query a database. Here, we'll use our CSV.
    try:
        # The input should tell us which year's data to use as features
        # Ensure year is an integer (handle string input gracefully)
        year_to_use = int(input_data["year"])
        latest_year_data = historical_data[historical_data["Year"] == year_to_use]

        if latest_year_data.empty:
            return jsonify({"error": f"No data found for year {year_to_use}"}), 400

        # Build lagged features that match the training schema (…_lag1)
        base_feature_cols = [
            "Subject employees",
            "Denied claims",
            "Fatality claims",
            "Rate: accepted disabling claims per 100 employees",
        ]

        # Validate required base columns exist
        missing = [c for c in base_feature_cols if c not in latest_year_data.columns]
        if missing:
            return (
                jsonify({"error": f"Missing required columns in data: {missing}"}),
                500,
            )

        # Map base features to lagged feature names using the provided year's values
        lagged_feature_row = {
            f"{col}_lag1": latest_year_data.iloc[0][col] for col in base_feature_cols
        }
        feature_vector = pd.DataFrame([lagged_feature_row])

        # Reorder columns to match the scaler's expected order if available
        if hasattr(scaler, "feature_names_in_"):
            expected_cols = list(scaler.feature_names_in_)
            # Ensure all expected columns are present
            missing_expected = [
                c for c in expected_cols if c not in feature_vector.columns
            ]
            if missing_expected:
                return (
                    jsonify(
                        {
                            "error": f"Prepared features missing expected columns: {missing_expected}"
                        }
                    ),
                    500,
                )
            feature_vector = feature_vector[expected_cols]

        # Scale the features using our pre-fitted scaler
        scaled_feature_vector = scaler.transform(feature_vector)

        # Make a prediction
        prediction = model.predict(scaled_feature_vector)

        # The prediction is a numpy array, so we get the first element
        forecast = prediction[0]

        # --- Create the JSON response ---
        response = {
            "forecast_for_year": year_to_use + 1,
            "predicted_disabling_claims": round(forecast, 2),
        }
        return jsonify(response)

    except Exception as e:
        # Handle potential errors in the input data or processing
        return jsonify({"error": str(e)}), 400


# 4. A simple root endpoint to check if the server is running
@app.route("/")
def home():
    return "WCF Forecasting API is running!"


# 5. Run the app
if __name__ == "__main__":
    # The app will be accessible at http://127.0.0.1:5000
    app.run(port=5000, debug=True)
