# ui.py

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
# This should be the first Streamlit command in your script.
st.set_page_config(
    page_title="WCF Claims Forecaster",
    page_icon="🇹🇿",  # Tanzania flag emoji
    layout="centered",
)


# --- LOAD MODEL AND DATA ---
# Use st.cache_data to load the model and data only once
@st.cache_data
def load_resources():
    """Loads the ML model, scaler, and historical data."""
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
        historical_data = pd.read_csv(data_path)
        return model, scaler, historical_data
    except FileNotFoundError:
        st.error(
            "Model or data files not found. Make sure the 'models' and 'data' directories are in the same folder as this script."
        )
        return None, None, None


model, scaler, historical_data = load_resources()


# --- WEB PAGE LAYOUT ---
st.title("🇹🇿 WCF Annual Claims Forecaster")
st.markdown(
    "An AI-powered tool to forecast the number of accepted disabling claims for the upcoming year, built for the Workers Compensation Fund of Tanzania."
)

if model is not None:
    # --- USER INPUT ---
    st.sidebar.header("Forecasting Options")

    # Get the list of available years from our data
    available_years = historical_data["Year"].unique()

    # Create a dropdown for the user to select the year to use for prediction
    # Default to the most recent year available
    year_to_use = st.sidebar.selectbox(
        "Select the last full year of data to use for the forecast:",
        options=available_years,
        index=len(available_years) - 1,  # Default to the last year
    )

    # --- MODEL PREDICTION ---
    if st.sidebar.button("Generate Forecast"):

        # 1. Get the data for the selected year
        latest_year_data = historical_data[historical_data["Year"] == year_to_use]

        if latest_year_data.empty:
            st.error(
                f"No data available for the year {year_to_use}. Please select another year."
            )
        else:
            # 2. Create the lagged feature vector expected by the scaler/model
            base_feature_cols = [
                "Subject employees",
                "Denied claims",
                "Fatality claims",
                "Rate: accepted disabling claims per 100 employees",
            ]

            # Validate required columns exist in the dataset
            missing = [
                c for c in base_feature_cols if c not in latest_year_data.columns
            ]
            if missing:
                st.error(f"Missing required columns in data: {missing}")
                st.stop()

            # Construct single-row DataFrame with lagged column names using selected year's values
            lagged_feature_row = {
                f"{col}_lag1": latest_year_data.iloc[0][col]
                for col in base_feature_cols
            }
            feature_vector = pd.DataFrame([lagged_feature_row])

            # Reorder to match scaler's expected order if available
            if hasattr(scaler, "feature_names_in_"):
                expected_cols = list(scaler.feature_names_in_)
                missing_expected = [
                    c for c in expected_cols if c not in feature_vector.columns
                ]
                if missing_expected:
                    st.error(
                        f"Prepared features missing expected columns: {missing_expected}"
                    )
                    st.stop()
                feature_vector = feature_vector[expected_cols]

            # 3. Scale the features
            scaled_feature_vector = scaler.transform(feature_vector)

            # 4. Make a prediction
            prediction = model.predict(scaled_feature_vector)
            forecast = prediction[0]

            # --- DISPLAY RESULTS ---
            st.header(f"📈 Forecast for {year_to_use + 1}")

            col1, col2 = st.columns(2)

            # Display the forecast as a large metric
            col1.metric(
                label="Predicted Disabling Claims",
                value=f"{int(forecast):,}",  # Format with commas
                help="This is the model's prediction based on data from the previous year.",
            )

            # Show the previous year's actuals for comparison
            previous_year_actual = latest_year_data["Accepted disabling claims"].iloc[0]
            col2.metric(
                label=f"Actual Claims in {year_to_use}",
                value=f"{int(previous_year_actual):,}",
                delta=f"{int(forecast - previous_year_actual):,}",
                delta_color="normal",
                help="This is the change from the previous year's actual number of claims.",
            )

    # --- VISUALIZATION ---
    st.header("Historical Data Trends")

    # Use Plotly for a nice interactive chart
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(
        go.Scatter(
            x=historical_data["Year"],
            y=historical_data["Accepted disabling claims"],
            mode="lines+markers",
            name="Actual Claims",
            line=dict(color="royalblue"),
        )
    )

    fig.update_layout(
        title="Accepted Disabling Claims (1968-2023)",
        xaxis_title="Year",
        yaxis_title="Number of Claims",
        legend_title="Legend",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Could not load the forecasting model. The application cannot proceed.")
