"""
app.py
------
STEP 4 of the project: the Web Application Dashboard.

What this app does:
1. Loads the trained model (and scaler) from the Model Registry.
2. Fetches the upcoming weather forecast for the next few days.
3. Uses the model to predict AQI hour-by-hour for the forecast window,
   feeding each prediction back in as the "lag" feature for the next
   hour (a recursive / multi-step forecasting strategy).
4. Displays everything on an interactive Streamlit dashboard, with a
   clear alert whenever hazardous AQI levels are expected.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import config
import utils
from utils import FEATURE_COLUMNS

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")


# =========================================================
# LOADING (cached so the app stays fast on every interaction)
# =========================================================

@st.cache_resource
def load_model_and_metadata():
    if config.USE_HOPSWORKS:
        import hopsworks
        project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME,
        )
        mr = project.get_model_registry()
        hops_model = mr.get_model(config.MODEL_NAME)
        model_dir = hops_model.download()
        import joblib, json, os
        model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        with open(os.path.join(model_dir, "model_meta.json")) as f:
            metadata = json.load(f)
        return model, scaler, metadata
    else:
        return utils.load_model_locally()


@st.cache_data(ttl=600)  # refresh at most every 10 minutes
def load_recent_history():
    if config.USE_HOPSWORKS:
        import hopsworks
        project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME,
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        df = fg.read()
    else:
        df = utils.load_features_locally()
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=600)
def load_weather_forecast():
    weather_json = utils.fetch_weather(
        config.LATITUDE, config.LONGITUDE, forecast_days=config.FORECAST_DAYS
    )
    hourly = weather_json["hourly"]
    forecast_df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "pressure": hourly["surface_pressure"],
    })
    # Only keep timestamps from now onward
    now = pd.Timestamp.now()
    return forecast_df[forecast_df["timestamp"] >= now].reset_index(drop=True)


# =========================================================
# RECURSIVE MULTI-STEP FORECASTING
# =========================================================

def forecast_future_aqi(model, scaler, history_df, forecast_weather_df):
    """
    Predict AQI for every hour in `forecast_weather_df`.

    Since our model needs aqi_lag_1 / aqi_lag_2 / aqi_change_rate as
    inputs, but we obviously don't have real future AQI values, we
    predict one hour at a time and feed each prediction back in as
    the "known" lag value for the following hour. This is called
    recursive forecasting.
    """
    last_known_aqi_1 = history_df["aqi"].iloc[-1]
    last_known_aqi_2 = history_df["aqi"].iloc[-2]

    predictions = []
    for _, row in forecast_weather_df.iterrows():
        change_rate = last_known_aqi_1 - last_known_aqi_2

        feature_row = pd.DataFrame([{
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "wind_speed": row["wind_speed"],
            "pressure": row["pressure"],
            "hour": row["timestamp"].hour,
            "day": row["timestamp"].day,
            "month": row["timestamp"].month,
            "day_of_week": row["timestamp"].dayofweek,
            "aqi_lag_1": last_known_aqi_1,
            "aqi_lag_2": last_known_aqi_2,
            "aqi_change_rate": change_rate,
        }])[FEATURE_COLUMNS]

        X_scaled = scaler.transform(feature_row.values)
        predicted_aqi = float(np.ravel(model.predict(X_scaled))[0])
        predicted_aqi = max(0, predicted_aqi)  # AQI cannot be negative

        predictions.append({"timestamp": row["timestamp"], "predicted_aqi": predicted_aqi})

        # Shift the lag window forward for the next iteration
        last_known_aqi_2 = last_known_aqi_1
        last_known_aqi_1 = predicted_aqi

    return pd.DataFrame(predictions)


# =========================================================
# DASHBOARD LAYOUT
# =========================================================

st.title("🌫️ Pearls AQI Predictor")
st.caption(f"3-day Air Quality Index forecast for {config.CITY_NAME}")

try:
    model, scaler, metadata = load_model_and_metadata()
    history_df = load_recent_history()
    forecast_weather_df = load_weather_forecast()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `backfill_historical.py` then `training_pipeline.py` first.")
    st.stop()

forecast_df = forecast_future_aqi(model, scaler, history_df, forecast_weather_df)
forecast_df["category"] = forecast_df["predicted_aqi"].apply(utils.categorize_aqi)

# ---- Top row: current status + model info ----
col1, col2, col3 = st.columns(3)
current_aqi = history_df["aqi"].iloc[-1]
col1.metric("Current AQI (latest known)", f"{current_aqi:.0f}", utils.categorize_aqi(current_aqi))
col2.metric("Model in use", metadata["model_type"].replace("_", " ").title())
col3.metric("Model RMSE (test set)", f"{metadata['metrics']['rmse']:.2f}")

# ---- Hazard alert ----
hazardous_hours = forecast_df[forecast_df["predicted_aqi"] >= config.HAZARDOUS_AQI_THRESHOLD]
if len(hazardous_hours) > 0:
    first_time = hazardous_hours["timestamp"].iloc[0]
    st.error(
        f"⚠️ ALERT: Hazardous AQI levels (≥{config.HAZARDOUS_AQI_THRESHOLD}) are "
        f"predicted starting {first_time.strftime('%A, %b %d at %I:%M %p')}. "
        "Consider limiting outdoor activity."
    )
else:
    st.success("✅ No hazardous AQI levels predicted in the next few days.")

# ---- Forecast chart ----
st.subheader("Hourly AQI Forecast")
chart_df = forecast_df.set_index("timestamp")[["predicted_aqi"]]
st.line_chart(chart_df)

# ---- Forecast table ----
with st.expander("See detailed hourly forecast table"):
    display_df = forecast_df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["predicted_aqi"] = display_df["predicted_aqi"].round(1)
    st.dataframe(display_df, use_container_width=True)

# ---- Recent historical trend ----
st.subheader("Recent Historical AQI Trend")
recent_history = history_df.tail(24 * 7).set_index("timestamp")[["aqi"]]
st.line_chart(recent_history)

st.caption(
    "Built for the Pearls AQI Predictor project | Data source: Open-Meteo | "
    f"Feature store: {'Hopsworks' if config.USE_HOPSWORKS else 'Local (fallback)'}"
)
