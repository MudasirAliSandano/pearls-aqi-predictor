"""
utils.py
--------
Shared helper functions used across the feature pipeline, training
pipeline, and the dashboard app.

Keeping all this logic in ONE file means we never write the same
feature-engineering code twice, so the training data and the
real-time prediction data are always built the exact same way.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

import config


# =========================================================
# DATA FETCHING FUNCTIONS
# =========================================================

def fetch_air_pollution_history(latitude, longitude, start_dt, end_dt):
    """
    Fetch HISTORICAL hourly pollutant data (PM2.5, PM10, CO, NO2, O3, SO2,
    NH3) from the OpenWeather Air Pollution History API.
    `start_dt` and `end_dt` must be timezone-aware or naive UTC datetimes.
    """
    params = {
        "lat": latitude,
        "lon": longitude,
        "start": int(start_dt.timestamp()),
        "end": int(end_dt.timestamp()),
        "appid": config.OPENWEATHER_API_KEY,
    }
    response = requests.get(config.OPENWEATHER_POLLUTION_HISTORY_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_air_quality(latitude, longitude, start_dt=None, end_dt=None, past_days=None):
    """
    Universal wrapper to fetch air quality history supporting explicit 
    start/end datetimes, a `past_days` integer count, or general fallback defaults.
    """
    if past_days is not None and (start_dt is None or end_dt is None):
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=int(past_days))
    elif start_dt is None or end_dt is None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=30)

    return fetch_air_pollution_history(latitude, longitude, start_dt, end_dt)


def fetch_air_pollution_forecast(latitude, longitude):
    """Fetch the upcoming (~4 day) hourly pollutant FORECAST from OpenWeather."""
    params = {"lat": latitude, "lon": longitude, "appid": config.OPENWEATHER_API_KEY}
    response = requests.get(config.OPENWEATHER_POLLUTION_FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_air_pollution_current(latitude, longitude):
    """Fetch the CURRENT hourly pollutant reading from OpenWeather."""
    params = {"lat": latitude, "lon": longitude, "appid": config.OPENWEATHER_API_KEY}
    response = requests.get(config.OPENWEATHER_POLLUTION_CURRENT_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_weather(latitude, longitude, past_days=None, forecast_days=None):
    """
    Fetch hourly weather data (temperature, humidity, wind, pressure)
    from the Open-Meteo Weather Forecast API. Timezone is fixed to UTC
    so timestamps line up exactly with the OpenWeather pollutant data.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "timezone": "UTC",
    }
    if past_days is not None:
        params["past_days"] = past_days
    if forecast_days is not None:
        params["forecast_days"] = forecast_days

    response = requests.get(config.WEATHER_API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# =========================================================
# EPA FORMULA: CONVERT RAW PM2.5 CONCENTRATION -> US AQI
# =========================================================
_EPA_PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),        # Good
    (12.1, 35.4, 51, 100),     # Moderate
    (35.5, 55.4, 101, 150),    # Unhealthy for Sensitive Groups
    (55.5, 150.4, 151, 200),   # Unhealthy
    (150.5, 250.4, 201, 300),  # Very Unhealthy
    (250.5, 350.4, 301, 400),  # Hazardous
    (350.5, 500.4, 401, 500),  # Hazardous
]


def compute_us_aqi_from_pm25(pm25_concentration):
    """Convert a raw PM2.5 concentration (ug/m3) into a standard US AQI value."""
    pm25 = max(0.0, float(pm25_concentration))
    for bp_lo, bp_hi, aqi_lo, aqi_hi in _EPA_PM25_BREAKPOINTS:
        if bp_lo <= pm25 <= bp_hi:
            return round((aqi_hi - aqi_lo) / (bp_hi - bp_lo) * (pm25 - bp_lo) + aqi_lo)
    return 500  # anything above the scale is capped at the maximum


# =========================================================
# RAW JSON -> DATAFRAME
# =========================================================

def air_pollution_json_to_df(pollution_json):
    """
    Convert a raw OpenWeather Air Pollution JSON response (history,
    forecast, or current - they all share the same "list" structure)
    into a clean DataFrame, computing our own US AQI target column.
    """
    records = pollution_json["list"]
    rows = []
    for record in records:
        timestamp = pd.to_datetime(record["dt"], unit="s", utc=True).tz_localize(None)
        components = record["components"]
        pm2_5 = components["pm2_5"]
        rows.append({
            "timestamp": timestamp,
            "aqi": compute_us_aqi_from_pm25(pm2_5),
            "pm2_5": pm2_5,
            "pm10": components["pm10"],
            "carbon_monoxide": components["co"],
            "nitrogen_dioxide": components["no2"],
            "ozone": components["o3"],
            "sulphur_dioxide": components["so2"],
            "ammonia": components["nh3"],
        })
    return pd.DataFrame(rows)


def weather_json_to_df(weather_json):
    """Convert the raw Open-Meteo weather JSON response into a DataFrame."""
    hourly = weather_json["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "pressure": hourly["surface_pressure"],
    })
    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def merge_air_quality_and_weather(aq_df, weather_df):
    """Join air quality and weather data on their shared timestamp column."""
    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def add_time_features(df):
    """Add simple time-based features: hour, day, month, day of week."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def add_derived_features(df):
    """Add derived features that describe how AQI has been changing over time."""
    df = df.copy()
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_2"] = df["aqi"].shift(2)
    df["aqi_change_rate"] = df["aqi_lag_1"] - df["aqi_lag_2"]
    return df


def build_feature_dataframe(aq_json, weather_json):
    """Full feature engineering pipeline."""
    aq_df = air_pollution_json_to_df(aq_json)
    weather_df = weather_json_to_df(weather_json)

    merged = merge_air_quality_and_weather(aq_df, weather_df)
    merged = add_time_features(merged)
    merged = add_derived_features(merged)

    return merged


FEATURE_COLUMNS = [
    "temperature", "humidity", "wind_speed", "pressure",
    "hour", "day", "month", "day_of_week",
    "aqi_lag_1", "aqi_lag_2", "aqi_change_rate",
]


# =========================================================
# AQI CATEGORY HELPER
# =========================================================

def categorize_aqi(aqi_value):
    """Convert a numeric US AQI value into a human-readable category."""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# =========================================================
# LOCAL STORAGE HELPERS
# =========================================================

def save_features_locally(df):
    """Append new feature rows to a local CSV file."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if os.path.exists(config.LOCAL_FEATURES_FILE):
        existing = pd.read_csv(config.LOCAL_FEATURES_FILE, parse_dates=["timestamp"])
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp", keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = df

    combined.to_csv(config.LOCAL_FEATURES_FILE, index=False)
    return combined


def load_features_locally():
    """Load all stored features from the local CSV fallback file."""
    if not os.path.exists(config.LOCAL_FEATURES_FILE):
        raise FileNotFoundError(
            f"No local feature file found at {config.LOCAL_FEATURES_FILE}. "
            "Run feature_pipeline.py or backfill_historical.py first."
        )
    return pd.read_csv(config.LOCAL_FEATURES_FILE, parse_dates=["timestamp"])


# =========================================================
# MODEL REGISTRY HELPERS
# =========================================================

import json
import joblib


def save_model_locally(model, scaler, metadata):
    """Save the trained model, its scaler, and metadata as local files."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(model, config.LOCAL_MODEL_FILE)
    joblib.dump(scaler, config.LOCAL_SCALER_FILE)
    with open(config.LOCAL_MODEL_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model to '{config.LOCAL_MODEL_FILE}' "
          f"and metadata to '{config.LOCAL_MODEL_META_FILE}'.")


def load_model_locally():
    """Load the most recently trained local model, scaler, and metadata."""
    if not os.path.exists(config.LOCAL_MODEL_FILE):
        raise FileNotFoundError(
            f"No trained model found at {config.LOCAL_MODEL_FILE}. "
            "Run training_pipeline.py first."
        )
    model = joblib.load(config.LOCAL_MODEL_FILE)
    scaler = joblib.load(config.LOCAL_SCALER_FILE)
    with open(config.LOCAL_MODEL_META_FILE) as f:
        metadata = json.load(f)
    return model, scaler, metadata