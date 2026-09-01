"""
config.py
---------
Central configuration file for the Pearls AQI Predictor project.
Change the values below to match your city and your own account setup.

Nothing else in the project should contain hardcoded settings -
everything reads from this file so it stays easy to maintain.
"""

import os

# =========================================================
# 1. LOCATION SETTINGS
# =========================================================
# Default location: Sukkur, Pakistan. Change latitude/longitude
# to your own city if needed (you can find these on Google Maps).
CITY_NAME = "Sukkur"
LATITUDE = 27.7052
LONGITUDE = 68.8574

# =========================================================
# 2. DATA SOURCE SETTINGS
# =========================================================
# POLLUTANT DATA -> OpenWeather Air Pollution API.
# You can paste your OpenWeather API key directly below inside the quotes,
# or set it as an environment variable.
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "Aap_Apni_API_Key_Yahan_Paste_Kar_Sakte_Hain")
OPENWEATHER_POLLUTION_CURRENT_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_POLLUTION_FORECAST_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
OPENWEATHER_POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# WEATHER DATA -> Open-Meteo (free, no API key needed).
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# How many past days of historical data to pull during backfill.
BACKFILL_PAST_DAYS = 30

# How many past hours to re-fetch every time the hourly feature
# pipeline runs (keeps a small overlap so lag features never break).
FEATURE_PIPELINE_PAST_HOURS = 48

# How many days ahead to forecast in the dashboard.
FORECAST_DAYS = 3

# =========================================================
# 3. FEATURE STORE / MODEL REGISTRY SETTINGS (Hopsworks)
# =========================================================
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "pearls_aqi")

# If a Hopsworks API key is found, the pipeline will use the real
# Feature Store / Model Registry. If not, it automatically falls
# back to local CSV / pickle files.
USE_HOPSWORKS = bool(HOPSWORKS_API_KEY)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_predictor_model"
TARGET_COLUMN = "aqi"

# =========================================================
# 4. LOCAL FALLBACK STORAGE PATHS
# =========================================================
DATA_DIR = "data"
LOCAL_FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")

MODEL_DIR = "models"
LOCAL_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pkl")
LOCAL_MODEL_META_FILE = os.path.join(MODEL_DIR, "model_meta.json")
LOCAL_SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")

# =========================================================
# 5. FLASK API SETTINGS
# =========================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# =========================================================
# 6. ALERT SETTINGS
# =========================================================
# US AQI scale reference:
#   0-50    Good
#   51-100  Moderate
#   101-150 Unhealthy for Sensitive Groups
#   151-200 Unhealthy
#   201-300 Very Unhealthy
#   301+    Hazardous
# We raise a dashboard alert when predicted AQI crosses this value.
HAZARDOUS_AQI_THRESHOLD = 150