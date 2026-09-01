"""
feature_pipeline.py
--------------------
STEP 1 of the project: the Feature Pipeline.

What this script does:
1. Fetches the latest raw weather and air-quality data from Open-Meteo.
2. Computes model-ready features (time-based + derived features).
3. Stores those features in the Feature Store (Hopsworks if configured,
   otherwise a local CSV file as a fallback).

This script is meant to be run automatically every hour
(see .github/workflows/feature_pipeline.yml).

Run manually with:
    python feature_pipeline.py
"""

import sys
import config
import utils


def get_feature_store():
    """
    Connect to the Hopsworks Feature Store if an API key is configured.
    Returns None if Hopsworks is not set up (local fallback will be used).
    """
    if not config.USE_HOPSWORKS:
        return None

    import hopsworks  # imported here so the package is only required if actually used

    project = hopsworks.login(
        api_key_value=config.HOPSWORKS_API_KEY,
        project=config.HOPSWORKS_PROJECT_NAME,
    )
    return project.get_feature_store()


def save_to_feature_store(df, feature_store):
    """Insert the new feature rows into the Hopsworks Feature Group."""
    feature_group = feature_store.get_or_create_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for AQI forecasting",
        primary_key=["timestamp"],
        event_time="timestamp",
    )
    feature_group.insert(df, write_options={"wait_for_job": True})
    print(f"Inserted {len(df)} rows into Hopsworks feature group "
          f"'{config.FEATURE_GROUP_NAME}' (v{config.FEATURE_GROUP_VERSION}).")


def run_feature_pipeline(past_days=2):
    """
    Main entry point: fetch recent data and store the computed features.
    `past_days=2` keeps a small overlap window so the aqi_lag_1 feature
    is always computed correctly, even if the script is a bit late to run.
    """
    print(f"Fetching last {past_days} day(s) of data for {config.CITY_NAME}...")

    aq_json = utils.fetch_air_quality(
        config.LATITUDE, config.LONGITUDE, past_days=past_days
    )
    weather_json = utils.fetch_weather(
        config.LATITUDE, config.LONGITUDE, past_days=past_days
    )

    features_df = utils.build_feature_dataframe(aq_json, weather_json)
    print(f"Computed {len(features_df)} rows of features.")

    feature_store = get_feature_store()

    if feature_store is not None:
        save_to_feature_store(features_df, feature_store)
    else:
        saved_df = utils.save_features_locally(features_df)
        print(f"[Local fallback mode] Saved features to "
              f"'{config.LOCAL_FEATURES_FILE}'. Total rows now: {len(saved_df)}")
        print("Tip: set the HOPSWORKS_API_KEY environment variable to switch "
              "to the real Hopsworks Feature Store.")

    print("Feature pipeline finished successfully.")


if __name__ == "__main__":
    try:
        run_feature_pipeline()
    except Exception as error:
        print(f"Feature pipeline failed: {error}")
        sys.exit(1)
