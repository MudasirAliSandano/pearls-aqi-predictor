"""
backfill_historical.py
-----------------------
STEP 2 of the project: Historical Data Backfill.

A model cannot be trained on just a couple of hours of data - it needs
weeks of history to learn real patterns. This script fetches the
maximum available historical window from Open-Meteo in one go and
stores it, giving the training pipeline enough data to work with.

Run manually with:
    python backfill_historical.py

You normally only need to run this ONCE at the start of the project.
After that, feature_pipeline.py (running hourly) keeps adding new data.
"""

import sys
import config
import utils
from feature_pipeline import get_feature_store, save_to_feature_store


def run_backfill(past_days=None):
    past_days = past_days or config.BACKFILL_PAST_DAYS
    print(f"Backfilling the last {past_days} days of historical data "
          f"for {config.CITY_NAME}...")

    aq_json = utils.fetch_air_quality(
        config.LATITUDE, config.LONGITUDE, past_days=past_days
    )
    weather_json = utils.fetch_weather(
        config.LATITUDE, config.LONGITUDE, past_days=past_days
    )

    features_df = utils.build_feature_dataframe(aq_json, weather_json)
    # The first couple of rows will always be missing lag features -> drop them
    features_df = features_df.dropna(subset=["aqi_lag_1", "aqi_lag_2"]).reset_index(drop=True)

    print(f"Built {len(features_df)} rows of historical training data.")

    feature_store = get_feature_store()

    if feature_store is not None:
        save_to_feature_store(features_df, feature_store)
    else:
        saved_df = utils.save_features_locally(features_df)
        print(f"[Local fallback mode] Saved backfilled data to "
              f"'{config.LOCAL_FEATURES_FILE}'. Total rows now: {len(saved_df)}")

    print("Backfill finished successfully. You can now run training_pipeline.py")


if __name__ == "__main__":
    try:
        run_backfill()
    except Exception as error:
        print(f"Backfill failed: {error}")
        sys.exit(1)
