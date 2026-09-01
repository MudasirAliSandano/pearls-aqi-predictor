"""
eda.py
------
Exploratory Data Analysis (EDA) script, as required by the project's
"Advanced Analytics Features" section.

Generates and saves several plots into the eda_outputs/ folder so they
can be included directly in your final report:
1. AQI trend over time
2. Distribution of AQI values
3. Average AQI by hour of day (finds daily pollution patterns)
4. Correlation heatmap between weather variables and AQI

Run with:
    python eda.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
import utils

OUTPUT_DIR = "eda_outputs"


def load_data():
    if config.USE_HOPSWORKS:
        import hopsworks
        project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME,
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        return fg.read().sort_values("timestamp")
    return utils.load_features_locally().sort_values("timestamp")


def plot_aqi_trend(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["aqi"], linewidth=0.8)
    plt.title(f"AQI Trend Over Time - {config.CITY_NAME}")
    plt.xlabel("Date")
    plt.ylabel("US AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aqi_trend.png", dpi=150)
    plt.close()


def plot_aqi_distribution(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(df["aqi"], bins=30, kde=True)
    plt.title("Distribution of AQI Values")
    plt.xlabel("US AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aqi_distribution.png", dpi=150)
    plt.close()


def plot_avg_aqi_by_hour(df):
    hourly_avg = df.groupby("hour")["aqi"].mean()
    plt.figure(figsize=(8, 4))
    hourly_avg.plot(kind="bar", color="steelblue")
    plt.title("Average AQI by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Average AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/avg_aqi_by_hour.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    cols = ["aqi", "temperature", "humidity", "wind_speed", "pressure"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation: Weather Variables vs AQI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()


def run_eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} rows for EDA.")

    plot_aqi_trend(df)
    plot_aqi_distribution(df)
    plot_avg_aqi_by_hour(df)
    plot_correlation_heatmap(df)

    print(f"Saved 4 EDA plots to '{OUTPUT_DIR}/':")
    print("  - aqi_trend.png")
    print("  - aqi_distribution.png")
    print("  - avg_aqi_by_hour.png")
    print("  - correlation_heatmap.png")


if __name__ == "__main__":
    run_eda()
