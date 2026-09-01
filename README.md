# Pearls AQI Predictor

An end-to-end, serverless machine learning system that predicts the Air
Quality Index (AQI) for the next 3 days, built following the FTI
(Feature / Training / Inference) pipeline architecture.

## 1. Architecture

```
Weather & Pollution API  --->  Feature Pipeline  --->  Feature Store
      (Open-Meteo)              (feature_pipeline.py)   (Hopsworks)
                                                              |
                                                              v
                                                      Training Pipeline
                                                      (training_pipeline.py)
                                                              |
                                                              v
                                                      Model Registry
                                                       (Hopsworks)
                                                              |
                                                              v
                                                        Web Dashboard
                                                          (app.py)
```

## 2. Why Open-Meteo instead of AQICN/OpenWeather?

The project brief explicitly allows exploring alternatives to
AQICN/OpenWeather. **Open-Meteo** was chosen because:
- It's completely free and requires **no API key / signup**.
- It already computes the **US AQI** value for us directly.
- It provides both historical data (backfill) and forecast data
  (needed for the dashboard) from the same provider.

This removes a major setup blocker so the whole pipeline can be built
and run the same day.

## 3. Project Structure

```
pearls_aqi_predictor/
├── config.py                    # All settings in one place
├── utils.py                     # Shared data fetching + feature engineering
├── feature_pipeline.py          # STEP 1: hourly feature pipeline
├── backfill_historical.py       # STEP 2: historical data backfill
├── training_pipeline.py         # STEP 3: trains & compares 3 models
├── eda.py                       # Exploratory Data Analysis plots
├── app.py                       # STEP 4: Streamlit dashboard
├── requirements.txt
├── .env.example
├── .github/workflows/
│   ├── feature_pipeline.yml     # Runs feature_pipeline.py every hour
│   └── training_pipeline.yml    # Runs training_pipeline.py every day
├── data/                        # Local feature store fallback (CSV)
├── models/                      # Local model registry fallback
└── eda_outputs/                 # Generated plots (EDA + SHAP)
```

## 4. Setup Instructions

### 4.1 Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 (Recommended) Set up Hopsworks Feature Store & Model Registry
The project requires a real Feature Store / Model Registry. To use the
real Hopsworks service instead of the local fallback:

1. Go to https://app.hopsworks.ai and create a **free account**.
2. Create a new project (e.g. `pearls_aqi`).
3. Go to **Account Settings -> API Keys -> Create new API key**.
4. Set it as an environment variable:
   ```bash
   export HOPSWORKS_API_KEY="your_key_here"
   export HOPSWORKS_PROJECT_NAME="pearls_aqi"
   ```

**Note:** If you don't set `HOPSWORKS_API_KEY`, every script automatically
falls back to storing features/models as local CSV/pickle files in
`data/` and `models/`. This lets you build and test the entire system
immediately, and switch to Hopsworks whenever your account is ready -
no code changes needed.

### 4.3 Change your city (optional)
Edit `CITY_NAME`, `LATITUDE`, and `LONGITUDE` in `config.py`.

## 5. Running the Pipeline (in order)

```bash
# Step 1: Get a large historical dataset to train on (run once)
python backfill_historical.py

# Step 2: Train and compare models, save the best one
python training_pipeline.py

# Step 3: Generate EDA plots for your report
python eda.py

# Step 4: Launch the live dashboard
streamlit run app.py

# (Ongoing) Keep collecting new data every hour
python feature_pipeline.py
```

## 6. Automating with GitHub Actions

1. Push this project to a GitHub repository.
2. Go to **Settings -> Secrets and variables -> Actions** and add:
   - `HOPSWORKS_API_KEY`
   - `HOPSWORKS_PROJECT_NAME`
3. The two workflows in `.github/workflows/` will now run automatically:
   - `feature_pipeline.yml` -> every hour
   - `training_pipeline.yml` -> every day at 03:00 UTC
4. You can also trigger them manually from the **Actions** tab
   ("Run workflow" button) to test them immediately.

## 7. Key Design Decisions

### 7.1 Avoiding data leakage
Derived features like `aqi_change_rate` are built **only from past
values** (`aqi_lag_1 - aqi_lag_2`), never from the current row's own
AQI. Using the current AQI to build a feature would leak the answer
into the model and produce misleadingly perfect training results that
completely fail in real-world forecasting.

### 7.2 Time-based train/test split
The data is split by time (oldest 80% train, newest 20% test) rather
than randomly shuffled, since shuffling a time series lets the model
"see the future" during evaluation and inflates its apparent accuracy.

### 7.3 Recursive multi-step forecasting
To forecast 3 days (72 hours) ahead, the app predicts one hour at a
time and feeds each prediction back in as the lag feature for the
next hour, using the real future weather forecast (from Open-Meteo)
as the other model inputs at each step.

### 7.4 Model comparison
Three models are trained and compared using RMSE, MAE, and R²:
- **Ridge Regression** - simple, fast statistical baseline
- **Random Forest** - tree-based ensemble, captures non-linear patterns
- **Neural Network (Keras/TensorFlow)** - deep learning approach

The model with the lowest test RMSE is automatically selected and
saved to the Model Registry.

### 7.5 Explainability
`training_pipeline.py` generates a SHAP summary plot
(`eda_outputs/shap_summary.png`) using a Random Forest explainer,
showing which features (temperature, wind speed, etc.) influence the
AQI prediction the most.

### 7.6 Hazard alerts
The dashboard raises a visible warning whenever the predicted AQI for
any upcoming hour reaches the "Unhealthy" threshold (US AQI ≥ 150,
configurable in `config.py`).

## 8. Final Submission Checklist

- [x] End-to-end AQI prediction system (feature -> training -> inference)
- [x] Scalable, automated pipeline (GitHub Actions, hourly + daily)
- [x] Interactive dashboard with real-time + forecasted AQI (Streamlit)
- [ ] Detailed report documenting your results (fill in your own
      metrics, screenshots, and findings using this README as a base)

## 9. Known Limitations / Possible Improvements

- Recursive forecasting can accumulate error over 72 hours; a
  direct multi-output model is a possible future improvement.
- Only weather-based features are used; adding nearby traffic or
  industrial activity data could improve accuracy further.
- The neural network uses a small architecture for speed; with more
  historical data, a larger network or an LSTM could be explored.
