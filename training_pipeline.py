"""
training_pipeline.py
---------------------
STEP 3 of the project: the Training Pipeline.

What this script does:
1. Loads all historical (features, target) data from the Feature Store.
2. Trains and compares three different models:
     - Ridge Regression        (simple statistical baseline)
     - Random Forest Regressor (tree-based ensemble model)
     - A small Neural Network  (deep learning model, via TensorFlow/Keras)
3. Evaluates every model with RMSE, MAE, and R^2 on a held-out test set.
4. Saves the BEST performing model to the Model Registry (Hopsworks
   if configured, otherwise local files).
5. Generates a SHAP feature-importance plot for explainability.

Run manually with:
    python training_pipeline.py

In production this should run automatically once a day
(see .github/workflows/training_pipeline.yml).
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
import utils
from utils import FEATURE_COLUMNS


# =========================================================
# 1. LOAD DATA
# =========================================================

def load_training_data():
    """Load the full historical dataset from Hopsworks or the local fallback."""
    if config.USE_HOPSWORKS:
        import hopsworks
        project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME,
        )
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group(
            name=config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION
        )
        df = feature_group.read()
    else:
        df = utils.load_features_locally()

    df = df.dropna(subset=FEATURE_COLUMNS + [config.TARGET_COLUMN])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def time_based_split(df, test_fraction=0.2):
    """
    Split data by TIME instead of randomly. This matters a lot for
    time-series problems like AQI forecasting: testing on randomly
    shuffled rows would let the model 'cheat' by seeing information
    from the future, giving misleadingly good scores.
    """
    split_index = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    return train_df, test_df


# =========================================================
# 2. TRAIN + EVALUATE MODELS
# =========================================================

def evaluate(y_true, y_pred):
    """Compute the three metrics required by the project spec."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def build_neural_network(input_dim):
    """A small, fast Multi-Layer Perceptron for AQI regression."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_compare_models(train_df, test_df):
    """Train all three models and return their trained objects + metrics."""
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[config.TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[config.TARGET_COLUMN].values

    # Scale features - helps Ridge Regression and the Neural Network converge well.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ---- Model 1: Ridge Regression (statistical baseline) ----
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_preds = ridge.predict(X_test_scaled)
    results["ridge_regression"] = {
        "model": ridge,
        "metrics": evaluate(y_test, ridge_preds),
    }

    # ---- Model 2: Random Forest (tree-based ensemble) ----
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    results["random_forest"] = {
        "model": rf,
        "metrics": evaluate(y_test, rf_preds),
    }

    # ---- Model 3: Neural Network (deep learning) ----
    nn = build_neural_network(input_dim=X_train_scaled.shape[1])
    nn.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        verbose=0,
    )
    nn_preds = nn.predict(X_test_scaled, verbose=0).flatten()
    results["neural_network"] = {
        "model": nn,
        "metrics": evaluate(y_test, nn_preds),
    }

    return results, scaler


def pick_best_model(results):
    """Choose the model with the lowest RMSE (lower error = better)."""
    best_name = min(results, key=lambda name: results[name]["metrics"]["rmse"])
    return best_name, results[best_name]


# =========================================================
# 3. EXPLAINABILITY (SHAP)
# =========================================================

def generate_shap_plot(train_df):
    """
    Generate a SHAP summary plot to explain which features matter most.
    We always explain using a Random Forest, because tree-based models
    work with SHAP's fast TreeExplainer regardless of which model
    ultimately 'wins' on accuracy - this keeps the explanation reliable.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")  # no GUI needed, just save the image to disk
    import matplotlib.pyplot as plt

    X = train_df[FEATURE_COLUMNS]
    y = train_df[config.TARGET_COLUMN]

    explain_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    explain_model.fit(X, y)

    explainer = shap.TreeExplainer(explain_model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("eda_outputs/shap_summary.png", dpi=150)
    plt.close()
    print("Saved SHAP feature-importance plot to 'eda_outputs/shap_summary.png'.")


# =========================================================
# 4. SAVE BEST MODEL
# =========================================================

def save_best_model(best_name, best_result, scaler):
    metadata = {
        "model_type": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": config.TARGET_COLUMN,
        "metrics": best_result["metrics"],
    }

    if config.USE_HOPSWORKS:
        import hopsworks
        project = hopsworks.login(
            api_key_value=config.HOPSWORKS_API_KEY,
            project=config.HOPSWORKS_PROJECT_NAME,
        )
        mr = project.get_model_registry()
        # Hopsworks models are saved from a local directory, so we still
        # write local files first, then upload them.
        utils.save_model_locally(best_result["model"], scaler, metadata)
        hops_model = mr.python.create_model(
            name=config.MODEL_NAME,
            metrics=best_result["metrics"],
            description=f"Best AQI model ({best_name})",
        )
        hops_model.save(config.MODEL_DIR)
        print(f"Uploaded model to Hopsworks Model Registry as '{config.MODEL_NAME}'.")
    else:
        utils.save_model_locally(best_result["model"], scaler, metadata)


# =========================================================
# MAIN
# =========================================================

def run_training_pipeline():
    print("Loading training data...")
    df = load_training_data()
    print(f"Loaded {len(df)} rows.")

    if len(df) < 50:
        print("WARNING: Very little data available. Run backfill_historical.py "
              "to get a proper training set before trusting these results.")

    train_df, test_df = time_based_split(df)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    print("Training Ridge Regression, Random Forest, and Neural Network...")
    results, scaler = train_and_compare_models(train_df, test_df)

    print("\nModel comparison (lower RMSE/MAE is better, higher R^2 is better):")
    for name, result in results.items():
        m = result["metrics"]
        print(f"  {name:18s} -> RMSE: {m['rmse']:.3f} | MAE: {m['mae']:.3f} | R^2: {m['r2']:.3f}")

    best_name, best_result = pick_best_model(results)
    print(f"\nBest model: {best_name} (RMSE = {best_result['metrics']['rmse']:.3f})")

    print("\nGenerating SHAP explainability plot...")
    generate_shap_plot(train_df)

    print("\nSaving best model to the Model Registry...")
    save_best_model(best_name, best_result, scaler)

    print("\nTraining pipeline finished successfully.")


if __name__ == "__main__":
    try:
        run_training_pipeline()
    except Exception as error:
        print(f"Training pipeline failed: {error}")
        sys.exit(1)
