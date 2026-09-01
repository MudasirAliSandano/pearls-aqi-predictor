import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model():
    print("Loading training data...")
    if not os.path.exists('data/features.csv'):
        print("Training pipeline failed: No local feature file found at data/features.csv. Run backfill_historical.py first.")
        return
        
    df = pd.read_csv('data/features.csv')
    
    # Exact 11 features matching app.py
    feature_cols = [
        'pm2_5', 'pm10', 'co', 'no2', 'so2', 
        'hour', 'day', 'month', 'aqi_lag1', 
        'temperature', 'wind_speed'
    ]
    
    # Ensure all columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    target_col = 'aqi'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Fit scaler with all 11 features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Model trained successfully! RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Saved model and scaler successfully!")

if __name__ == "__main__":
    train_model()