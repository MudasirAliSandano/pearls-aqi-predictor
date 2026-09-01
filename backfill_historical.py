import os
import pandas as pd
import requests
from datetime import datetime, timedelta

def backfill_data():
    print("Backfilling historical air quality data using Open-Meteo...")
    
    # Coordinates for Sukkur
    lat, lon = 27.7052, 68.8574
    
    # Fetching past 30 days of hourly data from Open-Meteo
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide&past_days=30"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.text}")
        return

    data = response.json()
    hourly = data.get("hourly", {})
    
    df = pd.DataFrame({
        'timestamp': hourly.get('time'),
        'pm2_5': hourly.get('pm2_5'),
        'pm10': hourly.get('pm10'),
        'co': hourly.get('carbon_monoxide'),
        'no2': hourly.get('nitrogen_dioxide'),
        'so2': hourly.get('sulphur_dioxide')
    })
    
    # Handle missing values and create dummy AQI calculation for target
    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['aqi'] = (df['pm2_5'] * 1.5).fillna(50) # Derived proxy for AQI target
    
    # Feature engineering steps
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['aqi_lag1'] = df['aqi'].shift(1).fillna(df['aqi'].mean())
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/features.csv', index=False)
    print("Successfully generated data/features.csv!")

if __name__ == "__main__":
    backfill_data()