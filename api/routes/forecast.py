import traceback
from fastapi import APIRouter, HTTPException, Query
from typing import List
from api.schemas import AirQualityForecast
from models.forecaster import AQIForecaster
from data.fetcher import AirQualityDataFetcher
from utils.helpers import classify_aqi, get_health_recommendations, get_lat_lon
from datetime import datetime, timedelta
from api.routes.model_loader import loaded_models
import pandas as pd

router = APIRouter()
fetcher = AirQualityDataFetcher()

def format_health_recommendation(health_data):
    if isinstance(health_data, dict):
        general = health_data.get("general", "")
        sensitive = health_data.get("sensitive_groups", "")
        tips = health_data.get("tips", [])
        tips_str = " | ".join(tips) if isinstance(tips, list) else str(tips)
        return f"{general} {sensitive} {tips_str}".strip()
    return str(health_data)

@router.get("/api/v1/forecast/{location}", response_model=List[AirQualityForecast])
def get_forecast(location: str, hours: int = Query(24, enum=[1, 6, 24])):
    if hours not in loaded_models:
        raise HTTPException(status_code=400, detail=f"No model for {hours}h forecast")
    
    try:
        forecaster = loaded_models[hours]
        
        # Step 1: Get coordinates
        lat, lon = get_lat_lon(location)
        
        # Step 2: Fetch Open-Meteo forecast
        forecast_data = fetcher.fetch_openmeteo_forecast(lat, lon, days=2)
        if not forecast_data:
            raise HTTPException(status_code=502, detail="Failed to fetch forecast from Open-Meteo")
        
        # Step 3: Convert to DataFrame
        df = pd.DataFrame(forecast_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['city'] = location
        
        print("Forecast DataFrame shape:", df.shape)
        print("Forecast DataFrame columns:", df.columns.tolist())
        print("First few rows:\n", df.head())
        
        # Step 4: Extract CURRENT CONDITIONS as a dictionary (NOT the whole DataFrame!)
        # Use the first row or most recent data point
        current_row = df.iloc[0] if len(df) > 0 else {}
        
        current_data = {
            'aqi': float(current_row.get('aqi', 50)) if 'aqi' in current_row else 50.0,
            'pm25': float(current_row.get('pm25', 25)) if 'pm25' in current_row else 25.0,
            'pm10': float(current_row.get('pm10', 50)) if 'pm10' in current_row else 50.0,
            'no2': float(current_row.get('no2', 20)) if 'no2' in current_row else 20.0,
            'temperature': float(current_row.get('temperature', 25)) if 'temperature' in current_row else 25.0,
            'humidity': float(current_row.get('humidity', 60)) if 'humidity' in current_row else 60.0,
            'wind_speed': float(current_row.get('wind_speed', 5)) if 'wind_speed' in current_row else 5.0,
            'city': location
        }
        
        # Step 5: Run model prediction with the DICT, not DataFrame
        predictions = forecaster.forecast(current_data, hours=hours)
        
        # Step 6: Format API response
        forecasts = []
        for pred in predictions:
            forecasts.append(AirQualityForecast(
                timestamp=datetime.fromisoformat(pred['timestamp']),
                aqi=int(pred['aqi']),
                pm25=pred['pm25'],
                no2=pred['no2'],
                risk_level=pred['risk_level'],
                health_recommendation=pred['health_recommendation'],
                confidence=pred['confidence']
            ))
        
        return forecasts
    
    except Exception as e:
        print(f"[ERROR] Forecast error for {location}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))