from fastapi import APIRouter, HTTPException
from typing import List
from api.schemas import AirQualityForecast
from models.forecaster import AQIForecaster
from utils.helpers import classify_aqi, get_health_recommendations
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/api/v1/forecast/{location}", response_model=List[AirQualityForecast])
def get_forecast(location: str, hours: int = 24):
    """Get air quality forecast"""
    # Simplified version - use mock data
    forecasts = []
    base_aqi = 120  # Mock
    
    for i in range(hours):
        aqi = base_aqi + (i % 6) * 5
        forecasts.append(AirQualityForecast(
            timestamp=datetime.now() + timedelta(hours=i),
            aqi=aqi,
            pm25=aqi * 0.4,
            no2=aqi * 0.3,
            risk_level=classify_aqi(aqi),
            health_recommendation=f"Forecast for hour {i+1}",
            confidence=0.85
        ))
    
    return forecasts