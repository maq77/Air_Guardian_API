from fastapi import APIRouter, HTTPException
import pandas as pd
from api.schemas import HealthAlert
from utils.helpers import get_health_recommendations, get_lat_lon
from data.fetcher import AirQualityDataFetcher

router = APIRouter()
fetcher = AirQualityDataFetcher()

@router.get("/api/v1/health-alert/{location}", response_model=HealthAlert)
def get_health_alert(location: str):
    """Get health recommendations based on current AQI"""
    try:
        # Get coordinates
        lat, lon = get_lat_lon(location)
        
        # Fetch current air quality data
        forecast_data = fetcher.fetch_openmeteo_forecast(lat, lon, days=1)
        
        if not forecast_data or len(forecast_data) == 0:
            raise HTTPException(status_code=502, detail="Failed to fetch air quality data")
        
        # Get current AQI from first data point
        df = pd.DataFrame(forecast_data)
        current_aqi = float(df.iloc[0].get('aqi', 50)) if 'aqi' in df.columns else 50.0
        
        # Get health recommendations
        recs = get_health_recommendations(current_aqi)
        
        return HealthAlert(
            risk_level=recs['risk_level'],
            activities_safe=recs['activities_safe'],
            activities_avoid=recs['activities_avoid'],
            sensitive_groups_warning=recs['sensitive_groups_message']
        )
    
    except Exception as e:
        print(f"[ERROR] Health alert error for {location}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
