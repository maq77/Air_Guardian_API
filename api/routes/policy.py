from fastapi import APIRouter, HTTPException
import pandas as pd
from api.schemas import PolicyRecommendation
from models.policy_engine import PolicyRecommendationEngine
from utils.helpers import get_lat_lon
from data.fetcher import AirQualityDataFetcher

router = APIRouter()
fetcher = AirQualityDataFetcher()
engine = PolicyRecommendationEngine()

@router.get("/api/v1/policy-recommendations/{location}", response_model=PolicyRecommendation)
def get_policy_recommendations(location: str):
    """Get government policy recommendations based on forecast"""
    try:
        lat, lon = get_lat_lon(location)
        
        forecast_data = fetcher.fetch_openmeteo_forecast(lat, lon, days=2)
        
        if not forecast_data or len(forecast_data) < 3:
            raise HTTPException(status_code=502, detail="Insufficient forecast data")
        
        df = pd.DataFrame(forecast_data)
        forecast_aqi = []
        
        for i in range(min(24, len(df))):
            aqi_value = float(df.iloc[i].get('aqi', 100)) if 'aqi' in df.columns else 100.0
            forecast_aqi.append(aqi_value)
        
        recs = engine.get_recommendations(forecast_aqi, location)
        
        return PolicyRecommendation(
            location=recs['location'],
            priority=recs['priority'],
            forecast_severity=recs['severity_level'],
            recommendations=recs['recommended_actions'],
            estimated_impact=recs['estimated_impact']
        )
    
    except Exception as e:
        print(f"[ERROR] Policy recommendations error for {location}: {e}")
        raise HTTPException(status_code=500, detail=str(e))