from fastapi import APIRouter, HTTPException
import pandas as pd
from api.schemas import HealthAlert
from utils.helpers import get_lat_lon
from data.fetcher import AirQualityDataFetcher
from models.classifier import RiskClassifier
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()
fetcher = AirQualityDataFetcher()
risk_classifier = RiskClassifier(method='rule_based')

#### Mock User , E.g : usr

class UserProfile(BaseModel):
    age: int
    conditions: List[str] = []
    activity_level: str = 'moderate'


@router.get("/api/v1/health-alert/{location}", response_model=HealthAlert)
def get_health_alert(location: str):
    """Get health recommendations based on current AQI"""
    try:
        lat, lon = get_lat_lon(location)
        
        forecast_data = fetcher.fetch_openmeteo_forecast(lat, lon, days=1)
        
        if not forecast_data or len(forecast_data) == 0:
            raise HTTPException(status_code=502, detail="Failed to fetch air quality data")
        
        df = pd.DataFrame(forecast_data)

        if 'aqi' not in df.columns and 'pm25' in df.columns:
            df['aqi'] = df['pm25'].apply(lambda x: min(500, max(0, x * 2.5)))
        
        current_aqi = float(df.iloc[0].get('aqi', 50))
        
        
        recs = risk_classifier.classify_rule_based(current_aqi)
        
        return HealthAlert(
            risk_level=recs['risk_level'],
            activities_safe=recs['activities_safe'],
            activities_avoid=recs['activities_avoid'],
            sensitive_groups_warning=recs['sensitive_groups_message']
        )
    
    except Exception as e:
        print(f"[ERROR] Health alert error for {location}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/v1/health-alert-detailed/{location}")
def get_detailed_health_alert(location: str, hour: int = None):
    """Get detailed health alert with time-specific recommendations"""
    try:
        from datetime import datetime
        
        lat, lon = get_lat_lon(location)
        forecast_data = fetcher.fetch_openmeteo_forecast(lat, lon, days=1)
        
        if not forecast_data:
            raise HTTPException(status_code=502, detail="Failed to fetch data")
        
        df = pd.DataFrame(forecast_data)
        if 'aqi' not in df.columns and 'pm25' in df.columns:
            df['aqi'] = df['pm25'].apply(lambda x: min(500, max(0, x * 2.5)))
        
        current_aqi = float(df.iloc[0].get('aqi', 50))
        current_hour = hour if hour is not None else datetime.now().hour
        
        detailed_alert = risk_classifier.get_detailed_health_alert(current_aqi, current_hour)
        
        return detailed_alert
    
    except Exception as e:
        print(f"[ERROR] Detailed health alert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/v1/health-advice-personalized/{location}")
def get_personalized_health_advice(location: str, profile: UserProfile):
    """Get personalized health advice based on user profile"""
    try:
        from models.classifier import HealthAdvisor
        
        lat, lon = get_lat_lon(location)
        forecast_data = fetcher.fetch_openmeteo_forecast(lat, lon, days=1)
        
        if not forecast_data:
            raise HTTPException(status_code=502, detail="Failed to fetch data")
        
        df = pd.DataFrame(forecast_data)
        if 'aqi' not in df.columns and 'pm25' in df.columns:
            df['aqi'] = df['pm25'].apply(lambda x: min(500, max(0, x * 2.5)))
        
        current_aqi = float(df.iloc[0].get('aqi', 50))
        
        advisor = HealthAdvisor()
        advice = advisor.get_personalized_advice(
            current_aqi, 
            profile.dict()
        )
        
        return advice
    
    except Exception as e:
        print(f"[ERROR] Personalized advice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))