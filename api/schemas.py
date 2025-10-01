from pydantic import BaseModel
from datetime import datetime
from typing import List

class AirQualityForecast(BaseModel):
    timestamp: datetime
    aqi: int
    pm25: float
    no2: float
    risk_level: str
    health_recommendation: str
    confidence: float

class HealthAlert(BaseModel):
    risk_level: str
    activities_safe: List[str]
    activities_avoid: List[str]
    sensitive_groups_warning: str

class PolicyRecommendation(BaseModel):
    location: str
    priority: str
    forecast_severity: str
    recommendations: List[str]
    estimated_impact: str