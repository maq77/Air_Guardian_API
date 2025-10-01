from fastapi import APIRouter
from api.schemas import PolicyRecommendation
from models.policy_engine import PolicyRecommendationEngine

router = APIRouter()

@router.get("/api/v1/policy-recommendations/{location}", response_model=PolicyRecommendation)
def get_policy_recommendations(location: str):
    """Get government policy recommendations"""
    engine = PolicyRecommendationEngine()
    forecast_aqi = [120, 130, 140]  # Mock
    recs = engine.get_recommendations(forecast_aqi, location)
    
    return PolicyRecommendation(
        location=recs['location'],
        priority=recs['priority'],
        forecast_severity=recs['severity_level'],
        recommendations=recs['recommended_actions'],
        estimated_impact=recs['estimated_impact']
    )