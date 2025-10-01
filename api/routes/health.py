from fastapi import APIRouter
from api.schemas import HealthAlert
from utils.helpers import get_health_recommendations

router = APIRouter()

@router.get("/api/v1/health-alert/{location}", response_model=HealthAlert)
def get_health_alert(location: str):
    """Get health recommendations"""
    aqi = 120  # Mock - replace with real data
    recs = get_health_recommendations(aqi)
    
    return HealthAlert(
        risk_level=recs['risk_level'],
        activities_safe=recs['activities_safe'],
        activities_avoid=recs['activities_avoid'],
        sensitive_groups_warning=recs['sensitive_groups_message']
    )