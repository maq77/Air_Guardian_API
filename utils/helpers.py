"""
Helper Functions for AirGuardian API
Utility functions used across the application
"""

from fastapi import HTTPException
import numpy as np
from datetime import datetime
from typing import Union, List
from utils.constants import AQI_BREAKPOINTS, RISK_LEVELS, HEALTH_MESSAGES
from data.fetcher import AirQualityDataFetcher

fetcher = AirQualityDataFetcher()

CITY_COORDINATES = {
    'cairo': {'lat': 30.0444, 'lon': 31.2357},
    'alexandria': {'lat': 31.2001, 'lon': 29.9187},
    'giza': {'lat': 30.0131, 'lon': 31.2089},
    'aswan': {'lat': 24.0889, 'lon': 32.8998},
    'luxor': {'lat': 25.6872, 'lon': 32.6396},
    'port said': {'lat': 31.2653, 'lon': 32.3019},
    'suez': {'lat': 29.9668, 'lon': 32.5498},
    'mansoura': {'lat': 31.0409, 'lon': 31.3785},
    'tanta': {'lat': 30.7865, 'lon': 31.0004},
    'asyut': {'lat': 27.1809, 'lon': 31.1837}
}

def get_lat_lon(city: str) -> tuple:
    city_normalized = city.lower().strip()
    
    if city_normalized in CITY_COORDINATES:
        coords = CITY_COORDINATES[city_normalized]
        print(f"[INFO] Using hardcoded coordinates for {city}: {coords['lat']}, {coords['lon']}")
        return coords['lat'], coords['lon']
    
    try:
        data = fetcher.geocode_city(city)
        if data:
            print(f"[INFO] Using geocoded coordinates for {city}: {data['lat']}, {data['lon']}")
            return data['lat'], data['lon']
    except Exception as e:
        print(f"[WARNING] Geocoding failed for {city}: {e}")
    
    raise HTTPException(
        status_code=404, 
        detail=f"City '{city}' not found in database or geocoding service"
    )



def calculate_aqi_from_pm25(pm25: float) -> int:
    """
    Calculate AQI from PM2.5 concentration
    Uses EPA AQI calculation formula
    
    Args:
        pm25: PM2.5 concentration in μg/m³
        
    Returns:
        AQI value (0-500)
    """
    # EPA breakpoints for PM2.5
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            # Linear interpolation
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            return int(round(aqi))
    
    # If PM2.5 is above 500.4, return max AQI
    return 500


def classify_aqi(aqi: int) -> str:
    """
    Classify AQI into risk category
    
    Args:
        aqi: Air Quality Index value
        
    Returns:
        Risk level string (e.g., "Good", "Unhealthy")
    """
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_risk_category_key(aqi: int) -> str:
    """
    Get category key for looking up messages
    
    Args:
        aqi: Air Quality Index value
        
    Returns:
        Category key (e.g., "good", "unhealthy")
    """
    if aqi <= 50:
        return "good"
    elif aqi <= 100:
        return "moderate"
    elif aqi <= 150:
        return "unhealthy_sensitive"
    elif aqi <= 200:
        return "unhealthy"
    elif aqi <= 300:
        return "very_unhealthy"
    else:
        return "hazardous"


def get_health_recommendations(aqi: int, time_of_day: str = None) -> dict:
    """
    Get health recommendations based on AQI
    
    Args:
        aqi: Air Quality Index value
        time_of_day: Optional time period (morning/afternoon/evening)
        
    Returns:
        Dictionary with health guidance
    """
    category = get_risk_category_key(aqi)
    messages = HEALTH_MESSAGES[category]
    
    result = {
        "risk_level": classify_aqi(aqi),
        "general_message": messages["general"],
        "sensitive_groups_message": messages["sensitive"],
        "activities_safe": messages["activities_safe"],
        "activities_avoid": messages["activities_avoid"],
    }
    
    # Add time-specific message if provided
    if time_of_day:
        from utils.constants import TIME_SPECIFIC_MESSAGES
        if time_of_day in TIME_SPECIFIC_MESSAGES:
            result["time_message"] = TIME_SPECIFIC_MESSAGES[time_of_day][category]
    
    return result


def get_time_of_day(hour: int = None) -> str:
    """
    Determine time of day period
    
    Args:
        hour: Hour (0-23), if None uses current hour
        
    Returns:
        Time period string (morning/afternoon/evening)
    """
    if hour is None:
        hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"


def format_timestamp(dt: datetime, format: str = "iso") -> str:
    """
    Format datetime for API responses
    
    Args:
        dt: Datetime object
        format: Format type (iso/readable/time_only)
        
    Returns:
        Formatted string
    """
    if format == "iso":
        return dt.isoformat()
    elif format == "readable":
        return dt.strftime("%B %d, %Y at %I:%M %p")
    elif format == "time_only":
        return dt.strftime("%I:%M %p")
    else:
        return str(dt)


def calculate_confidence(forecast_hours: int, model_accuracy: float = 0.85) -> float:
    """
    Calculate prediction confidence based on forecast horizon
    
    Args:
        forecast_hours: Hours ahead being forecasted
        model_accuracy: Base model accuracy (0-1)
        
    Returns:
        Confidence score (0-1)
    """
    # Confidence decreases with time
    decay_rate = 0.02
    confidence = model_accuracy * np.exp(-decay_rate * forecast_hours)
    return round(max(0.5, min(1.0, confidence)), 2)


def interpolate_missing_values(data: List[float], method: str = "linear") -> List[float]:
    """
    Fill missing values in time series data
    
    Args:
        data: List of values (None for missing)
        method: Interpolation method
        
    Returns:
        List with interpolated values
    """
    import pandas as pd
    series = pd.Series(data)
    
    if method == "linear":
        filled = series.interpolate(method='linear')
    elif method == "ffill":
        filled = series.fillna(method='ffill')
    elif method == "bfill":
        filled = series.fillna(method='bfill')
    else:
        filled = series.fillna(series.mean())
    
    return filled.tolist()


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude values
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        True if valid, False otherwise
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two coordinates in kilometers
    
    Args:
        lat1, lon1: First coordinate
        lat2, lon2: Second coordinate
        
    Returns:
        Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return round(distance, 2)


def smooth_forecast(values: List[float], window: int = 3) -> List[float]:
    """
    Apply moving average smoothing to forecast
    
    Args:
        values: List of forecast values
        window: Window size for smoothing
        
    Returns:
        Smoothed values
    """
    if len(values) < window:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    
    return smoothed


def detect_anomalies(values: List[float], threshold: float = 2.0) -> List[int]:
    """
    Detect anomalous values using z-score
    
    Args:
        values: List of values
        threshold: Z-score threshold
        
    Returns:
        List of indices with anomalies
    """
    mean = np.mean(values)
    std = np.std(values)
    
    if std == 0:
        return []
    
    anomalies = []
    for i, val in enumerate(values):
        z_score = abs((val - mean) / std)
        if z_score > threshold:
            anomalies.append(i)
    
    return anomalies


def aggregate_pollutants(pollutant_data: dict) -> int:
    """
    Calculate overall AQI from multiple pollutants
    
    Args:
        pollutant_data: Dict with pollutant concentrations
        
    Returns:
        Overall AQI (maximum of all pollutants)
    """
    aqis = []
    
    if "pm25" in pollutant_data:
        aqis.append(calculate_aqi_from_pm25(pollutant_data["pm25"]))
    
    # Add other pollutant calculations here
    # For now, use PM2.5 as primary indicator
    
    return max(aqis) if aqis else 0


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between different units
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
    """
    conversions = {
        ("ug/m3", "mg/m3"): lambda x: x / 1000,
        ("mg/m3", "ug/m3"): lambda x: x * 1000,
        ("ppb", "ppm"): lambda x: x / 1000,
        ("ppm", "ppb"): lambda x: x * 1000,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        return conversions[key](value)
    
    return value  # No conversion needed


def create_error_response(message: str, status_code: int = 400) -> dict:
    """
    Create standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Error response dict
    """
    return {
        "error": True,
        "message": message,
        "status_code": status_code,
        "timestamp": datetime.now().isoformat()
    }


def create_success_response(data: dict, message: str = "Success") -> dict:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Success message
        
    Returns:
        Success response dict
    """
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }


def is_rush_hour(hour: int = None) -> bool:
    """
    Determine if current time is rush hour
    
    Args:
        hour: Hour (0-23), if None uses current
        
    Returns:
        True if rush hour
    """
    if hour is None:
        hour = datetime.now().hour
    
    # Morning: 7-9 AM, Evening: 5-7 PM
    return hour in [7, 8, 17, 18, 19]


def get_season(month: int = None) -> str:
    """
    Determine season from month (Northern Hemisphere)
    
    Args:
        month: Month (1-12), if None uses current
        
    Returns:
        Season name
    """
    if month is None:
        month = datetime.now().month
    
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"


def calculate_trend(values: List[float]) -> str:
    """
    Determine trend direction from values
    
    Args:
        values: List of sequential values
        
    Returns:
        Trend string (improving/worsening/stable)
    """
    if len(values) < 2:
        return "stable"
    
    # Calculate slope
    x = list(range(len(values)))
    slope = np.polyfit(x, values, 1)[0]
    
    if slope < -2:
        return "improving"
    elif slope > 2:
        return "worsening"
    else:
        return "stable"


def parse_location(location: str) -> dict:
    """
    Parse location string and return metadata
    
    Args:
        location: Location name or coordinates
        
    Returns:
        Dict with parsed location info
    """
    from config.settings import settings
    
    location_lower = location.lower().strip()
    
    # Check if it's a known city
    city_info = settings.get_city_coords(location_lower)
    if city_info:
        return {
            "type": "city",
            "name": location_lower.title(),
            "lat": city_info["lat"],
            "lon": city_info["lon"],
            "country": city_info["country"]
        }
    
    # Try to parse as coordinates (lat,lon)
    if "," in location:
        try:
            parts = location.split(",")
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            if validate_coordinates(lat, lon):
                return {
                    "type": "coordinates",
                    "name": f"{lat}, {lon}",
                    "lat": lat,
                    "lon": lon,
                    "country": "unknown"
                }
        except ValueError:
            pass
    
    return None


def log_api_request(endpoint: str, location: str, response_time: float):
    """
    Log API request for monitoring
    
    Args:
        endpoint: API endpoint called
        location: Location requested
        response_time: Response time in seconds
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"API Request: {endpoint} | Location: {location} | Time: {response_time:.3f}s")


def cache_key(location: str, hours: int = None) -> str:
    """
    Generate cache key for requests
    
    Args:
        location: Location name
        hours: Optional forecast hours
        
    Returns:
        Cache key string
    """
    key = f"aqi_{location.lower()}"
    if hours:
        key += f"_{hours}h"
    return key