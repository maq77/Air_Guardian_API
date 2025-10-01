"""
Constants used throughout the AirGuardian API
AQI thresholds, health messages, and other fixed values
"""

# ==================== AQI Thresholds ====================

AQI_BREAKPOINTS = {
    "good": {"min": 0, "max": 50, "color": "#00e400"},
    "moderate": {"min": 51, "max": 100, "color": "#ffff00"},
    "unhealthy_sensitive": {"min": 101, "max": 150, "color": "#ff7e00"},
    "unhealthy": {"min": 151, "max": 200, "color": "#ff0000"},
    "very_unhealthy": {"min": 201, "max": 300, "color": "#8f3f97"},
    "hazardous": {"min": 301, "max": 500, "color": "#7e0023"},
}

# ==================== Risk Level Names ====================

RISK_LEVELS = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy for Sensitive Groups",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous"
}

# ==================== Health Messages ====================

HEALTH_MESSAGES = {
    "good": {
        "general": "Air quality is excellent. Ideal for outdoor activities.",
        "sensitive": "Perfect conditions for everyone, including sensitive groups.",
        "activities_safe": ["Running", "Cycling", "Outdoor sports", "Walking", "Playground activities"],
        "activities_avoid": [],
    },
    "moderate": {
        "general": "Air quality is acceptable for most people.",
        "sensitive": "Unusually sensitive individuals should consider limiting prolonged outdoor exertion.",
        "activities_safe": ["Walking", "Light jogging", "Outdoor dining", "Casual cycling"],
        "activities_avoid": ["Prolonged intense exercise"],
    },
    "unhealthy_sensitive": {
        "general": "Sensitive groups may experience health effects.",
        "sensitive": "Children, elderly, and people with heart/lung disease should reduce prolonged outdoor exertion.",
        "activities_safe": ["Indoor activities", "Short walks", "Light outdoor activity"],
        "activities_avoid": ["Prolonged outdoor sports", "Intense exercise", "Marathon running"],
    },
    "unhealthy": {
        "general": "Everyone may begin to experience health effects.",
        "sensitive": "Sensitive groups should avoid prolonged outdoor exertion. Everyone else should limit it.",
        "activities_safe": ["Indoor activities", "Brief outdoor errands"],
        "activities_avoid": ["All outdoor sports", "Prolonged outdoor activity", "Opening windows for long periods"],
    },
    "very_unhealthy": {
        "general": "Health alert: Everyone may experience serious health effects.",
        "sensitive": "Everyone should avoid all outdoor exertion. Sensitive groups should stay indoors.",
        "activities_safe": ["Indoor activities only", "Air-conditioned spaces"],
        "activities_avoid": ["Any outdoor activity", "Opening windows", "Commuting without protection"],
    },
    "hazardous": {
        "general": "Health warning of emergency conditions: everyone affected.",
        "sensitive": "Everyone should avoid all outdoor exposure. Emergency conditions.",
        "activities_safe": ["Stay indoors", "Use air purifiers", "Seal windows and doors"],
        "activities_avoid": ["All outdoor activities", "Going outside without N95 mask", "Physical exertion"],
    },
}

# ==================== Time-Based Messages ====================

TIME_SPECIFIC_MESSAGES = {
    "morning": {
        "good": "Perfect morning for exercise!",
        "moderate": "Good morning air quality for most activities.",
        "unhealthy_sensitive": "Sensitive individuals should limit morning exercise.",
        "unhealthy": "Consider indoor morning workout today.",
        "very_unhealthy": "Stay indoors this morning.",
        "hazardous": "Do not go outside this morning.",
    },
    "afternoon": {
        "good": "Great afternoon for outdoor activities!",
        "moderate": "Afternoon air quality is acceptable.",
        "unhealthy_sensitive": "Sensitive groups should stay indoors this afternoon.",
        "unhealthy": "Limit afternoon outdoor exposure.",
        "very_unhealthy": "Stay indoors this afternoon.",
        "hazardous": "Emergency conditions - stay inside.",
    },
    "evening": {
        "good": "Lovely evening for a walk!",
        "moderate": "Evening air quality is moderate.",
        "unhealthy_sensitive": "Consider indoor activities this evening.",
        "unhealthy": "Best to stay indoors this evening.",
        "very_unhealthy": "Do not go outside this evening.",
        "hazardous": "Emergency - remain indoors.",
    },
}

# ==================== Pollutant Information ====================

POLLUTANTS = {
    "pm25": {
        "name": "PM2.5",
        "full_name": "Fine Particulate Matter",
        "unit": "μg/m³",
        "description": "Particles less than 2.5 micrometers",
        "health_effects": "Respiratory and cardiovascular issues",
        "safe_limit": 12.0,  # WHO guideline
    },
    "pm10": {
        "name": "PM10",
        "full_name": "Coarse Particulate Matter",
        "unit": "μg/m³",
        "description": "Particles less than 10 micrometers",
        "health_effects": "Respiratory irritation",
        "safe_limit": 45.0,
    },
    "no2": {
        "name": "NO2",
        "full_name": "Nitrogen Dioxide",
        "unit": "ppb",
        "description": "Traffic and industrial emissions",
        "health_effects": "Respiratory inflammation",
        "safe_limit": 25.0,
    },
    "o3": {
        "name": "O3",
        "full_name": "Ozone",
        "unit": "ppb",
        "description": "Formed by sunlight reacting with pollutants",
        "health_effects": "Lung irritation",
        "safe_limit": 100.0,
    },
    "so2": {
        "name": "SO2",
        "full_name": "Sulfur Dioxide",
        "unit": "ppb",
        "description": "Industrial emissions",
        "health_effects": "Respiratory issues",
        "safe_limit": 20.0,
    },
    "co": {
        "name": "CO",
        "full_name": "Carbon Monoxide",
        "unit": "ppm",
        "description": "Combustion byproduct",
        "health_effects": "Reduces oxygen delivery",
        "safe_limit": 4.0,
    },
}

# ==================== Policy Priority Levels ====================

POLICY_PRIORITIES = {
    "LOW": {
        "color": "#28a745",
        "description": "Routine monitoring",
        "response_time": "Normal operations",
    },
    "MEDIUM": {
        "color": "#ffc107",
        "description": "Enhanced monitoring",
        "response_time": "24-48 hours",
    },
    "HIGH": {
        "color": "#fd7e14",
        "description": "Active intervention required",
        "response_time": "12-24 hours",
    },
    "CRITICAL": {
        "color": "#dc3545",
        "description": "Emergency response",
        "response_time": "Immediate action",
    },
}

# ==================== Data Sources ====================

DATA_SOURCES = {
    "tempo": {
        "name": "NASA TEMPO",
        "description": "Satellite-based NO2 and aerosol measurements",
        "coverage": "North America (expanding)",
        "frequency": "Hourly",
        "url": "https://tempo.si.edu/",
    },
    "openaq": {
        "name": "OpenAQ",
        "description": "Global ground-based air quality monitoring",
        "coverage": "Worldwide",
        "frequency": "Real-time",
        "url": "https://openaq.org/",
    },
    "openweather": {
        "name": "OpenWeather",
        "description": "Meteorological data",
        "coverage": "Worldwide",
        "frequency": "Real-time",
        "url": "https://openweathermap.org/",
    },
}

# ==================== Feature Names ====================

FEATURE_NAMES = {
    "temporal": ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour"],
    "cyclical": ["hour_sin", "hour_cos", "month_sin", "month_cos"],
    "lag": ["pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24"],
    "rolling": ["pm25_rolling_mean_3", "pm25_rolling_mean_6", "pm25_rolling_std_6"],
    "weather": ["temperature", "humidity", "pressure", "wind_speed", "wind_direction"],
    "interactions": ["temp_humidity_interaction", "wind_dispersion_factor"],
}

# ==================== Model Hyperparameters (Default) ====================

MODEL_PARAMS = {
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
    },
    "lstm": {
        "units": 64,
        "dropout": 0.2,
        "epochs": 50,
        "batch_size": 32,
    },
}

# ==================== API Response Messages ====================

API_MESSAGES = {
    "success": "Request processed successfully",
    "not_found": "Location not found",
    "invalid_hours": "Invalid forecast hours. Must be between 1 and 72",
    "model_error": "Error processing prediction",
    "data_unavailable": "Data temporarily unavailable",
    "rate_limit": "Rate limit exceeded. Please try again later",
}

# ==================== Egyptian City Metadata ====================

EGYPTIAN_CITIES = {
    "cairo": {
        "name_ar": "القاهرة",
        "population": 20900000,
        "elevation": 23,
        "timezone": "Africa/Cairo",
        "major_sources": ["traffic", "industrial", "dust"],
    },
    "alexandria": {
        "name_ar": "الإسكندرية",
        "population": 5200000,
        "elevation": 5,
        "timezone": "Africa/Cairo",
        "major_sources": ["traffic", "port", "industrial"],
    },
    "giza": {
        "name_ar": "الجيزة",
        "population": 8800000,
        "elevation": 30,
        "timezone": "Africa/Cairo",
        "major_sources": ["traffic", "construction", "dust"],
    },
}