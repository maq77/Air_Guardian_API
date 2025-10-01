"""
Configuration Management for AirGuardian API
Loads environment variables and provides centralized settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "trained_models"
    DATA_DIR = BASE_DIR / "data_storage"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # API Configuration
    API_TITLE = "AirGuardian API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "AI-powered air quality forecasting and decision support"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # External API Keys
    NASA_EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN", "")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
    
    # External API URLs
    OPENAQ_BASE_URL = "https://api.openaq.org/v2"
    NASA_EARTHDATA_URL = "https://urs.earthdata.nasa.gov"
    OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    # Model Configuration
    FORECASTER_1H_PATH = MODELS_DIR / "forecaster_1h.pkl"
    FORECASTER_6H_PATH = MODELS_DIR / "forecaster_6h.pkl"
    CLASSIFIER_PATH = MODELS_DIR / "risk_classifier.pkl"
    POLICY_ENGINE_PATH = MODELS_DIR / "policy_engine.pkl"
    
    # Forecast Settings
    MAX_FORECAST_HOURS = 72
    DEFAULT_FORECAST_HOURS = 24
    CONFIDENCE_THRESHOLD = 0.7
    
    # Egyptian Cities (default supported locations)
    CITIES = {
        "cairo": {"lat": 30.0444, "lon": 31.2357, "country": "EG"},
        "alexandria": {"lat": 31.2001, "lon": 29.9187, "country": "EG"},
        "giza": {"lat": 30.0131, "lon": 31.2089, "country": "EG"},
        "aswan": {"lat": 24.0889, "lon": 32.8998, "country": "EG"},
        "luxor": {"lat": 25.6872, "lon": 32.6396, "country": "EG"},
    }
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS Settings (for frontend)
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    
    # Cache Settings (optional - for performance)
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "False").lower() == "true"
    CACHE_TTL = 300  # 5 minutes
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_city_coords(cls, city_name: str):
        """Get coordinates for a city"""
        city = city_name.lower()
        if city in cls.CITIES:
            return cls.CITIES[city]
        return None
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and warn about missing keys"""
        warnings = []
        
        if not cls.NASA_EARTHDATA_TOKEN:
            warnings.append("NASA_EARTHDATA_TOKEN not set - will use mock TEMPO data")
        
        if not cls.OPENWEATHER_API_KEY:
            warnings.append("OPENWEATHER_API_KEY not set - will use mock weather data")
        
        return warnings


# Create singleton instance
settings = Settings()

# Create directories on import
settings.create_directories()

# Validate configuration
config_warnings = settings.validate_config()
if config_warnings:
    print("⚠️  Configuration Warnings:")
    for warning in config_warnings:
        print(f"   - {warning}")