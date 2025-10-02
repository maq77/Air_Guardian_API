# utils/fetch_realtime_data.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import os

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_coordinates(location: str):
    """Get latitude and longitude from city name using OpenWeather's Geocoding API"""
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={OPENWEATHER_API_KEY}"
    resp = requests.get(url)
    data = resp.json()
    
    if not data:
        raise ValueError(f"Location '{location}' not found")
    
    return data[0]['lat'], data[0]['lon']

def fetch_realtime_air_quality(location: str, hours: int = 48) -> pd.DataFrame:
    """Fetch past 48h AQI + weather data from OpenWeatherMap"""
    lat, lon = get_coordinates(location)

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history"
    end = int(datetime.utcnow().timestamp())
    start = end - (hours * 3600)

    params = {
        "lat": lat,
        "lon": lon,
        "start": start,
        "end": end,
        "appid": OPENWEATHER_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'list' not in data:
        raise ValueError("No air quality data returned")

    # Extract and format
    records = []
    for item in data['list']:
        dt = datetime.utcfromtimestamp(item['dt'])
        components = item['components']
        main = item.get('main', {})

        records.append({
            'timestamp': dt,
            'pm25': components.get('pm2_5'),
            'no2': components.get('no2'),
            'temperature': None,  # You can fetch this from separate weather API
            'humidity': None,
            'wind_speed': None,
            'city': location
        })

    df = pd.DataFrame(records)
    return df
