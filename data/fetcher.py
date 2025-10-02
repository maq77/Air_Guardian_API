"""
Unified Air Quality Data Fetcher
Combines multiple data sources with improved error handling and efficiency
Sources: OpenWeatherMap, WAQI, OpenAQ, Open-Meteo, Google AQ, NASA TEMPO
"""

import sys
import numpy as np
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import settings

load_dotenv()


class AirQualityDataFetcher:
    """ fetcher for air quality data from multiple reliable sources"""
    
    def __init__(self):
        # API Keys
        self.openweather_key = os.getenv("OPENWEATHER_API_KEY")
        self.waqi_token = os.getenv("WAQI_API_TOKEN")
        self.openaq_key = os.getenv("OPENAQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.nasa_earthdata_token = os.getenv("NASA_EARTHDATA_TOKEN")
        
        # Base URLs
        self.openweather_base = "https://api.openweathermap.org/data/2.5"
        self.waqi_base = "https://api.waqi.info"
        self.openaq_base = "https://api.openaq.org/v3"
        self.openmeteo_base = "https://air-quality-api.open-meteo.com/v1"
        self.google_base = "https://airquality.googleapis.com/v1"
        
        self._print_status()
    
    def _print_status(self):
        """Print initialization status"""
        print("="*70)
        print("🌍 Unified Air Quality Fetcher Initialized")
        print("="*70)
        print(f"OpenWeatherMap: {'✓' if self.openweather_key else '✗'}")
        print(f"WAQI:           {'✓' if self.waqi_token else '✗'}")
        print(f"OpenAQ:         {'✓' if self.openaq_key else '✗'}")
        print(f"Google AQ:      {'✓' if self.google_api_key else '✗'}")
        print(f"NASA TEMPO:     {'✓' if self.nasa_earthdata_token else '✗ (using mock)'}")
        print(f"Open-Meteo:     ✓ (no key needed)")
        print("="*70 + "\n")
    
    def geocode_city(self, city: str) -> Optional[Dict[str, float]]:
        url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {"q": city, "limit": 1, "appid": self.openweather_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json()
            if not results:
                return None
            return {"lat": results[0]['lat'], "lon": results[0]['lon']}
        except Exception as e:
            print(f"✗ Geocode error: {e}")
            return None

    # ==================== OpenWeatherMap ====================
    
    def fetch_openweather_current(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch current air quality from OpenWeatherMap"""
        if not self.openweather_key:
            return None
        
        url = f"{self.openweather_base}/air_pollution"
        params = {"lat": lat, "lon": lon, "appid": self.openweather_key}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('list'):
                air_data = data['list'][0]
                components = air_data['components']
                aqi = air_data['main']['aqi']
                
                # Convert OpenWeather AQI (1-5) to US EPA AQI (0-500)
                aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 275}
                us_aqi = aqi_mapping.get(aqi, 100)
                
                return {
                    'source': 'OpenWeatherMap',
                    'aqi': us_aqi,
                    'aqi_category': self._get_aqi_category(us_aqi),
                    'pm25': components.get('pm2_5', 0),
                    'pm10': components.get('pm10', 0),
                    'no2': components.get('no2', 0),
                    'o3': components.get('o3', 0),
                    'so2': components.get('so2', 0),
                    'co': components.get('co', 0),
                    'nh3': components.get('nh3', 0),
                    'timestamp': datetime.fromtimestamp(air_data['dt']).isoformat()
                }
        except Exception as e:
            print(f"  ✗ OpenWeatherMap error: {e}")
            return None
    
    def fetch_openweather_forecast(self, lat: float, lon: float) -> List[Dict]:
        """Fetch 4-day air quality forecast from OpenWeatherMap"""
        if not self.openweather_key:
            return []
        
        url = f"{self.openweather_base}/air_pollution/forecast"
        params = {"lat": lat, "lon": lon, "appid": self.openweather_key}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 275}
            
            for item in data.get('list', []):
                components = item['components']
                aqi = item['main']['aqi']
                
                forecasts.append({
                    'timestamp': datetime.fromtimestamp(item['dt']).isoformat(),
                    'aqi': aqi_mapping.get(aqi, 100),
                    'pm25': components.get('pm2_5', 0),
                    'pm10': components.get('pm10', 0),
                    'no2': components.get('no2', 0)
                })
            
            return forecasts
        except Exception as e:
            print(f"  ✗ OpenWeather forecast error: {e}")
            return []
    
    # ==================== WAQI ====================
    
    def fetch_waqi_data(self, city: str = None, lat: float = None, lon: float = None) -> Optional[Dict]:
        """Fetch data from World Air Quality Index"""
        if not self.waqi_token:
            return None
        
        if city:
            url = f"{self.waqi_base}/feed/{city}/"
        elif lat and lon:
            url = f"{self.waqi_base}/feed/geo:{lat};{lon}/"
        else:
            return None
        
        params = {"token": self.waqi_token}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok':
                station_data = data['data']
                iaqi = station_data.get('iaqi', {})
                
                return {
                    'source': 'WAQI',
                    'aqi': station_data.get('aqi', 0),
                    'aqi_category': self._get_aqi_category(station_data.get('aqi', 0)),
                    'pm25': iaqi.get('pm25', {}).get('v', 0),
                    'pm10': iaqi.get('pm10', {}).get('v', 0),
                    'no2': iaqi.get('no2', {}).get('v', 0),
                    'o3': iaqi.get('o3', {}).get('v', 0),
                    'so2': iaqi.get('so2', {}).get('v', 0),
                    'co': iaqi.get('co', {}).get('v', 0),
                    'station': station_data.get('city', {}).get('name'),
                    'timestamp': station_data.get('time', {}).get('iso')
                }
        except Exception as e:
            print(f"  ✗ WAQI error: {e}")
            return None
    
    # ==================== Open-Meteo ====================
    
    def fetch_openmeteo_current(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch from Open-Meteo (FREE, no API key needed)"""
        url = f"{self.openmeteo_base}/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,uv_index,european_aqi,us_aqi",
            "timezone": "auto"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            current = data.get('current', {})
            
            return {
                'source': 'Open-Meteo',
                'aqi': current.get('us_aqi', 0),
                'european_aqi': current.get('european_aqi', 0),
                'aqi_category': self._get_aqi_category(current.get('us_aqi', 0)),
                'pm25': current.get('pm2_5', 0),
                'pm10': current.get('pm10', 0),
                'no2': current.get('nitrogen_dioxide', 0),
                'o3': current.get('ozone', 0),
                'so2': current.get('sulphur_dioxide', 0),
                'co': current.get('carbon_monoxide', 0),
                'dust': current.get('dust', 0),
                'uv_index': current.get('uv_index', 0),
                'timestamp': current.get('time')
            }
        except Exception as e:
            print(f"  ✗ Open-Meteo error: {e}")
            return None
    
    def fetch_openmeteo_forecast(self, lat: float, lon: float, days: int = 5) -> List[Dict]:
        """Fetch air quality forecast from Open-Meteo"""
        url = f"{self.openmeteo_base}/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5,pm10,us_aqi",
            "forecast_days": days,
            "timezone": "auto"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            pm25 = hourly.get('pm2_5', [])
            aqi = hourly.get('us_aqi', [])
            
            return [
                {
                    'timestamp': times[i],
                    'pm25': pm25[i],
                    'aqi': aqi[i]
                }
                for i in range(len(times))
            ]
        except Exception as e:
            print(f"  ✗ Open-Meteo forecast error: {e}")
            return []
    
    # ==================== OpenAQ ====================
    
    def fetch_openaq_latest(self, city: str, country: str = "EG") -> Optional[Dict]:
        """Fetch latest measurements from OpenAQ"""
        if not self.openaq_key:
            return None
        
        headers = {"X-API-Key": self.openaq_key}
        
        # Step 1: Get locations
        url = f"{self.openaq_base}/locations"
        params = {"country": country, "city": city, "limit": 100}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                return None
            
            location_ids = [loc['id'] for loc in data['results'][:10]]
            
            # Step 2: Get measurements
            measurements_data = self._fetch_openaq_measurements(location_ids)
            
            if measurements_data:
                return self._process_openaq_data(measurements_data)
            return None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"  ✗ OpenAQ authentication failed")
            else:
                print(f"  ✗ OpenAQ error: {e}")
            return None
        except Exception as e:
            print(f"  ✗ OpenAQ error: {e}")
            return None
    
    def fetch_openaq_historical(self, city: str, days: int = 30, country: str = "EG") -> pd.DataFrame:
        """Fetch historical data from OpenAQ"""
        if not self.openaq_key:
            return pd.DataFrame()
        
        headers = {"X-API-Key": self.openaq_key}
        
        # Get locations
        locations_url = f"{self.openaq_base}/locations"
        params = {"country": country, "city": city, "limit": 100}
        
        try:
            response = requests.get(locations_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                return pd.DataFrame()
            
            location_ids = [loc['id'] for loc in data['results'][:20]]
            
            # Fetch measurements
            measurements_url = f"{self.openaq_base}/measurements"
            date_to = datetime.now()
            date_from = date_to - timedelta(days=days)
            
            params = {
                "location_id": ",".join(map(str, location_ids)),
                "date_from": date_from.strftime("%Y-%m-%d"),
                "date_to": date_to.strftime("%Y-%m-%d"),
                "limit": 10000
            }
            
            response = requests.get(measurements_url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                return pd.DataFrame()
            
            # Convert to DataFrame
            records = []
            for result in data['results']:
                parameter_info = result.get('parameter', {})
                datetime_info = result.get('datetime', {})
                location_info = result.get('location', {})
                
                records.append({
                    'timestamp': datetime_info.get('utc'),
                    'parameter': parameter_info.get('name'),
                    'value': result.get('value'),
                    'unit': parameter_info.get('units'),
                    'location': location_info.get('name'),
                    'location_id': location_info.get('id'),
                    'city': city,
                    'country': country
                })
            
            df = pd.DataFrame(records)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            print(f"  ✗ OpenAQ historical error: {e}")
            return pd.DataFrame()
    
    def _fetch_openaq_measurements(self, location_ids: List[int]) -> List[Dict]:
        """Fetch latest measurements for given location IDs"""
        if not location_ids:
            return []
        
        url = f"{self.openaq_base}/latest"
        headers = {"X-API-Key": self.openaq_key}
        params = {
            "location_id": ",".join(map(str, location_ids)),
            "limit": 1000
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except:
            return []
    
    def _process_openaq_data(self, results: List[Dict]) -> Dict:
        """Process OpenAQ v3 API results"""
        measurements = {}
        
        for result in results:
            location_name = result.get('location', {}).get('name', 'Unknown')
            
            for sensor in result.get('sensors', []):
                parameter_info = sensor.get('parameter', {})
                parameter = parameter_info.get('name')
                
                latest = sensor.get('latest', {})
                if not latest:
                    continue
                
                value = latest.get('value')
                unit = parameter_info.get('units', 'unknown')
                
                if parameter and value is not None:
                    if parameter not in measurements:
                        measurements[parameter] = []
                    
                    measurements[parameter].append({
                        'value': value,
                        'unit': unit,
                        'location': location_name,
                        'timestamp': latest.get('datetime', {}).get('utc')
                    })
        
        # Calculate averages
        averaged = {}
        for param, values in measurements.items():
            if values:
                avg_value = sum([v['value'] for v in values]) / len(values)
                averaged[param] = {
                    'value': round(avg_value, 2),
                    'unit': values[0]['unit'],
                    'sample_count': len(values),
                    'locations': list(set([v['location'] for v in values]))
                }
        
        return averaged
    
    # ==================== Google Air Quality ====================
    
    def fetch_google_aq(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch from Google Air Quality API"""
        if not self.google_api_key:
            return None
        
        url = f"{self.google_base}/currentConditions:lookup"
        params = {"key": self.google_api_key}
        payload = {"location": {"latitude": lat, "longitude": lon}}
        
        try:
            response = requests.post(url, params=params, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            indexes = data.get('indexes', [])
            if indexes:
                us_aqi_data = next((idx for idx in indexes if idx.get('code') == 'uaqi'), None)
                if us_aqi_data:
                    return {
                        'source': 'Google',
                        'aqi': us_aqi_data.get('aqi', 0),
                        'aqi_category': us_aqi_data.get('category'),
                        'dominant_pollutant': us_aqi_data.get('dominantPollutant'),
                        'timestamp': data.get('dateTime')
                    }
        except Exception as e:
            print(f"  ✗ Google AQ error: {e}")
            return None
    
    # ==================== NASA TEMPO ====================
    
    def fetch_tempo_no2(self, lat: float, lon: float) -> Optional[float]:
        """Fetch NO2 data from NASA TEMPO (mock implementation)"""
        import random
        
        print("  ⚠️  Using mock TEMPO data (real API requires NASA Earthdata auth)")
        
        # Generate realistic NO2 values
        base_no2 = 30
        hour = datetime.now().hour
        
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base_no2 += random.uniform(10, 25)
        else:
            base_no2 += random.uniform(-5, 10)
        
        no2 = base_no2 + random.uniform(-10, 10)
        return round(max(5, no2), 2)
    
    # ==================== Smart Multi-Source Fetcher ====================
    
    def fetch_best_available(self, city: str, lat: float, lon: float) -> Dict:
        """Fetch from multiple sources and return combined data"""
        print(f"\n📡 Fetching air quality data for {city}...")
        print(f"   Coordinates: {lat:.4f}, {lon:.4f}\n")
        
        result = {
            'city': city,
            'coordinates': {'lat': lat, 'lon': lon},
            'timestamp': datetime.now().isoformat(),
            'sources': [],
            'data': {}
        }
        
        sources_data = {}
        
        # Try all sources
        print("Querying data sources...")
        
        # 1. OpenWeatherMap
        ow_data = self.fetch_openweather_current(lat, lon)
        if ow_data:
            sources_data['openweather'] = ow_data
            result['sources'].append('OpenWeatherMap')
            print("  ✓ OpenWeatherMap")
        
        # 2. WAQI
        waqi_data = self.fetch_waqi_data(city=city) or self.fetch_waqi_data(lat=lat, lon=lon)
        if waqi_data:
            sources_data['waqi'] = waqi_data
            result['sources'].append('WAQI')
            print("  ✓ WAQI")
        
        # 3. Open-Meteo
        om_data = self.fetch_openmeteo_current(lat, lon)
        if om_data:
            sources_data['openmeteo'] = om_data
            result['sources'].append('Open-Meteo')
            print("  ✓ Open-Meteo")
        
        # 4. OpenAQ
        openaq_data = self.fetch_openaq_latest(city)
        if openaq_data:
            sources_data['openaq'] = openaq_data
            result['sources'].append('OpenAQ')
            print("  ✓ OpenAQ")
        
        # 5. Google
        if self.google_api_key:
            google_data = self.fetch_google_aq(lat, lon)
            if google_data:
                sources_data['google'] = google_data
                result['sources'].append('Google')
                print("  ✓ Google")
        
        # 6. TEMPO
        tempo_no2 = self.fetch_tempo_no2(lat, lon)
        if tempo_no2:
            sources_data['tempo'] = {'no2': tempo_no2, 'unit': 'ppb'}
            result['sources'].append('NASA TEMPO')
            print("  ✓ NASA TEMPO")
        
        # Aggregate data
        if sources_data:
            result['data'] = self._aggregate_sources(sources_data)
            print(f"\n✓ Successfully retrieved data from {len(sources_data)} source(s)")
        else:
            print(f"\n⚠️  No data available from any source")
        
        return result

    def fetch_training_dataset(self, cities: List[dict], days: int = 90) -> pd.DataFrame:
        """
        Fetch comprehensive training dataset from ALL sources in the file
        
        Args:
            cities: List of dicts with 'name', 'lat', 'lon', 'country'
            days: Number of days of historical data
            
        Returns:
            DataFrame ready for model training with all features
        """
        print("\n" + "="*70)
        print("🎯 FETCHING TRAINING DATASET FROM ALL SOURCES")
        print("="*70)
        print(f"Cities: {len(cities)}")
        print(f"Time range: {days} days")
        print("="*70 + "\n")
        
        all_records = []
        
        for idx, city_info in enumerate(cities, 1):
            print(f"\n[{idx}/{len(cities)}] Processing {city_info['name']}, {city_info.get('country', 'Unknown')}...")
            
            city_name = city_info['name']
            lat = city_info['lat']
            lon = city_info['lon']
            country = city_info.get('country', 'EG')
            
            # # ==================== 1. OPENAQ HISTORICAL ====================
            # print("  [1/6] Fetching OpenAQ historical...")
            # if self.openaq_key:
            #     try:
            #         openaq_df = self.fetch_openaq_historical(city_name, days, country)
            #         if not openaq_df.empty:
            #             # Pivot to wide format
            #             openaq_pivot = openaq_df.pivot_table(
            #                 index='timestamp',
            #                 columns='parameter',
            #                 values='value',
            #                 aggfunc='mean'
            #             ).reset_index()
                        
            #             openaq_pivot['city'] = city_name
            #             openaq_pivot['lat'] = lat
            #             openaq_pivot['lon'] = lon
            #             openaq_pivot['source'] = 'OpenAQ'
            #             all_records.append(openaq_pivot)
            #             print(f"      ✓ OpenAQ: {len(openaq_pivot)} records")
            #         else:
            #             print(f"      ✗ OpenAQ: No data")
            #     except Exception as e:
            #         print(f"      ✗ OpenAQ error: {e}")
            # else:
            #     print(f"      ⚠ OpenAQ: No API key")
            
            # ==================== 2. OPEN-METEO HISTORICAL ====================
            print("  [2/6] Fetching Open-Meteo historical air quality...")
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                url = f"{self.openmeteo_base}/air-quality"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi,european_aqi,dust,uv_index",
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "timezone": "auto"
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                hourly = data.get('hourly', {})
                openmeteo_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(hourly.get('time', [])),
                    'aqi': hourly.get('us_aqi', []),
                    'pm25': hourly.get('pm2_5', []),
                    'pm10': hourly.get('pm10', []),
                    'no2': hourly.get('nitrogen_dioxide', []),
                    'o3': hourly.get('ozone', []),
                    'so2': hourly.get('sulphur_dioxide', []),
                    'co': hourly.get('carbon_monoxide', []),
                    'eu_aqi': hourly.get('european_aqi', []),
                    'dust': hourly.get('dust', []),
                    'uv_index': hourly.get('uv_index', []),
                })
                
                openmeteo_df['city'] = city_name
                openmeteo_df['lat'] = lat
                openmeteo_df['lon'] = lon
                openmeteo_df['source'] = 'Open-Meteo'
                all_records.append(openmeteo_df)
                print(f"      ✓ Open-Meteo AQ: {len(openmeteo_df)} records")
            except Exception as e:
                print(f"      ✗ Open-Meteo AQ error: {e}")
            
            # ==================== 3. OPEN-METEO WEATHER ====================
            print("  [3/6] Fetching Open-Meteo weather data...")
            try:
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure,cloud_cover",
                    "timezone": "auto"
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                hourly = data.get('hourly', {})
                weather_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(hourly.get('time', [])),
                    'temperature': hourly.get('temperature_2m', []),
                    'humidity': hourly.get('relative_humidity_2m', []),
                    'precipitation': hourly.get('precipitation', []),
                    'wind_speed': hourly.get('wind_speed_10m', []),
                    'wind_direction': hourly.get('wind_direction_10m', []),
                    'pressure': hourly.get('surface_pressure', []),
                    'cloud_cover': hourly.get('cloud_cover', []),
                })
                
                weather_df['city'] = city_name
                weather_df['lat'] = lat
                weather_df['lon'] = lon
                all_records.append(weather_df)
                print(f"      ✓ Weather: {len(weather_df)} records")
            except Exception as e:
                print(f"      ✗ Weather error: {e}")
            
            # # ==================== 4. OPENWEATHER FORECAST ====================
            # print("  [4/6] Fetching OpenWeather forecast...")
            # if self.openweather_key:
            #     try:
            #         forecasts = self.fetch_openweather_forecast(lat, lon)
            #         if forecasts:
            #             ow_forecast_df = pd.DataFrame(forecasts)
            #             ow_forecast_df['timestamp'] = pd.to_datetime(ow_forecast_df['timestamp'])
            #             ow_forecast_df['city'] = city_name
            #             ow_forecast_df['lat'] = lat
            #             ow_forecast_df['lon'] = lon
            #             ow_forecast_df['source'] = 'OpenWeather_Forecast'
            #             ow_forecast_df = ow_forecast_df.rename(columns={
            #                 'aqi': 'aqi_ow',
            #                 'pm25': 'pm25_ow',
            #                 'pm10': 'pm10_ow',
            #                 'no2': 'no2_ow'
            #             })
            #             all_records.append(ow_forecast_df)
            #             print(f"      ✓ OpenWeather Forecast: {len(ow_forecast_df)} records")
            #         else:
            #             print(f"      ✗ OpenWeather Forecast: No data")
            #     except Exception as e:
            #         print(f"      ✗ OpenWeather Forecast error: {e}")
            # else:
            #     print(f"      ⚠ OpenWeather: No API key")
            
            # ==================== 5. OPEN-METEO FORECAST ====================
            print("  [5/6] Fetching Open-Meteo forecast...")
            try:
                om_forecasts = self.fetch_openmeteo_forecast(lat, lon, days=5)
                if om_forecasts:
                    om_forecast_df = pd.DataFrame(om_forecasts)
                    om_forecast_df['timestamp'] = pd.to_datetime(om_forecast_df['timestamp'])
                    om_forecast_df['city'] = city_name
                    om_forecast_df['lat'] = lat
                    om_forecast_df['lon'] = lon
                    om_forecast_df['source'] = 'Open-Meteo_Forecast'
                    om_forecast_df = om_forecast_df.rename(columns={
                        'pm25': 'pm25_forecast',
                        'aqi': 'aqi_forecast'
                    })
                    all_records.append(om_forecast_df)
                    print(f"      ✓ Open-Meteo Forecast: {len(om_forecast_df)} records")
                else:
                    print(f"      ✗ Open-Meteo Forecast: No data")
            except Exception as e:
                print(f"      ✗ Open-Meteo Forecast error: {e}")
            
            # ==================== 6. CURRENT DATA FROM ALL SOURCES ====================
            # print("  [6/6] Fetching current conditions from all sources...")
            # try:
            #     current_data = self.fetch_best_available(city_name, lat, lon)
            #     if current_data.get('data') and current_data['data'].get('individual_sources'):
            #         current_records = []
                    
            #         for source_name, source_data in current_data['data']['individual_sources'].items():
            #             if isinstance(source_data, dict):
            #                 record = {
            #                     'timestamp': pd.Timestamp.now(),
            #                     'city': city_name,
            #                     'lat': lat,
            #                     'lon': lon,
            #                     'source': f'Current_{source_name}'
            #                 }
                            
            #                 # Add all available fields from source
            #                 for key, value in source_data.items():
            #                     if key not in ['source', 'timestamp', 'aqi_category', 'station']:
            #                         record[f'{key}_{source_name}'] = value
                            
            #                 current_records.append(record)
                    
            #         if current_records:
            #             current_df = pd.DataFrame(current_records)
            #             all_records.append(current_df)
            #             print(f"      ✓ Current data: {len(current_records)} source(s)")
            # except Exception as e:
            #     print(f"      ✗ Current data error: {e}")
            
            # Rate limiting between cities
            if idx < len(cities):
                time.sleep(2)
        
        # ==================== COMBINE ALL DATA ====================
        print("\n" + "="*70)
        print("📊 COMBINING DATA FROM ALL SOURCES...")
        print("="*70)
        
        if not all_records:
            print("⚠️  No data collected from any source")
            return pd.DataFrame()
        
        # Merge all dataframes
        print(f"Merging {len(all_records)} dataframes...")
        
        # Group by city
        city_dfs = {}
        for df in all_records:
            if df.empty:
                continue
            city = df['city'].iloc[0]
            if city not in city_dfs:
                city_dfs[city] = []
            city_dfs[city].append(df)
        
        # Merge each city's data
        merged_city_dfs = []
        for city, dfs in city_dfs.items():
            print(f"\n  Merging data for {city}...")
            
            # Start with first df
            merged = dfs[0].copy()
            
            # Merge rest
            for df in dfs[1:]:
                if 'timestamp' in df.columns and 'timestamp' in merged.columns:
                    merged = pd.merge(
                        merged,
                        df,
                        on=['timestamp', 'city', 'lat', 'lon'],
                        how='outer',
                        suffixes=('', '_dup')
                    )
                    
                    # Remove duplicate columns
                    dup_cols = [col for col in merged.columns if col.endswith('_dup')]
                    if dup_cols:
                        merged = merged.drop(columns=dup_cols)
            
            merged_city_dfs.append(merged)
            print(f"    ✓ {city}: {len(merged)} records")
        
        # Combine all cities
        combined_df = pd.concat(merged_city_dfs, ignore_index=True)
        
        # ==================== ADD TEMPORAL FEATURES ====================
        print("\n⏰ Adding temporal features...")
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        
        # Time components
        combined_df['year'] = combined_df['timestamp'].dt.year
        combined_df['month'] = combined_df['timestamp'].dt.month
        combined_df['day'] = combined_df['timestamp'].dt.day
        combined_df['hour'] = combined_df['timestamp'].dt.hour
        combined_df['dayofweek'] = combined_df['timestamp'].dt.dayofweek
        combined_df['dayofyear'] = combined_df['timestamp'].dt.dayofyear
        combined_df['week'] = combined_df['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding
        combined_df['hour_sin'] = np.sin(2 * np.pi * combined_df['hour'] / 24)
        combined_df['hour_cos'] = np.cos(2 * np.pi * combined_df['hour'] / 24)
        combined_df['month_sin'] = np.sin(2 * np.pi * combined_df['month'] / 12)
        combined_df['month_cos'] = np.cos(2 * np.pi * combined_df['month'] / 12)
        combined_df['day_sin'] = np.sin(2 * np.pi * combined_df['dayofweek'] / 7)
        combined_df['day_cos'] = np.cos(2 * np.pi * combined_df['dayofweek'] / 7)
        
        # Categorical features
        combined_df['is_weekend'] = (combined_df['dayofweek'] >= 5).astype(int)
        combined_df['is_rush_hour'] = ((combined_df['hour'] >= 7) & (combined_df['hour'] <= 9) | 
                                        (combined_df['hour'] >= 17) & (combined_df['hour'] <= 19)).astype(int)
        combined_df['season'] = (combined_df['month'] % 12 + 3) // 3
        
        # ==================== CLEAN DATASET ====================
        print("🧹 Cleaning dataset...")
        
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=['timestamp', 'city'], keep='first')
        
        # Sort by timestamp
        combined_df = combined_df.sort_values(['city', 'timestamp']).reset_index(drop=True)
        
        # Handle missing values for key pollutants
        pollutant_patterns = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co', 'aqi']
        pollutant_cols = [col for col in combined_df.columns 
                         if any(pattern in col.lower() for pattern in pollutant_patterns)]
        
        # Forward and backward fill within each city
        for col in pollutant_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df.groupby('city')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
        
        # Fill remaining numeric columns with median
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if combined_df[col].isnull().any():
                median_val = combined_df[col].median()
                combined_df[col] = combined_df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        # Remove outliers (cap at 99.9th percentile)
        for col in pollutant_cols:
            if col in combined_df.columns and combined_df[col].std() > 0:
                threshold = combined_df[col].quantile(0.999)
                combined_df.loc[combined_df[col] > threshold, col] = threshold
        
        # ==================== SUMMARY ====================
        print("\n" + "="*70)
        print("✅ DATASET READY FOR TRAINING")
        print("="*70)
        print(f"Total records: {len(combined_df):,}")
        print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"Cities: {combined_df['city'].nunique()}")
        print(f"Features: {len(combined_df.columns)}")
        print(f"\nSources used: {combined_df['source'].nunique() if 'source' in combined_df.columns else 'N/A'}")
        
        # Show available pollutant columns
        print(f"\nPollutant columns ({len(pollutant_cols)}):")
        for col in sorted(pollutant_cols)[:10]:
            print(f"  - {col}")
        if len(pollutant_cols) > 10:
            print(f"  ... and {len(pollutant_cols) - 10} more")
        
        print(f"\nMissing values:")
        missing = combined_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            print(missing.head(10))
        else:
            print("  None!")
        
        print("="*70 + "\n")
        
        return combined_df

    def _aggregate_sources(self, sources_data: Dict) -> Dict:
        """Aggregate data from multiple sources"""
        aqis, pm25s, pm10s, no2s = [], [], [], []
        
        for source_name, data in sources_data.items():
            if isinstance(data, dict):
                if data.get('aqi'):
                    aqis.append(data['aqi'])
                if data.get('pm25'):
                    pm25s.append(data['pm25'])
                if data.get('pm10'):
                    pm10s.append(data['pm10'])
                if data.get('no2'):
                    no2s.append(data['no2'])
        
        avg_aqi = round(sum(aqis) / len(aqis)) if aqis else 0
        
        return {
            'aqi': avg_aqi,
            'aqi_category': self._get_aqi_category(avg_aqi),
            'pm25': round(sum(pm25s) / len(pm25s), 2) if pm25s else 0,
            'pm10': round(sum(pm10s) / len(pm10s), 2) if pm10s else 0,
            'no2': round(sum(no2s) / len(no2s), 2) if no2s else 0,
            'sources_count': len(sources_data),
            'individual_sources': sources_data
        }
    
    def _get_aqi_category(self, aqi: int) -> str:
        """Convert AQI to category"""
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


# ==================== Usage Example ====================

if __name__ == "__main__":
    fetcher = AirQualityDataFetcher()
    
    # Test with Cairo
    cairo_data = fetcher.fetch_best_available(
        city="Cairo",
        lat=30.0444,
        lon=31.2357
    )
    
    print("\n" + "="*70)
    print("CAIRO AIR QUALITY SUMMARY")
    print("="*70)
    print(f"Sources: {', '.join(cairo_data['sources'])}")
    
    if cairo_data['data']:
        data = cairo_data['data']
        print(f"\n📊 Aggregated Results:")
        print(f"  AQI:   {data['aqi']} ({data['aqi_category']})")
        print(f"  PM2.5: {data['pm25']} μg/m³")
        print(f"  PM10:  {data['pm10']} μg/m³")
        print(f"  NO2:   {data['no2']} μg/m³")
    
    print("\n" + "="*70)

    cities = [
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "country": "EG"},
    ]
    # Fetch training dataset
    df = fetcher.fetch_training_dataset(
        cities,
        days=30
    )
    # Save combined training data
    data_path = settings.RAW_DATA_DIR / 'multi_city_training_data.csv'
    df.to_csv(data_path, index=False)
    print(f"\n✓ Multi-city training data saved to {data_path}")
    