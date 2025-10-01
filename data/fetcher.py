"""
Data Fetcher for AirGuardian API
Fetches data from NASA TEMPO, OpenAQ, and weather APIs
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class AirQualityDataFetcher:
    """Fetch air quality data from multiple sources"""
    
    def __init__(self):
        self.openaq_base = "https://api.openaq.org/v2"
        self.nasa_earthdata_token = os.getenv("NASA_EARTHDATA_TOKEN")
        self.openweather_key = os.getenv("OPENWEATHER_API_KEY")
    
    # ==================== OpenAQ Integration ====================
    
    def fetch_openaq_latest(self, city: str, country: str = "EG") -> Dict:
        """
        Fetch latest measurements from OpenAQ
        
        Args:
            city: City name (e.g., "Cairo")
            country: Country code (default: "EG" for Egypt)
            
        Returns:
            Dictionary with averaged measurements by parameter
        """
        url = f"{self.openaq_base}/latest"
        params = {
            "city": city,
            "country": country,
            "limit": 100
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['results']:
                return self._process_openaq_data(data['results'])
            else:
                print(f"No data found for {city}, {country}")
                return None
        except Exception as e:
            print(f"Error fetching OpenAQ data: {e}")
            return None
    
    def _process_openaq_data(self, results: List) -> Dict:
        """Process OpenAQ results into usable format"""
        measurements = {}
        
        for result in results:
            for measurement in result.get('measurements', []):
                parameter = measurement['parameter']
                value = measurement['value']
                unit = measurement['unit']
                
                if parameter not in measurements:
                    measurements[parameter] = []
                
                measurements[parameter].append({
                    'value': value,
                    'unit': unit,
                    'location': result['location'],
                    'timestamp': measurement.get('lastUpdated')
                })
        
        # Calculate averages
        averaged = {}
        for param, values in measurements.items():
            avg_value = sum([v['value'] for v in values]) / len(values)
            averaged[param] = {
                'value': round(avg_value, 2),
                'unit': values[0]['unit'],
                'sample_count': len(values)
            }
        
        return averaged
    
    def fetch_openaq_historical(self, city: str, days: int = 30, 
                               country: str = "EG") -> pd.DataFrame:
        """
        Fetch historical data for model training
        
        Args:
            city: City name
            days: Number of days of historical data
            country: Country code
            
        Returns:
            DataFrame with historical measurements
        """
        url = f"{self.openaq_base}/measurements"
        
        date_to = datetime.now()
        date_from = date_to - timedelta(days=days)
        
        params = {
            "city": city,
            "country": country,
            "date_from": date_from.isoformat(),
            "date_to": date_to.isoformat(),
            "limit": 10000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                print(f"No historical data found for {city}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            records = []
            for result in data['results']:
                records.append({
                    'timestamp': result['date']['utc'],
                    'parameter': result['parameter'],
                    'value': result['value'],
                    'unit': result['unit'],
                    'location': result['location'],
                    'city': result.get('city', city),
                    'country': result.get('country', country)
                })
            
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print(f"Fetched {len(df)} historical measurements for {city}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def fetch_openaq_by_coordinates(self, lat: float, lon: float, 
                                   radius: int = 25000) -> Dict:
        """
        Fetch data by coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
            radius: Radius in meters (default 25km)
            
        Returns:
            Latest measurements near coordinates
        """
        url = f"{self.openaq_base}/latest"
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": radius,
            "limit": 100
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['results']:
                return self._process_openaq_data(data['results'])
            return None
            
        except Exception as e:
            print(f"Error fetching data by coordinates: {e}")
            return None
    
    # ==================== Weather Data ====================
    
    def fetch_weather_data(self, lat: float, lon: float) -> Dict:
        """
        Fetch current weather data from OpenWeather
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Weather data dictionary
        """
        if not self.openweather_key:
            print("OpenWeather API key not found, using mock data")
            return self._mock_weather_data()
        
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.openweather_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "temperature": data['main']['temp'],
                "humidity": data['main']['humidity'],
                "pressure": data['main']['pressure'],
                "wind_speed": data['wind']['speed'],
                "wind_direction": data['wind'].get('deg', 0),
                "description": data['weather'][0]['description'],
                "clouds": data.get('clouds', {}).get('all', 0),
                "visibility": data.get('visibility', 10000)
            }
            
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return self._mock_weather_data()
    
    def fetch_weather_forecast(self, lat: float, lon: float) -> List[Dict]:
        """
        Fetch weather forecast
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            List of hourly forecasts
        """
        if not self.openweather_key:
            return []
        
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.openweather_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data.get('list', []):
                forecasts.append({
                    'timestamp': item['dt_txt'],
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'wind_speed': item['wind']['speed'],
                    'description': item['weather'][0]['description']
                })
            
            return forecasts
            
        except Exception as e:
            print(f"Error fetching weather forecast: {e}")
            return []
    
    def _mock_weather_data(self) -> Dict:
        """Mock weather data for testing"""
        return {
            "temperature": 28,
            "humidity": 45,
            "pressure": 1013,
            "wind_speed": 3.5,
            "wind_direction": 180,
            "description": "clear sky",
            "clouds": 10,
            "visibility": 10000
        }
    
    # ==================== NASA TEMPO Data ====================
    
    def fetch_tempo_no2(self, lat: float, lon: float, date: str = None) -> Optional[float]:
        """
        Fetch NO2 data from NASA TEMPO
        
        Note: This requires NASA Earthdata credentials
        Register at: https://urs.earthdata.nasa.gov/
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            NO2 concentration in ppb
        """
        if not self.nasa_earthdata_token:
            print("NASA Earthdata token not found. Using mock data.")
            return self._mock_tempo_data()
        
        # TEMPO data endpoint
        # Note: Actual implementation depends on NASA's specific API
        # For now, this is a placeholder
        
        # Example API call (adjust based on actual TEMPO API):
        # url = "https://disc.gsfc.nasa.gov/api/tempo/no2"
        # headers = {"Authorization": f"Bearer {self.nasa_earthdata_token}"}
        # params = {"lat": lat, "lon": lon, "date": date or datetime.now().strftime("%Y-%m-%d")}
        
        # For hackathon, use mock data or pre-downloaded files
        return self._mock_tempo_data()
    
    def _mock_tempo_data(self) -> float:
        """Mock TEMPO data for testing"""
        import random
        return round(random.uniform(20, 80), 2)  # NO2 in ppb
    
    # ==================== Combined Data Fetch ====================
    
    def fetch_complete_data(self, city: str, lat: float, lon: float) -> Dict:
        """
        Fetch all available data for a location
        
        Args:
            city: City name
            lat: Latitude
            lon: Longitude
            
        Returns:
            Combined data from all sources
        """
        result = {
            "city": city,
            "coordinates": {"lat": lat, "lon": lon},
            "timestamp": datetime.now().isoformat(),
            "data_sources": []
        }
        
        # Fetch OpenAQ data
        print(f"Fetching OpenAQ data for {city}...")
        openaq_data = self.fetch_openaq_latest(city)
        if openaq_data:
            result["openaq"] = openaq_data
            result["data_sources"].append("OpenAQ")
        
        # Fetch weather data
        print(f"Fetching weather data...")
        weather_data = self.fetch_weather_data(lat, lon)
        if weather_data:
            result["weather"] = weather_data
            result["data_sources"].append("OpenWeather")
        
        # Fetch TEMPO data
        print(f"Fetching TEMPO data...")
        tempo_no2 = self.fetch_tempo_no2(lat, lon)
        if tempo_no2:
            result["tempo"] = {"no2": tempo_no2, "unit": "ppb"}
            result["data_sources"].append("NASA TEMPO")
        
        return result
    
    def fetch_training_dataset(self, city: str, lat: float, lon: float, 
                               days: int = 90) -> pd.DataFrame:
        """
        Fetch complete training dataset combining all sources
        
        Args:
            city: City name
            lat: Latitude
            lon: Longitude
            days: Days of historical data
            
        Returns:
            Combined DataFrame ready for model training
        """
        print(f"\n{'='*60}")
        print(f"Fetching training dataset for {city}")
        print(f"{'='*60}\n")
        
        # Fetch historical air quality data
        df = self.fetch_openaq_historical(city, days=days)
        
        if df.empty:
            print("No historical data available")
            return pd.DataFrame()
        
        # Pivot to get one row per timestamp with columns for each parameter
        df_pivot = df.pivot_table(
            index='timestamp',
            columns='parameter',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Add weather data (sample at regular intervals)
        print("Adding weather data...")
        weather_samples = []
        
        for idx in range(0, len(df_pivot), 24):  # Sample daily
            timestamp = df_pivot.iloc[idx]['timestamp']
            weather = self.fetch_weather_data(lat, lon)
            
            weather_samples.append({
                'timestamp': timestamp,
                **weather
            })
        
        weather_df = pd.DataFrame(weather_samples)
        
        # Merge with air quality data
        df_combined = pd.merge(
            df_pivot,
            weather_df,
            on='timestamp',
            how='left'
        )
        
        # Forward fill weather data
        df_combined = df_combined.fillna(method='ffill')
        
        # Calculate AQI if pm25 exists
        if 'pm25' in df_combined.columns:
            from utils.helpers import calculate_aqi_from_pm25
            df_combined['aqi'] = df_combined['pm25'].apply(calculate_aqi_from_pm25)
        
        print(f"\nDataset summary:")
        print(f"  Rows: {len(df_combined)}")
        print(f"  Columns: {list(df_combined.columns)}")
        print(f"  Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
        print(f"  Missing values: {df_combined.isnull().sum().sum()}")
        
        return df_combined


# ==================== Utility Functions ====================

def save_data_to_csv(df: pd.DataFrame, filename: str, directory: str = "data_storage/raw"):
    """Save fetched data to CSV"""
    import os
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Initialize fetcher
    fetcher = AirQualityDataFetcher()
    
    # Egyptian cities coordinates
    cities = {
        "Cairo": (30.0444, 31.2357),
        "Alexandria": (31.2001, 29.9187),
        "Giza": (30.0131, 31.2089)
    }
    
    # Example 1: Fetch current data for Cairo
    print("\n" + "="*60)
    print("Example 1: Fetching current data for Cairo")
    print("="*60)
    
    cairo_data = fetcher.fetch_complete_data("Cairo", *cities["Cairo"])
    
    print(f"\nCairo Air Quality Data:")
    print(f"Timestamp: {cairo_data['timestamp']}")
    print(f"Data Sources: {', '.join(cairo_data['data_sources'])}")
    
    if 'openaq' in cairo_data:
        print("\nOpenAQ Measurements:")
        for param, data in cairo_data['openaq'].items():
            print(f"  {param}: {data['value']} {data['unit']}")
    
    if 'weather' in cairo_data:
        print("\nWeather Conditions:")
        print(f"  Temperature: {cairo_data['weather']['temperature']}°C")
        print(f"  Humidity: {cairo_data['weather']['humidity']}%")
        print(f"  Wind Speed: {cairo_data['weather']['wind_speed']} m/s")
    
    if 'tempo' in cairo_data:
        print("\nNASA TEMPO Data:")
        print(f"  NO2: {cairo_data['tempo']['no2']} {cairo_data['tempo']['unit']}")
    
    # Example 2: Fetch historical data
    print("\n" + "="*60)
    print("Example 2: Fetching historical data (30 days)")
    print("="*60)
    
    historical_df = fetcher.fetch_openaq_historical("Cairo", days=30)
    
    if not historical_df.empty:
        print(f"\nRetrieved {len(historical_df)} measurements")
        print(f"Parameters: {historical_df['parameter'].unique()}")
        print(f"Date range: {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}")
        
        # Save to CSV
        save_data_to_csv(historical_df, "cairo_historical_30days.csv")
    
    # Example 3: Fetch complete training dataset
    print("\n" + "="*60)
    print("Example 3: Fetching training dataset (90 days)")
    print("="*60)
    
    training_df = fetcher.fetch_training_dataset("Cairo", *cities["Cairo"], days=7)  # Use 7 for demo
    
    if not training_df.empty:
        print("\nTraining dataset ready!")
        print(training_df.head())
        
        # Save for model training
        save_data_to_csv(training_df, "cairo_training_dataset.csv", "data_storage/processed")