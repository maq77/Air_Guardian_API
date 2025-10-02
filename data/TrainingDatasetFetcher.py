"""
Training Dataset Fetcher for Air Quality Prediction Model
Fetches historical data from multiple sources and prepares it for ML training
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import time

import requests

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetcher import AirQualityDataFetcher


class TrainingDatasetFetcher(AirQualityDataFetcher):
    """Extended fetcher for creating training datasets"""
    
    def fetch_training_dataset(
        self,
        cities: List[dict],
        days: int = 90,
        include_weather: bool = True,
        include_temporal: bool = True
    ) -> pd.DataFrame:
        """
        Fetch comprehensive training dataset
        
        Args:
            cities: List of dicts with 'name', 'lat', 'lon', 'country'
            days: Number of days of historical data
            include_weather: Include weather features
            include_temporal: Include time-based features
            
        Returns:
            DataFrame ready for model training
        """
        print("\n" + "="*70)
        print("🎯 FETCHING TRAINING DATASET")
        print("="*70)
        print(f"Cities: {len(cities)}")
        print(f"Time range: {days} days")
        print(f"Weather features: {include_weather}")
        print(f"Temporal features: {include_temporal}")
        print("="*70 + "\n")
        
        all_data = []
        
        for idx, city in enumerate(cities, 1):
            print(f"\n[{idx}/{len(cities)}] Processing {city['name']}, {city.get('country', 'Unknown')}...")
            
            city_data = self._fetch_city_historical(
                city=city,
                days=days,
                include_weather=include_weather
            )
            
            if not city_data.empty:
                all_data.append(city_data)
                print(f"  ✓ Retrieved {len(city_data)} records")
            else:
                print(f"  ✗ No data available")
            
            # Rate limiting
            time.sleep(1)
        
        if not all_data:
            print("\n⚠️  No data collected from any city")
            return pd.DataFrame()
        
        # Combine all data
        print("\n📊 Combining data from all cities...")
        df = pd.concat(all_data, ignore_index=True)
        
        # Add temporal features
        if include_temporal:
            print("⏰ Adding temporal features...")
            df = self._add_temporal_features(df)
        
        # Clean and prepare
        print("🧹 Cleaning dataset...")
        df = self._clean_dataset(df)
        
        # Summary
        print("\n" + "="*70)
        print("✅ DATASET READY")
        print("="*70)
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Cities: {df['city'].nunique()}")
        print(f"Features: {len(df.columns)}")
        print(f"\nColumns: {', '.join(df.columns.tolist())}")
        print("="*70 + "\n")
        
        return df
    
    def _fetch_city_historical(
        self,
        city: dict,
        days: int,
        include_weather: bool
    ) -> pd.DataFrame:
        """Fetch historical data for a single city"""
        
        all_records = []
        
        # 1. OpenAQ Historical Data (most reliable for historical)
        if self.openaq_key:
            print("  Fetching OpenAQ historical...")
            openaq_df = self.fetch_openaq_historical(
                city=city['name'],
                days=days,
                country=city.get('country', 'EG')
            )
            
            if not openaq_df.empty:
                # Pivot to wide format
                pivot_df = openaq_df.pivot_table(
                    index=['timestamp', 'city'],
                    columns='parameter',
                    values='value',
                    aggfunc='mean'
                ).reset_index()
                
                pivot_df['source'] = 'OpenAQ'
                pivot_df['lat'] = city['lat']
                pivot_df['lon'] = city['lon']
                all_records.append(pivot_df)
                print(f"    ✓ OpenAQ: {len(pivot_df)} records")
        
        # 2. Open-Meteo Historical (free, reliable)
        print("  Fetching Open-Meteo historical...")
        openmeteo_df = self._fetch_openmeteo_historical(
            lat=city['lat'],
            lon=city['lon'],
            days=days
        )
        
        if not openmeteo_df.empty:
            openmeteo_df['city'] = city['name']
            openmeteo_df['source'] = 'Open-Meteo'
            all_records.append(openmeteo_df)
            print(f"    ✓ Open-Meteo: {len(openmeteo_df)} records")
        
        # 3. Weather data (if requested)
        if include_weather:
            print("  Fetching weather data...")
            weather_df = self._fetch_weather_historical(
                lat=city['lat'],
                lon=city['lon'],
                days=days
            )
            
            if not weather_df.empty:
                weather_df['city'] = city['name']
                all_records.append(weather_df)
                print(f"    ✓ Weather: {len(weather_df)} records")
        
        if not all_records:
            return pd.DataFrame()
        
        # Merge all sources by timestamp
        combined_df = self._merge_city_data(all_records)
        return combined_df
    
    def _fetch_openmeteo_historical(
        self,
        lat: float,
        lon: float,
        days: int
    ) -> pd.DataFrame:
        """Fetch historical air quality from Open-Meteo"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.openmeteo_base}/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi,european_aqi",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "timezone": "auto"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get('hourly', {})
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(hourly.get('time', [])),
                'pm25': hourly.get('pm2_5', []),
                'pm10': hourly.get('pm10', []),
                'no2': hourly.get('nitrogen_dioxide', []),
                'o3': hourly.get('ozone', []),
                'so2': hourly.get('sulphur_dioxide', []),
                'co': hourly.get('carbon_monoxide', []),
                'aqi': hourly.get('us_aqi', []),
                'european_aqi': hourly.get('european_aqi', []),
                'lat': lat,
                'lon': lon
            })
            
            return df
            
        except Exception as e:
            print(f"    ✗ Open-Meteo error: {e}")
            return pd.DataFrame()
    
    def _fetch_weather_historical(
        self,
        lat: float,
        lon: float,
        days: int
    ) -> pd.DataFrame:
        """Fetch historical weather data from Open-Meteo"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure",
            "timezone": "auto"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get('hourly', {})
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(hourly.get('time', [])),
                'temperature': hourly.get('temperature_2m', []),
                'humidity': hourly.get('relative_humidity_2m', []),
                'precipitation': hourly.get('precipitation', []),
                'wind_speed': hourly.get('wind_speed_10m', []),
                'wind_direction': hourly.get('wind_direction_10m', []),
                'pressure': hourly.get('surface_pressure', []),
                'lat': lat,
                'lon': lon
            })
            
            return df
            
        except Exception as e:
            print(f"    ✗ Weather error: {e}")
            return pd.DataFrame()
    
    def _merge_city_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple dataframes for a city"""
        
        if not dataframes:
            return pd.DataFrame()
        
        # Start with first dataframe
        merged = dataframes[0].copy()
        
        # Merge others on timestamp
        for df in dataframes[1:]:
            if 'timestamp' in df.columns:
                merged = pd.merge(
                    merged,
                    df,
                    on=['timestamp', 'city'],
                    how='outer',
                    suffixes=('', '_dup')
                )
                
                # Remove duplicate columns
                dup_cols = [col for col in merged.columns if col.endswith('_dup')]
                merged = merged.drop(columns=dup_cols)
        
        return merged
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Date/time components
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['dayofyear'] = df['timestamp'].dt.dayofyear
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Categorical features
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                              (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        df['season'] = (df['month'] % 12 + 3) // 3
        
        return df
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataset"""
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp', 'city'], keep='first')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle missing values for key pollutants
        pollutant_cols = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co', 'aqi']
        
        for col in pollutant_cols:
            if col in df.columns:
                # Forward fill then backward fill
                df[col] = df.groupby('city')[col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with all pollutants missing
        if any(col in df.columns for col in pollutant_cols):
            existing_pollutants = [col for col in pollutant_cols if col in df.columns]
            df = df.dropna(subset=existing_pollutants, how='all')
        
        # Fill remaining numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove outliers (values > 99.9th percentile for pollutants)
        for col in pollutant_cols:
            if col in df.columns and df[col].std() > 0:
                threshold = df[col].quantile(0.999)
                df.loc[df[col] > threshold, col] = threshold
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "aq_training_data.csv"):
        """Save dataset to CSV"""
        df.to_csv(filename, index=False)
        print(f"\n💾 Dataset saved to: {filename}")
        print(f"   Size: {len(df):,} rows × {len(df.columns)} columns")


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Initialize fetcher
    fetcher = TrainingDatasetFetcher()
    
    # Define cities for training
    cities = [
        {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "country": "EG"},
        {"name": "Alexandria", "lat": 31.2001, "lon": 29.9187, "country": "EG"},
        {"name": "Giza", "lat": 30.0131, "lon": 31.2089, "country": "EG"},
    ]
    
    # Fetch training dataset
    training_df = fetcher.fetch_training_dataset(
        cities=cities,
        days=90,  # 3 months of data
        include_weather=True,
        include_temporal=True
    )
    
    # Display sample
    if not training_df.empty:
        print("\n📋 Dataset Preview:")
        print(training_df.head(10))
        
        print("\n📊 Dataset Statistics:")
        print(training_df.describe())
        
        print("\n🔍 Missing Values:")
        print(training_df.isnull().sum())
        
        # Save dataset
        fetcher.save_dataset(training_df, "aq_training_data.csv")
    else:
        print("\n⚠️  Failed to create dataset")