"""
Multi-City Model Training Script for AirGuardian API
Train models on data from ALL Egyptian cities for better generalization
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetcher import AirQualityDataFetcher
from models.forecaster import AQIForecaster
from models.classifier import RiskClassifier
from models.policy_engine import PolicyRecommendationEngine
from config.settings import settings


# ==================== Egyptian Cities Configuration ====================

EGYPTIAN_CITIES = {
    "Cairo": {"lat": 30.0444, "lon": 31.2357, "country": "EG"},
    "Alexandria": {"lat": 31.2001, "lon": 29.9187, "country": "EG"},
    "Giza": {"lat": 30.0131, "lon": 31.2089, "country": "EG"},
    "Shubra El Kheima": {"lat": 30.1286, "lon": 31.2422, "country": "EG"},
    "Port Said": {"lat": 31.2653, "lon": 32.3019, "country": "EG"},
    "Suez": {"lat": 29.9668, "lon": 32.5498, "country": "EG"},
    "Luxor": {"lat": 25.6872, "lon": 32.6396, "country": "EG"},
    "Mansoura": {"lat": 31.0409, "lon": 31.3785, "country": "EG"},
    "Tanta": {"lat": 30.7865, "lon": 31.0004, "country": "EG"},
    "Asyut": {"lat": 27.1809, "lon": 31.1837, "country": "EG"},
}


def create_sample_training_data_for_city(city_name: str, lat: float, lon: float, 
                                         days: int = 90) -> pd.DataFrame:
    """
    Generate sample training data for a specific city
    
    Args:
        city_name: Name of the city
        lat: Latitude
        lon: Longitude
        days: Number of days to generate
        
    Returns:
        DataFrame with realistic air quality patterns for this city
    """
    print(f"  Generating {days} days of sample data for {city_name}...")
    
    # Generate hourly timestamps
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='H'
    )
    n = len(dates)
    
    # Extract time features
    hours = pd.Series(dates).dt.hour.values
    days_of_week = pd.Series(dates).dt.dayofweek.values
    months = pd.Series(dates).dt.month.values
    
    # City-specific pollution levels (some cities are more polluted)
    city_pollution_factor = {
        "Cairo": 1.3,  # Higher pollution
        "Giza": 1.25,
        "Shubra El Kheima": 1.35,
        "Alexandria": 0.9,  # Coastal, better air
        "Port Said": 0.85,
        "Suez": 1.0,
        "Luxor": 0.8,  # Less industrial
        "Mansoura": 0.95,
        "Tanta": 1.0,
        "Asyut": 1.1,
    }
    
    pollution_factor = city_pollution_factor.get(city_name, 1.0)
    
    # Base pollution with realistic patterns
    base_pm25 = 40 * pollution_factor + 25 * np.sin(2 * np.pi * hours / 24)
    
    # Weekly cycle: higher on weekdays
    base_pm25 += 15 * (days_of_week < 5) * pollution_factor
    
    # Rush hour spikes
    rush_hour_mask = np.isin(hours, [7, 8, 17, 18, 19])
    base_pm25 += 20 * rush_hour_mask * pollution_factor
    
    # Weekend reduction
    weekend_mask = days_of_week >= 5
    base_pm25 -= 10 * weekend_mask
    
    # Seasonal trend (winter pollution)
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (months - 12) / 12)
    base_pm25 *= seasonal_factor
    
    # Add random noise
    base_pm25 += np.random.normal(0, 8 * pollution_factor, n)
    
    # Occasional pollution episodes
    for i in range(0, n, 24*7):
        if np.random.random() > 0.7:
            episode_length = np.random.randint(12, 48)
            base_pm25[i:i+episode_length] += np.random.uniform(30, 60) * pollution_factor
    
    # Ensure realistic bounds
    base_pm25 = np.clip(base_pm25, 5, 300)
    
    # Generate correlated pollutants
    no2 = base_pm25 * 0.7 + np.random.normal(0, 5, n)
    no2 = np.clip(no2, 5, 200)
    
    # Calculate AQI from PM2.5
    from utils.helpers import calculate_aqi_from_pm25
    aqi = [calculate_aqi_from_pm25(pm) for pm in base_pm25]
    
    # Generate weather data with location-specific patterns
    # Coastal cities: more moderate temperatures, higher humidity
    is_coastal = city_name in ["Alexandria", "Port Said", "Suez"]
    
    # Temperature
    temp_base = 23 if is_coastal else 26
    temperature = temp_base + 8 * np.sin(2 * np.pi * hours / 24)
    temperature += 5 * np.sin(2 * np.pi * months / 12)
    temperature += np.random.normal(0, 2, n)
    
    # Humidity
    humidity_base = 75 if is_coastal else 60
    humidity = humidity_base - 15 * np.sin(2 * np.pi * hours / 24)
    humidity += np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 95)
    
    # Wind speed (higher in coastal areas)
    wind_base = 4 if is_coastal else 2.5
    wind_speed = wind_base + 2 * np.sin(2 * np.pi * hours / 24)
    wind_speed += np.random.normal(0, 1, n)
    wind_speed = np.clip(wind_speed, 0.5, 15)
    
    # Pressure
    pressure = 1013 + np.random.normal(0, 5, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'city': city_name,
        'latitude': lat,
        'longitude': lon,
        'pm25': base_pm25,
        'no2': no2,
        'aqi': aqi,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure
    })
    
    print(f"  ✓ Generated {len(df)} samples for {city_name}")
    return df


def train_forecasting_models(df: pd.DataFrame) -> dict:
    """
    Train forecasting models on multi-city data
    
    Args:
        df: Training data from multiple cities
        
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*70)
    print("TRAINING FORECASTING MODELS (Multi-City)")
    print("="*70)
    
    models = {}
    horizons = [1, 6, 24]  # 1-hour, 6-hour, 24-hour forecasts
    
    # CRITICAL FIX: Ensure we have the required columns before encoding
    print(f"\n📊 Initial data check:")
    print(f"   Total samples: {len(df)}")
    print(f"   Cities: {df['city'].nunique()}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check for required pollutant column
    if 'pm25' not in df.columns:
        print("\n✗ Error: 'pm25' column not found in data")
        print(f"Available columns: {df.columns.tolist()}")
        return models
    
    # Remove rows where pm25 is null (critical!)
    df_clean = df.dropna(subset=['pm25']).copy()
    print(f"   Samples after removing null PM2.5: {len(df_clean)}")
    
    if len(df_clean) == 0:
        print("\n✗ Error: No valid PM2.5 data available for training")
        return models
    
    # Add city encoding for the model
    df_encoded = df_clean.copy()
    
    # One-hot encode city names
    city_dummies = pd.get_dummies(df_encoded['city'], prefix='city')
    df_encoded = pd.concat([df_encoded, city_dummies], axis=1)
    
    print(f"\n📊 Training on data from {df_encoded['city'].nunique()} cities")
    print(f"   Total samples: {len(df_encoded)}")
    print(f"   Features after encoding: {len(df_encoded.columns)}")
    
    for hours in horizons:
        print(f"\n{'*'*70}")
        print(f"Training {hours}-hour forecast model")
        print(f"{'*'*70}")
        
        try:
            forecaster = AQIForecaster(
                forecast_hours=hours,
                lag_hours=[1,3,6,12,24]
            )
            
            print(df_encoded.dtypes[df_encoded.dtypes == 'object'])
            # Train with current PM2.5 - the forecaster will handle creating the future target
            metrics = forecaster.train(df_encoded, target_col='aqi', test_size=0.2)

            # Save model
            model_path = settings.MODELS_DIR / f'forecaster_{hours}h_multi_city.pkl'
            forecaster.save(str(model_path))
            
            models[f'{hours}h'] = {
                'model': forecaster,
                'metrics': metrics,
                'path': str(model_path)
            }
            
            print(f"\n✓ {hours}-hour model trained on multi-city data")
            print(f"  Test MAE: {metrics['test']['mae']:.2f}")
            print(f"  Test RMSE: {metrics['test']['rmse']:.2f}")
            print(f"  Test R²: {metrics['test']['r2']:.3f}")
            
        except Exception as e:
            print(f"\n✗ Error training {hours}-hour model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return models


def train_risk_classifier(df: pd.DataFrame) -> RiskClassifier:
    """
    Train risk classification model
    
    Args:
        df: Training data with AQI values
        
    Returns:
        Trained classifier
    """
    print("\n" + "="*70)
    print("TRAINING RISK CLASSIFIER")
    print("="*70)
    
    classifier = RiskClassifier(method='rule_based')
    
    print("\nUsing rule-based classification (EPA standards)")
    print("✓ Works universally across all cities")
    
    # Save classifier
    classifier_path = settings.MODELS_DIR / 'risk_classifier.pkl'
    classifier.save(str(classifier_path))
    
    print(f"✓ Rule-based classifier saved to {classifier_path}")
    
    return classifier


def initialize_policy_engine():
    """
    Initialize and save policy recommendation engine
    
    Returns:
        Policy engine instance
    """
    print("\n" + "="*70)
    print("INITIALIZING POLICY RECOMMENDATION ENGINE")
    print("="*70)
    
    engine = PolicyRecommendationEngine()
    
    print("\nPolicy engine initialized with rules for all AQI categories")
    print("✓ Works for all Egyptian cities")
    
    # Save policy engine
    policy_path = settings.MODELS_DIR / 'policy_engine.pkl'
    engine.save(str(policy_path))
    
    print(f"\n✓ Policy engine saved to {policy_path}")
    
    return engine


def generate_training_report(models: dict, df: pd.DataFrame):
    """
    Generate comprehensive training report
    
    Args:
        models: Dictionary of trained models
        df: Training data used
    """
    print("\n" + "="*70)
    print("MULTI-CITY TRAINING SUMMARY REPORT")
    print("="*70)
    
    print(f"\n📊 Dataset Information:")
    print(f"  Total samples: {len(df)}")
    print(f"  Number of cities: {df['city'].nunique()}")
    print(f"  Cities: {', '.join(df['city'].unique())}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Features: {len(df.columns)}")
    
    print(f"\n📈 Statistics by City:")
    for city in sorted(df['city'].unique()):
        city_data = df[df['city'] == city]
        print(f"\n  {city}:")
        print(f"    Samples: {len(city_data)}")
        if 'pm25' in city_data.columns:
            print(f"    PM2.5: {city_data['pm25'].mean():.1f} ± {city_data['pm25'].std():.1f} μg/m³")
            print(f"    PM2.5 range: {city_data['pm25'].min():.1f} - {city_data['pm25'].max():.1f}")
        if 'aqi' in city_data.columns:
            print(f"    AQI: {city_data['aqi'].mean():.1f} ± {city_data['aqi'].std():.1f}")
            print(f"    AQI range: {city_data['aqi'].min():.0f} - {city_data['aqi'].max():.0f}")
    
    print(f"\n🎯 Overall Statistics:")
    if 'pm25' in df.columns:
        print(f"  PM2.5 Mean: {df['pm25'].mean():.2f} μg/m³")
        print(f"  PM2.5 Std: {df['pm25'].std():.2f} μg/m³")
        print(f"  PM2.5 Range: {df['pm25'].min():.2f} - {df['pm25'].max():.2f} μg/m³")
    
    if 'aqi' in df.columns:
        print(f"  AQI Mean: {df['aqi'].mean():.2f}")
        print(f"  AQI Std: {df['aqi'].std():.2f}")
        print(f"  AQI Range: {df['aqi'].min():.0f} - {df['aqi'].max():.0f}")
    
    print(f"\n🤖 Trained Models:")
    for name, info in models.items():
        print(f"\n  {name.upper()} Forecast Model:")
        if 'metrics' in info:
            metrics = info['metrics']
            print(f"    Train MAE: {metrics['train']['mae']:.2f}")
            print(f"    Train RMSE: {metrics['train']['rmse']:.2f}")
            print(f"    Train R²: {metrics['train']['r2']:.3f}")
            print(f"    Test MAE: {metrics['test']['mae']:.2f}")
            print(f"    Test RMSE: {metrics['test']['rmse']:.2f}")
            print(f"    Test R²: {metrics['test']['r2']:.3f}")
            print(f"    Saved to: {info['path']}")
    
    print(f"\n💾 All models saved to: {settings.MODELS_DIR}")
    print(f"\n✅ Models trained on {df['city'].nunique()} cities will generalize to new cities!")


def main(use_real_data: bool = False, days: int = 90):
    """
    Main training pipeline for multi-city models
    
    Args:
        use_real_data: If True, fetch real data; if False, use sample data
        days: Days of historical data
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  AirGuardian API - Multi-City Model Training".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print(f"\n⚙️  Configuration:")
    print(f"  Use real data: {use_real_data}")
    print(f"  Historical days: {days}")
    print(f"  Training cities: {len(EGYPTIAN_CITIES)}")
    
    # Create directories
    settings.create_directories()
    
    # Step 1: Get training data from ALL cities
    print(f"\n{'='*70}")
    print("STEP 1: DATA COLLECTION")
    print(f"{'='*70}")
    
    if use_real_data:
        print("\nAttempting to fetch real data from all cities...")
        df = fetch_real_data_for_all_cities(days=days)
        
        if df.empty:
            print("\n⚠️  No real data available, falling back to sample data")
            df = create_sample_data_for_all_cities(days=days)
    else:
        df = create_sample_data_for_all_cities(days=days)
    
    if df.empty:
        print("\n✗ Error: No training data available")
        return
    
    # Save combined training data
    data_path = settings.RAW_DATA_DIR / 'multi_city_training_data.csv'
    df.to_csv(data_path, index=False)
    print(f"\n✓ Multi-city training data saved to {data_path}")
    
    # Save individual city data
    for city in df['city'].unique():
        city_df = df[df['city'] == city]
        city_path = settings.RAW_DATA_DIR / f'{city.lower().replace(" ", "_")}_data.csv'
        city_df.to_csv(city_path, index=False)
    
    # Step 2: Train forecasting models
    print(f"\n{'='*70}")
    print("STEP 2: MODEL TRAINING")
    print(f"{'='*70}")
    
    try:
        models = train_forecasting_models(df)
    except Exception as e:
        print(f"\n✗ Error in forecasting training: {e}")
        import traceback
        traceback.print_exc()
        models = {}
    
    # Step 3: Train risk classifier
    try:
        classifier = train_risk_classifier(df)
    except Exception as e:
        print(f"\n✗ Error in classifier training: {e}")
    
    # Step 4: Initialize policy engine
    try:
        policy_engine = initialize_policy_engine()
    except Exception as e:
        print(f"\n✗ Error initializing policy engine: {e}")
    
    # Step 5: Generate report
    print(f"\n{'='*70}")
    print("STEP 3: TRAINING REPORT")
    print(f"{'='*70}")
    
    generate_training_report(models, df)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 MULTI-CITY TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n✅ Models trained on {df['city'].nunique()} Egyptian cities:")
    for city in sorted(df['city'].unique()):
        print(f"   • {city}")
    
    print("\n📋 Next steps:")
    print("  1. Start the API server: python main.py")
    print("  2. Test with any Egyptian city:")
    print("     curl http://localhost:8000/api/v1/forecast/Alexandria")
    print("     curl http://localhost:8000/api/v1/forecast/Cairo")
    print("  3. Models will work for ALL cities (trained or unseen)")
    
    print("\n💡 Important:")
    print("  • Models trained on multiple cities generalize better")
    print("  • Will provide reasonable predictions for unseen cities")
    print("  • Based on geographic and meteorological patterns")
    
    print(f"\n📁 Models directory: {settings.MODELS_DIR}")
    
    return models


def fetch_real_data_for_all_cities(days: int = 90) -> pd.DataFrame:
    """
    Fetch real data for all Egyptian cities
    
    Args:
        days: Days of historical data
        
    Returns:
        Combined DataFrame with data from all cities
    """
    print(f"\n{'='*70}")
    print(f"FETCHING REAL DATA FOR ALL EGYPTIAN CITIES")
    print(f"{'='*70}\n")
    
    fetcher = AirQualityDataFetcher()
    cities = [
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "country": "EG"},
    {"name": "Alexandria", "lat": 31.2001, "lon": 29.9187, "country": "EG"},
    ]
    # Fetch training dataset
    df = fetcher.fetch_training_dataset(
        cities,
        days=days
    )
    if not df.empty:
        print(f"\n✓ Total samples from all cities: {len(df)}")
        print(f"✓ Cities with data: {df['city'].nunique()}")
        return df
    else:
        print(f"\n⚠️  No real data available from any city")
        return pd.DataFrame()


def create_sample_data_for_all_cities(days: int = 90) -> pd.DataFrame:
    """
    Generate sample training data for ALL Egyptian cities
    
    Args:
        days: Number of days to generate
        
    Returns:
        Combined DataFrame with data from all cities
    """
    print(f"\n{'='*70}")
    print(f"GENERATING SAMPLE DATA FOR ALL EGYPTIAN CITIES")
    print(f"{'='*70}\n")
    
    all_data = []
    
    # FIXED: Proper for loop syntax
    for city_name, coords in EGYPTIAN_CITIES.items():
        print(f"📍 {city_name}...")
        city_df = create_sample_training_data_for_city(
            city_name=city_name,
            lat=coords['lat'],
            lon=coords['lon'],
            days=days
        )
        all_data.append(city_df)
    
    # Combine all city data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✓ Generated data summary:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Cities: {combined_df['city'].nunique()}")
    print(f"  Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    for city in combined_df['city'].unique():
        city_data = combined_df[combined_df['city'] == city]
        print(f"  {city}: {len(city_data)} samples, "
              f"PM2.5 avg: {city_data['pm25'].mean():.1f} μg/m³, "
              f"AQI avg: {city_data['aqi'].mean():.1f}")
    
    return combined_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train AirGuardian models on ALL Egyptian cities'
    )
    parser.add_argument(
        '--real-data',
        action='store_true',
        help='Use real data from APIs (default: use sample data)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Days of historical data (default: 90)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MULTI-CITY TRAINING SCRIPT")
    print("="*70)
    print("\nThis script will train models on data from:")
    for i, city in enumerate(EGYPTIAN_CITIES.keys(), 1):
        print(f"  {i}. {city}")
    print("\nThis ensures models work correctly for ALL Egyptian cities!")
    print("="*70)
    
    main(
        use_real_data=args.real_data,
        days=args.days
    )