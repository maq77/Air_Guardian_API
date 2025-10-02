"""
Model Training Script for AirGuardian API
Train forecasting, classification, and policy recommendation models
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fetcher import AirQualityDataFetcher
from models.forecaster import AQIForecaster
from models.classifier import RiskClassifier
from models.policy_engine import PolicyRecommendationEngine
from config.settings import settings


def create_sample_training_data(days: int = 90) -> pd.DataFrame:
    """
    Generate sample training data for demonstration
    Use this if real data is unavailable
    
    Args:
        days: Number of days to generate
        
    Returns:
        DataFrame with realistic air quality patterns
    """
    print(f"Generating {days} days of sample training data...")
    
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
    
    # Base pollution with realistic patterns
    # Daily cycle: higher during day, lower at night
    base_pm25 = 50 + 25 * np.sin(2 * np.pi * hours / 24)
    
    # Weekly cycle: higher on weekdays
    base_pm25 += 15 * (days_of_week < 5)
    
    # Rush hour spikes (7-9am, 5-7pm)
    rush_hour_mask = np.isin(hours, [7, 8, 17, 18, 19])
    base_pm25 += 20 * rush_hour_mask
    
    # Weekend reduction
    weekend_mask = days_of_week >= 5
    base_pm25 -= 10 * weekend_mask
    
    # Seasonal trend (optional - simulates winter pollution)
    months = pd.Series(dates).dt.month.values
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (months - 12) / 12)
    base_pm25 *= seasonal_factor
    
    # Add random noise and occasional spikes
    base_pm25 += np.random.normal(0, 8, n)
    
    # Occasional pollution episodes
    for i in range(0, n, 24*7):  # Weekly episodes
        if np.random.random() > 0.7:
            episode_length = np.random.randint(12, 48)
            base_pm25[i:i+episode_length] += np.random.uniform(30, 60)
    
    # Ensure realistic bounds
    base_pm25 = np.clip(base_pm25, 5, 250)
    
    # Generate correlated pollutants
    no2 = base_pm25 * 0.7 + np.random.normal(0, 5, n)
    no2 = np.clip(no2, 5, 150)
    
    # Calculate AQI from PM2.5
    from utils.helpers import calculate_aqi_from_pm25
    aqi = [calculate_aqi_from_pm25(pm) for pm in base_pm25]
    
    # Generate weather data with correlations
    # Temperature: higher during day, seasonal variation
    temperature = 25 + 8 * np.sin(2 * np.pi * hours / 24)
    temperature += 5 * np.sin(2 * np.pi * months / 12)
    temperature += np.random.normal(0, 2, n)
    
    # Humidity: inverse to temperature
    humidity = 70 - 15 * np.sin(2 * np.pi * hours / 24)
    humidity += np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 95)
    
    # Wind speed: generally higher during day, helps disperse pollution
    wind_speed = 3 + 2 * np.sin(2 * np.pi * hours / 24)
    wind_speed += np.random.normal(0, 1, n)
    wind_speed = np.clip(wind_speed, 0.5, 15)
    
    # Pressure
    pressure = 1013 + np.random.normal(0, 5, n)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'pm25': base_pm25,
        'no2': no2,
        'aqi': aqi,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure
    })
    
    print(f"Generated {len(df)} samples")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"PM2.5 range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f}")
    print(f"AQI range: {min(aqi)} - {max(aqi)}")
    
    return df


def fetch_real_training_data(city: str = "Cairo", days: int = 90) -> pd.DataFrame:
    """
    Fetch real training data from APIs
    
    Args:
        city: City name
        days: Days of historical data
        
    Returns:
        DataFrame with real data
    """
    print(f"\nFetching real data for {city}...")
    
    fetcher = AirQualityDataFetcher()
    
    # Get city coordinates
    city_info = settings.get_city_coords(city.lower())
    if not city_info:
        print(f"City {city} not found in settings")
        return pd.DataFrame()
    
    cities = [
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "country": "EG"},
    ]
    # Fetch training dataset
    df = fetcher.fetch_training_dataset(
        cities,
        days=days
    )
    
    return df


def train_forecasting_models(df: pd.DataFrame) -> dict:
    """
    Train forecasting models for different time horizons
    
    Args:
        df: Training data
        
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*70)
    print("TRAINING FORECASTING MODELS")
    print("="*70)
    
    models = {}
    horizons = [1, 6, 24]  # 1-hour, 6-hour, 24-hour forecasts
    
    for hours in horizons:
        print(f"\n{'*'*70}")
        print(f"Training {hours}-hour forecast model")
        print(f"{'*'*70}")
        
        try:
            forecaster = AQIForecaster(
                model_type='xgboost',
                forecast_hours=hours
            )
            
            metrics = forecaster.train(df, target_col='pm25', test_size=0.2)
            
            # Save model
            model_path = settings.MODELS_DIR / f'forecaster_{hours}h.pkl'
            forecaster.save(str(model_path))
            
            models[f'{hours}h'] = {
                'model': forecaster,
                'metrics': metrics,
                'path': str(model_path)
            }
            
            print(f"\n✓ {hours}-hour model saved successfully")
            
        except Exception as e:
            print(f"\n✗ Error training {hours}-hour model: {e}")
            continue
    
    return models


def train_risk_classifier(df: pd.DataFrame) -> RiskClassifier:
    """
    Train risk classification model (optional ML-based)
    
    Args:
        df: Training data with AQI values
        
    Returns:
        Trained classifier
    """
    print("\n" + "="*70)
    print("TRAINING RISK CLASSIFIER")
    print("="*70)
    
    # For this system, we primarily use rule-based classification
    # But we can train an ML classifier as backup
    
    classifier = RiskClassifier(method='rule_based')
    
    print("\nUsing rule-based classification (EPA standards)")
    print("No ML training needed - using threshold-based rules")
    
    # Optionally train ML classifier
    if len(df) > 1000 and 'aqi' in df.columns:
        print("\nTraining optional ML classifier...")
        ml_classifier = RiskClassifier(method='ml_based')
        
        try:
            accuracy = ml_classifier.train_ml_model(df)
            
            # Save ML classifier
            ml_path = settings.MODELS_DIR / 'risk_classifier_ml.pkl'
            ml_classifier.save(str(ml_path))
            
            print(f"✓ ML classifier saved to {ml_path}")
        except Exception as e:
            print(f"✗ Error training ML classifier: {e}")
    
    # Save rule-based classifier
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
    
    print("\nPolicy engine initialized with rules for:")
    print("  - Good (AQI 0-50)")
    print("  - Moderate (AQI 51-100)")
    print("  - Unhealthy for Sensitive Groups (AQI 101-150)")
    print("  - Unhealthy (AQI 151-200)")
    print("  - Very Unhealthy (AQI 201-300)")
    print("  - Hazardous (AQI 301+)")
    
    # Test with sample forecast
    print("\nTesting policy engine...")
    sample_forecast = [120, 135, 145, 155, 150, 140]
    recommendations = engine.get_recommendations(sample_forecast, "Cairo")
    
    print(f"\nSample recommendations for AQI forecast {sample_forecast}:")
    print(f"  Priority: {recommendations['priority']}")
    print(f"  Actions: {len(recommendations['recommended_actions'])} recommendations")
    print(f"  First 3 actions:")
    for action in recommendations['recommended_actions'][:3]:
        print(f"    - {action}")
    
    # Save policy engine
    policy_path = settings.MODELS_DIR / 'policy_engine.pkl'
    engine.save(str(policy_path))
    
    print(f"\n✓ Policy engine saved to {policy_path}")
    
    return engine


def generate_training_report(models: dict, df: pd.DataFrame):
    """
    Generate summary report of training results
    
    Args:
        models: Dictionary of trained models
        df: Training data used
    """
    print("\n" + "="*70)
    print("TRAINING SUMMARY REPORT")
    print("="*70)
    
    print(f"\nDataset Information:")
    print(f"  Total samples: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Features: {list(df.columns)}")
    
    if 'pm25' in df.columns:
        print(f"\nPM2.5 Statistics:")
        print(f"  Mean: {df['pm25'].mean():.2f} μg/m³")
        print(f"  Std: {df['pm25'].std():.2f} μg/m³")
        print(f"  Min: {df['pm25'].min():.2f} μg/m³")
        print(f"  Max: {df['pm25'].max():.2f} μg/m³")
    
    if 'aqi' in df.columns:
        print(f"\nAQI Statistics:")
        print(f"  Mean: {df['aqi'].mean():.2f}")
        print(f"  Std: {df['aqi'].std():.2f}")
        print(f"  Min: {df['aqi'].min():.2f}")
        print(f"  Max: {df['aqi'].max():.2f}")
    
    print(f"\nTrained Models:")
    for name, info in models.items():
        print(f"\n  {name.upper()} Forecast:")
        if 'metrics' in info:
            metrics = info['metrics']
            print(f"    Test MAE: {metrics['test']['mae']:.2f}")
            print(f"    Test RMSE: {metrics['test']['rmse']:.2f}")
            print(f"    Test R²: {metrics['test']['r2']:.3f}")
            print(f"    Saved to: {info['path']}")
    
    print(f"\nAll models saved to: {settings.MODELS_DIR}")


def main(use_real_data: bool = False, city: str = "Cairo", days: int = 90):
    """
    Main training pipeline
    
    Args:
        use_real_data: If True, fetch real data; if False, use sample data
        city: City name for real data
        days: Days of historical data
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  AirGuardian API - Model Training Pipeline".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    print(f"\nConfiguration:")
    print(f"  Use real data: {use_real_data}")
    print(f"  City: {city}")
    print(f"  Historical days: {days}")
    
    # Create directories
    settings.create_directories()
    
    # Step 1: Get training data
    if use_real_data:
        df = fetch_real_training_data(city=city, days=days)
        
        if df.empty:
            print("\n⚠️  No real data available, falling back to sample data")
            df = create_sample_training_data(days=days)
    else:
        df = create_sample_training_data(days=days)
    
    if df.empty:
        print("\n✗ Error: No training data available")
        return
    
    # Save raw training data
    data_path = settings.RAW_DATA_DIR / 'training_data.csv'
    df.to_csv(data_path, index=False)
    print(f"\n✓ Training data saved to {data_path}")
    
    # Step 2: Train forecasting models
    try:
        models = train_forecasting_models(df)
    except Exception as e:
        print(f"\n✗ Error in forecasting training: {e}")
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
    generate_training_report(models, df)
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. Start the API server: python main.py")
    print("  2. Test endpoints: curl http://localhost:8000/api/v1/forecast/Cairo")
    print("  3. Open demo frontend: frontend/demo.html")
    
    print("\nTrained models are ready to use in the API!")
    print(f"Models directory: {settings.MODELS_DIR}")
    
    return models


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AirGuardian models')
    parser.add_argument(
        '--real-data',
        action='store_true',
        help='Use real data from APIs (default: use sample data)'
    )
    parser.add_argument(
        '--city',
        type=str,
        default='Cairo',
        help='City name for real data (default: Cairo)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Days of historical data (default: 90)'
    )
    
    args = parser.parse_args()
    
    # Run training
    main(
        use_real_data=args.real_data,
        city=args.city,
        days=args.days
    )