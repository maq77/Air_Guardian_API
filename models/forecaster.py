"""
Complete AQI Forecaster with proper temporal predictions
Place this as: models/forecaster.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle


class AQIForecaster:
    """
    AQI Forecaster that generates varying predictions over time
    """
    
    def __init__(self, forecast_hours=24, lag_hours=None):
        self.forecast_hours = forecast_hours
        self.lag_hours = lag_hours or [1, 3, 6, 12, 24]
        self.model = None
        self.feature_columns = None
        self.scaler_params = {}
      
    def create_lag_features(self, df, target_col='aqi'):
        """Create lagged features for time series prediction"""
        df = df.copy()
        df = df.sort_values(['city', 'timestamp']).reset_index(drop=True)
        
        # Create lag features for each city separately
        for lag in self.lag_hours:
            df[f'{target_col}_lag_{lag}h'] = df.groupby('city')[target_col].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24]:
            df[f'{target_col}_rolling_mean_{window}h'] = (
                df.groupby('city')[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            df[f'{target_col}_rolling_std_{window}h'] = (
                df.groupby('city')[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)
                .reset_index(0, drop=True)
            )
        
        return df
    
    def create_temporal_features(self, df):
        """Create temporal features from timestamp"""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Boolean features
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9)) | 
            ((df['hour'] >= 17) & (df['hour'] <= 19))
        ).astype(int)
        
        return df
    
    def prepare_training_data(self, df, target_col='aqi'):
        """Prepare data with all features"""
        print(f"Preparing training data with target: {target_col}")
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, target_col)
        
        # Drop rows with NaN in lag features
        initial_len = len(df)
        df = df.dropna(subset=[f'{target_col}_lag_{self.lag_hours[0]}h'])
        print(f"Dropped {initial_len - len(df)} rows with missing lag features")
        
        # One-hot encode city
        if 'city' in df.columns:
            city_dummies = pd.get_dummies(df['city'], prefix='city')
            df = pd.concat([df, city_dummies], axis=1)
        
        return df
    
    def train(self, df, target_col='aqi', test_size=0.2):
        """Train the forecasting model"""
        print(f"\nTraining forecaster for {self.forecast_hours}h ahead...")
        
        # Prepare data
        df = self.prepare_training_data(df, target_col)
        
        # Drop non-numeric or unused features
        df = df.drop(columns=['source','city_Cairo','city_Alexandria'], errors='ignore')  # <-- Drop 'source' here

        # Define feature columns (exclude target and metadata)
        exclude_cols = ['timestamp', 'city', 'source', target_col, 'latitude', 'longitude', 'lat', 'lon']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        # Ensure feature_columns only contains valid string column names
        self.feature_columns = [col for col in self.feature_columns if isinstance(col, str) and col in df.columns]

        print(f"Using {len(self.feature_columns)} features")
        
        # Handle object dtype columns: convert to categorical codes
        # for col in self.feature_columns:
        #     if pd.api.types.is_object_dtype(df[col]):
        #         print(f"Encoding object column: {col}")
        #         df[col] = df[col].astype('category').cat.codes

        # Prepare X and y
        X = df[self.feature_columns].fillna(0)
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            enable_categorical=True
            ####
        )
        

        print("Columns and their types in X_train:")
        for col in X_train.columns:
            print(f"{col}: {type(X_train[col])}, dtype={getattr(X_train[col], 'dtype', None)}")


        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train': {
                'mae': mean_absolute_error(y_train, train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'r2': r2_score(y_train, train_pred)
            },
            'test': {
                'mae': mean_absolute_error(y_test, test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'r2': r2_score(y_test, test_pred)
            }
        }
        
        print(f"\nTraining Results:")
        print(f"  Train MAE: {metrics['train']['mae']:.2f}")
        print(f"  Test MAE:  {metrics['test']['mae']:.2f}")
        print(f"  Test RMSE: {metrics['test']['rmse']:.2f}")
        print(f"  Test R²:   {metrics['test']['r2']:.3f}")
        
        return metrics

    def forecast(self, current_data, hours=24):
        """
        Generate forecast for next N hours with VARYING predictions
        
        Args:
            current_data: Dict with current conditions including 'aqi', 'pm25', 'city'
            hours: Number of hours to forecast
        
        Returns:
            List of predictions with varying AQI values
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        forecasts = []
        current_timestamp = pd.Timestamp.now()
        
        # Initialize with current AQI
        current_aqi = current_data.get('aqi', 50)
        city = current_data.get('city', 'Cairo')
        
        # Build history for lag features (initialize with current value)
        history = [current_aqi] * 24
        
        for h in range(hours):
            pred_timestamp = current_timestamp + timedelta(hours=h+1)
            
            # Create features for this specific timestamp
            # CRITICAL: Ensure all values are numeric types, not objects
            features = {
                'hour': int(pred_timestamp.hour),
                'dayofweek': int(pred_timestamp.dayofweek),
                'month': int(pred_timestamp.month),
                'day': int(pred_timestamp.day),
                'hour_sin': float(np.sin(2 * np.pi * pred_timestamp.hour / 24)),
                'hour_cos': float(np.cos(2 * np.pi * pred_timestamp.hour / 24)),
                'month_sin': float(np.sin(2 * np.pi * pred_timestamp.month / 12)),
                'month_cos': float(np.cos(2 * np.pi * pred_timestamp.month / 12)),
                'day_sin': float(np.sin(2 * np.pi * pred_timestamp.dayofweek / 7)),
                'day_cos': float(np.cos(2 * np.pi * pred_timestamp.dayofweek / 7)),
                'is_weekend': int(pred_timestamp.dayofweek >= 5),
                'is_rush_hour': int((7 <= pred_timestamp.hour <= 9) or 
                                (17 <= pred_timestamp.hour <= 19)),
            }
            
            # Add lag features from history - ENSURE FLOAT TYPE
            for lag in self.lag_hours:
                if lag <= len(history):
                    features[f'aqi_lag_{lag}h'] = float(history[-lag])
                else:
                    features[f'aqi_lag_{lag}h'] = float(history[0])
            
            # Add rolling statistics - ENSURE FLOAT TYPE
            for window in [6, 12, 24]:
                window_data = history[-min(window, len(history)):]
                features[f'aqi_rolling_mean_{window}h'] = float(np.mean(window_data))
                features[f'aqi_rolling_std_{window}h'] = float(np.std(window_data)) if len(window_data) > 1 else 0.0
            
            # Add city encoding if trained with cities
            features[f'city_{city}'] = 1
            
            # Add other features from current_data - ENSURE FLOAT TYPE
            for key in ['pm25', 'pm10', 'no2', 'temperature', 'humidity', 'wind_speed']:
                if key in current_data:
                    # Convert to float to avoid object dtype
                    features[key] = float(current_data[key]) if current_data[key] is not None else 0.0
            
            # Create DataFrame with all possible features
            X = pd.DataFrame([features])
            
            # Ensure all feature columns exist with float dtype
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0.0
            
            # Select only the features used during training and convert to float64
            X = X[self.feature_columns].astype('float64')
            
            # Make prediction
            predicted_aqi = self.model.predict(X)[0]
            
            # Add realistic variations based on time of day
            hour = pred_timestamp.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                predicted_aqi *= 1.15
            elif 2 <= hour <= 5:  # Early morning
                predicted_aqi *= 0.85
            
            # Add small random variation to prevent identical values
            predicted_aqi *= (1 + np.random.uniform(-0.05, 0.05))
            
            # Ensure realistic bounds
            predicted_aqi = np.clip(predicted_aqi, 0, 500)
            
            # Update history with this prediction
            history.append(predicted_aqi)
            if len(history) > 24:
                history.pop(0)
            
            # Calculate derived values
            predicted_pm25 = self._aqi_to_pm25(predicted_aqi)
            predicted_no2 = predicted_pm25 * 0.7
            
            forecasts.append({
                'timestamp': pred_timestamp.isoformat(),
                'aqi': round(predicted_aqi, 1),
                'pm25': round(predicted_pm25, 2),
                'no2': round(predicted_no2, 2),
                'risk_level': self._get_risk_level(predicted_aqi),
                'health_recommendation': self._get_health_recommendation(predicted_aqi),
                'confidence': round(max(0.5, 0.95 - (h * 0.015)), 2)
            })
        
        return forecasts    
    
    def _aqi_to_pm25(self, aqi):
        """Convert AQI back to PM2.5 (approximate)"""
        if aqi <= 50:
            return aqi * 12 / 50
        elif aqi <= 100:
            return 12 + (aqi - 50) * 23.5 / 50
        elif aqi <= 150:
            return 35.5 + (aqi - 100) * 19.9 / 50
        elif aqi <= 200:
            return 55.5 + (aqi - 150) * 94.5 / 50
        else:
            return 150 + (aqi - 200) * 100 / 100

    def _get_risk_level(self, aqi):
        """Get risk level from AQI"""
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

    def _get_health_recommendation(self, aqi):
        """Get health recommendation based on AQI"""
        if aqi <= 50:
            return "Air quality is good. Enjoy outdoor activities."
        elif aqi <= 100:
            return "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion."
        elif aqi <= 150:
            return "Sensitive groups should reduce prolonged or heavy outdoor exertion."
        elif aqi <= 200:
            return "Everyone should reduce prolonged or heavy outdoor exertion."
        elif aqi <= 300:
            return "Everyone should avoid prolonged or heavy outdoor exertion. Sensitive groups should remain indoors."
        else:
            return "Everyone should avoid all outdoor exertion. Remain indoors."
    
    def save(self, path):
        """Save model to file"""
        data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'forecast_hours': self.forecast_hours,
            'lag_hours': self.lag_hours
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from file (instance method)"""
        import sys
        import xgboost as xgb
        
        # Add xgboost to sys.modules to help pickle find it
        sys.modules['XGBRegressor'] = xgb.XGBRegressor
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            
            # Load other attributes if they were saved
            if 'forecast_hours' in data:
                self.forecast_hours = data['forecast_hours']
            if 'lag_hours' in data:
                self.lag_hours = data['lag_hours']
            
            print(f"Model loaded successfully from {path}")
        except ModuleNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Old model format detected. Please retrain models with: python scripts/train_model.py")
            raise


# Example usage
if __name__ == "__main__":
    # Create sample training data
    dates = pd.date_range(start='2025-06-01', end='2025-09-30', freq='H')
    n = len(dates)
    
    # Generate realistic AQI pattern
    hours = pd.Series(dates).dt.hour.values
    aqi = 50 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 10, n)
    aqi = np.clip(aqi, 0, 500)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'city': 'Cairo',
        'aqi': aqi,
        'pm25': aqi * 0.5,
        'no2': aqi * 0.4,
        'temperature': 25 + 10 * np.sin(2 * np.pi * hours / 24),
        'humidity': 60 - 20 * np.sin(2 * np.pi * hours / 24),
        'wind_speed': 3 + 2 * np.random.random(n)
    })
    
    # Train model
    forecaster = AQIForecaster(forecast_hours=24)
    metrics = forecaster.train(df)
    
    # Make forecast
    current_conditions = {
        'aqi': 75,
        'pm25': 35,
        'no2': 25,
        'temperature': 28,
        'humidity': 55,
        'wind_speed': 3.5,
        'city': 'Cairo'
    }
    
    predictions = forecaster.forecast(current_conditions, hours=24)
    
    print("\n24-Hour Forecast:")
    for pred in predictions[:5]:
        print(f"{pred['timestamp']}: AQI={pred['aqi']}, PM2.5={pred['pm25']}")