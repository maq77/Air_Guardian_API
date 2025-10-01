"""
Data Preprocessing for AirGuardian API
Feature engineering and data preparation for ML models
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib


class AQIDataPreprocessor:
    """
    Preprocess air quality data for model training and prediction
    Handles feature engineering, cleaning, and transformation
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        
        # Binary indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 17, 18, 19]).astype(int)
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cyclical features (hour, month) using sin/cos transformation
        This helps models understand that hour 23 is close to hour 0
        
        Args:
            df: DataFrame with temporal features
            
        Returns:
            DataFrame with cyclical encodings
        """
        df = df.copy()
        
        # Hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of week (7-day cycle)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'pm25', 
                           lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create lag features (previous time steps)
        Essential for time series forecasting
        
        Args:
            df: DataFrame sorted by timestamp
            target_col: Column to create lags for
            lags: List of lag periods (hours)
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        if target_col in df.columns:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Also create lags for NO2 if available
        if 'no2' in df.columns:
            for lag in [1, 3, 6, 12]:
                df[f'no2_lag_{lag}'] = df['no2'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'pm25',
                               windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Create rolling window statistics
        Captures short-term trends and volatility
        
        Args:
            df: DataFrame sorted by timestamp
            target_col: Column to calculate statistics for
            windows: List of window sizes (hours)
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        if target_col in df.columns:
            for window in windows:
                # Rolling mean
                df[f'{target_col}_rolling_mean_{window}'] = (
                    df[target_col].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling standard deviation (volatility)
                df[f'{target_col}_rolling_std_{window}'] = (
                    df[target_col].rolling(window=window, min_periods=1).std()
                )
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features and interactions
        
        Args:
            df: DataFrame with weather columns
            
        Returns:
            DataFrame with weather features
        """
        df = df.copy()
        
        # Interaction features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        # Wind dispersion factor (inverse relationship)
        if 'wind_speed' in df.columns:
            df['wind_dispersion_factor'] = 1 / (df['wind_speed'] + 1)
        
        return df
    
    def create_difference_features(self, df: pd.DataFrame, target_col: str = 'pm25') -> pd.DataFrame:
        """
        Create difference features (rate of change)
        
        Args:
            df: DataFrame with target column
            target_col: Column to calculate differences
            
        Returns:
            DataFrame with difference features
        """
        df = df.copy()
        
        if target_col in df.columns:
            # First difference (change from previous hour)
            df[f'{target_col}_diff_1'] = df[target_col].diff(1)
            
            # Percentage change
            df[f'{target_col}_pct_change'] = df[target_col].pct_change()
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            df: DataFrame with potential missing values
            method: Method to use (interpolate/ffill/drop)
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if method == 'interpolate':
            # Linear interpolation for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        elif method == 'ffill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        elif method == 'drop':
            df = df.dropna()
        
        return df
    
    def create_all_features(self, df: pd.DataFrame, target_col: str = 'pm25') -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Raw DataFrame with timestamp and pollutant data
            target_col: Target variable for prediction
            
        Returns:
            DataFrame with all engineered features
        """
        print("Creating features...")
        
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply all transformations
        df = self.create_temporal_features(df)
        df = self.create_cyclical_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_weather_features(df)
        df = self.create_difference_features(df, target_col)
        
        # Handle missing values created by lag/rolling features
        df = self.handle_missing_values(df, method='drop')
        
        print(f"Created {len(df.columns)} features from {len(df)} samples")
        
        return df
    
    def prepare_train_test_split(self, df: pd.DataFrame, target_col: str = 'pm25',
                                 test_size: float = 0.2, forecast_hours: int = 1) -> Tuple:
        """
        Prepare training and test sets
        
        Args:
            df: Preprocessed DataFrame
            target_col: Target variable
            test_size: Proportion for test set
            forecast_hours: Hours ahead to predict
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Create target (future value)
        df['target'] = df[target_col].shift(-forecast_hours)
        df = df.dropna(subset=['target'])
        
        # Select feature columns (exclude non-numeric and metadata)
        exclude_cols = ['timestamp', 'target', target_col]
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df['target']
        
        # Time series split (no shuffling)
        split_idx = int(len(df) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def fit_scaler(self, X_train: pd.DataFrame):
        """
        Fit scaler on training data
        
        Args:
            X_train: Training features
        """
        self.scaler.fit(X_train)
        self.is_fitted = True
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler
        
        Args:
            X: Features to transform
            
        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        return self.scaler.transform(X)
    
    def save(self, filepath: str):
        """Save preprocessor state"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor state"""
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.feature_columns = data['feature_columns']
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor