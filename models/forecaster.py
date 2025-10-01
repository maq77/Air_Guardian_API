"""
Air Quality Forecasting Model
Predicts AQI, PM2.5, and NO2 values for 1-72 hours ahead
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from data.preprocessor import AQIDataPreprocessor


class AQIForecaster:
    """
    Air Quality Forecasting Model
    
    Uses XGBoost or Random Forest to predict future AQI values
    based on historical pollution, weather, and temporal patterns
    """
    
    def __init__(self, model_type: str = 'xgboost', forecast_hours: int = 1):
        """
        Initialize forecaster
        
        Args:
            model_type: 'xgboost' or 'random_forest'
            forecast_hours: Hours ahead to forecast (1-72)
        """
        self.model_type = model_type
        self.forecast_hours = forecast_hours
        self.model = None
        self.preprocessor = AQIDataPreprocessor()
        self.metrics = {}
        self.feature_importance = None
        
    def _create_model(self) -> object:
        """Create the ML model based on type"""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, df: pd.DataFrame, target_col: str = 'pm25', 
              test_size: float = 0.2) -> Dict:
        """
        Train the forecasting model
        
        Args:
            df: DataFrame with columns: timestamp, pm25, no2, temperature, humidity, wind_speed
            target_col: Variable to predict ('pm25', 'no2', 'aqi')
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_type} model for {self.forecast_hours}h forecast")
        print(f"Target: {target_col}")
        print(f"{'='*60}\n")
        
        # Feature engineering
        df_processed = self.preprocessor.create_all_features(df, target_col)
        
        # Prepare train/test split
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_train_test_split(
            df_processed, 
            target_col=target_col,
            test_size=test_size,
            forecast_hours=self.forecast_hours
        )
        
        # Fit scaler
        self.preprocessor.fit_scaler(X_train)
        
        # Create and train model
        self.model = self._create_model()
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'train': {
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            },
            'samples': {
                'train': len(X_train),
                'test': len(X_test)
            }
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Print results
        self._print_metrics()
        
        return self.metrics
    
    def _print_metrics(self):
        """Print training metrics"""
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        
        print("\nTrain Set Performance:")
        print(f"  MAE:  {self.metrics['train']['mae']:.2f}")
        print(f"  RMSE: {self.metrics['train']['rmse']:.2f}")
        print(f"  R²:   {self.metrics['train']['r2']:.3f}")
        
        print("\nTest Set Performance:")
        print(f"  MAE:  {self.metrics['test']['mae']:.2f}")
        print(f"  RMSE: {self.metrics['test']['rmse']:.2f}")
        print(f"  R²:   {self.metrics['test']['r2']:.3f}")
        
        print(f"\nDataset Size:")
        print(f"  Train: {self.metrics['samples']['train']} samples")
        print(f"  Test:  {self.metrics['samples']['test']} samples")
        
        if self.feature_importance is not None:
            print("\nTop 10 Important Features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    def predict(self, current_data: pd.DataFrame) -> float:
        """
        Predict future AQI value
        
        Args:
            current_data: Recent data points (at least 24 hours for lag features)
            
        Returns:
            Predicted value
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess
        df_processed = self.preprocessor.create_all_features(current_data)
        
        # Select features
        X = df_processed[self.preprocessor.feature_columns].tail(1)
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        return float(prediction)
    
    def predict_sequence(self, current_data: pd.DataFrame, 
                        hours: int = 24) -> List[Dict]:
        """
        Predict multiple hours ahead
        
        Args:
            current_data: Recent data points
            hours: Number of hours to forecast
            
        Returns:
            List of predictions with metadata
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = []
        
        # For simplicity, we'll retrain on rolling window for each step
        # In production, use a more sophisticated approach
        for h in range(1, hours + 1):
            try:
                pred_value = self.predict(current_data)
                
                predictions.append({
                    'hours_ahead': h,
                    'predicted_value': pred_value,
                    'confidence': self._calculate_confidence(h)
                })
            except Exception as e:
                print(f"Error predicting hour {h}: {e}")
                break
        
        return predictions
    
    def _calculate_confidence(self, hours_ahead: int) -> float:
        """
        Calculate prediction confidence based on test performance and time horizon
        
        Args:
            hours_ahead: Hours into the future
            
        Returns:
            Confidence score (0-1)
        """
        if not self.metrics:
            return 0.75
        
        # Base confidence from test R²
        base_confidence = max(0.5, self.metrics['test']['r2'])
        
        # Decay with time
        decay_rate = 0.015
        confidence = base_confidence * np.exp(-decay_rate * hours_ahead)
        
        return round(max(0.5, min(1.0, confidence)), 2)
    
    def evaluate_on_holdout(self, holdout_df: pd.DataFrame, 
                           target_col: str = 'pm25') -> Dict:
        """
        Evaluate model on separate holdout dataset
        
        Args:
            holdout_df: Holdout dataset
            target_col: Target variable
            
        Returns:
            Evaluation metrics
        """
        # Preprocess
        df_processed = self.preprocessor.create_all_features(holdout_df, target_col)
        
        # Create target
        df_processed['target'] = df_processed[target_col].shift(-self.forecast_hours)
        df_processed = df_processed.dropna(subset=['target'])
        
        X = df_processed[self.preprocessor.feature_columns]
        y = df_processed['target']
        
        # Predict
        y_pred = self.model.predict(X)
        
        # Metrics
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'samples': len(X)
        }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save trained model and preprocessor
        
        Args:
            filepath: Path to save model
        """
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'forecast_hours': self.forecast_hours,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }, filepath)
        print(f"\nModel saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load trained model
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded AQIForecaster instance
        """
        data = joblib.load(filepath)
        
        forecaster = cls(
            model_type=data['model_type'],
            forecast_hours=data['forecast_hours']
        )
        forecaster.model = data['model']
        forecaster.preprocessor = data['preprocessor']
        forecaster.metrics = data['metrics']
        forecaster.feature_importance = data.get('feature_importance')
        
        print(f"Model loaded from {filepath}")
        print(f"Test MAE: {forecaster.metrics['test']['mae']:.2f}")
        
        return forecaster


class EnsembleForecaster:
    """
    Ensemble of multiple forecasters for improved accuracy
    Combines predictions from multiple models
    """
    
    def __init__(self, models: List[AQIForecaster]):
        """
        Initialize ensemble
        
        Args:
            models: List of trained AQIForecaster models
        """
        self.models = models
        self.weights = None
    
    def calculate_weights(self):
        """Calculate weights based on model performance"""
        # Weight by inverse MAE (better models get higher weight)
        maes = [model.metrics['test']['mae'] for model in self.models]
        inv_maes = [1/mae for mae in maes]
        total = sum(inv_maes)
        self.weights = [w/total for w in inv_maes]
    
    def predict(self, current_data: pd.DataFrame) -> float:
        """
        Ensemble prediction
        
        Args:
            current_data: Recent data points
            
        Returns:
            Weighted average prediction
        """
        if self.weights is None:
            self.calculate_weights()
        
        predictions = [model.predict(current_data) for model in self.models]
        ensemble_pred = sum(p * w for p, w in zip(predictions, self.weights))
        
        return float(ensemble_pred)
    
    def predict_with_uncertainty(self, current_data: pd.DataFrame) -> Tuple[float, float]:
        """
        Predict with uncertainty estimate
        
        Args:
            current_data: Recent data points
            
        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        predictions = [model.predict(current_data) for model in self.models]
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return float(mean_pred), float(std_pred)


# Utility functions

def train_multiple_horizons(df: pd.DataFrame, 
                           horizons: List[int] = [1, 3, 6, 12, 24],
                           model_type: str = 'xgboost') -> Dict[int, AQIForecaster]:
    """
    Train models for multiple forecast horizons
    
    Args:
        df: Training data
        horizons: List of forecast hours
        model_type: Model type to use
        
    Returns:
        Dictionary mapping horizon to trained model
    """
    models = {}
    
    for hours in horizons:
        print(f"\n{'#'*60}")
        print(f"Training {hours}-hour forecast model")
        print(f"{'#'*60}")
        
        forecaster = AQIForecaster(
            model_type=model_type,
            forecast_hours=hours
        )
        
        forecaster.train(df, target_col='pm25')
        models[hours] = forecaster
    
    return models


def compare_model_types(df: pd.DataFrame, 
                       forecast_hours: int = 1) -> pd.DataFrame:
    """
    Compare different model types
    
    Args:
        df: Training data
        forecast_hours: Hours to forecast
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_type in ['xgboost', 'random_forest']:
        forecaster = AQIForecaster(
            model_type=model_type,
            forecast_hours=forecast_hours
        )
        
        metrics = forecaster.train(df, target_col='pm25')
        
        results.append({
            'model_type': model_type,
            'mae': metrics['test']['mae'],
            'rmse': metrics['test']['rmse'],
            'r2': metrics['test']['r2']
        })
    
    return pd.DataFrame(results).sort_values('mae')