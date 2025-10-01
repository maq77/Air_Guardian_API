"""
Risk Classification Model
Classifies AQI values into health risk categories
Provides health recommendations for citizens
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils.constants import RISK_LEVELS, HEALTH_MESSAGES
from utils.helpers import classify_aqi, get_risk_category_key, get_health_recommendations


class RiskClassifier:
    """
    Classifies air quality into risk categories and generates health advice
    
    Can use either rule-based or ML-based classification
    """
    
    def __init__(self, method: str = 'rule_based'):
        """
        Initialize classifier
        
        Args:
            method: 'rule_based' or 'ml_based'
        """
        self.method = method
        self.model = None
        self.accuracy = None
    
    def _aqi_to_category_number(self, aqi: float) -> int:
        """Convert AQI to category number (0-5)"""
        if aqi <= 50:
            return 0
        elif aqi <= 100:
            return 1
        elif aqi <= 150:
            return 2
        elif aqi <= 200:
            return 3
        elif aqi <= 300:
            return 4
        else:
            return 5
    
    def classify_rule_based(self, aqi: float) -> Dict:
        """
        Rule-based classification (EPA standards)
        
        Args:
            aqi: Air Quality Index value
            
        Returns:
            Classification result with health recommendations
        """
        risk_level = classify_aqi(aqi)
        category_key = get_risk_category_key(aqi)
        messages = HEALTH_MESSAGES[category_key]
        
        return {
            'aqi': aqi,
            'risk_level': risk_level,
            'category_number': self._aqi_to_category_number(aqi),
            'general_message': messages['general'],
            'sensitive_groups_message': messages['sensitive'],
            'activities_safe': messages['activities_safe'],
            'activities_avoid': messages['activities_avoid']
        }
    
    def train_ml_model(self, df: pd.DataFrame) -> float:
        """
        Train ML-based classifier
        
        Args:
            df: DataFrame with features (pm25, no2, temp, humidity, wind_speed) and aqi
            
        Returns:
            Test accuracy
        """
        print("\nTraining ML-based risk classifier...")
        
        # Create category labels
        df['category'] = df['aqi'].apply(self._aqi_to_category_number)
        
        # Features
        feature_cols = ['pm25', 'no2', 'temperature', 'humidity', 'wind_speed']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols]
        y = df['category']
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classification Accuracy: {self.accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=list(RISK_LEVELS.values())))
        
        return self.accuracy
    
    def classify_ml_based(self, features: Dict) -> Dict:
        """
        ML-based classification
        
        Args:
            features: Dict with pm25, no2, temperature, humidity, wind_speed
            
        Returns:
            Classification result
        """
        if self.model is None:
            raise ValueError("ML model not trained!")
        
        # Prepare features
        feature_values = [
            features.get('pm25', 0),
            features.get('no2', 0),
            features.get('temperature', 25),
            features.get('humidity', 50),
            features.get('wind_speed', 3)
        ]
        
        # Predict
        category_num = self.model.predict([feature_values])[0]
        risk_level = RISK_LEVELS[category_num]
        
        # Get corresponding AQI (approximate)
        aqi = features.get('aqi', features.get('pm25', 0) * 1.2)
        category_key = get_risk_category_key(aqi)
        messages = HEALTH_MESSAGES[category_key]
        
        return {
            'aqi': aqi,
            'risk_level': risk_level,
            'category_number': category_num,
            'general_message': messages['general'],
            'sensitive_groups_message': messages['sensitive'],
            'activities_safe': messages['activities_safe'],
            'activities_avoid': messages['activities_avoid']
        }
    
    def classify(self, aqi: float = None, features: Dict = None) -> Dict:
        """
        Classify air quality (uses configured method)
        
        Args:
            aqi: AQI value (for rule-based)
            features: Feature dict (for ML-based)
            
        Returns:
            Classification result
        """
        if self.method == 'rule_based':
            if aqi is None:
                raise ValueError("AQI value required for rule-based classification")
            return self.classify_rule_based(aqi)
        
        elif self.method == 'ml_based':
            if features is None:
                raise ValueError("Features required for ML-based classification")
            return self.classify_ml_based(features)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_detailed_health_alert(self, aqi: float, hour: int = None) -> Dict:
        """
        Generate detailed health alert with time-specific recommendations
        
        Args:
            aqi: Air Quality Index
            hour: Hour of day (0-23)
            
        Returns:
            Detailed health alert
        """
        from utils.helpers import get_time_of_day
        from utils.constants import TIME_SPECIFIC_MESSAGES
        
        classification = self.classify_rule_based(aqi)
        time_period = get_time_of_day(hour)
        category_key = get_risk_category_key(aqi)
        
        result = {
            **classification,
            'time_of_day': time_period,
            'time_specific_message': TIME_SPECIFIC_MESSAGES[time_period][category_key],
            'should_wear_mask': aqi > 150,
            'should_use_air_purifier': aqi > 100,
            'window_recommendations': {
                'open_windows': aqi <= 50,
                'close_windows': aqi > 100,
                'use_air_filter': aqi > 150
            }
        }
        
        return result
    
    def save(self, filepath: str):
        """Save classifier"""
        joblib.dump({
            'method': self.method,
            'model': self.model,
            'accuracy': self.accuracy
        }, filepath)
        print(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load classifier"""
        data = joblib.load(filepath)
        classifier = cls(method=data['method'])
        classifier.model = data['model']
        classifier.accuracy = data['accuracy']
        return classifier


class HealthAdvisor:
    """
    Provides personalized health advice based on user profile
    """
    
    SENSITIVE_GROUPS = ['children', 'elderly', 'asthma', 'heart_disease', 'pregnant']
    
    def __init__(self):
        self.classifier = RiskClassifier(method='rule_based')
    
    def get_personalized_advice(self, aqi: float, user_profile: Dict) -> Dict:
        """
        Get personalized health advice
        
        Args:
            aqi: Current AQI
            user_profile: User info (age, conditions, activity_level)
            
        Returns:
            Personalized recommendations
        """
        base_classification = self.classifier.classify(aqi=aqi)
        
        # Check if user is in sensitive group
        is_sensitive = self._is_sensitive_group(user_profile)
        
        # Adjust thresholds for sensitive groups
        if is_sensitive:
            adjusted_aqi = aqi * 1.2  # More conservative for sensitive groups
            adjusted_classification = self.classifier.classify(aqi=adjusted_aqi)
            
            advice = {
                **adjusted_classification,
                'is_sensitive_group': True,
                'specific_concerns': self._get_specific_concerns(user_profile, aqi),
                'activity_recommendations': self._get_activity_recommendations(adjusted_aqi, user_profile)
            }
        else:
            advice = {
                **base_classification,
                'is_sensitive_group': False,
                'activity_recommendations': self._get_activity_recommendations(aqi, user_profile)
            }
        
        return advice
    
    def _is_sensitive_group(self, profile: Dict) -> bool:
        """Check if user is in sensitive group"""
        age = profile.get('age', 30)
        conditions = profile.get('conditions', [])
        
        if age < 12 or age > 65:
            return True
        
        if any(condition in conditions for condition in self.SENSITIVE_GROUPS):
            return True
        
        return False
    
    def _get_specific_concerns(self, profile: Dict, aqi: float) -> List[str]:
        """Get condition-specific concerns"""
        concerns = []
        conditions = profile.get('conditions', [])
        
        if 'asthma' in conditions and aqi > 100:
            concerns.append("High risk of asthma symptoms. Keep rescue inhaler nearby.")
        
        if 'heart_disease' in conditions and aqi > 150:
            concerns.append("Elevated cardiovascular stress. Avoid physical exertion.")
        
        if 'pregnant' in conditions and aqi > 100:
            concerns.append("Increased risk for pregnancy. Minimize outdoor exposure.")
        
        age = profile.get('age', 30)
        if age > 65 and aqi > 100:
            concerns.append("Elderly individuals face higher health risks. Stay indoors.")
        
        if age < 12 and aqi > 100:
            concerns.append("Children are more vulnerable. Limit outdoor play.")
        
        return concerns
    
    def _get_activity_recommendations(self, aqi: float, profile: Dict) -> Dict:
        """Get activity-specific recommendations"""
        activity_level = profile.get('activity_level', 'moderate')
        
        recommendations = {
            'walking': aqi <= 100,
            'jogging': aqi <= 50,
            'intense_exercise': aqi <= 50,
            'outdoor_sports': aqi <= 100,
            'cycling': aqi <= 100,
        }
        
        # Adjust for high activity level
        if activity_level == 'high' and aqi <= 150:
            recommendations['light_exercise'] = True
        
        return recommendations