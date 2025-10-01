"""
Policy Recommendation Engine
Generates government-level policy recommendations based on air quality forecasts
"""

import numpy as np
from typing import Dict, List
from utils.constants import POLICY_PRIORITIES
from utils.helpers import get_risk_category_key
import joblib


class PolicyRecommendationEngine:
    """
    Generates policy recommendations for government and city authorities
    Based on forecasted air quality levels
    """
    
    def __init__(self):
        self.policy_rules = self._create_policy_rules()
        self.intervention_costs = self._define_intervention_costs()
    
    def _create_policy_rules(self) -> Dict:
        """Define policy recommendation rules for each severity level"""
        return {
            'good': {
                'priority': 'LOW',
                'actions': [
                    'Continue routine air quality monitoring',
                    'Maintain current environmental regulations',
                    'Public awareness campaigns on air quality'
                ],
                'estimated_impact': 'No immediate action needed',
                'response_time': 'Normal operations',
                'cost_estimate': 'Low'
            },
            'moderate': {
                'priority': 'LOW',
                'actions': [
                    'Increase monitoring frequency to every 2 hours',
                    'Launch public awareness campaign',
                    'Encourage use of public transportation',
                    'Monitor industrial emissions more closely',
                    'Prepare emergency response protocols'
                ],
                'estimated_impact': 'Preventive measures - maintain current levels',
                'response_time': '24-48 hours',
                'cost_estimate': 'Low-Medium'
            },
            'unhealthy_sensitive': {
                'priority': 'MEDIUM',
                'actions': [
                    'Issue health advisory for sensitive groups (children, elderly, respiratory conditions)',
                    'Increase public transportation frequency by 20%',
                    'Reduce public transport fares temporarily',
                    'Monitor industrial emissions in real-time',
                    'Activate air quality display boards city-wide',
                    'Restrict construction activities during peak hours',
                    'Coordinate with hospitals for potential surge'
                ],
                'estimated_impact': 'Moderate reduction expected within 24-48 hours',
                'response_time': '12-24 hours',
                'cost_estimate': 'Medium'
            },
            'unhealthy': {
                'priority': 'HIGH',
                'actions': [
                    'Issue public health warning to all citizens',
                    'Implement odd-even vehicle restrictions',
                    'Make public transportation free',
                    'Temporary reduction of industrial operations (20-30%)',
                    'Ban outdoor construction and burning',
                    'Restrict heavy truck movement to nighttime only',
                    'Close outdoor markets during peak pollution hours',
                    'Deploy mobile air quality monitoring units',
                    'Set up emergency medical response teams',
                    'Distribute masks at public locations'
                ],
                'estimated_impact': 'Significant reduction (20-30%) expected within 12-24 hours',
                'response_time': '6-12 hours',
                'cost_estimate': 'High'
            },
            'very_unhealthy': {
                'priority': 'HIGH',
                'actions': [
                    'Declare air quality alert',
                    'Mandatory odd-even vehicle restrictions with penalties',
                    'Shutdown high-emission industries temporarily',
                    'Ban all construction activities',
                    'Restrict school outdoor activities',
                    'Close outdoor recreational facilities',
                    'Emergency response protocol activated',
                    'Distribute N95 masks to vulnerable populations',
                    'Set up air-filtered public shelters',
                    'Coordinate with healthcare system for emergency capacity',
                    'Deploy water sprinklers on major roads',
                    'Emergency meeting with environmental agencies'
                ],
                'estimated_impact': 'Substantial reduction (30-40%) expected within 6-12 hours',
                'response_time': '3-6 hours',
                'cost_estimate': 'Very High'
            },
            'hazardous': {
                'priority': 'CRITICAL',
                'actions': [
                    '⚠️ DECLARE AIR QUALITY EMERGENCY',
                    'Close schools and non-essential businesses',
                    'Ban ALL non-emergency vehicles',
                    'Mandatory industrial shutdown (except essential services)',
                    'Activate emergency medical services',
                    'Open public shelters with HEPA air filtration',
                    'Deploy emergency air quality improvement units',
                    'Free distribution of N95/N99 masks',
                    'Emergency transport for medical cases',
                    'Coordinate with military for support',
                    'Implement emergency traffic management AI',
                    'Deploy medical teams to affected areas',
                    'Activate inter-city coordination protocols'
                ],
                'estimated_impact': 'Critical emergency response - 40-50% reduction target within immediate timeframe',
                'response_time': 'IMMEDIATE (< 3 hours)',
                'cost_estimate': 'Critical'
            }
        }
    
    def _define_intervention_costs(self) -> Dict:
        """Define relative costs of interventions"""
        return {
            'public_transport_increase': {'cost': 'medium', 'impact': 'moderate'},
            'vehicle_restrictions': {'cost': 'high', 'impact': 'high'},
            'industrial_shutdown': {'cost': 'very_high', 'impact': 'very_high'},
            'construction_ban': {'cost': 'high', 'impact': 'moderate'},
            'public_awareness': {'cost': 'low', 'impact': 'low'},
            'mask_distribution': {'cost': 'medium', 'impact': 'moderate'},
            'air_filtration_shelters': {'cost': 'very_high', 'impact': 'high'}
        }
    
    def get_recommendations(self, forecast_aqi: List[float], location: str) -> Dict:
        """
        Generate policy recommendations based on AQI forecast
        
        Args:
            forecast_aqi: List of forecasted AQI values
            location: City/region name
            
        Returns:
            Policy recommendations with priority and actions
        """
        max_aqi = max(forecast_aqi)
        avg_aqi = np.mean(forecast_aqi)
        trend = self._calculate_trend(forecast_aqi)
        
        # Determine severity level
        category_key = get_risk_category_key(max_aqi)
        rules = self.policy_rules[category_key]
        
        # Build response
        response = {
            'location': location,
            'timestamp': self._get_current_time(),
            'forecast_summary': {
                'max_aqi': int(max_aqi),
                'avg_aqi': int(avg_aqi),
                'min_aqi': int(min(forecast_aqi)),
                'trend': trend,
                'hours_forecasted': len(forecast_aqi)
            },
            'priority': rules['priority'],
            'severity_level': category_key.replace('_', ' ').title(),
            'recommended_actions': rules['actions'],
            'estimated_impact': rules['estimated_impact'],
            'response_time': rules['response_time'],
            'cost_estimate': rules['cost_estimate']
        }
        
        # Add specific interventions based on trend
        if trend == 'worsening':
            response['additional_measures'] = self._get_preventive_measures(max_aqi)
        
        # Add target metrics
        response['target_metrics'] = self._calculate_target_metrics(max_aqi)
        
        return response
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from forecast values"""
        if len(values) < 3:
            return 'stable'
        
        # Linear regression slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 3:
            return 'worsening'
        elif slope < -3:
            return 'improving'
        else:
            return 'stable'
    
    def _get_preventive_measures(self, aqi: float) -> List[str]:
        """Get additional preventive measures for worsening conditions"""
        measures = [
            'Implement emergency traffic management',
            'Deploy rapid response teams',
            'Coordinate with neighboring cities for support'
        ]
        
        if aqi > 200:
            measures.extend([
                'Prepare for potential school closures',
                'Alert healthcare facilities',
                'Activate emergency communication protocols'
            ])
        
        return measures
    
    def _calculate_target_metrics(self, current_aqi: float) -> Dict:
        """Calculate target reduction metrics"""
        if current_aqi <= 100:
            target_reduction = 0
        elif current_aqi <= 150:
            target_reduction = 20
        elif current_aqi <= 200:
            target_reduction = 30
        else:
            target_reduction = 40
        
        target_aqi = max(50, current_aqi - target_reduction)
        
        return {
            'current_aqi': int(current_aqi),
            'target_aqi': int(target_aqi),
            'reduction_needed': int(target_reduction),
            'reduction_percentage': f"{(target_reduction / current_aqi * 100):.1f}%"
        }
    
    def get_intervention_impact_estimate(self, aqi: float, interventions: List[str]) -> Dict:
        """
        Estimate impact of specific interventions
        
        Args:
            aqi: Current AQI
            interventions: List of intervention names
            
        Returns:
            Impact estimate
        """
        total_impact = 0
        total_cost = 0
        
        impact_map = {
            'vehicle_restrictions': 15,
            'industrial_reduction': 20,
            'construction_ban': 10,
            'public_transport_increase': 5,
        }
        
        for intervention in interventions:
            if intervention in impact_map:
                total_impact += impact_map[intervention]
        
        estimated_new_aqi = max(50, aqi - total_impact)
        
        return {
            'current_aqi': int(aqi),
            'estimated_aqi_after': int(estimated_new_aqi),
            'estimated_reduction': int(total_impact),
            'time_to_effect': '6-24 hours',
            'interventions_applied': interventions
        }
    
    def get_best_time_for_intervention(self, hourly_forecast: List[Dict]) -> Dict:
        """
        Determine optimal timing for interventions
        
        Args:
            hourly_forecast: List of hourly forecasts with AQI
            
        Returns:
            Optimal timing recommendations
        """
        # Find peak pollution hours
        peak_hours = []
        for forecast in hourly_forecast:
            if forecast['aqi'] > 150:
                peak_hours.append(forecast['hour'])
        
        if not peak_hours:
            return {
                'intervention_needed': False,
                'message': 'Air quality within acceptable range'
            }
        
        return {
            'intervention_needed': True,
            'peak_pollution_hours': peak_hours,
            'recommended_start_time': max(0, min(peak_hours) - 2),
            'recommended_duration': f"{len(peak_hours)} hours",
            'critical_actions': [
                'Implement vehicle restrictions 2 hours before peak',
                'Increase public transport during peak hours',
                'Alert citizens 3 hours in advance'
            ]
        }
    
    def generate_public_announcement(self, aqi: float, location: str) -> Dict:
        """
        Generate public announcement text
        
        Args:
            aqi: Current or forecasted AQI
            location: Location name
            
        Returns:
            Announcement text for different channels
        """
        category_key = get_risk_category_key(aqi)
        risk_level = category_key.replace('_', ' ').title()
        
        announcements = {
            'good': {
                'short': f"Good air quality in {location}. Enjoy outdoor activities!",
                'detailed': f"Air quality in {location} is excellent (AQI: {int(aqi)}). No health concerns for any group.",
                'sms': f"AQ Alert {location}: GOOD (AQI {int(aqi)}). Safe for all activities."
            },
            'moderate': {
                'short': f"Moderate air quality in {location}. Most people can enjoy outdoor activities.",
                'detailed': f"Air quality in {location} is acceptable (AQI: {int(aqi)}). Sensitive individuals should consider limiting prolonged outdoor exertion.",
                'sms': f"AQ Alert {location}: MODERATE (AQI {int(aqi)}). Sensitive groups take care."
            },
            'unhealthy_sensitive': {
                'short': f"Air quality alert for {location}. Sensitive groups should limit outdoor activities.",
                'detailed': f"⚠️ Air quality in {location} is unhealthy for sensitive groups (AQI: {int(aqi)}). Children, elderly, and those with respiratory conditions should reduce prolonged outdoor activities.",
                'sms': f"⚠️ AQ ALERT {location}: UNHEALTHY (AQI {int(aqi)}). Limit outdoor activity."
            },
            'unhealthy': {
                'short': f"⚠️ Unhealthy air quality in {location}. Everyone should limit outdoor exposure.",
                'detailed': f"⚠️ AIR QUALITY WARNING for {location}: Unhealthy (AQI: {int(aqi)}). Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors.",
                'sms': f"⚠️ AQ WARNING {location}: UNHEALTHY (AQI {int(aqi)}). Stay indoors if possible."
            },
            'very_unhealthy': {
                'short': f"🚨 Very unhealthy air in {location}. Avoid all outdoor activities.",
                'detailed': f"🚨 HEALTH ALERT for {location}: Very Unhealthy (AQI: {int(aqi)}). Everyone should avoid all outdoor activities. Health effects may be experienced by the general public.",
                'sms': f"🚨 HEALTH ALERT {location}: VERY UNHEALTHY (AQI {int(aqi)}). STAY INDOORS."
            },
            'hazardous': {
                'short': f"🚨 EMERGENCY: Hazardous air quality in {location}. Stay indoors!",
                'detailed': f"🚨🚨 EMERGENCY AIR QUALITY WARNING for {location}: HAZARDOUS (AQI: {int(aqi)}). This is a health emergency. Everyone must avoid all outdoor exposure. Seek emergency help if experiencing symptoms.",
                'sms': f"🚨🚨 EMERGENCY {location}: HAZARDOUS (AQI {int(aqi)}). DO NOT GO OUTSIDE."
            }
        }
        
        return announcements.get(category_key, announcements['moderate'])
    
    def compare_policy_scenarios(self, current_aqi: float) -> List[Dict]:
        """
        Compare different policy scenarios and their outcomes
        
        Args:
            current_aqi: Current AQI level
            
        Returns:
            List of scenarios with outcomes
        """
        scenarios = [
            {
                'name': 'No Intervention',
                'actions': ['Continue monitoring'],
                'estimated_aqi_24h': current_aqi,
                'cost': 'None',
                'public_impact': 'No change'
            },
            {
                'name': 'Light Intervention',
                'actions': ['Public awareness', 'Encourage public transport'],
                'estimated_aqi_24h': current_aqi - 5,
                'cost': 'Low',
                'public_impact': 'Minimal disruption'
            },
            {
                'name': 'Moderate Intervention',
                'actions': ['Vehicle restrictions', 'Free public transport', 'Construction limits'],
                'estimated_aqi_24h': current_aqi - 20,
                'cost': 'Medium',
                'public_impact': 'Some disruption'
            },
            {
                'name': 'Aggressive Intervention',
                'actions': ['Full vehicle ban', 'Industrial shutdown', 'Emergency measures'],
                'estimated_aqi_24h': current_aqi - 40,
                'cost': 'Very High',
                'public_impact': 'Significant disruption'
            }
        ]
        
        # Filter scenarios based on AQI
        if current_aqi <= 100:
            return [scenarios[0], scenarios[1]]
        elif current_aqi <= 150:
            return scenarios[:3]
        else:
            return scenarios
    
    def _get_current_time(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save(self, filepath: str):
        """Save policy engine"""
        joblib.dump({
            'policy_rules': self.policy_rules,
            'intervention_costs': self.intervention_costs
        }, filepath)
        print(f"Policy engine saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load policy engine"""
        data = joblib.load(filepath)
        engine = cls()
        engine.policy_rules = data['policy_rules']
        engine.intervention_costs = data['intervention_costs']
        return engine


class CostBenefitAnalyzer:
    """
    Analyze cost-benefit of different policy interventions
    """
    
    def __init__(self):
        self.health_cost_per_aqi_point = 10000  # USD per AQI point (example)
    
    def calculate_health_cost(self, aqi: float, population: int, duration_hours: int = 24) -> float:
        """
        Calculate health cost of air pollution
        
        Args:
            aqi: Air Quality Index
            population: Affected population
            duration_hours: Duration of exposure
            
        Returns:
            Estimated health cost in USD
        """
        if aqi <= 50:
            cost_multiplier = 0
        elif aqi <= 100:
            cost_multiplier = 0.5
        elif aqi <= 150:
            cost_multiplier = 1.5
        elif aqi <= 200:
            cost_multiplier = 3
        else:
            cost_multiplier = 5
        
        health_cost = (aqi * cost_multiplier * population * duration_hours) / 1000
        return health_cost
    
    def calculate_intervention_roi(self, current_aqi: float, intervention_cost: float,
                                  expected_reduction: float, population: int) -> Dict:
        """
        Calculate ROI of intervention
        
        Args:
            current_aqi: Current AQI
            intervention_cost: Cost of intervention
            expected_reduction: Expected AQI reduction
            population: Affected population
            
        Returns:
            ROI analysis
        """
        health_cost_before = self.calculate_health_cost(current_aqi, population)
        health_cost_after = self.calculate_health_cost(current_aqi - expected_reduction, population)
        health_savings = health_cost_before - health_cost_after
        
        net_benefit = health_savings - intervention_cost
        roi = (net_benefit / intervention_cost * 100) if intervention_cost > 0 else 0
        
        return {
            'intervention_cost': intervention_cost,
            'health_cost_avoided': health_savings,
            'net_benefit': net_benefit,
            'roi_percentage': f"{roi:.1f}%",
            'recommendation': 'Proceed' if net_benefit > 0 else 'Reconsider'
        }