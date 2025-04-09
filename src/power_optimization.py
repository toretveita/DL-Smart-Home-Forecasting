import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import json
import pandas as pd
from train_lstm_model import LSTMModel  # Import the LSTM model class

print("Loading power_optimization.py...")

class PowerOptimizationAnalyzer:
    def __init__(self, device_settings_path, power_consumption_path, model_path):
        print("Initializing PowerOptimizationAnalyzer...")
        
        # Load data
        print("Loading data files...")
        self.device_settings = pd.read_csv(device_settings_path)
        self.power_consumption = pd.read_csv(power_consumption_path)
        
        # Initialize feature mappings and scalers
        print("Initializing feature mappings...")
        self.feature_mappings = self._initialize_feature_mappings()
        self.feature_scaler = StandardScaler()
        self.power_scalers = {}
        
        # Fit scalers with all available data
        print("Fitting scalers...")
        self._fit_scalers()
        
        # Load and prepare the trained LSTM model
        print(f"Loading LSTM model from {model_path}...")
        self.model = LSTMModel(input_size=13, hidden_size=512, num_layers=3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode
        
        print("PowerOptimizationAnalyzer initialization complete")
    
    def predict_power_consumption(self, device_data):
        """Predict power consumption using the trained LSTM model."""
        device_type = device_data['device_type'].iloc[0]
        print(f"\nPredicting power consumption for {device_type}...")
        
        # Special handling for lighting - use brightness-based calculation
        if device_type == 'Light':
            brightness = float(device_data['brightness'].iloc[0])
            # Scale power based on brightness percentage
            base_power = 100  # Base power for 100% brightness
            predicted_power = (brightness / 100.0) * base_power
            predicted_power = np.clip(predicted_power, 5, 200)
            print(f"Brightness-based prediction for {device_type}: {predicted_power:.2f}W")
            return predicted_power
        
        # For other devices, use the LSTM model
        # Prepare sequence for prediction
        sequence = self._prepare_sequence(device_data)
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
            
        # Ensure prediction is in the right shape for inverse transform
        prediction = prediction.numpy().reshape(-1, 1)
        
        # Get device type and corresponding scaler
        power_scaler = self.power_scalers.get(device_type)
        if power_scaler is None:
            print(f"Warning: No scaler found for {device_type}, using AC scaler")
            power_scaler = self.power_scalers['AC']
        
        # Scale back to original scale
        prediction = power_scaler.inverse_transform(prediction)
        predicted_power = float(prediction[0][0])  # Convert to Python float
        print(f"Raw prediction for {device_type}: {predicted_power:.2f}W")
        
        # Validate prediction based on device type and clip to reasonable ranges
        if device_type == 'AC':
            predicted_power = np.clip(predicted_power, 500, 3000)
        elif device_type == 'Water_Heater':
            predicted_power = np.clip(predicted_power, 1000, 4000)
        elif device_type == 'Refrigerator':
            predicted_power = np.clip(predicted_power, 100, 400)
            
        print(f"Final prediction for {device_type}: {predicted_power:.2f}W")
        return predicted_power
    
    def generate_recommendations(self, device_id):
        """Generate optimization recommendations for a device."""
        device_data = self.device_settings[self.device_settings['device_id'] == device_id]
        if device_data.empty:
            return []
        
        device_type = device_data['device_type'].iloc[0]
        recommendations = []
        
        # Special handling for lighting
        if device_type == 'Light':
            brightness = float(device_data['brightness'].iloc[0])
            schedule = device_data['schedule'].iloc[0]
            
            # Calculate expected power based on brightness
            base_power = 100  # Base power for traditional bulb at 100% brightness
            expected_power = (brightness / 100.0) * base_power
            
            # Generate lighting-specific recommendations
            if expected_power > 150:
                recommendations.append("Switch to LED bulbs immediately (up to 90% energy savings)")
                recommendations.append("Install motion sensors in low-traffic areas")
                recommendations.append("Consider using task lighting instead of overhead lighting")
            elif expected_power > 100:
                recommendations.append("Switch to LED bulbs if not already using them")
                recommendations.append("Use natural light during daytime when possible")
            elif expected_power > 50:
                recommendations.append("Consider installing dimmers for flexible control")
                recommendations.append("Adjust brightness based on time of day")
            
            # Add schedule-based recommendations
            if schedule == 'always_on':
                recommendations.append("Consider using motion sensors or timers to reduce unnecessary usage")
            elif schedule == 'off_peak':
                recommendations.append("Verify off-peak schedule matches actual usage patterns")
            
            return recommendations
        
        # For other devices, use the model prediction
        current_power = self.predict_power_consumption(device_data)
        
        if device_type == 'AC':
            # AC-specific recommendations
            if current_power > 2500:  # Very high consumption
                recommendations.append("Consider increasing temperature by 2-3°C during peak hours")
                recommendations.append("Schedule AC to turn off 30 minutes before leaving")
                recommendations.append("Check for any air leaks or insulation issues")
            elif current_power > 2000:  # High consumption
                recommendations.append("Consider increasing temperature by 1-2°C during peak hours")
                recommendations.append("Use ceiling fans to improve air circulation")
            elif current_power > 1500:  # Moderate consumption
                recommendations.append("Check air filter condition and clean if necessary")
                recommendations.append("Ensure windows and doors are properly sealed")
        
        elif device_type == 'Water_Heater':
            # Water heater recommendations
            if current_power > 3500:  # Very high consumption
                recommendations.append("Lower water temperature by 10°F")
                recommendations.append("Check for leaks and insulation issues")
                recommendations.append("Consider upgrading to a more efficient model")
            elif current_power > 3000:  # High consumption
                recommendations.append("Lower water temperature by 5°F")
                recommendations.append("Install a timer to reduce off-hours operation")
            elif current_power > 2500:  # Moderate consumption
                recommendations.append("Add insulation to hot water pipes")
                recommendations.append("Use cold water for laundry when possible")
        
        elif device_type == 'Refrigerator':
            # Refrigerator recommendations
            if current_power > 300:  # Very high consumption
                recommendations.append("Check door seals for leaks")
                recommendations.append("Ensure proper ventilation around the unit")
                recommendations.append("Consider upgrading to an ENERGY STAR model")
            elif current_power > 250:  # High consumption
                recommendations.append("Clean condenser coils")
                recommendations.append("Check temperature settings")
            elif current_power > 200:  # Moderate consumption
                recommendations.append("Keep refrigerator well-stocked for better efficiency")
                recommendations.append("Avoid placing hot food directly in refrigerator")
        
        return recommendations
    
    def analyze_device(self, device_id):
        """Analyze a single device and return optimization insights."""
        device_data = self.device_settings[self.device_settings['device_id'] == device_id]
        if device_data.empty:
            return None
        
        device_type = device_data['device_type'].iloc[0]
        
        # Special handling for lighting
        if device_type == 'Light':
            brightness = float(device_data['brightness'].iloc[0])
            # Calculate power based on brightness
            base_power = 100  # Base power for traditional bulb at 100% brightness
            current_power = (brightness / 100.0) * base_power
            current_power = np.clip(current_power, 5, 200)
        else:
            # For other devices, use the model prediction
            current_power = self.predict_power_consumption(device_data)
        
        recommendations = self.generate_recommendations(device_id)
        
        return {
            'device_id': device_id,
            'device_type': device_type,
            'current_power': current_power,
            'recommendations': recommendations
        }
    
    def generate_report(self):
        """Generate a comprehensive power optimization report."""
        report = []
        
        for device_id in self.device_settings['device_id'].unique():
            analysis = self.analyze_device(device_id)
            if analysis:
                report.append(analysis)
        
        return report
    
    def _initialize_feature_mappings(self):
        """Initialize feature mappings for device settings."""
        feature_mappings = {}
        
        # Add numerical features
        numerical_features = ['temperature', 'brightness', 'fan_speed']
        for i, feature in enumerate(numerical_features):
            feature_mappings[feature] = {'type': 'numerical', 'index': i}
        
        # Add categorical features
        categorical_features = {
            'device_type': ['AC', 'Light'],
            'schedule': ['always_on', 'off_peak', 'motion_sensor']
        }
        
        current_idx = len(numerical_features)
        for feature, values in categorical_features.items():
            feature_mappings[feature] = {
                'type': 'categorical',
                'values': values,
                'index': current_idx,
                'size': len(values)
            }
            current_idx += len(values)
        
        return feature_mappings
    
    def _fit_scalers(self):
        """Fit scalers with all available data."""
        # Prepare all features
        all_features = []
        device_type_power = {}  # Track power consumption by device type
        
        # First, collect all features without scaling
        for device_id in self.device_settings['device_id'].unique():
            device_data = self.device_settings[self.device_settings['device_id'] == device_id]
            device_type = device_data['device_type'].iloc[0]
            
            # Initialize feature array
            features = np.zeros((len(device_data), 13))  # 13 features as per training
            
            # 1. Original features
            features[:, 0] = device_data['temperature'].values
            features[:, 1] = device_data['brightness'].values
            
            # 2. Fan speed one-hot encoding (3 features)
            if 'fan_speed' in device_data.columns:
                features[:, 2] = (device_data['fan_speed'] == 'low').astype(int)
                features[:, 3] = (device_data['fan_speed'] == 'medium').astype(int)
                features[:, 4] = (device_data['fan_speed'] == 'high').astype(int)
            
            # 3. Schedule one-hot encoding (3 features)
            features[:, 5] = (device_data['schedule'] == 'always_on').astype(int)
            features[:, 6] = (device_data['schedule'] == 'off_peak').astype(int)
            features[:, 7] = (device_data['schedule'] == 'motion_sensor').astype(int)
            
            # 4. Temporal features (5 features)
            if 'timestamp' in device_data.columns:
                timestamps = pd.to_datetime(device_data['timestamp'])
                features[:, 8] = timestamps.dt.hour.values / 24  # Normalized hour
                features[:, 9] = timestamps.dt.dayofweek.values / 7  # Normalized day
                features[:, 10] = timestamps.dt.month.values / 12  # Normalized month
                features[:, 11] = ((timestamps.dt.hour >= 17) & (timestamps.dt.hour <= 21)).astype(int)  # Peak hours
                features[:, 12] = ((timestamps.dt.hour < 6) | (timestamps.dt.hour >= 22)).astype(int)  # Night
            
            all_features.append(features)
            
            # Get corresponding power consumption
            power_data = self.power_consumption[self.power_consumption['device_id'] == device_id]
            if not power_data.empty:
                if device_type not in device_type_power:
                    device_type_power[device_type] = []
                device_type_power[device_type].extend(power_data['power_consumption'].values)
        
        # Fit feature scaler
        all_features = np.vstack(all_features)
        self.feature_scaler.fit(all_features)
        
        # Initialize power scalers for each device type
        self.power_scalers = {}
        for device_type, power_values in device_type_power.items():
            scaler = StandardScaler()
            power_values = np.array(power_values).reshape(-1, 1)
            
            # Apply reasonable bounds before fitting scaler
            if device_type == 'AC':
                power_values = np.clip(power_values, 500, 3000)
            elif device_type == 'Light':
                power_values = np.clip(power_values, 5, 200)
            elif device_type == 'Water_Heater':
                power_values = np.clip(power_values, 1000, 4000)
            elif device_type == 'Refrigerator':
                power_values = np.clip(power_values, 100, 400)
                
            scaler.fit(power_values)
            self.power_scalers[device_type] = scaler
    
    def _prepare_sequence(self, device_data):
        """Prepare input sequence for the LSTM model."""
        # Initialize feature array
        features = np.zeros((len(device_data), 13))  # 13 features as per training
        
        # 1. Original features
        features[:, 0] = device_data['temperature'].values
        features[:, 1] = device_data['brightness'].values
        
        # 2. Fan speed one-hot encoding (3 features)
        if 'fan_speed' in device_data.columns:
            features[:, 2] = (device_data['fan_speed'] == 'low').astype(int)
            features[:, 3] = (device_data['fan_speed'] == 'medium').astype(int)
            features[:, 4] = (device_data['fan_speed'] == 'high').astype(int)
        
        # 3. Schedule one-hot encoding (3 features)
        features[:, 5] = (device_data['schedule'] == 'always_on').astype(int)
        features[:, 6] = (device_data['schedule'] == 'off_peak').astype(int)
        features[:, 7] = (device_data['schedule'] == 'motion_sensor').astype(int)
        
        # 4. Temporal features (5 features)
        if 'timestamp' in device_data.columns:
            timestamps = pd.to_datetime(device_data['timestamp'])
            features[:, 8] = timestamps.dt.hour.values / 24  # Normalized hour
            features[:, 9] = timestamps.dt.dayofweek.values / 7  # Normalized day
            features[:, 10] = timestamps.dt.month.values / 12  # Normalized month
            features[:, 11] = ((timestamps.dt.hour >= 17) & (timestamps.dt.hour <= 21)).astype(int)  # Peak hours
            features[:, 12] = ((timestamps.dt.hour < 6) | (timestamps.dt.hour >= 22)).astype(int)  # Night
        
        # Scale features
        features = self.feature_scaler.transform(features)
        
        return features

def main():
    """Main function to demonstrate usage"""
    # Initialize analyzer with data paths and LSTM model
    analyzer = PowerOptimizationAnalyzer(
        device_settings_path='data/device_settings.csv',
        power_consumption_path='data/power_consumption.csv',
        model_path='models/lstm_best_model.pth'  # Updated to use the correct model file extension
    )
    
    # Generate report for all devices
    report = analyzer.generate_report()
    
    # Save report
    print("\nSaving report...")
    with open('reports/power_optimization_report.csv', 'w') as f:
        # Write header
        f.write("device_id,device_type,current_power,recommendations\n")
        
        # Write data
        for row in report:
            # Convert recommendations to string
            recs = []
            for rec in row['recommendations']:
                recs.append(rec)
            recommendations = "; ".join(recs)
            
            # Write row
            f.write(f"{row['device_id']},{row['device_type']},{row['current_power']:.2f},\"{recommendations}\"\n")
    
    # Print summary
    print("\nPower Optimization Analysis Summary:")
    print(f"Total devices analyzed: {len(report)}")
    total_power = sum(row['current_power'] for row in report)
    avg_power = total_power / len(report) if report else 0
    print(f"Total current power consumption: {total_power:.2f} kWh")
    print(f"Average power consumption per device: {avg_power:.2f} kWh")
    print("\nReport saved to 'reports/power_optimization_report.csv'")

if __name__ == "__main__":
    main() 