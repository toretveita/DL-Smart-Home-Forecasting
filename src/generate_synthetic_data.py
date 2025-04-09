import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_device_settings(num_devices=10, num_days=30):
    """Generate synthetic device settings data more efficiently."""
    device_types = ['AC', 'Refrigerator', 'Water_Heater', 'Lighting']
    fan_speeds = ['low', 'medium', 'high']
    schedules = ['always_on', 'off_peak', 'motion_sensor']
    
    # Generate timestamps for all readings at once
    base_date = datetime(2023, 1, 1)
    timestamps = [base_date + timedelta(hours=h) for h in range(24 * num_days)]
    
    devices = []
    for device_id in range(1, num_devices + 1):
        device_type = random.choice(device_types)
        
        # Base settings based on device type
        if device_type == 'AC':
            base_temp = random.uniform(22, 26)
            base_fan = random.choice(fan_speeds)
            base_schedule = random.choice(schedules)
        elif device_type == 'Refrigerator':
            base_temp = random.uniform(3, 5)
            base_fan = 'medium'
            base_schedule = 'always_on'
        elif device_type == 'Water_Heater':
            base_temp = random.uniform(55, 65)
            base_fan = 'low'
            base_schedule = random.choice(['always_on', 'off_peak'])
        else:  # Lighting
            base_temp = 0
            base_fan = 'low'
            base_schedule = random.choice(schedules)
        
        # Generate all readings for this device at once
        device_readings = []
        for timestamp in timestamps:
            current_hour = timestamp.hour
            current_month = timestamp.month
            is_weekend = timestamp.weekday() >= 5
            
            # Calculate temperature with seasonal variation
            if device_type == 'AC':
                seasonal_variation = 2 * np.sin(2 * np.pi * (current_month - 6) / 12)
                current_temp = base_temp + seasonal_variation
                if current_hour < 6 or current_hour >= 22:
                    current_temp += 1
            elif device_type == 'Refrigerator':
                seasonal_variation = 0.5 * np.sin(2 * np.pi * (current_month - 6) / 12)
                current_temp = base_temp + seasonal_variation
            else:
                current_temp = base_temp
            
            # Add small random variation
            current_temp += random.uniform(-0.5, 0.5)
            
            # Ensure temperature stays within bounds
            if device_type == 'AC':
                current_temp = max(18, min(30, current_temp))
            elif device_type == 'Refrigerator':
                current_temp = max(1, min(10, current_temp))
            elif device_type == 'Water_Heater':
                current_temp = max(40, min(80, current_temp))
            
            # Determine fan speed
            if device_type == 'AC' and 17 <= current_hour <= 21:
                current_fan = 'high' if random.random() > 0.3 else 'medium'
            else:
                current_fan = base_fan
            
            # Determine schedule
            if is_weekend and random.random() > 0.7:
                current_schedule = 'always_on'
            else:
                current_schedule = base_schedule
            
            # Calculate brightness for lighting
            if device_type == 'Lighting':
                base_brightness = 100 if 6 <= current_hour <= 18 else 30
                brightness = base_brightness * random.uniform(0.9, 1.1)
            else:
                brightness = 0
            
            device_readings.append({
                'timestamp': timestamp,
                'device_id': device_id,
                'device_type': device_type,
                'temperature': round(current_temp, 1),
                'fan_speed': current_fan,
                'schedule': current_schedule,
                'brightness': round(brightness, 1) if device_type == 'Lighting' else 0
            })
        
        devices.extend(device_readings)
        print(f"Generated settings for device {device_id}")
    
    return pd.DataFrame(devices)

def generate_power_consumption(device_settings, num_days=30):
    """Generate synthetic power consumption data more efficiently."""
    power_data = []
    
    # Group by device for more efficient processing
    for device_id, device_group in device_settings.groupby('device_id'):
        device_type = device_group['device_type'].iloc[0]
        
        # Base power consumption based on device type
        if device_type == 'AC':
            base_power = 1000
            temp_factor = 50
            fan_factor = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
            schedule_factor = {'always_on': 1.0, 'off_peak': 0.7, 'motion_sensor': 0.5}
        elif device_type == 'Refrigerator':
            base_power = 150
            temp_factor = 20
            fan_factor = {'low': 0.9, 'medium': 1.0, 'high': 1.1}
            schedule_factor = {'always_on': 1.0, 'off_peak': 1.0, 'motion_sensor': 1.0}
        elif device_type == 'Water_Heater':
            base_power = 2000
            temp_factor = 30
            fan_factor = {'low': 1.0, 'medium': 1.0, 'high': 1.0}
            schedule_factor = {'always_on': 1.0, 'off_peak': 0.6, 'motion_sensor': 0.8}
        else:  # Lighting
            base_power = 50
            temp_factor = 0
            fan_factor = {'low': 1.0, 'medium': 1.0, 'high': 1.0}
            schedule_factor = {'always_on': 1.0, 'off_peak': 0.8, 'motion_sensor': 0.3}
        
        # Process all readings for this device at once
        device_power = []
        for _, row in device_group.iterrows():
            timestamp = row['timestamp']
            current_hour = timestamp.hour
            current_month = timestamp.month
            is_weekend = timestamp.weekday() >= 5
            is_night = current_hour < 6 or current_hour >= 22
            is_peak_hours = 17 <= current_hour <= 21
            
            # Calculate base power
            temp_diff = 24 - row['temperature'] if device_type == 'AC' else row['temperature'] - 4
            power = base_power + (temp_factor * max(0, temp_diff))
            power *= fan_factor[row['fan_speed']]
            power *= schedule_factor[row['schedule']]
            
            # Add temporal patterns
            if device_type == 'AC':
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (current_month - 6) / 12)
                peak_factor = 1.2 if is_peak_hours else 1.0
                night_factor = 0.7 if is_night else 1.0
                power *= seasonal_factor * peak_factor * night_factor
            elif device_type == 'Lighting':
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (current_month - 12) / 12)
                night_factor = 1.5 if is_night else 0.5
                weekend_factor = 0.8 if is_weekend else 1.0
                power *= seasonal_factor * night_factor * weekend_factor
            elif device_type == 'Water_Heater':
                time_factor = 1.3 if (6 <= current_hour <= 8) or (17 <= current_hour <= 19) else 1.0
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (current_month - 12) / 12)
                power *= time_factor * seasonal_factor
            elif device_type == 'Refrigerator':
                time_factor = 1.1 if 8 <= current_hour <= 20 else 1.0
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * (current_month - 6) / 12)
                power *= time_factor * seasonal_factor
            
            # Add brightness factor for lighting
            if device_type == 'Lighting':
                power *= row['brightness'] / 100
            
            # Add small random noise
            power *= random.uniform(0.98, 1.02)
            
            device_power.append({
                'timestamp': timestamp,
                'device_id': device_id,
                'power_consumption': round(power, 2)
            })
        
        power_data.extend(device_power)
        print(f"Generated power data for device {device_id}")
    
    return pd.DataFrame(power_data)

def main():
    print("Generating synthetic data...")
    
    # Generate device settings
    print("Generating device settings...")
    device_settings = generate_device_settings()
    device_settings.to_csv('data/device_settings.csv', index=False)
    print(f"Generated {len(device_settings)} device settings")
    
    # Generate power consumption data
    print("Generating power consumption data...")
    power_consumption = generate_power_consumption(device_settings)
    power_consumption.to_csv('data/power_consumption.csv', index=False)
    print(f"Generated {len(power_consumption)} power consumption readings")
    
    print("Data generation complete!")

if __name__ == "__main__":
    main() 