import sys
import os
import numpy as np

print("Starting script...")

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Importing PowerOptimizationAnalyzer...")
from src.power_optimization_v2 import PowerOptimizationAnalyzer

def main():
    print("Creating reports directory...")
    # Create reports directory if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    print("Initializing analyzer...")
    # Initialize analyzer with data paths and LSTM model
    analyzer = PowerOptimizationAnalyzer(
        device_settings_path='data/device_settings.csv',
        power_consumption_path='data/power_consumption.csv',
        model_path='models/lstm_best_model.pth'  # Using our best LSTM model
    )
    
    # Generate report for all devices
    print("\nGenerating power optimization report...")
    report = analyzer.generate_report()
    
    # Save report
    print("Saving report...")
    with open('reports/power_optimization_report.csv', 'w') as f:
        # Write header
        f.write("device_id,device_type,current_power,recommendations\n")
        
        # Write data
        for row in report:
            # Convert recommendations to string
            recs = "; ".join(row['recommendations'])
            
            # Write row
            f.write(f"{row['device_id']},{row['device_type']},{row['current_power']:.2f},\"{recs}\"\n")
    
    # Print summary
    print("\nPower Optimization Analysis Summary:")
    print(f"Total devices analyzed: {len(report)}")
    total_power = sum(row['current_power'] for row in report)
    avg_power = total_power / len(report) if report else 0
    print(f"Total current power consumption: {total_power:.2f} W")
    print(f"Average power consumption per device: {avg_power:.2f} W")
    print("\nReport saved to 'reports/power_optimization_report.csv'")

if __name__ == "__main__":
    main() 