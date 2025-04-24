import sys
import os
import numpy as np

print("Starting script...")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Importing PowerOptimizationAnalyzer...")
from src.power_optimization_v2 import PowerOptimizationAnalyzer

def main():
    print("Creating reports directory...")
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    print("Initializing analyzer...")
    analyzer = PowerOptimizationAnalyzer(
        device_settings_path='data/device_settings.csv',
        power_consumption_path='data/power_consumption.csv',
        model_path='models/lstm_best_model.pth'
    )
    
    # Generate report
    print("\nGenerating power optimization report...")
    report = analyzer.generate_report()
    
    # Save report
    print("Saving report...")
    with open('reports/power_optimization_report.csv', 'w') as f:
        f.write("device_id,device_type,current_power,recommendations\n")
        
        for row in report:
            recs = "; ".join(row['recommendations'])
            
            f.write(f"{row['device_id']},{row['device_type']},{row['current_power']:.2f},\"{recs}\"\n")
    
    # summary
    print("\nPower Optimization Analysis Summary:")
    print(f"Total devices analyzed: {len(report)}")
    total_power = sum(row['current_power'] for row in report)
    avg_power = total_power / len(report) if report else 0
    print(f"Total current power consumption: {total_power:.2f} W")
    print(f"Average power consumption per device: {avg_power:.2f} W")
    print("\nReport saved to 'reports/power_optimization_report.csv'")

if __name__ == "__main__":
    main() 