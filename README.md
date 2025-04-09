# Smart Home Power Consumption Analysis

## Project Overview
This project implements a deep learning-based system for forecasting and optimizing power consumption in smart homes. It uses LSTM networks to predict power consumption patterns and generate optimization recommendations for various household devices.

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

1. Clone the repository

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install required packages in requirements.txt:
```bash
pip install -r requirements.txt
```



## Running the Analysis

1. Ensure you have activated your virtual environment and installed all dependencies.

2. Run the power optimization analysis:
```bash
python src/run_power_optimization.py
```

This will:
- Load the trained model
- Analyze device power consumption
- Generate optimization recommendations
- Save the results to `reports/power_optimization_report.csv`

3. View the results:
- The analysis report will be saved in `reports/power_optimization_report.csv`
- The report includes:
  - Device-specific power consumption predictions
  - Optimization recommendations
  - Potential energy savings

## Expected Output
The analysis will generate predictions and recommendations for various devices:
- Air conditioners
- Lighting systems
- Water heaters
- Refrigerators

Each device will have:
- Current power consumption
- Predicted power consumption
- Optimization recommendations
- Potential savings
