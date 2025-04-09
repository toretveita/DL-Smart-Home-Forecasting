import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import time
from datetime import datetime
import sys
import traceback
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import warnings

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
# Also suppress other common warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def setup_directories():
    """Create necessary directories for outputs"""
    try:
        os.makedirs('visualizations/traditional_ml', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        print("Directories created successfully")
    except Exception as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)

def load_data(file_path):
    """
    Load and preprocess the energy consumption data
    """
    print(f"\nAttempting to load data from: {file_path}")
    start_time = time.time()
    
    try:
        # Read the data file
        print("Reading CSV file...")
        df = pd.read_csv(file_path, delimiter='\t', header=None)
        print(f"Successfully read {len(df)} rows")
        
        # Assuming the first column is timestamp and the rest are energy consumption values
        df.columns = ['timestamp'] + [f'device_{i}' for i in range(1, len(df.columns))]
        print(f"Columns: {df.columns.tolist()}")
        
        # Convert timestamp to datetime if needed
        print("Converting timestamp...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert device_2 to numeric, replacing any non-numeric values with NaN
        print("Processing device_2 data...")
        df['device_2'] = pd.to_numeric(df['device_2'], errors='coerce')
        
        # Drop device_4 as it contains all NaN values
        df = df.drop('device_4', axis=1)
        
        # Convert categorical variables to numeric using label encoding
        print("Encoding categorical variables...")
        le = LabelEncoder()
        df['device_1'] = le.fit_transform(df['device_1'])
        df['device_3'] = le.fit_transform(df['device_3'])
        
        # Add time-based features
        print("Adding time-based features...")
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        print(f"Final data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Error type: {type(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)

def create_features(df):
    """Create features for the traditional ML models"""
    print("\nCreating features...")
    
    try:
        # Create a copy of the dataframe
        data = df.copy()
        
        # Add lag features
        print("Adding lag features...")
        for col in ['device_1', 'device_2', 'device_3']:
            for lag in [1, 2, 3, 6, 12, 24]:
                data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Add rolling features
        print("Adding rolling features...")
        for col in ['device_2']:
            for window in [3, 6, 12, 24, 48]:
                data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                data[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()
                data[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window).max()
                data[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window).min()
        
        # Add exponential moving averages
        print("Adding exponential moving averages...")
        for col in ['device_2']:
            for span in [3, 6, 12, 24]:
                data[f'{col}_ema_{span}'] = data[col].ewm(span=span).mean()
        
        # Add cyclical time features
        print("Adding cyclical time features...")
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Drop rows with NaN values
        print("Dropping rows with NaN values...")
        original_len = len(data)
        data = data.dropna()
        print(f"Dropped {original_len - len(data)} rows with NaN values")
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col not in ['timestamp', 'device_2']]
        X = data[feature_columns]
        y = data['device_2']
        
        print(f"\nFeature creation completed:")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Feature names: {feature_columns}")
        print(f"Data shape: {X.shape}")
        
        return X, y, feature_columns
    except Exception as e:
        print(f"Error creating features: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple traditional ML models"""
    print("\nTraining and evaluating models...")
    
    try:
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data using time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Store results
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            try:
                # Perform cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    mse = mean_squared_error(y_val, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_val, y_pred)
                    
                    cv_scores.append({
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2
                    })
                
                # Calculate average metrics
                avg_metrics = {
                    'mse': np.mean([score['mse'] for score in cv_scores]),
                    'rmse': np.mean([score['rmse'] for score in cv_scores]),
                    'r2': np.mean([score['r2'] for score in cv_scores])
                }
                
                # Train final model on all data
                model.fit(X_scaled, y)
                
                # Store results
                results[name] = {
                    'model': model,
                    'metrics': avg_metrics,
                    'training_time': time.time() - start_time,
                    'feature_importance': dict(zip(X.columns, model.feature_importances_)) if hasattr(model, 'feature_importances_') else None
                }
                
                print(f"{name} training completed in {results[name]['training_time']:.2f} seconds")
                print(f"Average MSE: {avg_metrics['mse']:.4f}")
                print(f"Average RMSE: {avg_metrics['rmse']:.4f}")
                print(f"Average R2: {avg_metrics['r2']:.4f}")
            
            except Exception as e:
                print(f"Error training {name}: {e}")
                print("Traceback:")
                print(traceback.format_exc())
                continue
        
        return results, scaler
    except Exception as e:
        print(f"Error in train_and_evaluate_models: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)

def plot_results(results, X, y, scaler):
    """Create visualizations for model results"""
    print("\nCreating visualizations...")
    
    try:
        # Plot feature importance for each model
        plt.figure(figsize=(15, 5))
        for i, (name, result) in enumerate(results.items()):
            if result['feature_importance']:
                plt.subplot(1, 3, i+1)
                importance = pd.Series(result['feature_importance']).sort_values(ascending=True)
                importance.plot(kind='barh')
                plt.title(f'{name} Feature Importance')
                plt.tight_layout()
        plt.savefig('visualizations/traditional_ml/feature_importance.png')
        plt.close()
        
        # Plot actual vs predicted values
        plt.figure(figsize=(15, 5))
        for i, (name, result) in enumerate(results.items()):
            plt.subplot(1, 3, i+1)
            y_pred = result['model'].predict(scaler.transform(X))
            plt.plot(y.values[:100], label='Actual', alpha=0.7)
            plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
            plt.title(f'{name} Predictions')
            plt.legend()
            plt.tight_layout()
        plt.savefig('visualizations/traditional_ml/predictions.png')
        plt.close()
        
        # Plot error distributions
        plt.figure(figsize=(15, 5))
        for i, (name, result) in enumerate(results.items()):
            plt.subplot(1, 3, i+1)
            y_pred = result['model'].predict(scaler.transform(X))
            errors = y_pred - y
            sns.histplot(errors, kde=True)
            plt.title(f'{name} Error Distribution')
            plt.tight_layout()
        plt.savefig('visualizations/traditional_ml/error_distributions.png')
        plt.close()
        
        # Save metrics to file
        with open('visualizations/traditional_ml/metrics.txt', 'w') as f:
            f.write("Traditional ML Models Performance Metrics\n")
            f.write("=======================================\n\n")
            for name, result in results.items():
                f.write(f"{name}:\n")
                f.write(f"Training Time: {result['training_time']:.2f} seconds\n")
                f.write(f"MSE: {result['metrics']['mse']:.4f}\n")
                f.write(f"RMSE: {result['metrics']['rmse']:.4f}\n")
                f.write(f"R2 Score: {result['metrics']['r2']:.4f}\n")
                f.write("\n")
        
        print("Visualizations and metrics saved successfully")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)

def main():
    """Main execution function"""
    try:
        print("\nCreating directories...")
        setup_directories()
        
        print("\nLoading data...")
        data_path = os.path.join('data', 'shib010_data.txt')
        df = load_data(data_path)
        
        print("\nCreating features...")
        X, y, feature_columns = create_features(df)
        
        print("\nTraining and evaluating models...")
        results, scaler = train_and_evaluate_models(X, y)
        
        print("\nCreating visualizations...")
        plot_results(results, X, y, scaler)
        
        print("\nAll tasks completed successfully!")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        print(f"Error type: {type(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 