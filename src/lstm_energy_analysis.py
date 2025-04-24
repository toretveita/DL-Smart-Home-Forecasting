import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import os
import time
from datetime import datetime
import sys

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please ensure TensorFlow is installed correctly in your virtual environment.")
    TENSORFLOW_AVAILABLE = False
    sys.exit(1)

from sklearn.metrics import mean_squared_error, r2_score

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    print("Checking GPU availability...")
    try:
        # Disable TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Check if TensorFlow is available
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow is not available. Please check your installation.")
            return False
            
        # List physical devices
        physical_devices = tf.config.list_physical_devices()
        print(f"Available physical devices: {physical_devices}")
        
        # List GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found GPU(s): {gpus}")
            try:
                for gpu in gpus:
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # Set memory limit to 80% of available memory
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*13)]  # 13GB limit for 16GB GPU
                    )
                    print(f"Successfully configured GPU: {gpu}")
                return True
            except RuntimeError as e:
                print(f"Error configuring GPU: {e}")
                return False
        else:
            print("No GPU devices found. Running on CPU.")
            return False
    except Exception as e:
        print(f"Error checking GPU: {e}")
        print("Continuing with CPU...")
        return False

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
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)

def create_sequences(features, targets, sequence_length):
    """Improved sequence creation with better normalization and overlap"""
    X, y = [], []
    step = sequence_length // 4  # 75% overlap between sequences
    
    for i in range(0, len(features) - sequence_length, step):
        # Only create sequence if we have valid target data
        if not np.isnan(targets[i + sequence_length - 1]):
            X.append(features[i:(i + sequence_length)])
            y.append(targets[i + sequence_length - 1])
    
    return np.array(X), np.array(y)

def preprocess_data(df):
    """Enhanced data preprocessing with robust scaling and feature engineering"""
    df_processed = df.copy()
    
    if 'timestamp' in df_processed.columns:
        df_processed = df_processed.drop('timestamp', axis=1)
    
    # more temporal features
    df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
    df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
    df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
    df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
    df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
    
    # lag features
    df_processed['device_1_lag1'] = df_processed['device_1'].shift(1)
    df_processed['device_1_lag24'] = df_processed['device_1'].shift(24)
    df_processed['device_3_lag1'] = df_processed['device_3'].shift(1)
    df_processed['device_3_lag24'] = df_processed['device_3'].shift(24)
    
    # rolling statistics
    df_processed['device_1_rolling_mean'] = df_processed['device_1'].rolling(window=24).mean()
    df_processed['device_1_rolling_std'] = df_processed['device_1'].rolling(window=24).std()
    df_processed['device_3_rolling_mean'] = df_processed['device_3'].rolling(window=24).mean()
    df_processed['device_3_rolling_std'] = df_processed['device_3'].rolling(window=24).std()
    
    df_processed = df_processed.ffill().bfill()
    
    target = df_processed['device_2'].copy()
    df_processed = df_processed.drop('device_2', axis=1)
    
    # scaling of features
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()
    
    # Scale features and target
    df_processed = pd.DataFrame(
        scaler_x.fit_transform(df_processed),
        columns=df_processed.columns
    )
    
    target = scaler_y.fit_transform(target.values.reshape(-1, 1))
    
    return df_processed, target, scaler_x, scaler_y

def create_lstm_model(input_shape, output_shape):
    """Create an improved LSTM model with bidirectional layers and residual connections"""
    model = Sequential([
        # First LSTM layer
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second LSTM layer
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Third LSTM layer
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(output_shape)
    ])
    
    # Compile
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Enhanced training process with learning rate scheduling and early stopping"""
    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'models/lstm_best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Training with class weights
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            early_stopping,
            lr_scheduler,
            model_checkpoint
        ],
        verbose=1
    )
    
    return model, history

def analyze_data_distribution(data):
    """Analyze the distribution of data across time periods"""
    print("\nData Distribution Analysis:")
    print("-" * 50)
    
    print("\nSamples per hour:")
    print(data['hour'].value_counts().sort_index())
    print("\nSamples per day of week:")
    print(data['day_of_week'].value_counts().sort_index())
    print("\nSamples per month:")
    print(data['month'].value_counts().sort_index())
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data['hour'], bins=24)
    plt.title('Distribution by Hour')
    plt.subplot(2, 2, 2)
    sns.histplot(data['day_of_week'], bins=7)
    plt.title('Distribution by Day of Week')
    plt.subplot(2, 2, 3)
    sns.histplot(data['month'], bins=12)
    plt.title('Distribution by Month')
    plt.subplot(2, 2, 4)
    for device in ['device_1', 'device_2', 'device_3']:
        sns.kdeplot(data[device], label=device)
    plt.title('Device Values Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('visualizations', 'lstm', 'lstm_data_distribution.png'))
    plt.close()

def main():
    try:
        # Set random seed
        np.random.seed(42)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(42)
        
        print("Starting energy analysis script...")
        
        # Setup GPU
        gpu_available = setup_gpu()
        if not gpu_available:
            print("Warning: Running on CPU. This may be slower.")
        
        # Create necessary directories
        print("\nCreating directories...")
        model_name = "lstm"  # Model name for organization
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs(f'visualizations/{model_name}', exist_ok=True)
        
        # Load data
        data_path = os.path.join('data', 'shib010_data.txt')
        print(f"\nLoading data from: {data_path}")
        df = load_data(data_path)
        
        print("\nData Shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        print("\nData Info:")
        print(df.info())
        
        # Plot temperature over time
        print("\nCreating visualizations...")
        plt.figure(figsize=(15, 8))
        plt.plot(df['timestamp'], df['device_2'], label='Temperature')
        plt.title('Temperature Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.savefig(os.path.join('visualizations', model_name, f'{model_name}_temperature.png'))
        plt.close()
        print("Temperature plot saved")
        
        # Plot temperature distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='device_2', bins=30)
        plt.title('Temperature Distribution')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Count')
        plt.savefig(os.path.join('visualizations', model_name, f'{model_name}_temperature_distribution.png'))
        plt.close()
        print("Distribution plot saved")
        
        # Analyze data distribution
        analyze_data_distribution(df)
        
        # Use the improved preprocessing
        print("\nPreprocessing data...")
        features_scaled, targets_scaled, scaler_x, scaler_y = preprocess_data(df)
        
        print("\nCreating sequences...")
        X, y = create_sequences(features_scaled.values, targets_scaled, sequence_length=48)
        print(f"Created {len(X)} sequences")
        
        # Split with validation set
        print("\nSplitting data...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Create and train the improved model
        print("\nCreating and training model...")
        model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=1)
        model, history = train_model(model, X_train, y_train, X_val, y_val)
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred_scaled = model.predict(X_test)
        
        # Inverse transform predictions
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, y_pred)
        
        print(f"\nModel Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Plot training history
        print("\nCreating training history plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name.upper()} Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join('visualizations', model_name, f'{model_name}_training_history.png'))
        plt.close()
        print("Training history plot saved")
        
        # Plot actual vs predicted values
        print("Creating predictions plot...")
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual[:100], label='Actual')
        plt.plot(y_pred[:100], label='Predicted')
        plt.title(f'{model_name.upper()} Model: Actual vs Predicted Temperature (First 100 Predictions)')
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.savefig(os.path.join('visualizations', model_name, f'{model_name}_predictions.png'))
        plt.close()
        print("Predictions plot saved")
        
        # Save model performance metrics
        with open(os.path.join('visualizations', model_name, f'{model_name}_metrics.txt'), 'w') as f:
            f.write(f"Model: {model_name.upper()}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R2 Score: {r2:.4f}\n")
            f.write(f"Training Time: {time.time() - start_time:.2f} seconds\n")
            f.write(f"Number of Epochs: {len(history.history['loss'])}\n")
            f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
            f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
        
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 