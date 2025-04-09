import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime
import os

class DeviceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

class LSTMModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=512, num_layers=3, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=0.3 if num_layers > 1 else 0)
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def add_temporal_patterns(data):
    """Add realistic temporal patterns to the data."""
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Add time-based features
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['month'] = data['timestamp'].dt.month
    
    # Add seasonal patterns
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)
    data['is_peak_hours'] = ((data['hour'] >= 17) & (data['hour'] <= 21)).astype(int)
    
    # Add device-specific patterns
    for device_id in data['device_id'].unique():
        device_mask = data['device_id'] == device_id
        device_type = data[device_mask]['device_type'].iloc[0]
        
        if device_type == 'AC':
            # AC usage patterns
            data.loc[device_mask, 'temperature_impact'] = np.where(
                data.loc[device_mask, 'temperature'] > 25,
                (data.loc[device_mask, 'temperature'] - 25) * 0.1,
                0
            )
            # Higher usage during peak hours
            data.loc[device_mask, 'power_consumption'] *= (1 + data.loc[device_mask, 'is_peak_hours'] * 0.2)
            # Lower usage at night
            data.loc[device_mask, 'power_consumption'] *= (1 - data.loc[device_mask, 'is_night'] * 0.3)
            
        elif device_type == 'Light':
            # Light usage patterns
            data.loc[device_mask, 'brightness_impact'] = np.where(
                data.loc[device_mask, 'brightness'] < 50,
                (50 - data.loc[device_mask, 'brightness']) * 0.05,
                0
            )
            # Higher usage during night
            data.loc[device_mask, 'power_consumption'] *= (1 + data.loc[device_mask, 'is_night'] * 0.4)
            # Lower usage during weekends
            data.loc[device_mask, 'power_consumption'] *= (1 - data.loc[device_mask, 'is_weekend'] * 0.2)
    
    return data

def prepare_sequences(data, sequence_length=168, stride=24):  # 168 hours = 1 week, stride = 24 hours
    """Prepare sequences for LSTM input with overlapping windows."""
    print("Preparing sequences...")
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    sequences = []
    targets = []
    
    # Take a smaller sample of devices
    device_ids = data['device_id'].unique()
    num_devices = min(10, len(device_ids))  # Use at most 10 devices
    selected_devices = np.random.choice(device_ids, num_devices, replace=False)
    
    print(f"Processing {num_devices} devices with sequence length {sequence_length} hours and stride {stride} hours")
    
    for device_id in selected_devices:
        device_data = data[data['device_id'] == device_id].sort_values('timestamp')
        
        # Create overlapping sequences with stride
        num_sequences = (len(device_data) - sequence_length) // stride
        if num_sequences <= 0:
            continue
            
        # Pre-allocate arrays
        device_sequences = np.zeros((num_sequences, sequence_length, 13))  # 13 features per time step
        device_targets = np.zeros(num_sequences)
        
        for i in range(num_sequences):
            start_idx = i * stride
            end_idx = start_idx + sequence_length
            sequence = device_data.iloc[start_idx:end_idx]
            
            # Target is the average power consumption for the next 24 hours
            target_start = end_idx
            target_end = min(target_start + 24, len(device_data))
            target = device_data.iloc[target_start:target_end]['power_consumption'].mean()
            
            # Extract features for the entire sequence at once
            features = np.zeros((sequence_length, 13))
            
            # Original features
            features[:, 0] = sequence['temperature'].values
            features[:, 1] = sequence['brightness'].values
            
            # Categorical features
            if sequence['device_type'].iloc[0] == 'AC':
                features[:, 2] = (sequence['fan_speed'] == 'low').astype(int)
                features[:, 3] = (sequence['fan_speed'] == 'medium').astype(int)
                features[:, 4] = (sequence['fan_speed'] == 'high').astype(int)
            
            features[:, 5] = (sequence['schedule'] == 'always_on').astype(int)
            features[:, 6] = (sequence['schedule'] == 'off_peak').astype(int)
            features[:, 7] = (sequence['schedule'] == 'motion_sensor').astype(int)
            
            # Temporal features
            if 'timestamp' in sequence.columns:
                features[:, 8] = sequence['timestamp'].dt.hour.values / 24  # Normalized hour
                features[:, 9] = sequence['timestamp'].dt.dayofweek.values / 7  # Normalized day
                features[:, 10] = sequence['timestamp'].dt.month.values / 12  # Normalized month
                features[:, 11] = ((sequence['timestamp'].dt.hour >= 17) & (sequence['timestamp'].dt.hour <= 21)).astype(int)  # Peak hours
                features[:, 12] = ((sequence['timestamp'].dt.hour < 6) | (sequence['timestamp'].dt.hour >= 22)).astype(int)  # Night
            else:
                # If no timestamp, use default values
                features[:, 8:13] = 0
            
            device_sequences[i] = features
            device_targets[i] = target
        
        sequences.extend(device_sequences)
        targets.extend(device_targets)
        
        print(f"Completed device {device_id} with {num_sequences} sequences")
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    print(f"Prepared {len(sequences)} sequences in total")
    return sequences, targets

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):
    """Train the LSTM model."""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10  # Early stopping patience
    no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/lstm_best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
    
    print(f'Test Metrics:')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R2 Score: {r2:.2f}')
    
    # Save predictions and actuals to CSV for later analysis
    results_df = pd.DataFrame({
        'actual': actuals,
        'predicted': predictions
    })
    results_df.to_csv('reports/model_predictions.csv', index=False)
    
    return predictions, actuals

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    device_settings = pd.read_csv('data/device_settings.csv')
    power_consumption = pd.read_csv('data/power_consumption.csv')
    
    # Convert timestamps
    device_settings['timestamp'] = pd.to_datetime(device_settings['timestamp'])
    power_consumption['timestamp'] = pd.to_datetime(power_consumption['timestamp'])
    
    # Sort by timestamp to ensure proper sequence creation
    device_settings = device_settings.sort_values('timestamp')
    power_consumption = power_consumption.sort_values('timestamp')
    
    # Use all available data
    print(f"Using all {len(device_settings)} samples for training")
    
    # Merge data
    data = pd.merge(device_settings, power_consumption, on=['device_id', 'timestamp'])
    
    # Sort by device_id and timestamp to ensure proper sequence creation
    data = data.sort_values(['device_id', 'timestamp'])
    
    # Prepare sequences with overlapping windows for more training data
    sequences, targets = prepare_sequences(data, sequence_length=168, stride=24)  # 1 week sequences, 1 day stride
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(sequences, targets, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create datasets and dataloaders
    train_dataset = DeviceDataset(X_train, y_train)
    val_dataset = DeviceDataset(X_val, y_val)
    test_dataset = DeviceDataset(X_test, y_test)
    
    # Increase batch size for better batch normalization
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model with increased capacity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Each sequence has 13 features per time step
    input_size = 13
    model = LSTMModel(input_size=input_size, hidden_size=512, num_layers=3).to(device)  # Increased hidden size for longer sequences
    
    # Training setup with learning rate scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Reduced learning rate, increased weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=100, device=device, scheduler=scheduler  # Increased epochs
    )
    
    # Save losses to CSV for later plotting
    losses_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    losses_df.to_csv('reports/training_history.csv', index=False)
    
    # Evaluate model
    print("Evaluating model...")
    predictions, actuals = evaluate_model(model, test_loader, device)
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main() 