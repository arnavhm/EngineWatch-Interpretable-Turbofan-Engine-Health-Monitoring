"""Quick validation script for preprocess.py"""

from data.load import load_config, load_dataset
from data.preprocess import preprocess_data, remove_flat_sensors

# Load config and data
config = load_config("config/config.yaml")
train_df, test_df, rul = load_dataset(config)

# Get selected sensors from config
sensor_list = config["selected_sensors"]
print(f"\nSelected sensors from config: {len(sensor_list)} sensors")

# Check for flat sensors first
active_sensors = remove_flat_sensors(train_df, sensor_list)
print(f"Active sensors after variance filter: {len(active_sensors)} sensors")

# Preprocess data
train_proc, test_proc, scaler = preprocess_data(
    train_df, test_df, active_sensors, compute_train_rul=True, compute_test_rul=False
)

print(f"\n✅ Preprocessing successful!")
print(f"Train shape: {train_proc.shape}")
print(f"Test shape: {test_proc.shape}")
print(f'Train has RUL: {"rul" in train_proc.columns}')
print(f'Test has RUL: {"rul" in test_proc.columns}')
print(f'Train RUL range: [{train_proc["rul"].min()}, {train_proc["rul"].max()}]')
print(f"Scaler statistics available for {scaler.n_features_in_} sensors")
print(f"Scaler mean shape: {scaler.mean_.shape}")
print(f"Scaler std shape: {scaler.scale_.shape}")
