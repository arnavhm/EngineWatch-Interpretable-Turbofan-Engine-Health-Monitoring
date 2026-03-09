"""Quick validation script for preprocess.py"""

from data.load import load_config, load_dataset
from data.preprocess import preprocess_train, preprocess_test


def main() -> None:
    """Run a quick manual validation of preprocessing pipeline."""
    config = load_config("config/config.yaml")
    train_df, test_df, _ = load_dataset(config)

    train_proc, scaler, sensor_cols = preprocess_train(train_df, config)
    test_proc = preprocess_test(test_df, config, scaler)

    print(f"\n✅ Preprocessing successful!")
    print(f"Train shape: {train_proc.shape}")
    print(f"Test shape: {test_proc.shape}")
    print(f'Train has RUL: {"RUL" in train_proc.columns}')
    print(f'Test has RUL: {"RUL" in test_proc.columns}')
    print(f'Train RUL range: [{train_proc["RUL"].min()}, {train_proc["RUL"].max()}]')
    print(f"Scaler statistics available for {scaler.n_features_in_} sensors")
    print(f"Scaler mean shape: {scaler.mean_.shape}")
    print(f"Scaler std shape: {scaler.scale_.shape}")
    print(f"Scaled sensor count: {len(sensor_cols)}")


if __name__ == "__main__":
    main()
