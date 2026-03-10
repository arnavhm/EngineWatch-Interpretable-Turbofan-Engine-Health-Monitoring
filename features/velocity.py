"""
features/velocity.py

Compute Health Index (HI) velocity — the rate of change of degradation per cycle.
Velocity is the slope of a linear regression of HI over a rolling window of cycles.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class VelocityArtifacts:
    """
    Container for velocity feature statistics and metadata.

    Attributes:
        window_size: Size of rolling window used for slope calculation
        mean_velocity: Mean slope across all engines and cycles
        std_velocity: Standard deviation of slopes
        max_velocity: Maximum (least negative) slope
        min_velocity: Minimum (most negative) slope
    """

    window_size: int
    mean_velocity: float
    std_velocity: float
    max_velocity: float
    min_velocity: float


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling linear regression slope over a pandas Series.

    Purpose:
        Apply numpy.polyfit over each full rolling window to extract the slope
        (velocity) of HI degradation. Velocity measures rate of change in health.

    Input:
        series: HI values for a single engine, indexed by position. Shape: (n_cycles,)
        window: Number of consecutive cycles per regression window. Must be > 1.

    Output:
        pd.Series of slopes (velocities), same length as input.
        First (window-1) values are NaN (rolling window cannot fit).

    Mathematical Definition:
        For each window position i to i+window:
            x = [0, 1, ..., window-1]  (relative cycle positions)
            y = HI values in window
            slope = polyfit(x, y, deg=1)[0]  (first element is slope)
        Units: change in HI per cycle (dimensionless)

    Assumptions:
        - Input is sorted chronologically
        - Health index values are in [0, 1]
        - Window size >= 2

    Failure Conditions:
        - ValueError raised if window < 2
        - Returns NaN for small engines (n_cycles < window)
    """
    if window < 2:
        raise ValueError(f"Window size must be >= 2, got {window}")

    x = np.arange(window)  # Relative cycle positions [0, 1, ..., window-1]

    def slope_of_window(y: np.ndarray) -> float:
        """Extract slope from window of HI values via polyfit."""
        return np.polyfit(x, y, 1)[0]

    return series.rolling(window=window, min_periods=window).apply(
        slope_of_window, raw=True
    )


def compute_velocity(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add HI_velocity column to DataFrame.

    Purpose:
        Compute per-engine rolling slope of health_index.
        Forward-fill NaN values at the start of each engine's window
        so downstream modules receive no missing values.

    Input:
        df: DataFrame with columns [unit, cycle, health_index].
            Shape: (n_rows, n_cols), e.g. (20631, 3+) for FD001 train
        config: Dict containing config["rolling"]["window_size"] (int, >= 2)

    Output:
        Same DataFrame with added column HI_velocity (float).
        Shape: (n_rows, n_cols+1)
        No rows dropped; NaN values forward-filled per engine.

    Assumptions:
        - health_index column exists and is in range [0, 1]
        - Sorted by unit then cycle (sort applied internally if needed)
        - Rolling window applied per engine (no cross-engine contamination)
        - Config has rolling.window_size key

    Failure Conditions:
        - KeyError: health_index column missing or config key missing
        - ValueError: window_size < 2 or health_index outside [0, 1]
    """
    # Validate required column
    if "health_index" not in df.columns:
        raise KeyError(
            "Column 'health_index' not found. "
            "Run features/health_index.py before computing velocity."
        )

    # Extract and validate window size
    try:
        window_size: int = config["rolling"]["window_size"]
    except KeyError:
        raise KeyError(
            "Config missing 'rolling.window_size'. "
            "Check config/config.yaml for rolling.window_size key."
        )

    if window_size < 2:
        raise ValueError(
            f"window_size must be >= 2, got {window_size}. "
            "Update config['rolling']['window_size']."
        )

    # Validate health_index range
    if not df["health_index"].between(0.0, 1.0).all():
        raise ValueError(
            "health_index contains values outside [0, 1]. "
            "Check health_index computation for normalization errors."
        )

    # Sort defensively — polyfit requires chronological order within each engine
    df = df.sort_values(["unit", "cycle"]).copy()

    # Compute rolling slope per engine
    # groupby prevents cross-engine contamination
    df["HI_velocity"] = df.groupby("unit")["health_index"].transform(
        lambda s: rolling_slope(s, window_size)
    )

    # Forward-fill NaN values at the start of each engine's window.
    # First (window_size - 1) cycles have NaN from rolling window.
    # bfill() propagates the first valid slope backward, providing
    # reasonable estimates for early cycles (which have nearly zero degradation).
    df["HI_velocity"] = df.groupby("unit")["HI_velocity"].transform(lambda s: s.bfill())

    return df


def build_velocity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, VelocityArtifacts]:
    """
    Build velocity feature for train and test sets.

    Purpose:
        Top-level API matching features/health_index.py pattern.
        Computes HI_velocity for both train and test, returning artifacts.

    Input:
        train_df: Training data with health_index column.
                  Shape: (n_train, n_cols+health_index)
        test_df: Test data with health_index column.
                 Shape: (n_test, n_cols+health_index)
        config: Configuration dict with rolling.window_size

    Output:
        train_velocity: DataFrame with added HI_velocity.
                        Shape: (n_train, n_cols+health_index+velocity)
        test_velocity: DataFrame with added HI_velocity.
                       Shape: (n_test, n_cols+health_index+velocity)
        artifacts: VelocityArtifacts with velocity statistics

    Assumptions:
        - Both DataFrames have health_index column
        - Config has rolling.window_size
        - health_index is normalized to [0, 1]

    Failure Conditions:
        - Raises KeyError/ValueError if validation fails (see compute_velocity)
    """
    # Compute velocity on both sets
    train_velocity = compute_velocity(train_df, config)
    test_velocity = compute_velocity(test_df, config)

    # Gather velocity statistics from combined data for artifact reporting
    all_velocity = pd.concat(
        [train_velocity["HI_velocity"], test_velocity["HI_velocity"]], ignore_index=True
    ).dropna()

    window_size: int = config["rolling"]["window_size"]

    artifacts = VelocityArtifacts(
        window_size=window_size,
        mean_velocity=float(all_velocity.mean()),
        std_velocity=float(all_velocity.std()),
        max_velocity=float(all_velocity.max()),
        min_velocity=float(all_velocity.min()),
    )

    return train_velocity, test_velocity, artifacts
