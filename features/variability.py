"""
features/variability.py

Purpose:
    Compute Health Index Variability — the rolling standard deviation of HI.
    Captures instability in the degradation trajectory.
    Healthy engines show low variability; engines near failure show higher variability.

Mathematical basis:
    For each window of `w` consecutive cycles within one engine:
        variability = std(HI_values in window)
    Uses Bessel's correction (ddof=1) — pandas rolling std default.

Input:
    DataFrame with columns: unit, cycle, health_index (output of health_index.py)
    config dict (window_size and min_periods from config["rolling"])

Output:
    Same DataFrame with added column: HI_variability (float, >= 0)
    Normalised to [0, 1] across the full training dataset.

Assumptions:
    - health_index column exists and is in [0, 1].
    - Rows are sorted by unit, then cycle (ascending).
    - Window is applied per engine — never across engine boundaries.
    - Normalisation parameters are fitted on training data only.

Failure conditions:
    - Missing health_index column → KeyError
    - window_size or min_periods not in config → KeyError
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class VariabilityArtifacts:
    """
    Container for variability feature statistics and normalisation parameters.

    Attributes:
        window_size: Size of rolling window used for std calculation
        min_periods: Minimum observations required for valid std
        var_min: Minimum raw variability (training data) — used for normalisation
        var_max: Maximum raw variability (training data) — used for normalisation
        mean_variability: Mean normalised variability across training set
        std_variability: Std of normalised variability across training set
    """

    window_size: int
    min_periods: int
    var_min: float
    var_max: float
    mean_variability: float
    std_variability: float


def _compute_raw_variability(
    df: pd.DataFrame, window: int, min_periods: int
) -> pd.Series:
    """
    Compute rolling std of health_index per engine.

    Purpose:
        Rolling std within each engine's lifecycle.
        min_periods allows partial windows at the start (avoids leading NaN).

    Input:  df with unit, health_index columns
            window      — full window size (e.g. 20)
            min_periods — minimum observations to compute std (e.g. 2)
    Output: pd.Series of raw variability values, same length as df

    Mathematical definition:
        var(t) = std(HI[t-w+1 : t+1])  using Bessel-corrected std (ddof=1)
    """
    return df.groupby("unit")["health_index"].transform(
        lambda s: s.rolling(window=window, min_periods=min_periods).std()
    )


def _normalise_variability(
    raw: pd.Series,
    var_min: Optional[float] = None,
    var_max: Optional[float] = None,
) -> tuple[pd.Series, float, float]:
    """
    Min-max normalise variability to [0, 1].

    Purpose:
        Make variability dimensionless and comparable to HI and velocity.
        Fitted on training data; applied identically to test data.

    Input:  raw     — raw rolling std values
            var_min — if None, compute from raw (training mode)
            var_max — if None, compute from raw (training mode)
    Output: (normalised Series, var_min used, var_max used)

    Failure: If var_max == var_min, raises ValueError (constant variability).
    """
    if var_min is None:
        var_min = float(raw.min())
    if var_max is None:
        var_max = float(raw.max())

    denom = var_max - var_min
    if denom == 0:
        raise ValueError(
            "Variability is constant across all cycles — cannot normalise. "
            "Check that health_index has non-zero variance."
        )

    normalised = (raw - var_min) / denom
    normalised = normalised.clip(
        0.0, 1.0
    )  # clip test data that may exceed training range
    return normalised, var_min, var_max


def compute_variability(
    df: pd.DataFrame,
    config: dict,
    artifacts: Optional[VariabilityArtifacts] = None,
) -> tuple[pd.DataFrame, VariabilityArtifacts]:
    """
    Add HI_variability column to DataFrame.

    Purpose:
        Compute per-engine rolling std of health_index, then normalise to [0, 1].
        If artifacts is None (training mode): fit normalisation on this data.
        If artifacts is provided (test mode): apply training normalisation params.

    Input:
        df: DataFrame with columns [unit, cycle, health_index].
            Shape: (n_rows, n_cols), e.g. (20631, 3+) for FD001 train
        config: Dict containing config["rolling"]["window_size"] and config["rolling"]["min_periods"]
        artifacts: VariabilityArtifacts from training run (None for train, required for test)

    Output:
        (DataFrame with HI_variability column, VariabilityArtifacts)
        Shape: (n_rows, n_cols+1)
        No rows dropped; NaN values filled with 0.0.

    Assumptions:
        - health_index column exists and is in range [0, 1]
        - Sorted by unit then cycle (sort applied internally if needed)
        - Rolling window applied per engine (no cross-engine contamination)
        - Config has rolling.window_size and rolling.min_periods keys

    Failure Conditions:
        - KeyError: health_index column missing or config keys missing
        - ValueError: window_size < 2, min_periods < 1, or variability is constant
    """
    # Validate required column
    if "health_index" not in df.columns:
        raise KeyError(
            "Column 'health_index' not found. "
            "Run features/health_index.py before computing variability."
        )

    # Extract and validate window parameters
    try:
        window: int = config["rolling"]["window_size"]
        min_periods: int = config["rolling"]["min_periods"]
    except KeyError:
        raise KeyError(
            "Config missing 'rolling.window_size' or 'rolling.min_periods'. "
            "Check config/config.yaml for required rolling settings."
        )

    if window < 2:
        raise ValueError(
            f"window_size must be >= 2, got {window}. "
            "Update config['rolling']['window_size']."
        )

    if min_periods < 1:
        raise ValueError(
            f"min_periods must be >= 1, got {min_periods}. "
            "Update config['rolling']['min_periods']."
        )

    # Validate health_index range
    if not df["health_index"].between(0.0, 1.0).all():
        raise ValueError(
            "health_index contains values outside [0, 1]. "
            "Check health_index computation for normalization errors."
        )

    # Sort defensively — rolling window requires chronological order within each engine
    df = df.sort_values(["unit", "cycle"]).copy()

    # Compute raw rolling std per engine
    raw_var = _compute_raw_variability(df, window, min_periods)

    # Fill any NaN from min_periods constraint with 0 (no variability yet)
    raw_var = raw_var.fillna(0.0)

    # Normalise — fit on train, apply fitted params on test
    if artifacts is None:
        # Training mode: fit normalisation parameters
        normalised, var_min, var_max = _normalise_variability(raw_var)
        mean_var = float(normalised.mean())
        std_var = float(normalised.std())
        artifacts = VariabilityArtifacts(
            window_size=window,
            min_periods=min_periods,
            var_min=var_min,
            var_max=var_max,
            mean_variability=mean_var,
            std_variability=std_var,
        )
    else:
        # Test mode: apply training normalisation parameters
        normalised, _, _ = _normalise_variability(
            raw_var, var_min=artifacts.var_min, var_max=artifacts.var_max
        )

    df["HI_variability"] = normalised
    return df, artifacts


# ------------------------------------------------------------------
# Top-level convenience function (matching health_index.py pattern)
# ------------------------------------------------------------------


def build_variability(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, VariabilityArtifacts]:
    """
    Build variability feature for train and test sets.

    Purpose:
        Top-level API matching features/health_index.py pattern.
        Fits normalisation on train, applies to test. Returns artifacts.

    Input:
        train_df: Training data with health_index column.
                  Shape: (n_train, n_cols+health_index)
        test_df: Test data with health_index column.
                 Shape: (n_test, n_cols+health_index)
        config: Configuration dict with rolling.window_size and rolling.min_periods

    Output:
        train_variability: DataFrame with added HI_variability.
                           Shape: (n_train, n_cols+health_index+variability)
        test_variability: DataFrame with added HI_variability.
                          Shape: (n_test, n_cols+health_index+variability)
        artifacts: VariabilityArtifacts with normalisation params and statistics

    Assumptions:
        - Both DataFrames have health_index column
        - Config has rolling.window_size and rolling.min_periods
        - health_index is normalized to [0, 1]

    Failure Conditions:
        - Raises KeyError/ValueError if validation fails (see compute_variability)
    """
    # Fit normalisation on training data
    train_variability, artifacts = compute_variability(train_df, config, artifacts=None)
    # Apply training normalisation to test data
    test_variability, _ = compute_variability(test_df, config, artifacts=artifacts)
    return train_variability, test_variability, artifacts
