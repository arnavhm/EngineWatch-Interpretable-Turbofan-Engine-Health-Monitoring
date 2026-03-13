"""
model/rul.py

Purpose:
    Predict Remaining Useful Life (RUL) using three interpretable regression baselines.
    Trained on engineered health features, not raw sensor data.

Input Features (from health monitoring pipeline):
    health_index, HI_velocity, HI_variability, risk_score

Target:
    RUL - clipped at max_rul_clip ( default 125 cycles) before training.

Training Protocol:
    - Train on all cycles of training dataset
    - RUL targets clipped at config["rul"]["max_rul_clip"] to focus on critical failure window.
    - Evaluate on last cycle of each test engine only (official CMAPSS protocol) for fair comparison.

Models:
    1. LinearRegression - interpretable baseline
    2.RandomForestRegressor - non-linear, feature importance insights
    3. GradientBoostingRegressor - powerful ensemble method, also provides feature importance, sequential boosting for improved performance

Evaluation Metrics:
    - RMSE - standard regression error metric, sensitive to large errors
    - NASA score - asymmetric penalty metric used in CMAPSS, penalizes late predictions more than early ones, reflects real-world cost of maintenance decisions
    e = predicted - actual
    e < 0: exp(-e/13) - 1
    e >= 0: exp(e/10) - 1

Public Interface:
    build_rul_model(train_df, test_df, config) -> (test_predictions_df, RULArtifacts)

Assumptions:
    - All four feature columns are present and NaN-free in the input dataframes.
    - RUL column exists in train_df (computed for preprocess.py).
    - test_df has true RUL values available for evaluation (last cycle of each engine).
    - models/ directory will be created if it does not exist.

Failure Conditions:
    - Missing feature columns -> KeyError
    - Missing RUL column in train_df -> KeyError
    - models/ directory not writable -> OSError

"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Feature columns - must match pipeline output exactly
RUL_FEATURES = ["health_index", "HI_velocity", "HI_variability", "risk_score"]


@dataclass
class RULArtifacts:
    """
    Container for all outputs and metadata from the RUL modeling process.

    Attributes:
        best_model.    - fitted model object with lowest RMSE
        best_model_name - string name of the best model
        all_models - dict of all three fitted models keyed by name
        feature_columns - list of input feature names
        evaluation_metrics - dict: model_name -> {"rmse": float, "nasa_score": float}
        feature_importance - dict: model_name -> pd.Series (RF and GB only)
        rul_clip - max RUL value used for clipping targets during training
        model_used - same as best_model_name (explicit field for Dashboard use)
    """

    best_model: object
    best_model_name: str
    all_models: dict
    feature_columns: list
    evaluation_metrics: dict
    feature_importance: dict
    rul_clip: int
    model_used: str


def _validate_features(df: pd.DataFrame, require_rul: bool = False):
    """
    Validate that required columns are present and NaN-free in the dataframe.

    Input: df       - DataFrame to validate
              require_rul - if True, also check for 'RUL' column
    Failure: KeyError if columns are missing, ValueError if NaNs are present
    """
    required = RUL_FEATURES + (["RUL"] if require_rul else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. Required: {required}"
            "  Ensure the full pipeline has run before calling build_rul_model()."
        )
    nan_counts = df[required].isna().sum()
    if nan_counts.any():
        raise ValueError(
            f"NaN values detected in required columns: {nan_counts[nan_counts > 0].to_dict()}. "
        )


def _nasa_score(errors: np.ndarray) -> float:
    """
    Compute the NASA asymettric scoring function.

    Purpose:
        Penalises late predictions (positive error) more heavily than early ones (negative error). Reflects aviation safety preference for conservative (early) maintenance decisions/predictions.

    Mathematical definition:
        e = predicted RUL - actual RUL
        score = sum(exp(-e/13) - 1) for e < 0 (early predictions)
        score = sum(exp(e/10) - 1) for e >= 0 (late predictions)

    Input: errors - numpy array of (predicted - true) values
    Output: float - total NASA score across all predictions (lower is better)
    """

    scores = np.where(errors < 0, np.exp(-errors / 13) - 1, np.exp(errors / 10) - 1)
    return float(np.sum(scores))


def _get_last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the last observed cycle for each test engine.

    Purpose:
        The official CMAPSS evaluation protocol focuses on the model's ability to predict RUL at the last observed cycle of each test engine, as this simulates the real-world scenario of making a maintenance decision based on the most recent data.

    Input: df - DataFrame containing test engine data with 'unit' and 'cycle' columns
    Output: DataFrame with one row per engine, containing the last cycle's data
    """

    return df.sort_values("cycle").groupby("unit").last().reset_index()


def _train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> dict:
    """
    Fit all three regression models on the training data.

    Input: X_train - feature matrix (n_rows, 4)
           y_train - clipped RUL targets (n_rows,)
           config  - loaded config dict (randoms state read from here)
    Output: dict mapping model name -> fitted model object
    """
    random_state = config["rul"]["random_state"]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=random_state
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def _evaluate_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, dict, str]:
    """
    Evaluate all models on last-cycle test data.

    Input: models - dict of fitted models
           X_test - test feature matrix (n_engines, 4)
           y_test - true RUL values (n_engines,)
    Output: (evaluation_metrics, predictions_per_model, best_model_name)

    evaluation_metrics: model_name -> {"rmse": float, "nasa_score": float}
    predictions_per_model: model_name -> np.array of predictions (n_engines,)
    """
    evaluation_metrics = {}
    predictions_per_model = {}
    best_rmse = float("inf")
    best_model_name = None

    for name, model in models.items():
        preds = model.predict(X_test)
        # Clip predictions to be non-negative, as RUL cannot be negative
        preds = np.clip(preds, 0, None)
        errors = preds - y_test

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        nasa = _nasa_score(errors)

        evaluation_metrics[name] = {"rmse": rmse, "nasa_score": nasa}
        predictions_per_model[name] = preds

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name

    return evaluation_metrics, predictions_per_model, best_model_name


def _extract_feature_importance(models: dict) -> dict:
    """
    Extract feature importance for tree-based models.

    Purpose:
        Provides interpretability - shows which health features drive RUL prediction.
        Linear regression has no feature importance in the sklearn sense.

    Input: models - dict of fitted models
    Output: dict mapping model name -> pd.Series of importances (RF and GB only)
    """
    importance = {}
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importance[name] = pd.Series(
                model.feature_importances_,
                index=RUL_FEATURES,
                name=f"{name}_importance",
            ).sort_values(ascending=False)
    return importance


def _save_artifacts(artifacts: RULArtifacts, save_path: str) -> None:
    """

    Persist RUL artifacts to disk using joblib.

    Purpose:
        Allows dashboard to load trained models without retraining.
        models/ directory will be created if it does not exist.

    Input: artifacts - RULArtifacts object
           save_path - directory path from config["rul"]["save_path"]
    """
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    # Save individual models
    for name, model in artifacts.all_models.items():
        joblib.dump(model, path / f"rul_{name}.joblib")

    # Save the entire artifacts object for easy loading in dashboard
    joblib.dump(artifacts, path / "rul_artifacts.joblib")


def build_rul_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, RULArtifacts]:
    """
    Top-level entry point for the RUL prediction pipeline.

    Purpose:
        1. Validates input data
        2. Clips RUL targets at max_rul_clip to focus on critical failure window
        3. Trains three regression models on all training cycles
        4. Evaluate on lastc cycle of each test engine (official protocol)
        5. Select best model based on RMSE
        6. Save artifacts to models/ directory
        7. Return predictions DataFrame and RULArtifacts for dashboard use

    Input:
        train_df - full training DataFrame with RUL column (20631 rows for FD001)
        test_df - full test DataFrame with RUL column (13096 rows for FD001)
        config - loaded config dict with rul parameters

    Output:
        test_predictions_df - DataFrame with columns: unit, true_RUL, predicted_RUL, error, model_used
        RULArtifacts - full artifact object

    Failure Conditions:
        - Missing feature or RUL columns -> KeyError
        - NaN values in features -> ValueError
        - models/ not writable -> OSError

    """
    # Validate both DataFrames
    _validate_features(train_df, require_rul=True)
    _validate_features(test_df, require_rul=True)

    rul_clip = config["rul"]["max_rul_clip"]
    save_path = config["rul"]["save_path"]

    # Prepare training data - clip RUL targets
    X_train = train_df[RUL_FEATURES].values
    y_train = np.clip(train_df["RUL"].values, 0, rul_clip)

    # Prepare test data - last cycle per engine only (official evaluation protocol)
    test_last_cycle = _get_last_cycle_per_engine(test_df)
    X_test = test_last_cycle[RUL_FEATURES].values
    y_test = test_last_cycle["RUL"].values

    # Train all three models
    models = _train_models(X_train, y_train, config)

    # Evaluate models and select best one
    evaluation_metrics, predictions_per_model, best_model_name = _evaluate_models(
        models, X_test, y_test
    )

    # Extract feature importance for tree-based models
    feature_importance = _extract_feature_importance(models)

    # Build predictions DataFrame using best model
    best_preds = predictions_per_model[best_model_name]
    test_predictions_df = pd.DataFrame(
        {
            "unit": test_last_cycle["unit"].values,
            "true_RUL": y_test,
            "predicted_RUL": np.round(best_preds).astype(int),
            "error": np.round(best_preds - y_test, 2),
            "model_used": best_model_name,
        }
    )

    # Assemble artifacts
    artifacts = RULArtifacts(
        best_model=models[best_model_name],
        best_model_name=best_model_name,
        all_models=models,
        feature_columns=RUL_FEATURES,
        evaluation_metrics=evaluation_metrics,
        feature_importance=feature_importance,
        rul_clip=rul_clip,
        model_used=best_model_name,
    )

    # Save artifacts to disk
    _save_artifacts(artifacts, save_path)

    return test_predictions_df, artifacts
