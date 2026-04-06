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
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Feature columns - must match pipeline output exactly
RUL_FEATURES = ["health_index", "HI_velocity", "HI_variability", "risk_score"]


def _compute_piecewise_rul(
    train_df: pd.DataFrame,
    rul_clip: int,
    onset_threshold: float,
) -> np.ndarray:
    """
    Compute piecewise linear RUL targets for training.

    Purpose:
        The flat clip at max_rul_clip treats all early-life cycles as equally
        uninformative by assigning them the same target value. The piecewise
        approach is more physically honest:
        - Before degradation onset: RUL = max_rul_clip (flat, no information)
        - After degradation onset: RUL decreases linearly to 0 at failure

        Degradation onset is identified per engine as the first cycle where
        the health index drops below its initial value by more than a threshold
        (configurable normalized units). This avoids clipping on an arbitrary cycle
        number and grounds the target in the actual health signal.

    Mathematical definition:
        For each engine:
            onset_cycle = first cycle where HI < HI_initial - onset_threshold
            RUL(cycle) = min(max_rul_clip, max_cycle - cycle)
                         but capped at max_rul_clip for all cycles before onset

    Input:
        train_df — full training DataFrame with RUL and health_index columns
        rul_clip — maximum RUL value (from config)

    Output:
        numpy array of piecewise RUL targets, shape (n_train_rows,)

    Assumptions:
        - health_index column is present (requires pipeline to have run to Phase 3)
        - RUL column is present (computed by preprocess.py)
        - Engines are identified by unit column

    Failure conditions:
        - Missing health_index column -> KeyError
        - Missing RUL column -> KeyError
    """
    if "health_index" not in train_df.columns:
        raise KeyError(
            "health_index column required for piecewise RUL computation. "
            "Ensure Phase 3 (health index) has run before calling build_rul_model()."
        )
    if "RUL" not in train_df.columns:
        raise KeyError("RUL column missing from train_df.")

    piecewise_targets = np.zeros(len(train_df))

    for unit, group in train_df.groupby("unit"):
        group = group.sort_values("cycle")
        idx = group.index

        # Identify degradation onset: first cycle where HI drops onset_threshold below
        # the engine's initial health index value
        hi_initial = group["health_index"].iloc[0]
        onset_mask = group["health_index"] < (hi_initial - onset_threshold)

        if onset_mask.any():
            onset_rows = group[onset_mask]
            onset_cycle = onset_rows["cycle"].iloc[0]
        else:
            # No detectable onset — treat entire trajectory as pre-onset
            onset_cycle = group["cycle"].max() + 1

        raw_rul = group["RUL"].to_numpy(dtype=float)

        # Apply piecewise: cap at rul_clip before onset, linear after
        clipped = np.minimum(raw_rul, float(rul_clip))

        # Before onset: force to rul_clip (flat region)
        before_onset = group["cycle"].values < onset_cycle
        clipped[before_onset] = rul_clip

        piecewise_targets[idx] = clipped

    return piecewise_targets


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
        confidence_intervals - dict: model_name -> interval stats or None
        rul_clip - max RUL value used for clipping targets during training
        model_used - same as best_model_name (explicit field for Dashboard use)
    """

    best_model: object
    best_model_name: str
    all_models: dict
    feature_columns: list
    evaluation_metrics: dict
    feature_importance: dict
    confidence_intervals: dict
    prediction_balance: dict
    rul_clip: int
    model_used: str


def _validate_features(df: pd.DataFrame, require_rul: bool = False) -> None:
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
    models_cfg = config["rul"].get("models", {})
    random_forest_n_estimators = models_cfg.get("random_forest_n_estimators", 50)
    gradient_boosting_n_estimators = models_cfg.get(
        "gradient_boosting_n_estimators", 100
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=random_forest_n_estimators,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=gradient_boosting_n_estimators,
            random_state=random_state,
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


def _compute_confidence_intervals(
    models: dict,
    X_test: np.ndarray,
    confidence: float = 1.0,
) -> dict:
    """
    Compute prediction confidence intervals using Random Forest tree variance.

    Purpose:
        A single RUL prediction carries no uncertainty information.
        Random Forest trains n_estimators individual trees, each producing
        a slightly different prediction. The standard deviation across trees
        quantifies how much the model disagrees with itself - a direct
        measure of prediction uncertainty.

        Wide interval = model is uncertain, inspect sooner.
        Narrow interval = model is confident in its prediction.

    Mathematical definition:
        For Random Forest with n trees producing predictions p_1...p_n:
            mean     = (1/n) * sum(p_i)
            std      = sqrt((1/n) * sum((p_i - mean)^2))
            interval = mean +/- (confidence * std)

    Input:
        models     - dict of fitted models
        X_test     - test feature matrix (n_engines, 4)
        confidence - multiplier for interval width (default 1.0 = +/- 1 std)

    Output:
        dict mapping model_name -> {"lower": array, "upper": array, "std": array}
        Only populated for models with tree-level predictions (Random Forest).
        Other models return None.

    Assumptions:
        - Random Forest is present in models dict
        - n_estimators >= 10 for meaningful variance estimate
    """
    intervals = {}

    for name, model in models.items():
        if isinstance(model, RandomForestRegressor):
            # Collect predictions from each individual tree
            tree_preds = np.array(
                [tree.predict(X_test) for tree in model.estimators_]
            )  # shape: (n_estimators, n_engines)

            mean_pred = tree_preds.mean(axis=0)
            std_pred = tree_preds.std(axis=0)

            intervals[name] = {
                "lower": np.clip(mean_pred - confidence * std_pred, 0, None),
                "upper": mean_pred + confidence * std_pred,
                "std": std_pred,
            }
        else:
            intervals[name] = None

    return intervals


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
    onset_threshold = config["rul"]["piecewise_onset_threshold"]
    save_path = config["rul"]["save_path"]

    # Prepare training data - clip RUL targets
    X_train = train_df[RUL_FEATURES].values
    y_train = _compute_piecewise_rul(train_df, rul_clip, onset_threshold)

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

    # Compute Random Forest confidence intervals for display in the dashboard
    confidence_intervals = _compute_confidence_intervals(models, X_test)

    # Build predictions DataFrame using best model
    best_preds = predictions_per_model[best_model_name]
    best_errors = best_preds - y_test
    prediction_balance = {
        "late": int(np.sum(best_errors > 0)),
        "early": int(np.sum(best_errors < 0)),
        "on_time": int(np.sum(best_errors == 0)),
    }
    rf_ci = confidence_intervals.get("random_forest")
    test_predictions_df = pd.DataFrame(
        {
            "unit": test_last_cycle["unit"].values,
            "true_RUL": y_test,
            "predicted_RUL": np.round(best_preds).astype(int),
            "error": np.round(best_errors, 2),
            "model_used": best_model_name,
            "ci_lower": np.round(rf_ci["lower"]).astype(int) if rf_ci else None,
            "ci_upper": np.round(rf_ci["upper"]).astype(int) if rf_ci else None,
            "ci_std": np.round(rf_ci["std"], 1) if rf_ci else None,
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
        confidence_intervals=confidence_intervals,
        prediction_balance=prediction_balance,
        rul_clip=rul_clip,
        model_used=best_model_name,
    )

    # Save artifacts to disk
    _save_artifacts(artifacts, save_path)

    return test_predictions_df, artifacts
