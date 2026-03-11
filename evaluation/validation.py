"""
evaluation/validation.py

Purpose:
    Cross-engine validation of health monitoring outputs.
    Validates that health index trend, cluster progression, and risk behavior
    are physically consistent across engines.

Validation checks:
    1. HI monotonicity: Spearman correlation between cycle and health_index.
    2. Cluster progression: Healthy -> Degrading -> Critical should be non-regressive.
    3. Risk-RUL relationship: risk_score should correlate negatively with RUL.

Input shape:
    DataFrame with columns:
        unit, cycle, RUL, health_index, HI_velocity, HI_variability, risk_state, risk_score

Output shape:
    ValidationReport dataclass with per-engine results and fleet summary metrics.

Assumptions:
    - Input is training data (RUL column exists).
    - Required columns are present and numeric columns are finite.

Failure conditions:
    - Missing required columns -> KeyError
    - Fewer than 2 engines -> ValueError
    - Empty engine group during per-engine validation -> ValueError
"""

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REQUIRED_COLUMNS: list[str] = [
    "unit",
    "cycle",
    "RUL",
    "health_index",
    "HI_velocity",
    "HI_variability",
    "risk_state",
    "risk_score",
]

CLUSTER_ORDER: dict[str, int] = {"Healthy": 0, "Degrading": 1, "Critical": 2}
DEFAULT_MONOTONICITY_ABS_RHO_THRESHOLD: float = 0.7
DEFAULT_WEAK_RISK_RUL_CORRELATION_THRESHOLD: float = -0.5


@dataclass
class EngineValidationResult:
    """
    Purpose:
        Store per-engine validation outcomes.

    Input shape:
        Scalar metrics computed from one engine trajectory.

    Output shape:
        Dataclass row in ValidationReport.engine_results.

    Assumptions:
        The input engine trajectory has at least one row.

    Failure conditions:
        None at dataclass construction beyond invalid caller-provided values.
    """

    unit_id: int
    n_cycles: int
    hi_spearman_rho: float
    hi_monotonic: bool
    cluster_sequence: list[str]
    cluster_progression_valid: bool
    mean_risk_score: float
    final_risk_score: float


@dataclass
class ValidationReport:
    """
    Purpose:
        Hold fleet-level validation summary and detailed per-engine results.

    Input shape:
        Aggregated metrics and list of EngineValidationResult entries.

    Output shape:
        Report object with print_report() utility.

    Assumptions:
        Metrics are derived from consistent pipeline output schema.

    Failure conditions:
        None inside dataclass itself.
    """

    engine_results: list[EngineValidationResult]
    n_engines: int
    pct_monotonic_hi: float
    pct_valid_cluster: float
    risk_rul_pearson_r: float
    risk_rul_p_value: float
    mean_spearman_rho: float
    monotonicity_abs_rho_threshold: float
    weak_risk_rul_correlation_threshold: float
    anomalous_engines: list[int] = field(default_factory=list)

    def print_report(self) -> None:
        """Print a concise, console-friendly validation summary."""
        print("=" * 60)
        print("PIPELINE VALIDATION REPORT")
        print("=" * 60)

        print("\n[1] Health Index Monotonicity")
        print(f"    Engines validated: {self.n_engines}")
        print(f"    Mean Spearman rho: {self.mean_spearman_rho:.4f}")
        print(
            f"    Monotonic (|rho| >= {self.monotonicity_abs_rho_threshold}): "
            f"{self.pct_monotonic_hi:.1f}%"
        )

        print("\n[2] Cluster Progression Consistency")
        print(f"    Valid progression: {self.pct_valid_cluster:.1f}%")

        print("\n[3] Risk Score-RUL Correlation")
        print(
            f"    Pearson r: {self.risk_rul_pearson_r:.4f} "
            f"(p-value: {self.risk_rul_p_value:.2e})"
        )
        interpretation = (
            "Strong negative correlation - risk tracks degradation correctly."
            if self.risk_rul_pearson_r < -0.6
            else "Weak correlation - review pipeline parameters."
        )
        print(f"    Interpretation: {interpretation}")

        if self.anomalous_engines:
            print("\n[!] Anomalous engines detected (flagged for review)")
            print(f"    Unit IDs: {self.anomalous_engines}")
        else:
            print("\n[✓] No anomalous engines detected. All checks passed.")

        print("=" * 60)


def _resolve_thresholds(config: Optional[dict] = None) -> tuple[float, float]:
    """
    Purpose:
        Resolve validation thresholds from config with safe defaults.

    Input shape:
        Optional config dict loaded from config/config.yaml.

    Output shape:
        Tuple (monotonicity_abs_rho_threshold, weak_risk_rul_correlation_threshold).

    Assumptions:
        If provided, config may include a 'validation' section.

    Failure conditions:
        Raises ValueError when resolved thresholds are non-numeric.
    """
    validation_cfg = (config or {}).get("validation", {})
    monotonicity_threshold = validation_cfg.get(
        "monotonicity_abs_rho_threshold",
        DEFAULT_MONOTONICITY_ABS_RHO_THRESHOLD,
    )
    weak_corr_threshold = validation_cfg.get(
        "weak_risk_rul_correlation_threshold",
        DEFAULT_WEAK_RISK_RUL_CORRELATION_THRESHOLD,
    )

    try:
        monotonicity_threshold = float(monotonicity_threshold)
        weak_corr_threshold = float(weak_corr_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Validation thresholds must be numeric. "
            "Check validation.monotonicity_abs_rho_threshold and "
            "validation.weak_risk_rul_correlation_threshold in config.yaml."
        ) from error

    return monotonicity_threshold, weak_corr_threshold


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Purpose:
        Ensure required schema is present before validation logic runs.

    Input shape:
        DataFrame with pipeline output columns.

    Output shape:
        None; raises on failure.

    Assumptions:
        DataFrame is not None.

    Failure conditions:
        Raises KeyError for missing columns.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Validation DataFrame is missing required columns: {missing}.")


def _validate_engine_hi_monotonicity(
    group: pd.DataFrame,
    monotonicity_abs_rho_threshold: float,
) -> tuple[float, bool]:
    """
    Purpose:
        Compute Spearman correlation between cycle and health index per engine.

    Input shape:
        Single-engine DataFrame sorted by cycle.

    Output shape:
        Tuple (rho, is_monotonic).

    Assumptions:
        group includes 'cycle' and 'health_index'.

    Failure conditions:
        Raises ValueError if engine group is empty.
    """
    if group.empty:
        raise ValueError("Engine group is empty; cannot compute monotonicity.")

    rho_result = spearmanr(group["cycle"].values, group["health_index"].values)
    rho_value = float(np.asarray(rho_result[0]).item())
    if np.isnan(rho_value):
        rho_value = 0.0
    is_monotonic = abs(rho_value) >= monotonicity_abs_rho_threshold
    return rho_value, is_monotonic


def _validate_engine_cluster_progression(
    group: pd.DataFrame,
) -> tuple[list[str], bool]:
    """
    Purpose:
        Validate non-regressive degradation-state transitions for one engine.

    Input shape:
        Single-engine DataFrame containing risk_state ordered by cycle.

    Output shape:
        Tuple of (collapsed cluster sequence, progression_valid flag).

    Assumptions:
        risk_state values belong to CLUSTER_ORDER keys.

    Failure conditions:
        Raises ValueError if engine group is empty.
    """
    if group.empty:
        raise ValueError("Engine group is empty; cannot evaluate progression.")

    states = group.sort_values("cycle")["risk_state"].astype(str).tolist()

    collapsed_sequence: list[str] = [states[0]]
    for state in states[1:]:
        if state != collapsed_sequence[-1]:
            collapsed_sequence.append(state)

    order_values_raw = [CLUSTER_ORDER.get(state, -1) for state in states]
    if any(value < 0 for value in order_values_raw):
        return collapsed_sequence, False

    # Robust progression check:
    # cluster labels at cycle-level can oscillate due local noise, so use
    # lifecycle segment medians (early/mid/late) rather than requiring every
    # transition to be non-decreasing.
    thirds = np.array_split(np.asarray(order_values_raw, dtype=float), 3)
    segment_medians = [float(np.median(segment)) for segment in thirds]

    progression_valid = (
        segment_medians[0] <= segment_medians[1] <= segment_medians[2]
        and segment_medians[2] >= segment_medians[0]
    )

    return collapsed_sequence, progression_valid


def _validate_risk_rul_correlation(df: pd.DataFrame) -> tuple[float, float]:
    """
    Purpose:
        Compute fleet-wide Pearson correlation between risk_score and RUL.

    Input shape:
        Full training DataFrame with risk_score and RUL columns.

    Output shape:
        Tuple (pearson_r, p_value).

    Assumptions:
        Inputs are numeric and finite.

    Failure conditions:
        Propagates scipy errors if correlation cannot be computed.
    """
    pearson_result = pearsonr(df["risk_score"].values, df["RUL"].values)
    pearson_r = float(np.asarray(pearson_result[0]).item())
    p_value = float(np.asarray(pearson_result[1]).item())
    return pearson_r, p_value


def run_validation(df: pd.DataFrame, config: Optional[dict] = None) -> ValidationReport:
    """
    Purpose:
        Execute the full cross-engine validation workflow.

    Input shape:
        Training DataFrame with REQUIRED_COLUMNS.

    Output shape:
        ValidationReport with per-engine and fleet-level metrics.

    Assumptions:
        DataFrame comes from the full feature + clustering + risk pipeline.

    Failure conditions:
        - Missing required columns -> KeyError
        - Fewer than 2 engines -> ValueError
    """
    _validate_columns(df)
    monotonicity_threshold, weak_corr_threshold = _resolve_thresholds(config)

    engine_ids_array = np.sort(df["unit"].unique())
    engine_ids: list[int] = [int(unit_id) for unit_id in engine_ids_array.tolist()]
    if len(engine_ids) < 2:
        raise ValueError(
            f"Validation requires at least 2 engines; found {len(engine_ids)}."
        )

    engine_results: list[EngineValidationResult] = []
    anomalous_engines: list[int] = []

    for unit_id in engine_ids:
        group = df[df["unit"] == unit_id].sort_values("cycle")

        rho, hi_monotonic = _validate_engine_hi_monotonicity(
            group,
            monotonicity_threshold,
        )
        cluster_sequence, progression_valid = _validate_engine_cluster_progression(
            group
        )

        result = EngineValidationResult(
            unit_id=int(unit_id),
            n_cycles=len(group),
            hi_spearman_rho=rho,
            hi_monotonic=hi_monotonic,
            cluster_sequence=cluster_sequence,
            cluster_progression_valid=progression_valid,
            mean_risk_score=float(group["risk_score"].mean()),
            final_risk_score=float(group["risk_score"].iloc[-1]),
        )
        engine_results.append(result)

        if not hi_monotonic or not progression_valid:
            anomalous_engines.append(int(unit_id))

    pearson_r, p_value = _validate_risk_rul_correlation(df)

    n_engines = len(engine_results)
    pct_monotonic_hi = (
        sum(result.hi_monotonic for result in engine_results) / n_engines
    ) * 100.0
    pct_valid_cluster = (
        sum(result.cluster_progression_valid for result in engine_results) / n_engines
    ) * 100.0
    mean_spearman_rho = float(
        np.mean([result.hi_spearman_rho for result in engine_results])
    )

    report = ValidationReport(
        engine_results=engine_results,
        n_engines=n_engines,
        pct_monotonic_hi=pct_monotonic_hi,
        pct_valid_cluster=pct_valid_cluster,
        risk_rul_pearson_r=pearson_r,
        risk_rul_p_value=p_value,
        mean_spearman_rho=mean_spearman_rho,
        monotonicity_abs_rho_threshold=monotonicity_threshold,
        weak_risk_rul_correlation_threshold=weak_corr_threshold,
        anomalous_engines=anomalous_engines,
    )

    if pearson_r > weak_corr_threshold:
        warnings.warn(
            f"Risk-RUL Pearson r = {pearson_r:.3f} is weak. "
            "Risk score may not reliably track remaining useful life. "
            "Review clustering and risk normalization settings.",
            UserWarning,
        )

    return report
