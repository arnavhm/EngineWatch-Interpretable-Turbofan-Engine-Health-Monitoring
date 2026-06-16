"""
model/risk.py

Purpose:
    Compute a continuous risk score in [0, 1] for each engine cycle.

Mathematical basis:
    Distance is computed from the operative health index axis:
        HPC-fault engines:  d = 1 - HI_hpc
        Fan-fault engines:  d = HI_fan  (already inverted — higher = worse)
        Unified (default):  d = 1 - min(HI_hpc, HI_fan)

    Risk score (no further inversion — higher d already means worse health):
        risk(x) = (d(x) - d_min) / (d_max - d_min)

    where d_min and d_max are fitted on training data only.

Input shape:
    DataFrame with columns: HI_hpc, HI_fan (health_index (single-axis, FD001/FD002)
    or operative-axis HI (HI_hpc or HI_fan for FD003/FD004)).
    ClusteringArtifacts fitted in model/clustering.py.

Output shape:
    DataFrame with added column risk_score (float in [0, 1]).
    RiskArtifacts storing d_min and d_max.

Assumptions:
    - ClusteringArtifacts are fitted on training data.
    - The clustering scaler is reused (no refit on test data).
    - Feature columns are present and NaN-free.

Failure conditions:
    - Missing required columns -> KeyError
    - Missing Critical label mapping -> KeyError
    - Unfitted clustering artifacts -> RuntimeError
    - Constant distances (d_max == d_min) -> ValueError
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from model.clustering import ClusteringArtifacts


@dataclass
class RiskArtifacts:
    """
    Purpose:
        Store risk normalisation parameters fitted on training data.

    Input shape:
        Scalars d_min and d_max derived from train distances.

    Output shape:
        Dataclass container used by transform() without refitting.

    Assumptions:
        Distances are computed in the same scaled feature space.

    Failure conditions:
        None at construction beyond Python type/value misuse.
    """

    d_min: float
    d_max: float


class RiskScorer:
    """
    Purpose:
        Compute continuous risk scores from fitted clustering artifacts.

    Input shape:
        DataFrame rows with CLUSTER_FEATURES.

    Output shape:
        DataFrame with risk_score and fitted/applied RiskArtifacts.

    Assumptions:
        ClusteringArtifacts come from build_clustering() on training data.

    Failure conditions:
        Raises RuntimeError if required fitted objects are missing.
    """

    def __init__(
        self,
        clustering_artifacts: ClusteringArtifacts,
        operative_axis: str = "min",
    ) -> None:
        """
        operative_axis: which HI axis to use for distance computation.
            'min'  — conservative minimum of both axes (original behaviour)
            'hpc'  — use HI_hpc only (for HPC-fault engines)
            'fan'  — use HI_fan only (for fan-fault engines)
        """
        if clustering_artifacts.kmeans is None or clustering_artifacts.scaler is None:
            raise RuntimeError(
                "ClusteringArtifacts are not fully fitted. "
                "Run build_clustering() or DegradationClusterer.fit_transform() first."
            )
        self._artifacts = clustering_artifacts
        self._operative_axis = operative_axis

    def _compute_distances(self, df: pd.DataFrame) -> np.ndarray:
        """
        Purpose:
            Compute degradation distance using the operative HI axis.

        Input shape:
            DataFrame (n_rows, n_cols) containing HI_hpc and HI_fan.

        Output shape:
            NumPy array (n_rows,) in [0, 1] prior to train normalization.

        Assumptions:
            For HPC-fault engines: distance = 1 - HI_hpc
            For fan-fault engines: distance = 1 - HI_fan (higher HI = healthier, same as hpc).
            For unified mode:      distance = 1 - min(HI_hpc, HI_fan)

        Failure conditions:
            Raises KeyError for missing HI axis columns.
        """
        required = ["HI_hpc", "HI_fan"]
        missing = [feature for feature in required if feature not in df.columns]
        if missing:
            raise KeyError(
                f"Required risk features missing: {missing}. "
                "Ensure dual-axis health index was built before risk scoring."
            )

        if self._operative_axis == "hpc":
            operative_health = df["HI_hpc"].to_numpy(dtype=float)
            # HI_hpc is naturally directed: higher = healthier
            distances = 1.0 - np.clip(operative_health, 0.0, 1.0)
        elif self._operative_axis == "fan":
            # Dead branch: all datasets route to 'hpc' via n_fault_modes_by_dataset=1 in config.
            # Formula kept symmetric with HPC (1 - HI_fan) so it is correct if re-enabled.
            operative_health = df["HI_fan"].to_numpy(dtype=float)
            distances = 1.0 - np.clip(operative_health, 0.0, 1.0)
        else:
            operative_health = np.minimum(
                df["HI_hpc"].to_numpy(dtype=float),
                df["HI_fan"].to_numpy(dtype=float),
            )
            distances = 1.0 - np.clip(operative_health, 0.0, 1.0)

        return distances

    def _normalise_and_invert(
        self,
        distances: np.ndarray,
        risk_artifacts: Optional[RiskArtifacts] = None,
    ) -> tuple[np.ndarray, RiskArtifacts]:
        """
        Purpose:
            Normalize distances to [0, 1], then invert so closer-to-critical means higher risk.

        Input shape:
            distances: (n_rows,)
            risk_artifacts: optional fitted d_min/d_max from training.

        Output shape:
            Tuple of (risk_scores: (n_rows,), RiskArtifacts).

        Assumptions:
            Distances are finite and computed in consistent feature space.

        Failure conditions:
            Raises ValueError when d_max == d_min.
        """
        if risk_artifacts is None:
            d_min = float(distances.min())
            d_max = float(distances.max())
            risk_artifacts = RiskArtifacts(d_min=d_min, d_max=d_max)
        else:
            d_min = risk_artifacts.d_min
            d_max = risk_artifacts.d_max

        denominator = d_max - d_min
        if denominator == 0:
            raise ValueError(
                "Cannot normalize risk distances because d_max equals d_min. "
                "Check clustering quality and feature variance."
            )

        # distance = 1 - HI, so higher distance = worse health = higher risk
        # Normalise to [0, 1] — no further inversion needed
        risk_scores = (distances - d_min) / denominator
        risk_scores = np.clip(risk_scores, 0.0, 1.0)
        return risk_scores, risk_artifacts

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, RiskArtifacts]:
        """
        Purpose:
            Fit risk normalization bounds on training data and return risk-scored DataFrame.

        Input shape:
            Training DataFrame (n_train_rows, n_cols) with CLUSTER_FEATURES.

        Output shape:
            (DataFrame with risk_score, fitted RiskArtifacts)

        Assumptions:
            Training data represents expected risk distance range.

        Failure conditions:
            Propagates helper validation errors.
        """
        distances = self._compute_distances(df)
        risk_scores, fitted_artifacts = self._normalise_and_invert(distances)

        result_df = df.copy()
        result_df["risk_score"] = risk_scores
        return result_df, fitted_artifacts

    def transform(
        self,
        df: pd.DataFrame,
        risk_artifacts: RiskArtifacts,
    ) -> tuple[pd.DataFrame, RiskArtifacts]:
        """
        Purpose:
            Apply fitted risk normalization bounds to non-training data.

        Input shape:
            DataFrame (n_rows, n_cols) with CLUSTER_FEATURES and fitted RiskArtifacts.

        Output shape:
            (DataFrame with risk_score, same RiskArtifacts)

        Assumptions:
            risk_artifacts originates from fit_transform() on training data.

        Failure conditions:
            Raises RuntimeError if risk_artifacts is None.
        """
        if risk_artifacts is None:
            raise RuntimeError(
                "risk_artifacts is None. Call fit_transform() on training data first."
            )

        distances = self._compute_distances(df)
        risk_scores, _ = self._normalise_and_invert(distances, risk_artifacts)

        result_df = df.copy()
        result_df["risk_score"] = risk_scores
        return result_df, risk_artifacts


def build_risk_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clustering_artifacts: ClusteringArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, RiskArtifacts]:
    """
    Purpose:
        End-to-end risk scoring for train and test sets.

    Input shape:
        train_df: clustering output for training set.
        test_df: clustering output for test set.
        clustering_artifacts: fitted artifacts from build_clustering().

    Output shape:
        (train_with_risk, test_with_risk, risk_artifacts)

    Assumptions:
        train_df and test_df share the same engineered feature schema.

    Failure conditions:
        Propagates RiskScorer and helper validation errors.
    """
    scorer = RiskScorer(clustering_artifacts)
    train_out, risk_artifacts = scorer.fit_transform(train_df)
    test_out, _ = scorer.transform(test_df, risk_artifacts)
    return train_out, test_out, risk_artifacts


def build_risk_score_per_fault_mode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    clusterers_by_mode: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Compute risk scores routing each engine to its fault-mode cluster model.

    Purpose:
        HPC-fault engines scored against HPC cluster centroids.
        Fan-fault engines scored against fan cluster centroids.
        This resolves the risk score inversion on FD004 where mixed-fault
        clustering produced a backwards risk signal.

    Input:
        train_df:           DataFrame with fault_mode and HI_hpc, HI_fan columns.
        test_df:            Same schema as train_df.
        clusterers_by_mode: Dict from build_clustering_per_fault_mode().
                            Maps 'hpc'/'fan' → ClusteringArtifacts.

    Output:
        train_with_risk: DataFrame with risk_score column.
        test_with_risk:  DataFrame with risk_score column.
        risk_artifacts_by_mode: Dict mapping mode → RiskArtifacts.

    Assumptions:
        fault_mode column present in both train_df and test_df.
        clusterers_by_mode keys match fault_mode values in df.

    Failure conditions:
        KeyError if fault_mode value has no entry in clusterers_by_mode.
        Falls back to unified scoring if fault_mode column absent.
    """
    if "fault_mode" not in train_df.columns:
        logging.warning(
            "fault_mode column not found. Falling back to unified risk scoring."
        )
        any_artifacts = next(iter(clusterers_by_mode.values()))
        train_out, test_out, risk_arts = build_risk_score(
            train_df, test_df, any_artifacts
        )
        return train_out, test_out, {"unified": risk_arts}

    fault_modes = train_df["fault_mode"].unique().tolist()
    risk_artifacts_by_mode: dict = {}
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for mode in fault_modes:
        train_mode = train_df[train_df["fault_mode"] == mode].copy()
        test_mode = test_df[test_df["fault_mode"] == mode].copy()

        cl_artifacts = clusterers_by_mode.get(mode)
        if cl_artifacts is None:
            raise KeyError(
                f"No clustering artifacts found for fault_mode='{mode}'. "
                f"Available modes: {list(clusterers_by_mode.keys())}"
            )

        operative_axis = mode  # 'hpc' or 'fan' — matches RiskScorer parameter
        scorer = RiskScorer(cl_artifacts, operative_axis=operative_axis)
        train_mode_out, risk_arts = scorer.fit_transform(train_mode)
        test_mode_out, _ = scorer.transform(test_mode, risk_arts)

        risk_artifacts_by_mode[mode] = risk_arts
        train_parts.append(train_mode_out)
        test_parts.append(test_mode_out)

    train_out = pd.concat(train_parts).sort_index()
    test_out = pd.concat(test_parts).sort_index()

    return train_out, test_out, risk_artifacts_by_mode


def apply_risk_score_per_fault_mode(
    df: pd.DataFrame,
    cluster_models_by_fault: dict,
    risk_artifacts_by_fault: dict,
) -> pd.DataFrame:
    """
    Purpose:     Append risk_score per row using PRE-FIT risk artifacts, routed
                 by fault_mode (via the operative axis). Transform-only.
    Input:       df — contains CLUSTER_FEATURES + 'fault_mode' (+ 'risk_state')
                 cluster_models_by_fault — {mode: ClusteringArtifacts}
                 risk_artifacts_by_fault — {mode: RiskArtifacts}
    Output:      df copy with 'risk_score' added, original row order preserved
    Assumptions: every fault_mode in df keys BOTH dicts; df has a sortable index
    Failure:     KeyError on a missing mode key in either dict
    """
    scored_parts = []
    for mode, group in df.groupby("fault_mode"):
        if mode not in cluster_models_by_fault or mode not in risk_artifacts_by_fault:
            raise KeyError(f"Missing clustering/risk artifacts for fault_mode '{mode}'")
        scorer = RiskScorer(cluster_models_by_fault[mode], operative_axis=mode)
        scored_df, _ = scorer.transform(group, risk_artifacts_by_fault[mode])
        scored_parts.append(scored_df)

    return pd.concat(scored_parts).sort_index()
