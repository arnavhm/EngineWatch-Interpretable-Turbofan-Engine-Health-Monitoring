"""
model/risk.py

Purpose:
    Compute a continuous risk score in [0, 1] for each engine cycle.

Mathematical basis:
    For each observation x (scaled feature vector):
        d(x) = ||x - c_critical||_2

    Risk score:
        risk(x) = 1 - (d(x) - d_min) / (d_max - d_min)

    where d_min and d_max are fitted on training data only.

Input shape:
    DataFrame with columns: health_index, HI_velocity, HI_variability.
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

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from model.clustering import CLUSTER_FEATURES, ClusteringArtifacts


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

    def __init__(self, clustering_artifacts: ClusteringArtifacts) -> None:
        if clustering_artifacts.kmeans is None or clustering_artifacts.scaler is None:
            raise RuntimeError(
                "ClusteringArtifacts are not fully fitted. "
                "Run build_clustering() or DegradationClusterer.fit_transform() first."
            )
        self._artifacts = clustering_artifacts

    def _get_critical_centroid_scaled(self) -> np.ndarray:
        """
        Purpose:
            Retrieve Critical cluster centroid in scaled feature space.

        Input shape:
            Uses in-memory ClusteringArtifacts.label_to_cluster and KMeans centroids.

        Output shape:
            NumPy array of shape (3,).

        Assumptions:
            label_to_cluster contains "Critical".

        Failure conditions:
            Raises KeyError if "Critical" label is absent.
        """
        label_to_cluster = self._artifacts.label_to_cluster
        if "Critical" not in label_to_cluster:
            raise KeyError(
                "Cluster label mapping does not contain 'Critical'. "
                "Verify clustering label mapping logic."
            )
        critical_cluster_idx = label_to_cluster["Critical"]
        return self._artifacts.kmeans.cluster_centers_[critical_cluster_idx]

    def _compute_distances(self, df: pd.DataFrame) -> np.ndarray:
        """
        Purpose:
            Compute Euclidean distance to the Critical centroid for each row.

        Input shape:
            DataFrame (n_rows, n_cols) containing CLUSTER_FEATURES.

        Output shape:
            NumPy array (n_rows,).

        Assumptions:
            Data is already prepared by upstream feature pipeline.

        Failure conditions:
            Raises KeyError for missing features.
        """
        missing = [feature for feature in CLUSTER_FEATURES if feature not in df.columns]
        if missing:
            raise KeyError(
                f"Required risk features missing: {missing}. "
                "Ensure clustering features were built before risk scoring."
            )

        x_values = df[CLUSTER_FEATURES].values
        x_scaled = self._artifacts.scaler.transform(x_values)
        critical_centroid = self._get_critical_centroid_scaled()

        diff = x_scaled - critical_centroid
        distances = np.sqrt((diff**2).sum(axis=1))
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

        normalised = (distances - d_min) / denominator
        risk_scores = 1.0 - normalised
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
