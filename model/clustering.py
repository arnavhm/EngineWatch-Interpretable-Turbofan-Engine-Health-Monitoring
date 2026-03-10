"""
model/clustering.py

Purpose:
    Classify each engine cycle into one of three degradation states using KMeans clustering.
    Input features: [health_index, HI_velocity, HI_variability]
    Output: cluster label mapped to a human-readable risk state:
            0 = Healthy, 1 = Degrading, 2 = Critical

Mathematical basis:
    KMeans minimises the within-cluster sum of squared distances:
        argmin_C  Σ_i  Σ_{x ∈ C_i}  ||x - μ_i||²
    where μ_i is the centroid of cluster C_i.
    Features are standardised before clustering so no single feature
    dominates due to scale differences.

Input:
    DataFrame with columns: health_index, HI_velocity, HI_variability
    config dict (n_clusters and random_state from config["clustering"])

Output:
    Same DataFrame with added column: risk_state (str: Healthy/Degrading/Critical)
    ClusteringArtifacts object for inspection and downstream use.

Assumptions:
    - All three feature columns are present and free of NaN.
    - StandardScaler is fitted on training data only.
    - KMeans is fitted on training data only.
    - Cluster-to-label mapping is derived from mean HI per cluster
      (highest HI = Healthy, lowest HI = Critical).

Failure conditions:
    - Missing feature columns → KeyError
    - NaN in feature columns → ValueError
    - Silhouette score < 0.3 → UserWarning (clusters may not be well-separated)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import Optional
import warnings


# The three features that describe degradation state in this system.
# These are the only approved inputs — do not add features here without
# updating the architecture charter.
CLUSTER_FEATURES = ["health_index", "HI_velocity", "HI_variability"]

# Human-readable label mapping — ordered from healthiest to most critical.
# These strings are used directly in the Streamlit dashboard.
RISK_LABELS = {0: "Healthy", 1: "Degrading", 2: "Critical"}


@dataclass
class ClusteringArtifacts:
    """
    Container for all outputs and metadata from the clustering step.

    Carrying everything in one structured object makes the pipeline
    state inspectable and avoids passing loose variables between modules.
    """

    scaler: StandardScaler  # Fitted on training data only
    kmeans: KMeans  # Fitted on training data only
    cluster_to_label: dict  # Maps raw KMeans label → risk string
    label_to_cluster: dict  # Inverse map: risk string → raw label
    silhouette: float  # Silhouette score on training data
    centroid_summary: pd.DataFrame  # Centroid values in original feature space
    cluster_counts: dict = field(default_factory=dict)  # Train distribution per cluster


class DegradationClusterer:
    """
    Fits and applies KMeans-based degradation state classification.

    Usage pattern (mirrors PCAHealthIndex):
        clusterer = DegradationClusterer(config)
        train_df = clusterer.fit_transform(train_df)   # fits scaler + KMeans
        test_df  = clusterer.transform(test_df)         # applies fitted params
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize clusterer with configuration parameters.

        Input:  config dict loaded from config/config.yaml
        Output: initialised (unfitted) DegradationClusterer

        Failure: KeyError if required config keys are missing
        """
        try:
            cl_cfg = config["clustering"]
            self.n_clusters: int = cl_cfg["n_clusters"]  # always 3
            self.random_state: int = cl_cfg["random_state"]  # always 42
            self.n_init: int = cl_cfg["n_init"]  # typically 10
        except KeyError as e:
            raise KeyError(
                f"Config missing required clustering key: {e}. "
                "Check config/config.yaml for clustering.n_clusters, clustering.random_state, and clustering.n_init."
            )

        if self.n_clusters != 3:
            raise ValueError(
                f"This architecture supports exactly 3 clusters (Healthy/Degrading/Critical). "
                f"Got n_clusters={self.n_clusters}. Update RISK_LABELS if changing cluster count."
            )

        # Placeholders — populated only after fit()
        self._scaler: Optional[StandardScaler] = None
        self._kmeans: Optional[KMeans] = None
        self._cluster_to_label: Optional[dict] = None
        self._artifacts: Optional[ClusteringArtifacts] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_features(self, df: pd.DataFrame) -> None:
        """
        Confirm all required feature columns are present and NaN-free.

        Purpose:  Fail loudly and informatively rather than silently producing
                  wrong cluster assignments or crashing inside sklearn.
        Input:    DataFrame to validate
        Failure:  KeyError if columns missing; ValueError if NaN detected.
        """
        missing = [f for f in CLUSTER_FEATURES if f not in df.columns]
        if missing:
            raise KeyError(
                f"Required clustering features missing: {missing}. "
                "Ensure health_index, velocity, and variability modules have run."
            )
        nan_counts = df[CLUSTER_FEATURES].isna().sum()
        if nan_counts.any():
            raise ValueError(
                f"NaN values detected in clustering features:\n{nan_counts[nan_counts > 0]}\n"
                "Resolve NaN before clustering — check edge case handling in velocity.py."
            )

    def _map_clusters_to_labels(self, df: pd.DataFrame, raw_labels: np.ndarray) -> dict:
        """
        Map raw KMeans integer labels to risk state strings.

        Purpose:
            KMeans assigns arbitrary integer labels (0, 1, 2) with no inherent meaning.
            We determine meaning by computing the mean HI within each cluster.
            The cluster with the highest mean HI is Healthy (engines are still strong).
            The cluster with the lowest mean HI is Critical (engines near failure).
            The middle cluster is Degrading.

        Input:  df         — DataFrame with health_index column
                raw_labels — numpy array of KMeans integer labels, same length as df
        Output: dict mapping raw integer label → risk string

        Justification:
            Using health_index as the ordering criterion is physically grounded —
            HI is our primary degradation signal, constructed directly from sensor
            physics via PCA. Ordering by velocity or variability alone would be
            less stable due to their wider variance.

        Failure:
            RuntimeError if cluster count != expected (can occur if KMeans collapses clusters)
        """
        temp = pd.DataFrame(
            {"cluster": raw_labels, "health_index": df["health_index"].values}
        )
        mean_hi_per_cluster = temp.groupby("cluster")["health_index"].mean()

        # Sort clusters by mean HI descending: highest HI = Healthy, lowest = Critical
        sorted_clusters = mean_hi_per_cluster.sort_values(
            ascending=False
        ).index.tolist()

        # Guard against cluster collapse (rare but possible with KMeans)
        if len(sorted_clusters) != self.n_clusters:
            raise RuntimeError(
                f"Expected {self.n_clusters} clusters but KMeans produced {len(sorted_clusters)}. "
                "This may indicate overlapping distributions or initialization issues. "
                "Try adjusting window parameters or feature scaling."
            )

        # sorted_clusters[0] has the highest mean HI → Healthy
        # sorted_clusters[1] has the middle mean HI → Degrading
        # sorted_clusters[2] has the lowest mean HI → Critical
        cluster_to_label = {
            sorted_clusters[0]: RISK_LABELS[0],  # Healthy
            sorted_clusters[1]: RISK_LABELS[1],  # Degrading
            sorted_clusters[2]: RISK_LABELS[2],  # Critical
        }
        return cluster_to_label

    def _build_centroid_summary(self) -> pd.DataFrame:
        """
        Return cluster centroids in original (unscaled) feature space.

        Purpose:
            Centroids in scaled space are uninterpretable. Inverse-transforming
            back to original units makes them physically meaningful and
            academically defensible.

        Input:  Fitted scaler and KMeans (internal state)
        Output: DataFrame with columns [risk_state, health_index, HI_velocity, HI_variability]

        Note: This method is only called from fit_transform() after fitting,
              so _kmeans, _scaler, and _cluster_to_label are guaranteed to be set.
        """
        # Type assertions for type checker (runtime guards ensure these are not None)
        assert self._kmeans is not None, "KMeans must be fitted"
        assert self._scaler is not None, "Scaler must be fitted"
        assert self._cluster_to_label is not None, "Cluster labels must be mapped"

        centroids_scaled = self._kmeans.cluster_centers_
        centroids_original = self._scaler.inverse_transform(centroids_scaled)

        summary = pd.DataFrame(centroids_original, columns=CLUSTER_FEATURES)
        summary["risk_state"] = [
            self._cluster_to_label[i] for i in range(self.n_clusters)
        ]
        return (
            summary[["risk_state"] + CLUSTER_FEATURES]
            .sort_values("health_index", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and KMeans on training data; return DataFrame with risk_state column.

        Purpose:
            1. Validate input features.
            2. Standardise features (fit StandardScaler on training data).
            3. Fit KMeans (k=3, random_state=42).
            4. Map raw cluster labels to risk state strings based on mean HI.
            5. Compute silhouette score as cluster quality metric.
            6. Build centroid summary in original feature space.
            7. Store all artefacts for later use in transform().

        Input:  Training DataFrame after velocity + variability steps
                Shape: (20631, n_cols) for FD001
        Output: Same DataFrame with added 'risk_state' column (str)

        Failure conditions:
            - Missing or NaN features → see _validate_features()
            - Silhouette score < 0.3 → UserWarning
        """
        self._validate_features(df)

        X = df[CLUSTER_FEATURES].values  # shape: (n_rows, 3)

        # Fit StandardScaler on training features only
        # This ensures test data is scaled relative to training distribution
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Fit KMeans — n_init runs multiple random initialisations, keeps best
        # This reduces sensitivity to initialisation (a known KMeans weakness)
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )
        raw_labels = self._kmeans.fit_predict(X_scaled)

        # Map integer labels to risk state strings
        self._cluster_to_label = self._map_clusters_to_labels(df, raw_labels)
        label_to_cluster = {v: k for k, v in self._cluster_to_label.items()}

        # Compute silhouette score — measures cluster separation quality
        # Range: [-1, 1] where higher is better; < 0.3 suggests poor separation
        sil = float(silhouette_score(X_scaled, raw_labels))
        if sil < 0.3:
            warnings.warn(
                f"Silhouette score is low ({sil:.3f}). "
                "Cluster boundaries may not be well-defined. "
                "Consider reviewing feature scaling or window parameters.",
                UserWarning,
            )

        # Attach risk_state labels to DataFrame
        result = df.copy()
        result["risk_state"] = pd.Categorical(
            [self._cluster_to_label[lbl] for lbl in raw_labels],
            categories=["Healthy", "Degrading", "Critical"],
            ordered=True,
        )

        # Count observations per cluster for reporting
        cluster_counts = result["risk_state"].value_counts().to_dict()

        # Build centroid summary in original feature space
        centroid_summary = self._build_centroid_summary()

        self._artifacts = ClusteringArtifacts(
            scaler=self._scaler,
            kmeans=self._kmeans,
            cluster_to_label=self._cluster_to_label,
            label_to_cluster=label_to_cluster,
            silhouette=sil,
            centroid_summary=centroid_summary,
            cluster_counts=cluster_counts,
        )

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scaler and KMeans to test (or validation) data.

        Purpose:
            Uses parameters from fit_transform() — no re-fitting.
            Ensures train/test cluster assignments are on the same basis.

        Input:  Test DataFrame after velocity + variability steps
                Shape: (13096, n_cols) for FD001
        Output: Same DataFrame with added 'risk_state' column

        Failure: Raises RuntimeError if called before fit_transform().
        """
        if self._kmeans is None:
            raise RuntimeError(
                "DegradationClusterer.transform() called before fit_transform(). "
                "Always fit on training data first."
            )

        self._validate_features(df)

        # Type assertions for type checker (runtime guard above ensures these are not None)
        assert self._scaler is not None, "Scaler must be fitted"
        assert self._cluster_to_label is not None, "Cluster labels must be mapped"

        X = df[CLUSTER_FEATURES].values
        X_scaled = self._scaler.transform(X)  # apply training scaler, no re-fitting
        raw_labels = self._kmeans.predict(X_scaled)

        result = df.copy()
        result["risk_state"] = pd.Categorical(
            [self._cluster_to_label[lbl] for lbl in raw_labels],
            categories=["Healthy", "Degrading", "Critical"],
            ordered=True,
        )
        return result

    def get_artifacts(self) -> ClusteringArtifacts:
        """
        Return fitted artifacts for logging, visualisation, and inspection.

        Failure: Raises RuntimeError if called before fit_transform().
        """
        if self._artifacts is None:
            raise RuntimeError("No artifacts available — call fit_transform() first.")
        return self._artifacts


# ------------------------------------------------------------------
# Module-level convenience function (mirrors health_index.py pattern)
# ------------------------------------------------------------------


def build_clustering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, ClusteringArtifacts]:
    """
    Top-level entry point for the clustering pipeline.

    Purpose:
        Fits KMeans on training data, applies to test data, returns artifacts.
        This is the recommended interface for the full feature pipeline.

    Input:
        train_df: Output of variability step (train). Must contain: health_index, HI_velocity, HI_variability
                  Shape: (n_train, n_cols) e.g. (20631, n_cols) for FD001
        test_df: Output of variability step (test). Must contain same features.
                 Shape: (n_test, n_cols) e.g. (13096, n_cols) for FD001
        config: Loaded config dict with clustering.n_clusters, clustering.random_state, clustering.n_init

    Output:
        train_with_clusters: Training DataFrame with added 'risk_state' column (categorical)
        test_with_clusters: Test DataFrame with added 'risk_state' column (categorical)
        artifacts: ClusteringArtifacts for inspection, logging, and downstream risk module

    Example:
        train_clust, test_clust, artifacts = build_clustering(train_var, test_var, config)
        print(artifacts.silhouette)  # e.g. 0.42
        print(artifacts.centroid_summary)

    Note:
        For production deployment, persist artifacts:
            joblib.dump(artifacts, config["clustering"]["artifacts_path"])
        Then load in dashboard/API without retraining.
    """
    clusterer = DegradationClusterer(config)
    train_out = clusterer.fit_transform(train_df)
    test_out = clusterer.transform(test_df)
    artifacts = clusterer.get_artifacts()
    return train_out, test_out, artifacts
