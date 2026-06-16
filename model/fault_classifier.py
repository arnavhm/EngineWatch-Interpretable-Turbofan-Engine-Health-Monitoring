"""
model/fault_classifier.py

Purpose:
    Classify each engine into HPC-fault or fan-fault mode using its
    late-life degradation fingerprint. Enables fault-mode-aware risk
    scoring that routes each engine to the correct cluster model.

Mathematical basis:
    For each engine, compute the late-life slope of the two health
    index axes: HI_hpc and HI_fan. This produces a two-dimensional
    fingerprint vector [HI_hpc_slope, HI_fan_slope]. KMeans(k=2)
    clusters these fingerprints. HPC-fault engines have steeply
    negative HI_hpc_slope and flatter HI_fan_slope; fan-fault engines
    show the reverse.

Input:
    DataFrame with unit, cycle, HI_hpc, HI_fan columns.
    config with fault_classifier block.

Output:
    DataFrame with fault_mode column added, hpc or fan.
    FaultClassifierArtifacts for persistence and inference.

Assumptions:
    Health index axes are already built before this step.
    late_life_window applies to training engines with full trajectories.
    For test engines, min(available_cycles, late_life_window) is used.
    Single-fault datasets FD001 and FD002 produce one dominant cluster.
    min_cluster_size gate detects this and uses unified fallback.

Failure conditions:
    Missing sensor columns raises KeyError.
    Fewer than 2 engines per cluster falls back to single-fault mode.
    Silhouette below threshold in block mode raises ValueError.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class FaultClassifierArtifacts:
    """
    Container for all fitted objects needed at inference time.

    kmeans:           Fitted KMeans(k=2) on training fingerprints.
    scaler:           StandardScaler fitted on training fingerprints.
    label_map:        Maps cluster integer → 'hpc' or 'fan'.
    single_fault_mode: True if dataset has only one fault mode detected.
    dominant_fault:   The fault mode label when single_fault_mode is True.
    silhouette:       Fingerprint cluster silhouette score.
    fault_counts:     Number of engines per fault mode in training data.
    late_life_window: Window size used — needed to reproduce at inference.
    """

    kmeans: KMeans
    scaler: StandardScaler
    label_map: dict
    single_fault_mode: bool
    dominant_fault: str
    silhouette: float
    fault_counts: dict = field(default_factory=dict)
    late_life_window: int = 30


def _compute_hi_slopes(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Compute per-engine late-life slope of HI_hpc and HI_fan.

    Purpose:
        Two-feature fingerprint using health index axes directly.
        HPC-fault engines: steeply negative HI_hpc_slope, flat HI_fan_slope.
        Fan-fault engines: flat HI_hpc_slope, steeply negative HI_fan_slope.
        This is cleaner than raw sensor slopes because PCA already extracted
        the fault-mode-specific signal into each axis.

    Input:
        df:     DataFrame with unit, cycle, HI_hpc, HI_fan columns.
        window: Number of late-life cycles to use.

    Output:
        DataFrame (n_engines, 2) with columns HI_hpc_slope, HI_fan_slope.
        Indexed by unit.

    Failure conditions:
        KeyError if HI_hpc or HI_fan missing from df.
    """
    for col in ["HI_hpc", "HI_fan"]:
        if col not in df.columns:
            raise KeyError(
                f"Column {col} missing. "
                "build_dual_health_index must run before fit_fault_classifier."
            )

    slopes = {}
    for unit_id, engine_df in df.groupby("unit"):
        engine_df = engine_df.sort_values("cycle")
        late_df = engine_df.tail(window)

        if len(late_df) < 2:
            slopes[unit_id] = {"HI_hpc_slope": 0.0, "HI_fan_slope": 0.0}
            continue

        n = len(late_df)
        hpc_slope = (late_df["HI_hpc"].iloc[-1] - late_df["HI_hpc"].iloc[0]) / n
        fan_slope = (late_df["HI_fan"].iloc[-1] - late_df["HI_fan"].iloc[0]) / n
        slopes[unit_id] = {"HI_hpc_slope": hpc_slope, "HI_fan_slope": fan_slope}

    return pd.DataFrame(slopes).T


def _label_clusters(
    kmeans: KMeans,
    scaler: StandardScaler,
) -> dict:
    """
    Assign 'hpc' or 'fan' label to each KMeans cluster.

    Purpose:
        KMeans returns integer labels with no inherent meaning.
        We determine which cluster is HPC-fault vs fan-fault by
        looking at the cluster centroids in HI slope space.

    Input:
        kmeans: Fitted KMeans(k=2).
        scaler: Fitted StandardScaler (to inverse-transform centroids).

    Output:
        dict mapping cluster integer (0 or 1) → 'hpc' or 'fan'.

    Failure conditions:
        None beyond unexpected centroid shapes.
    """
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(
        centroids_original,
        columns=["HI_hpc_slope", "HI_fan_slope"],
    )

    hpc_cluster = int(centroid_df["HI_hpc_slope"].idxmin())
    fan_cluster = 1 - hpc_cluster

    label_map = {hpc_cluster: "hpc", fan_cluster: "fan"}
    logging.info(
        "[FAULT CLASSIFIER] Centroid HPC slopes: %s | Fan slopes: %s",
        centroid_df["HI_hpc_slope"].to_dict(),
        centroid_df["HI_fan_slope"].to_dict(),
    )
    return label_map


def fit_fault_classifier(
    df: pd.DataFrame,
    config: dict,
) -> FaultClassifierArtifacts:
    """
    Fit the fault-mode classifier on training engine fingerprints.

    Purpose:
        1. Compute late-life degradation fingerprint per engine.
        2. Scale fingerprints with StandardScaler.
        3. Cluster into k=2 groups with KMeans.
        4. Label clusters as 'hpc' or 'fan' from centroid analysis.
        5. Check min_cluster_size gate — if one cluster is too small,
           activate single_fault_mode fallback.
        6. Compute silhouette score and apply gate.

    Input:
        df:     Training DataFrame (N, M) with unit, cycle, HI_hpc, HI_fan columns.
                Must include dual health index features.
        config: Config dict with fault_classifier block.

    Output:
        FaultClassifierArtifacts containing all fitted objects.

    Assumptions:
        config["fault_classifier"] contains:
            late_life_window, min_cluster_size, silhouette_gate.

    Failure conditions:
        KeyError if config keys missing.
        ValueError if silhouette gate in block mode and threshold not met.
    """
    fc_cfg = config["fault_classifier"]
    window: int = fc_cfg["late_life_window"]
    min_cluster_size: int = fc_cfg["min_cluster_size"]
    random_state: int = config["clustering"]["random_state"]

    n_modes = fc_cfg.get("n_fault_modes_by_dataset", {}).get(config.get("dataset_id"), None)
    if n_modes == 1:
        logging.info(
            "[FAULT CLASSIFIER] %s is single-fault by config; forcing all engines to 'hpc'.",
            config.get("dataset_id"),
        )
        # Dummy instances for type-safety since they are required but unused
        dummy_kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=1)
        dummy_scaler = StandardScaler()
        return FaultClassifierArtifacts(
            kmeans=dummy_kmeans,
            scaler=dummy_scaler,
            label_map={0: "hpc", 1: "hpc"},
            single_fault_mode=True,
            dominant_fault="hpc",
            silhouette=1.0,
            fault_counts={"hpc": df["unit"].nunique()},
            late_life_window=window,
        )

    logging.info(
        "[FAULT CLASSIFIER] Computing late-life HI slopes "
        "(window=%d cycles, %d engines)",
        window,
        df["unit"].nunique(),
    )

    fingerprints = _compute_hi_slopes(df, window)

    # Scale fingerprints — essential because sensors have different ranges
    scaler = StandardScaler()
    fingerprints_scaled = scaler.fit_transform(fingerprints.values)

    # Fit KMeans(k=2) — two fault modes
    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    raw_labels = kmeans.fit_predict(fingerprints_scaled)

    # Check silhouette
    sil = float(silhouette_score(fingerprints_scaled, raw_labels))
    gate_cfg = fc_cfg.get("silhouette_gate", {})
    gate_enabled = gate_cfg.get("enabled", True)
    gate_mode = gate_cfg.get("mode", "warn")
    min_sil = gate_cfg.get("min_silhouette", 0.25)

    logging.info(
        "[FAULT CLASSIFIER] Silhouette: %.4f (min required: %.2f)", sil, min_sil
    )

    if gate_enabled and sil < min_sil:
        msg = (
            f"Fault classifier silhouette {sil:.4f} < {min_sil}. "
            "Fault-mode separation may be weak."
        )
        if gate_mode == "block":
            raise ValueError(msg)
        else:
            warnings.warn(msg, UserWarning)

    # Count engines per cluster
    cluster_counts = {
        int(k): int(v) for k, v in zip(*np.unique(raw_labels, return_counts=True))
    }
    logging.info("[FAULT CLASSIFIER] Cluster distribution: %s", cluster_counts)

    # Check min_cluster_size — detect single-fault datasets
    single_fault_mode = False
    dominant_fault = "hpc"

    min_count = min(cluster_counts.values())
    if min_count < min_cluster_size:
        single_fault_mode = True
        # Dominant cluster is the larger one
        dominant_cluster = max(cluster_counts, key=cluster_counts.get)
        # Label the dominant cluster to get the dominant fault mode
        label_map_temp = _label_clusters(kmeans, scaler)
        dominant_fault = label_map_temp[dominant_cluster]
        logging.warning(
            "[FAULT CLASSIFIER] Single-fault mode detected. "
            "Smallest cluster has %d engines (min=%d). "
            "All engines assigned to '%s' fault mode.",
            min_count,
            min_cluster_size,
            dominant_fault,
        )
        label_map = {0: dominant_fault, 1: dominant_fault}
    else:
        label_map = _label_clusters(kmeans, scaler)

    # Log fault-mode split
    fault_counts = {}
    for cluster_id, label in label_map.items():
        count = cluster_counts.get(cluster_id, 0)
        fault_counts[label] = fault_counts.get(label, 0) + count

    logging.info("[FAULT CLASSIFIER] Fault-mode split: %s", fault_counts)

    return FaultClassifierArtifacts(
        kmeans=kmeans,
        scaler=scaler,
        label_map=label_map,
        single_fault_mode=single_fault_mode,
        dominant_fault=dominant_fault,
        silhouette=sil,
        fault_counts=fault_counts,
        late_life_window=window,
    )


def classify_engines(
    df: pd.DataFrame,
    artifacts: FaultClassifierArtifacts,
    config: dict,
) -> pd.DataFrame:
    """
    Assign fault_mode label to every row of df based on engine classification.

    Purpose:
        Apply fitted classifier to assign 'hpc' or 'fan' to each engine.
        Works on both full training trajectories and partial test data.
        Adds fault_mode column — one consistent value per engine (not per cycle).

    Input:
        df:        DataFrame with unit, cycle, HI_hpc, HI_fan columns.
        artifacts: Fitted FaultClassifierArtifacts from fit_fault_classifier().
        config:    Config dict (for sensor column names).

    Output:
        DataFrame with fault_mode column added.
        All rows belonging to the same engine get the same fault_mode value.

    Assumptions:
        Partial test trajectories are handled —
        min(available_cycles, late_life_window) cycles are used.

    Failure conditions:
        KeyError if sensor columns missing from df.
    """
    fc_cfg = config["fault_classifier"]
    window: int = artifacts.late_life_window

    # Single-fault fallback — assign all engines to dominant mode
    if artifacts.single_fault_mode:
        result = df.copy()
        result["fault_mode"] = artifacts.dominant_fault
        return result

    # Compute fingerprints using available cycles per engine
    fingerprints = _compute_hi_slopes(df, window)

    # Apply fitted scaler — no refit
    fingerprints_scaled = artifacts.scaler.transform(fingerprints.values)

    # Predict cluster for each engine
    raw_labels = artifacts.kmeans.predict(fingerprints_scaled)

    # Map cluster integer → fault mode string
    engine_fault_map = {
        unit_id: artifacts.label_map[raw_label]
        for unit_id, raw_label in zip(fingerprints.index, raw_labels)
    }

    result = df.copy()
    result["fault_mode"] = result["unit"].map(engine_fault_map)

    unmapped = result["fault_mode"].isna().sum()
    if unmapped > 0:
        logging.warning(
            "[FAULT CLASSIFIER] %d rows could not be classified. Defaulting to 'hpc'.",
            unmapped,
        )
        result["fault_mode"] = result["fault_mode"].fillna("hpc")

    return result
