#!/usr/bin/env python
"""Quick test: validate operative_axis parameter in RiskScorer."""

import sys
import os

sys.path.insert(
    0, "/Users/arnavhmutt/Desktop/aviation-ds-projects/project-2-predictive-maintenance"
)
os.environ["DATASET_NAME"] = "FD001"

from data.load import load_config, load_dataset
from data.preprocess import preprocess_train, preprocess_test
from features.health_index import build_dual_health_index, apply_dual_health_index
from features.velocity import build_velocity
from features.variability import build_variability
from model.fault_classifier import fit_fault_classifier, classify_engines
from model.clustering import build_clustering_per_fault_mode
from model.risk import build_risk_score_per_fault_mode

config = load_config()
dataset = "FD001"

# Load
train_raw, test_raw, test_rul_offsets = load_dataset(config)
print(f"[{dataset}] Train: {train_raw.shape}, Test: {test_raw.shape}")

# Preprocess
train_proc, scaler, _ = preprocess_train(train_raw, config, persist_outputs=False)
test_proc = preprocess_test(test_raw, config, scaler, persist_outputs=False)
print(
    f"[{dataset}] After preprocess: Train: {train_proc.shape}, Test: {test_proc.shape}"
)

# Build HI features
train_hi, pca_by_axis, scaler_by_axis = build_dual_health_index(train_proc, config)
test_hi = apply_dual_health_index(test_proc, pca_by_axis, scaler_by_axis, config)
print(f"[{dataset}] After dual HI: Train: {train_hi.shape}, Test: {test_hi.shape}")

# Build velocity
train_vel, test_vel, _ = build_velocity(train_hi, test_hi, config)
print(f"[{dataset}] After velocity: Train: {train_vel.shape}, Test: {test_vel.shape}")

# Build variability
train_var, test_var, _ = build_variability(train_vel, test_vel, config)
print(
    f"[{dataset}] After variability: Train: {train_var.shape}, Test: {test_var.shape}"
)

# Fault classification
print("\n=== FAULT CLASSIFICATION ===")
fault_artifacts = fit_fault_classifier(train_var, config)
print(f"Silhouette: {fault_artifacts.silhouette:.4f}")
print(f"Fault-mode split: {fault_artifacts.fault_counts}")

train_var_fc = classify_engines(train_var, fault_artifacts, config)
test_var_fc = classify_engines(test_var, fault_artifacts, config)
print(f"Train fault modes: {train_var_fc['fault_mode'].value_counts().to_dict()}")
print(f"Test fault modes: {test_var_fc['fault_mode'].value_counts().to_dict()}")

# Per-fault-mode clustering
print("\n=== CLUSTERING ===")
train_clust, test_clust, clusterers_by_mode = build_clustering_per_fault_mode(
    train_var_fc, test_var_fc, config
)
print(f"Clusterers by fault mode: {list(clusterers_by_mode.keys())}")

# Risk scoring
print("\n=== RISK SCORING (operative_axis) ===")
try:
    train_risk, test_risk, risk_arts_by_mode = build_risk_score_per_fault_mode(
        train_clust, test_clust, clusterers_by_mode
    )
    print(f"Risk arts by fault mode: {list(risk_arts_by_mode.keys())}")
    print(
        f"Train risk columns: {[c for c in train_risk.columns if 'risk' in c.lower()]}"
    )
    print("\n✓ operative_axis fix successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
