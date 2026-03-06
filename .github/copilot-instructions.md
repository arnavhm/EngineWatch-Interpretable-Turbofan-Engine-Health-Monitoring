# GitHub Copilot Instructions — CMAPSS Health Monitor

## Project Context

Turbofan engine degradation monitoring using NASA CMAPSS FD001.

This project implements a hybrid predictive maintenance system combining:

1. Interpretable health monitoring
2. Remaining Useful Life (RUL) prediction

The system prioritizes interpretable, physics-aligned, production-style Python code.
Health monitoring features provide explainability, while supervised models predict RUL.
Deep learning models should be avoided unless explicitly requested by the user.

## Strict Constraints — enforce always:

- Avoid suggesting deep learning models unless explicitly requested by the user
- Preferred baseline models for RUL prediction: Linear Regression, Random Forest, Gradient Boosting
- Do not automatically propose complex architectures
- No global variables or hardcoded constants
- No monolithic scripts or notebook-style production code
- No magic numbers — all constants from config/config.yaml

## Approved Feature Set:

Approved baseline features (serve both health monitoring and RUL prediction):

- Health Index (PCA-based, first principal component)
- Health Velocity (rolling window linear regression slope)
- Variability (rolling standard deviation)

These features support two subsystems:

Health Monitoring:

- KMeans clustering (k=3)
- Distance-based risk score

RUL Prediction:

- Supervised regression model predicting Remaining Useful Life

Additional features must not be introduced without explicit user approval.

## Architecture — always follow this structure:

- data/load.py — raw file ingestion only
- data/preprocess.py — scaling, sensor selection, RUL computation

- features/health_index.py — PCA health index
- features/velocity.py — rolling slope
- features/variability.py — rolling std

- model/clustering.py — KMeans k=3
- model/risk.py — continuous risk score
- model/rul.py — supervised regression model for RUL prediction

- evaluation/validation.py — monotonicity checks, RMSE, NASA scoring
- Trend monotonicity validation for health index
- Cluster stability analysis
- RUL prediction error distribution

- app/dashboard.py — Streamlit health monitoring + RUL prediction dashboard
- Training logic and inference logic must be implemented as separate functions.
  Model training must never occur inside the dashboard or notebook code.

## Code Standards — always enforce:

- Type hints on all function signatures
- Docstrings with: Purpose, Input shape, Output shape, Assumptions, Failure conditions
- All constants defined in config/config.yaml
- Fixed random_state=42 for all stochastic operations
- No hardcoded file paths
- Config must always be loaded via load_config() from data/load.py

## Mathematical Discipline:

- PCA: standardize sensors first, extract PC1, normalize output to [0,1]
- Velocity: compute slope using numpy polyfit inside rolling window
- Variability: rolling standard deviation
- Risk Score: Euclidean distance from healthy cluster centroid, normalized to [0,1]
- RUL Prediction: regression model trained on engineered features,
  evaluated using RMSE and NASA scoring function

## Dataset Facts:

- FD001: 100 engines, single operating condition, single fault mode (HPC degradation)
- 26 columns: unit, cycle, 3 op settings, 21 sensors
- Training trajectories run to failure (RUL=0 at last cycle)
- Test trajectories are truncated before failure
