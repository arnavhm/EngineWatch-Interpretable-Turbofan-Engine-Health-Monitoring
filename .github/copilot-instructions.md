# GitHub Copilot Instructions — CMAPSS Health Monitor

## Project Context

Turbofan engine degradation monitoring using NASA CMAPSS FD001.
Interpretable, physics-aligned, production-style Python. No deep learning.

## Strict Constraints — NEVER suggest these:

- LSTM, RNN, transformers, neural networks of any kind
- Ensemble stacks or boosting models
- Features outside the approved set
- Global variables or hardcoded constants
- Monolithic scripts or notebook-style code

## Approved Feature Set Only:

- Health Index (PCA-based, first principal component)
- Health Velocity (rolling window linear regression slope)
- Variability (rolling standard deviation)
- KMeans clustering (k=3)
- Distance-based Risk Score

## Architecture — always follow this structure:

- data/load.py — raw file ingestion only
- data/preprocess.py — scaling, sensor selection, RUL computation
- features/health_index.py — PCA health index
- features/velocity.py — rolling slope
- features/variability.py — rolling std
- model/clustering.py — KMeans k=3
- model/risk.py — continuous risk score
- evaluation/validation.py — monotonicity checks, plots
- app/dashboard.py — Streamlit only

## Code Standards — always enforce:

- Type hints on all function signatures
- Docstrings with: Purpose, Input shape, Output shape, Assumptions, Failure conditions
- No magic numbers — all constants from config/config.yaml
- Fixed random_state=42 for all stochastic operations
- No silent transformations — every step must be documented

## Config:

- All parameters live in config/config.yaml
- Always load config via load_config() from data/load.py
- Never hardcode file paths, window sizes, or cluster counts

## Mathematical Discipline:

- PCA: standardize first, extract PC1, normalize output to [0,1]
- Velocity: linear regression slope via numpy polyfit within rolling window
- Risk: Euclidean distance from healthy cluster centroid, normalized to [0,1]
- KMeans: StandardScaler input, silhouette score validation required

## Dataset Facts:

- FD001: 100 engines, 1 operating condition, 1 fault mode (HPC degradation)
- 26 columns: unit, cycle, 3 op settings, 21 sensors
- Training trajectories run to failure (RUL=0 at last cycle)
- Test trajectories are pruned before failure
