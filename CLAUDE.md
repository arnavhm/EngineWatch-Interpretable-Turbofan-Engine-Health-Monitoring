## Project Identity

- Project name: EngineWatch — Interpretable Turbofan Engine Health Monitoring
- Dataset: NASA CMAPSS FD001
- GitHub repo URL: git@github.com:arnavhm/EngineWatch-Interpretable-Turbofan-Engine-Health-Monitoring.git
- Current iteration status: Iteration 1 complete; Iteration 2 planned/in progress

## Architecture — Non-Negotiable

- Exact workspace structure in scope:

```text
config/
  config.yaml

data/
  __init__.py
  load.py
  preprocess.py
  raw/
    train_FD001.txt
    test_FD001.txt
    RUL_FD001.txt
  processed/

features/
  __init__.py
  health_index.py
  velocity.py
  variability.py

model/
  __init__.py
  clustering.py
  risk.py
  rul.py

evaluation/
  __init__.py
  validation.py

app/
  __init__.py
  dashboard.py
  theme.py
  components/
    fleet_overview.py
    engine_selector.py
    hi_plot.py
    dynamics_plots.py
    risk_gauge.py
    cluster_timeline.py
    rul_prediction.py
    model_evaluation.py
  utils/
    data_loader.py
    rul_artifacts.py

scripts/
  train_rul_artifacts.py

tests/
  __init__.py
  test_preprocess_pipeline.py
  test_iteration1_modules.py
  test_validation.py

notebooks/
  01_data_exploration.ipynb
  02_health_index_verification.ipynb
  03_velocity_variability_verification.ipynb
  04_rul_evaluation.ipynb
  clustering_checklist.ipynb
  risk_checklist.ipynb
  rul_verification_checklist.ipynb
  validation_checklist.ipynb
```

- Import pattern:
  - Core modules are imported via package paths from project root (for example `from data.load import ...`, `from model.rul import ...`).
  - Notebook scripts and `app/dashboard.py` add project root to `sys.path` for reliable local execution.
- Notebooks are exploration/verification only. Core logic must live in modules under `data/`, `features/`, `model/`, and `evaluation/`.

## Hard Constraints — Never Violate These

- No deep learning of any kind (LSTM, RNN, Transformer, neural network).
  - Reason: this codebase is explicitly interpretable/physics-aligned and currently built around classical features + regression baselines.
- No dashboard retraining. Dashboard is inference-only. Models are built by `scripts/train_rul_artifacts.py`.
- `StandardScaler` must be fit on training data only. Never on test data.
  - Reason: fitting on test data causes leakage and optimistic/unreliable evaluation.
- No hardcoded constants. All parameters from `config/config.yaml`.
- No global state.

## Approved Feature Set

- `health_index` (PCA PC1, normalized and direction-corrected): `features/health_index.py`
- `HI_velocity` (rolling linear slope via `numpy.polyfit`): `features/velocity.py`
- `HI_variability` (rolling standard deviation, normalized): `features/variability.py`
- `risk_state` (Healthy/Degrading/Critical via KMeans k=3): `model/clustering.py`
- `risk_score` (distance-based continuous score in [0,1]): `model/risk.py`
- RUL supervised inputs used by models: `health_index`, `HI_velocity`, `HI_variability`, `risk_score` in `model/rul.py`

## Key Results

Verified from current code execution + stored artifacts in this repository environment:

- PC1 explained variance: 0.6425 (64.25%)
- Health Index means:
  - Early-life mean HI: 0.7528
  - Late-life mean HI: 0.1812
- Silhouette score: 0.4005 (≈ 0.40)
- Mean Spearman rho (HI monotonicity): -0.9250
- Pearson r (risk vs RUL): -0.7683
- RMSE by model:
  - `linear_regression`: 19.49
  - `random_forest`: 19.32
  - `gradient_boosting`: 18.55
- NASA score by model:
  - `linear_regression`: 609.2
  - `random_forest`: 822.3
  - `gradient_boosting`: 694.4
- Best model: `gradient_boosting`
- Feature importances:
  - `random_forest`: `risk_score` 0.7675674339785266, `health_index` 0.13479824472119425, `HI_velocity` 0.054379040340413035, `HI_variability` 0.04325528095986608
  - `gradient_boosting`: `risk_score` 0.6955711519585897, `health_index` 0.21307491347855667, `HI_velocity` 0.08761191587245681, `HI_variability` 0.003742018690396893
- Anomalous engine IDs: 48, 51

## Code Standards

- Type hints are required on function signatures.
- Docstrings follow the project pattern centered on:
  - Purpose
  - Input shape
  - Output shape
  - Assumptions
  - Failure conditions
- Every transformation stage should document those five elements in module/function docstrings.

## Environment

- Python version in active project venv: 3.12.12
- Virtual environment path: `/Users/arnavhmutt/Desktop/aviation-ds-projects/.venvs/project-2`
- Activation command:
  - `source /Users/arnavhmutt/Desktop/aviation-ds-projects/.venvs/project-2/bin/activate`
- Pinned dependencies (`requirements.txt`):
  - numpy==1.26.4
  - pandas==2.2.2
  - scikit-learn==1.4.2
  - joblib==1.4.2
  - matplotlib==3.8.4
  - scipy==1.13.0
  - seaborn==0.13.2
  - plotly==5.22.0
  - streamlit==1.32.0
  - pyyaml==6.0.1
  - ipykernel==6.29.4
  - jupyter==1.0.0
  - black==24.4.2
  - pytest==8.2.0
- Rule: training (`scripts/train_rul_artifacts.py`) and dashboard (`app/dashboard.py`) must always run under the same interpreter/venv.

## Iteration 2 — Current Work

1. AOG cost simulator with synthetic benchmark dataset grounded in IATA MRO reports and DGCA data
2. Multi-dataset execution FD002-FD004 with operating condition normalisation
3. Wire confidence intervals into dashboard
4. Wire anomaly detection scatter plot into dashboard
5. Engine SVG graphic with sensor contribution hover
6. Agentic AI diagnostic narration
7. Streamlit Cloud deployment

## Ownership

- ML pipeline, `scripts/`, `data/`, `features/`, `model/`, `evaluation/` — owned by Arnav.
- Dashboard frontend, `app/components/`, `app/dashboard.py` — owned by Antigravity IDE.

## Notion Command Center

- Page ID: Not found in repository files.
- URL: Not found in repository files.
- Session rule: Always check the Notion command center for current project state before starting any session. If missing, obtain the active page ID/URL first.
