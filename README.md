# EngineWatch — Interpretable Turbofan Engine Health Monitoring

> Predictive maintenance prototype for NASA CMAPSS turbofan engines.  
> Built with interpretable ML — no deep learning, full diagnostic chain.

---

## What This System Does

EngineWatch monitors turbofan engine degradation and predicts Remaining Useful Life (RUL) using an interpretable pipeline grounded in aerospace physics. Given per-cycle engine telemetry, the system produces:

- A **Health Index** tracking degradation from healthy (0.75) to critical (0.18)
- **Health Velocity** — rate of decline per cycle
- **Health Variability** — instability signal preceding failure
- **Cluster-based health state** — Healthy / Degrading / Critical
- **Continuous Risk Score** — distance-based, normalised to [0, 1]
- **RUL Prediction** with confidence intervals — best model RMSE 18.40 cycles
- **Sensor contribution breakdown** — which sensors are driving degradation
- **Anomaly detection** — flags engines outside the training distribution

---

## Architecture

```text
Raw Telemetry (26 cols)
    ↓
Preprocessing (flat sensor removal, StandardScaler)
    ↓
PCA Health Index (PC1, 64.3% variance explained)
    ↓
Health Velocity (rolling linear regression slope)
Health Variability (rolling std, normalised)
    ↓
KMeans Clustering (k=3, silhouette 0.40)
    ↓
Risk Score (Euclidean distance to Critical centroid)
    ↓
RUL Prediction (Gradient Boosting, RMSE 18.40)
    ↓
Streamlit Dashboard
```

---

## Key Results

| Metric                       | Value             |
| ---------------------------- | ----------------- |
| PC1 explained variance       | 64.3%             |
| HI early life → late life    | 0.75 → 0.18       |
| Silhouette score             | 0.40              |
| Spearman ρ (HI monotonicity) | −0.925            |
| Risk–RUL Pearson r           | −0.768            |
| Best model                   | Gradient Boosting |
| RMSE                         | 18.40 cycles      |
| NASA score                   | 607.3             |
| Late predictions             | 46 / 100 engines  |
| Early predictions            | 54 / 100 engines  |

---

## Project Structure

```text
data/           load.py, preprocess.py
features/       health_index.py, velocity.py, variability.py
model/          clustering.py, risk.py, rul.py
evaluation/     validation.py
app/            dashboard.py, components/
config/         config.yaml
notebooks/      01_data_exploration.ipynb, 02_rul_evaluation.ipynb
reports/        rul_evaluation_plots.png
```

---

## Why Not Deep Learning?

This is a deliberate architectural decision, not a limitation:

1. **Dataset size** — ~20K rows, 100 engines. Insufficient for reliable LSTM/Transformer training
2. **Physics** — HPC degradation follows known exponential decay captured by PCA
3. **Interpretability** — every prediction has a traceable, explainable inference chain
4. **Deployment** — classical models are faster, lighter, and easier to audit

---

## Dashboard

The Streamlit dashboard provides:

- Fleet risk overview (bar chart, top 5 priority list, degradation heatmap)
- Per-engine health trajectory, velocity, variability, and risk gauge
- Predicted RUL with confidence interval
- Degradation state timeline
- Model evaluation panel (all three models compared)

Run locally:

```bash
pip install -r requirements.txt
```

Before training or launching the dashboard, activate the project virtual environment so artifacts are built and loaded with the same Python/scikit-learn stack.

```bash
source .venvs/project-2/bin/activate
```

### ▶️ Running the Project

Step 1 — Train models (required)

```bash
python train_rul_artifacts.py
```

Step 2 — Launch dashboard

```bash
streamlit run app/dashboard.py
```

Note: Running `python dashboard.py` is only for sanity checks.
The dashboard must be launched using Streamlit.

---

## Environment & Model Artifacts

Model artifacts are version-sensitive. Train and run the dashboard inside the same project virtual environment so the joblib files are created and loaded with the same Python and scikit-learn stack.

Use the project environment at:

```bash
source .venvs/project-2/bin/activate
```

Recommended run sequence:

```bash
pip install -r requirements.txt
python train_rul_artifacts.py
streamlit run app/dashboard.py
```

Do not mix interpreters between training and dashboard execution.

---

## Dataset

NASA CMAPSS FD001 — 100 training engines, 100 test engines, single operating condition, HPC degradation fault mode.

Reference: Saxena et al., _Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation_, PHM 2008.

---

## Status

**Iteration 1 — Complete**  
Full FD001 pipeline, dashboard, RUL prediction, evaluation, anomaly detection.

## Iteration 2 — AOG Cost Impact Simulator

### What's New

- **AOG Cost Impact Simulator** (`app/components/aog_cost_simulator.py`) —
  translates ML risk outputs into economic maintenance decisions
- **Decision formula:** Expected AOG cost = P(failure) × AOG cost per event.
  Act now if preventive maintenance cost < expected AOG cost
- **Dashboard panel** renders below RUL prediction — shows both cost scenarios,
  urgency badge, and estimated saving in Rs Crores
- **BTS Form 41 data pipeline** (`scripts/fetch_bts_benchmarks.py`) — fetches
  real engine maintenance cost data from US DOT transtats database

### Cost Benchmarks

All figures sourced from published primary sources — see `data/sources.md`:

- Go First NCLT IBC Filing, May 2023 (India narrowbody AOG: ~Rs 4.55 Cr/event)
- IATA MCX FY2024 Public Report (global average: $1,522/flight-hour)
- BTS Form 41 Schedule P-5.2 + T-2, FY2023 (engine cost per block-hour)
- Eurocontrol Standard Inputs Edition 10, May 2024

### Multi-Dataset Support

Pipeline validated across all four CMAPSS datasets:

- FD001 — 1 condition, 1 fault mode ✅ Full dashboard support
- FD002 — 6 conditions, 1 fault mode ✅ Pipeline validated
- FD003 — 1 condition, 2 fault modes ✅ Pipeline validated
- FD004 — 6 conditions, 2 fault modes ✅ Pipeline validated (NASA score 107,724 → 14,655)

Multi-condition dashboard visualization is in active development.

### Running the AOG Simulator

No retraining required. The simulator reads from pipeline outputs at runtime.
All cost parameters configurable in `config/config.yaml → aog_simulator`.
