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
- **RUL Prediction** with confidence intervals — best model RMSE 18.55 cycles
- **Sensor contribution breakdown** — which sensors are driving degradation
- **Anomaly detection** — flags engines outside the training distribution

---

## Architecture

```
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
RUL Prediction (Gradient Boosting, RMSE 18.55)
    ↓
Streamlit Dashboard
```

---

## Key Results

| Metric | Value |
|---|---|
| PC1 explained variance | 64.3% |
| HI early life → late life | 0.75 → 0.18 |
| Silhouette score | 0.40 |
| Spearman ρ (HI monotonicity) | −0.925 |
| Risk–RUL Pearson r | −0.768 |
| Best model | Gradient Boosting |
| RMSE | 18.55 cycles |
| NASA score | 694.4 |
| Late predictions | 53 / 100 engines |
| Early predictions | 47 / 100 engines |

---

## Project Structure

```
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
streamlit run app/dashboard.py
```

---

## Dataset

NASA CMAPSS FD001 — 100 training engines, 100 test engines, single operating condition, HPC degradation fault mode.

Reference: Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008.

---

## Status

**Iteration 1 — Complete**  
Full FD001 pipeline, dashboard, RUL prediction, evaluation, anomaly detection.

**Iteration 2 — Planned**  
Multi-dataset (FD002–FD004 with operating condition normalisation), fault localisation, AOG cost simulator, agentic diagnostic narration.
