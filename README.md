# EngineWatch — Interpretable Turbofan Engine Health Monitoring

> Deployed predictive maintenance system for NASA CMAPSS turbofan engines.
> Interpretable ML, zero-live-recompute API, full CI automation — no deep learning, full diagnostic chain.

**Live at:** [enginewatch.tech](https://enginewatch.tech)

---

## What This System Does

EngineWatch monitors turbofan engine degradation and predicts Remaining Useful Life (RUL) using an interpretable pipeline grounded in aerospace physics, across all four NASA CMAPSS datasets (FD001–FD004). Given per-cycle engine telemetry, the system produces:

- A **Health Index** tracking degradation from healthy to critical
- **Health Velocity** — rate of decline per cycle
- **Health Variability** — instability signal preceding failure
- **Cluster-based health state** — Healthy / Degrading / Critical (KMeans, k=3)
- **Continuous Risk Score** — normalized distance metric, fit on train only
- **RUL Prediction** with confidence intervals (RandomForest tree variance)
- **Sensor contribution breakdown** — which sensors are driving degradation
- **Anomaly detection** — flags engines outside the training distribution
- **Fleet-level analytics** — cross-engine risk ranking, trend analysis, handover summaries
- **Agentic narration** — plain-language diagnostic explanations per engine (Gemini 2.5 Flash runtime, never load-bearing for correctness)

Every prediction is served from pre-computed cached artifacts — the API layer performs zero live ML computation.

---

## Architecture

```text
Raw Telemetry (26 cols, FD001–FD004)
    ↓
Preprocessing (flat sensor removal, StandardScaler)
Regime Normalization (RegimeScaler — load-bearing for FD002/FD004, degenerates to 1 regime for FD001/FD003)
    ↓
PCA Health Index
    ↓
Health Velocity (rolling linear regression slope)
Health Variability (rolling std, normalised)
    ↓
KMeans Clustering (k=3)
    ↓
Risk Score — d = 1 - HI_hpc, normalized (d - d_min) / (d_max - d_min), fit on train only
    ↓
RUL Prediction — Monotonic HistGradientBoostingRegressor (sklearn 1.4.2)
  monotonic constraints: [health_index: 1, HI_velocity: 0, HI_variability: -1, risk_score: -1]
  RandomForest retained for confidence intervals only, never as point-prediction
    ↓
Pre-computed artifact cache (models/{dataset_id}/rul_artifacts.joblib)
    ↓
FastAPI backend (zero live recomputation) → React/Vite/TypeScript frontend
```

**Fault-mode routing:** HPC-only for all four datasets (`n_fault_modes_by_dataset: 1` everywhere). The dual-axis (HPC + Fan) code path is architecturally real and correct but currently inert in production — re-verified empirically that the fan-degradation axis is genuinely non-predictive for FD003/FD004 (Spearman 0.032 / 0.114) even after fixing two latent bugs found during that investigation. Reactivating it is a deliberate, not-yet-made decision — never re-enable without an explicit brief.

---

## Key Results — Canonical Metric Table (HPC-only, verified)

**Acceptance criterion is risk–RUL Spearman correlation, not NASA score or RMSE alone** — the deliberate trade-off is a traceable inference chain over benchmark-chasing.

| Dataset | Best Model | RMSE | NASA Score | Risk–RUL Spearman |
|---|---|---|---|---|
| FD001 | HistGBR (monotonic) | 18.459 | 617.5 | −0.750 |
| FD002 | HistGBR (monotonic) | 31.125 | 13,635 | −0.765 |
| FD003 | HistGBR (monotonic) | 22.798 | 1,995.7 | −0.816 |
| FD004 | HistGBR (monotonic) | 34.410 | 53,028 | −0.736 |

FD002/FD004 are documented weaker points, not clean wins — FD002 clustering silhouette (0.284) sits below the 0.30 threshold and is a disclosed limitation, not a bug.

**Canonical gate (Engine 34 / FD001, checked on every deploy):**

```
risk_score:    0.7402876566726511
rul_cycles:    3.698652753342952
health_index:  0.2597123433273489
risk_state:    Critical
rmse:          18.459221643626265
```

Single source of truth: `config/canonical_gate.json`, consumed directly by `ci-live-gate.yml`. If any document (including this README) disagrees with that file, the file is correct.

---

## Why Not Deep Learning?

A deliberate architectural decision, not a limitation:

1. **Dataset size** — ~20K rows per dataset, ~100 engines. Insufficient for reliable LSTM/Transformer training.
2. **Physics** — HPC degradation follows a known decay pattern captured well by PCA + monotonic gradient boosting.
3. **Interpretability** — every prediction has a traceable, explainable inference chain. No SHAP either — SHAP is a post-hoc approximation; this system is interpretable by construction (monotonic constraints + linear health-index decomposition), not explained after the fact.
4. **Deployment** — classical models are lighter, auditable, and reproducible without GPU infrastructure.
5. **Model scope discipline** — C-MAPSS models a 2-spool turbofan (GE90/PW4090-class). A 3-spool Rolls-Royce Trent architecture is explicitly out of scope for this dataset, since the intermediate-pressure shaft isn't modeled — the system does not claim generality beyond what the data supports.

The accepted cost: higher RMSE (~18.5 vs. ~13 achievable with an LSTM on FD001) in exchange for a fully traceable, monotonic, auditable inference chain.

---

## Deployment & Infrastructure

- **Backend:** FastAPI + uvicorn, `systemd` service `enginewatch`, fully cache-backed
- **Frontend:** React / Vite / TypeScript / Tailwind / Recharts, hand-rolled routing (~25 lines, zero new dependencies)
- **Reverse proxy:** Caddy, Let's Encrypt SSL, `enginewatch.tech`
- **Host:** DigitalOcean droplet (2GB/1vCPU), 2GB swap file active
- **Artifacts:** `models/{dataset_id}/rul_artifacts.joblib` (100–330MB each), deployed via `rsync`, not git (gitignored — exceeds GitHub's 100MB limit)
- **Deploy verification:** `/api/version` returns the deployed git commit hash; `ci-live-gate.yml` hits production on a schedule and hard-fails if unreachable; manual browser fresh-viewer check is still required per `DEPLOY.md` and is not replaced by automation

## CI Automation

- **`ci-static.yml`** — runs on every PR: AST-based `dataset_id` threading check, exact dependency pinning, regime-config duplication check, cache-tracking correctness
- **`ci-live-gate.yml`** — scheduled every 6 hours (and manually triggerable): hits production directly, validates against `config/canonical_gate.json`
- **`/api/version`** — makes "is the droplet running what I think" a `curl`, not an SSH session

## AI Coordination Contract

This project is built with a documented multi-agent workflow, not ad hoc prompting:

- **Claude** (this assistant) — architecture, correctness sign-off, brief authoring (~90% of reasoning work)
- **Antigravity** (Google, Gemini-based agentic IDE) — executes structured briefs for backend/ML/deploy and frontend work; reads `AGENTS.md` and auto-discovers `.agents/skills/`
- **Claude Code** (Anthropic CLI agent) — reasoning-heavy/cross-cutting execution; reads `CLAUDE.md` (includes `AGENTS.md`), no access to `.agents/skills/`
- **Claude Cowork** (Anthropic, autonomous batch agent) — narrow, pre-scoped, report-only-by-default tasks; does not auto-read `AGENTS.md`, constraints must be inline in the prompt
- **GitHub Copilot** — inline mechanical work only (renames, lint, boilerplate)
- **Gemini 2.5 Flash** — narration runtime only, never load-bearing for correctness

Governing documents: `AGENTS.md` (repo root), `CLAUDE.md`, `.github/copilot-instructions.md`, `.agents/skills/`, `.agents/test-prompts.md` (five regression tests, each grounded in a real caught incident, re-run cold after any contract change).

**Verification discipline:** "looks right," "confirmed working," and "deployment complete" are claims to check against real diff/curl/terminal output, never accepted as prose summaries — this pattern has caught multiple real regressions in production.

---

## Project Structure

```text
data/           load.py, preprocess.py, regime.py (RegimeScaler)
features/       health_index.py, velocity.py, variability.py
model/          clustering.py, risk.py, rul.py, fault_classifier.py
evaluation/     validation.py
app/            FastAPI backend, endpoint routers
frontend/       React/Vite/TypeScript — Fleet Command (Level 1), Engine Drill-Down (Level 2)
config/         config.yaml, canonical_gate.json
scripts/        deployment + data pipeline utilities
.agents/        skills/, test-prompts.md (Antigravity-only, auto-discovered)
AGENTS.md       shared multi-agent operating contract
CLAUDE.md       Claude Code entry point (includes AGENTS.md)
DEPLOY.md       authoritative manual deployment process
```

---

## Environment & Model Artifacts

Model artifacts are version-sensitive. Train and run the backend inside the same project virtual environment so joblib files are created and loaded against the same Python/scikit-learn stack.

```bash
# venv lives one directory above the repo root
source ../.venvs/project-2/bin/activate

# Always verify before trusting any measurement:
which python
pip show scikit-learn numpy joblib
```

Pinned versions: Python 3.12, scikit-learn 1.4.2, numpy 1.26.4, joblib 1.4.2.

Do not mix interpreters between training and serving. Never reuse a shell session where `source .../activate` previously failed — abandon it and open a fresh terminal; stale env vars silently corrupt subsequent `which python` / `pip show` output.

---

## Dataset

NASA CMAPSS FD001–FD004 — single- and multi-operating-condition turbofan degradation data, HPC fault mode (as currently routed in production).

Reference: Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008.

---

## Status

**v3.0 — Deployed, multi-dataset, fleet analytics live, AI coordination contract formalized, CI automation live.**

- ✅ Full FD001–FD004 pipeline, HPC-only monotonic HistGBR, live at enginewatch.tech
- ✅ Zero-live-recompute API surface across all endpoints
- ✅ Fleet-level analytics, agentic narration layer
- ✅ Full CI automation (`ci-static.yml`, `ci-live-gate.yml`, `/api/version`)
- ✅ Formal multi-agent coordination contract (`AGENTS.md`, `.agents/`)
- ✅ Frontend Level 1 (Fleet Command) shipped and live
- 🔄 Frontend Level 2 (Engine Drill-Down split-screen) — code-complete locally, not yet pushed/deployed; primary open item blocking the EngineWatch Stability Gate
- ⚪ AeroGraph (aviation knowledge-graph platform) — blueprint stage only, gated on the Stability Gate, not a committed date

Superseded — do not cite: any RMSE/NASA figures outside the canonical table above, any "dual-axis live" claim, any FD004 improvement-percentage figures. See `AGENTS.md` for the full dead-numbers list and incident history behind each hard constraint.