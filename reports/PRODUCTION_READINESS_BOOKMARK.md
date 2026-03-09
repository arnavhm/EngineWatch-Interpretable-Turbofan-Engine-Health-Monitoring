# Production Readiness Bookmark (Post-Feature-Engineering)

Use this checklist **after feature engineering and baseline model behavior are stable**.

## Recommended Timing

- Do now: keep writing modular code and tests for new feature modules.
- Do after FE freeze: CI, lint/type gates, CLI entrypoint, artifact/versioning hardening.

## Phase 1 — Quality Gates (first)

- [ ] Add CI workflow to run `pytest -q` on push/PR.
- [ ] Add lint/format/type checks:
  - [ ] `ruff`
  - [ ] `black --check`
  - [ ] `mypy` (start with core modules first)
- [ ] Make CI required before merge.

## Phase 2 — Complete Module Pipelines

- [ ] `features/health_index.py` (PCA PC1, normalized to [0,1])
- [ ] `features/velocity.py` (rolling slope via `numpy.polyfit`)
- [ ] `features/variability.py` (rolling std)
- [ ] `model/clustering.py` (KMeans, `k=3`, `random_state=42`)
- [ ] `model/risk.py` (distance-based normalized risk)
- [ ] `model/rul.py` (baseline regressors: Linear/RandomForest/GradientBoosting)
- [ ] `evaluation/validation.py`:
  - [ ] RMSE
  - [ ] NASA score
  - [ ] monotonicity checks

## Phase 3 — Entrypoints and Reproducibility

- [ ] Add single CLI/script entrypoint for train/infer.
- [ ] Persist all artifacts with version tags (config hash + timestamp):
  - [ ] scaler
  - [ ] trained model
  - [ ] feature metadata
  - [ ] evaluation metrics
- [ ] Add experiment run log (`reports/` or lightweight tracker).

## Definition of Done (8.5+/10)

- [ ] Every pipeline step is module-based and tested.
- [ ] CI gates pass on every PR.
- [ ] Train/infer reproducible from a single command.
- [ ] Artifacts and metrics are versioned and traceable.
