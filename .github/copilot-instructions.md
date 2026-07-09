@AGENTS.md

# GitHub Copilot Instructions — EngineWatch (CMAPSS Multi-Dataset Health Monitor)

## Project Context

Turbofan engine degradation monitoring across NASA CMAPSS FD001–FD004.
Hybrid predictive maintenance system: interpretable health monitoring +
Remaining Useful Life (RUL) prediction. Live in production at enginewatch.tech.
Deep learning is never used — interpretability and dataset size are hard
constraints, not preferences. Shared environment/verification invariants are
in `AGENTS.md` (included above) — this file covers architecture and
Copilot-specific mechanical conventions only, not those invariants again.

## Strict Constraints — enforce always

- No deep learning of any kind (LSTM, RNN, Transformer, neural nets). Not a
  suggestion — a hard architectural boundary.
- RUL models: LinearRegression, RandomForest, HistGradientBoostingRegressor
  (monotonic constraints) only. No other model families without explicit
  approval.
- No global mutable state, no hardcoded constants. All parameters from
  `config/config.yaml`.
- No monolithic scripts. Training and inference are always separate
  functions. **Never retrain inside the dashboard or API request path.**
- `dataset_id` must be threaded explicitly through every function/component
  that needs it — never rely on a default value silently. Before adding any
  new component or endpoint, grep for empty-parens calls to functions that
  take `dataset_id`:
  `grep -rn "some_function(\s*)" --include="*.py" app/ api/`
  Three separate bugs shipped in one day from exactly this pattern
  (`api/main.py` lifespan hardcoded to FD001, `model_evaluation.py` and
  `render_narration_panel` called with no dataset_id). This has recurred
  since — always grep before merging, not just the first time.

## Approved Feature Set (current — HPC-only routing, all four datasets)

- **Dual-axis Health Index**: separate PCA per axis exists in code —
  `HI_hpc` and `HI_fan` (`features/health_index.py::build_dual_health_index`
  / `apply_dual_health_index`). Each axis's HI is PC1, sign-aligned so it
  trends downward over engine life, min-max normalized to [0,1]. **This code
  path is correct and architecturally sound but currently INERT in
  production**: `n_fault_modes_by_dataset=1` for all four datasets forces
  every engine to `fault_mode="hpc"` via a dummy short-circuit
  (`model/fault_classifier.py:205-215` has an inline comment marking this as
  a dead branch). Fan-axis was confirmed genuinely non-predictive for FD003/FD004
  (Spearman 0.032/0.114) even after fixing two real latent bugs — this is a
  settled result, re-verified twice, not an open question. Do not re-enable
  fault-mode routing without an explicit instruction to do so, and don't
  write new code assuming the fan axis is live.
- **Health Velocity**: rolling-window slope (`np.polyfit`), computed per axis.
- **Health Variability**: rolling standard deviation, computed per axis.
- **Fault-mode classification**: `model/fault_classifier.py` code exists and
  is sound, but its output is currently overridden to always route to "hpc"
  (see above) — don't treat its classification as operative in the current
  deployed behavior.
- **Clustering**: KMeans k=3 (Healthy/Degrading/Critical),
  (`model/clustering.py::build_clustering_per_fault_mode`) — in current
  practice this means clustering per the single active ("hpc") fault mode,
  features = `[HI_hpc, HI_fan, HI_hpc_velocity, HI_fan_velocity]`.
- **Risk score**: normalized distance via `model/risk.py::RiskScorer`. As
  currently routed, every engine uses `d = 1 - HI_hpc` (the HPC-fault
  branch); the `d = HI_fan` / Fan-fault branch exists in code but is not
  reached by any engine under the current config. `risk = (d - d_min) /
  (d_max - d_min)`, `d_min`/`d_max` fit on training data only.
- **RUL prediction**: 3 models trained per dataset (LinearRegression,
  RandomForest, HistGBR monotonic) — HistGBR wins on all four datasets
  currently; check per-dataset when re-training, don't assume this is
  permanent.
- **Confidence intervals**: from RandomForest tree-variance, even on datasets
  where HistGBR is the point-prediction model — RF is retained specifically
  for this, not dead weight.

No new feature types without explicit approval. No re-enabling fault-mode
routing without an explicit instruction — see `AGENTS.md` Section 9.

## Multi-dataset regime handling — required for FD002/FD004

- `data/regime.py::RegimeScaler` replaces a bare `StandardScaler` whenever
  `config["regimes"]["enabled"]` is True. At `n_regimes=1` (FD001, FD003) it
  degenerates to a single global scaler. At `n_regimes=6` (FD002, FD004) it
  fits per-regime KMeans + StandardScaler on operating-condition columns
  before scaling sensors — without this, PCA's PC1 captures which flight
  regime an engine is in rather than actual degradation.
- **`resolve_regime_config(config, dataset_id)` is the ONLY place that sets
  `config["regimes"]["n_regimes"]`.** Every per-dataset config builder
  (`load_pipeline_data_uncached`, `train_all_datasets.py::_dataset_config`,
  `train_rul_artifacts.py::main`) must call this helper — never reimplement
  the override inline. Duplicated copies of this logic falling out of sync
  is exactly how a real bug shipped (FD002/FD004 silently ran with
  `n_regimes=1` for weeks).
- `RegimeScaler.fit()`'s silhouette check MUST stay bounded via
  `silhouette_sample_size` (config-driven, default 5000). Unbounded
  `silhouette_score` on FD002's ~53,700-row training set attempts a ~23GB
  pairwise distance matrix — do not remove this bound.
- `RegimeScaler.inverse_transform_df(df, sensor_cols)` — two required
  positional args. Requires original `setting_cols` still present
  (`transform_df` drops them — re-join by unit+cycle from raw data if
  calling downstream of it).

## RUL artifact loading — fail loud, never fall back silently

`app/utils/rul_artifacts.py::_load_rul_artifacts_uncached` loads ONLY from
`models/{dataset_id}/rul_artifacts.joblib`. No fallback chain to a flat
legacy path or a notebooks/ path. A file that exists but fails to unpickle is
a version-mismatch correctness bug (check `.venvs/project-2` is actually
activated) — it must raise immediately with a clear message, never silently
try a different path. A prior fallback chain did exactly this wrong and
silently served a stale model with no visible error.

## Environment discipline — enforced at process start, not just documented

Training, dashboard, and API must always run under `.venvs/project-2`
(Python 3.12, scikit-learn 1.4.2, joblib 1.4.2) — never base
miniforge/conda. `app/dashboard.py` raises at import time if
`sys.executable` isn't `.venvs/project-2`'s interpreter — do not remove that
guard. Version drift between training and serving has caused silent
unpickling failures and stale-model-served-with-no-error incidents twice
already.

## Architecture — always follow this structure

- `data/load.py` — raw file ingestion only
- `data/preprocess.py` — scaling (via `RegimeScaler` when regimes enabled),
  sensor selection, RUL computation
- `data/regime.py` — regime-aware scaling (see above)
- `features/health_index.py` — dual-axis PCA health index (HPC-only
  operative, per Approved Feature Set above)
- `features/velocity.py`, `features/variability.py` — per-axis rolling stats
- `model/fault_classifier.py` — HPC-fault vs Fan-fault classification (code
  sound, output currently overridden — see above)
- `model/clustering.py` — KMeans k=3
- `model/risk.py` — normalized risk score
- `model/rul.py` — RUL regression (3 models, best selected per dataset)
- `evaluation/validation.py` — monotonicity checks, RMSE, NASA scoring,
  Spearman risk-RUL correlation (the actual acceptance criterion, not RMSE)
- `app/dashboard.py` — Streamlit dashboard; training must NEVER occur here;
  secondary interface, not the primary deployed product
- `api/main.py` — FastAPI; loads pre-computed `.pkl` caches per dataset at
  startup, zero runtime ML computation on any request path
- `scripts/train_rul_artifacts.py`, `scripts/train_all_datasets.py` — the
  only places training happens, always on Mac under `.venvs/project-2`,
  never on the droplet

## Code Standards — always enforce

- Type hints on all function signatures
- Docstrings: Purpose, Input shape, Output shape, Assumptions, Failure
  conditions
- All constants from `config/config.yaml`, loaded via `load_config()`
- Fixed `random_state=42` for all stochastic operations
- No hardcoded file paths, no hardcoded dataset_id defaults relied upon
  silently

## Dataset Facts

- FD001: 100 train engines, 1 operating condition. Fault-mode label: HPC.
- FD002: 260 train engines, 6 operating conditions. Fault-mode label: HPC.
- FD003: 100 train engines, 1 operating condition. Fault-mode labels present
  in raw data: HPC + Fan — **but current routing sends every engine through
  the HPC branch regardless of label** (fan axis confirmed non-predictive,
  Spearman 0.032). Don't cite "FD003 has both fault modes" as describing
  current model behavior — it describes the raw dataset label only.
- FD004: 249 train engines, 6 operating conditions. Same caveat as FD003
  (fan axis Spearman 0.114 — non-predictive).
- All: 26 columns (unit, cycle, 3 op settings, 21 sensors).
- Training trajectories run to failure (RUL=0 at last cycle); test
  trajectories are truncated before failure.

## RESOLVED — artifact file sizes exceed GitHub's 100MB limit

`models/{FD002,FD004}/rul_artifacts.joblib` and similar files (~300MB each)
exceed GitHub's limit. **Resolved**: these deploy via `rsync` directly to the
droplet, never via `git` — gitignored as of commit `fa22891`. See
`DEPLOY.md`'s RUL Model Artifact Deployment section for the exact process.
Do not reopen this as a blocker; it's shipped, working architecture.

## Open — needs verification, not yet resolved

- `models/{dataset_id}/rul_random_forest.joblib` may be pure duplication of
  `artifacts.all_models["random_forest"]` already inside `rul_artifacts.joblib`
  — unconfirmed whether anything loads the standalone file independently.
- `models/FD001/rul_artifacts_full.joblib` (June 20 timestamp) looks like a
  stale naming-scheme leftover — unconfirmed whether anything still reads it.
