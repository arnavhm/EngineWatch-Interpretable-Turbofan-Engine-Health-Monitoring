# GitHub Copilot Instructions — EngineWatch (CMAPSS Multi-Dataset Health Monitor)

## Project Context

Turbofan engine degradation monitoring across NASA CMAPSS FD001–FD004.
Hybrid predictive maintenance system: interpretable health monitoring +
Remaining Useful Life (RUL) prediction. Live in production at enginewatch.tech.
Deep learning is never used — interpretability and dataset size are hard
constraints, not preferences.

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
  `narration_panel` called with no dataset_id).

## Approved Feature Set (current — dual-axis, per-fault-mode)

- **Dual-axis Health Index**: separate PCA per axis — `HI_hpc` and `HI_fan`
  (`features/health_index.py::build_dual_health_index` /
  `apply_dual_health_index`). Each axis's HI is PC1, sign-aligned so it
  trends downward over engine life, min-max normalized to [0,1].
- **Health Velocity**: rolling-window slope (`np.polyfit`), computed per axis.
- **Health Variability**: rolling standard deviation, computed per axis.
- **Fault-mode classification**: engines are classified HPC-fault vs Fan-fault
  (`model/fault_classifier.py`) before clustering/risk — this determines which
  axis is "operative" for that engine.
- **Clustering**: KMeans k=3 (Healthy/Degrading/Critical), fit PER fault mode
  (`model/clustering.py::build_clustering_per_fault_mode`), features =
  `[HI_hpc, HI_fan, HI_hpc_velocity, HI_fan_velocity]`.
- **Risk score**: NOT Euclidean distance from a centroid. Computed as a
  normalized distance from the operative axis
  (`model/risk.py::RiskScorer`): `d = 1 - HI_hpc` for HPC-fault engines,
  `d = HI_fan` for Fan-fault engines, `d = 1 - min(HI_hpc, HI_fan)` unified;
  then `risk = (d - d_min) / (d_max - d_min)` with `d_min`/`d_max` fit on
  training data only.
- **RUL prediction**: 3 models trained per dataset (LinearRegression,
  RandomForest, HistGBR monotonic) — HistGBR (`gradient_boosting` dict key,
  cosmetic label) is best on FD001/3/4; check per-dataset which wins, don't
  assume.
- **Confidence intervals**: from RandomForest tree-variance, even on datasets
  where HistGBR is the point-prediction model — RF is retained specifically
  for this, not dead weight.

No new feature types without explicit approval.

## Multi-dataset regime handling — required for FD002/FD004

- `data/regime.py::RegimeScaler` replaces a bare `StandardScaler` whenever
  `config["regimes"]["enabled"]` is True. At `n_regimes=1` (FD001, FD003) it
  degenerates to a single global scaler. At `n_regimes=6` (FD002, FD004) it
  fits per-regime KMeans + StandardScaler on operating-condition columns
  before scaling sensors — without this, PCA's PC1 captures which flight
  regime an engine is in rather than actual degradation.
- **`resolve_regime_config(config, dataset_id)` is the ONLY place that sets
  `config["regimes"]["n_regimes"]` from `by_dataset[dataset_id]`.** Every
  per-dataset config builder must call this helper — never reimplement the
  override inline. Duplicated copies of this logic falling out of sync is
  exactly how a real bug shipped (FD002/FD004 silently ran with
  `n_regimes=1` for a period).
- `RegimeScaler.fit()`'s silhouette check on regime clustering MUST stay
  bounded via `silhouette_sample_size` (config-driven, default 5000).
  Unbounded `silhouette_score` on a ~53,700-row training set (FD002)
  attempts a ~23GB pairwise distance matrix — do not remove this bound.
- `RegimeScaler.inverse_transform_df(df, sensor_cols)` takes two required
  positional args — used to recover physical sensor units for display.
  Requires original `setting_cols` still present (re-join by unit+cycle from
  raw data if calling downstream of `transform_df`, which drops them).

## RUL artifact loading — fail loud, never fall back silently

`app/utils/rul_artifacts.py` loads ONLY from
`models/{dataset_id}/rul_artifacts.joblib`. No fallback chain to a flat
legacy path or a notebooks/ path. A file that exists but fails to unpickle is
a version-mismatch correctness bug (sklearn version drift between training
and serving environment) — it must raise immediately with a clear message,
never silently try a different path. A prior fallback chain did exactly this
wrong and silently served a months-old, wrong-dataset model with no visible
error.

## Environment discipline

Training, dashboard, and API must always run under `.venvs/project-2`
(Python 3.12, scikit-learn 1.4.2) — never base miniforge/conda. Version drift
between training and serving causes silent unpickling failures or corrupted
model behavior. `app/dashboard.py` enforces this at import time; do not
remove that guard.

## Architecture — always follow this structure

- `data/load.py` — raw file ingestion only
- `data/preprocess.py` — scaling (via `RegimeScaler` when regimes enabled),
  sensor selection, RUL computation
- `data/regime.py` — regime-aware scaling (see above)
- `features/health_index.py` — dual-axis PCA health index
- `features/velocity.py`, `features/variability.py` — per-axis rolling stats
- `model/fault_classifier.py` — HPC-fault vs Fan-fault classification
- `model/clustering.py` — KMeans k=3, per fault mode
- `model/risk.py` — per-fault-mode normalized risk score
- `model/rul.py` — RUL regression (3 models, best selected per dataset)
- `evaluation/validation.py` — monotonicity checks, RMSE, NASA scoring,
  Spearman risk-RUL correlation (the actual acceptance criterion, not RMSE)
- `app/dashboard.py` — Streamlit dashboard; training must NEVER occur here
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

- FD001: 100 train engines, 1 operating condition, 1 fault mode (HPC)
- FD002: 260 train engines, 6 operating conditions, 1 fault mode (HPC)
- FD003: 100 train engines, 1 operating condition, 2 fault modes (HPC + Fan)
- FD004: 249 train engines, 6 operating conditions, 2 fault modes (HPC + Fan)
- All: 26 columns (unit, cycle, 3 op settings, 21 sensors)
- Training trajectories run to failure (RUL=0 at last cycle); test
  trajectories are truncated before failure

## Known open item — artifact file sizes exceed GitHub's 100MB limit

`models/{FD002,FD004}/rul_artifacts.joblib` and `rul_random_forest.joblib`
are ~300MB each. Do not attempt to git push these as-is or default to Git LFS
— investigate first whether the standalone RF file is redundant with
`artifacts.all_models["random_forest"]` already inside `rul_artifacts.joblib`,
and whether RF hyperparameters (n_estimators/max_depth) can shrink without
losing CI quality, before choosing LFS vs. deploying large caches outside git
entirely.

## Multi-dataset regime handling (added 2026-07-06)

- `data/regime.py` — `RegimeScaler` replaces bare `StandardScaler` when
  `config["regimes"]["enabled"]` is True. Degenerates to a plain global scaler
  at `n_regimes=1` (FD001, FD003). Fits per-regime KMeans + StandardScaler at
  `n_regimes=6` (FD002, FD004).
- **`resolve_regime_config(config, dataset_id)` is the ONLY place that sets
  `config["regimes"]["n_regimes"]` from `config["regimes"]["by_dataset"]`.**
  Every function that builds a per-dataset config (`load_pipeline_data_uncached`,
  `train_all_datasets.py::_dataset_config`, `train_rul_artifacts.py::main`) must
  call this helper. Do not reimplement this override inline anywhere — that
  exact duplication caused a silent bug where FD002/FD004 ran with n_regimes=1
  for weeks because one caller's copy of the override logic fell out of sync.
- `RegimeScaler.fit()` computes `silhouette_score` on the regime clustering.
  This is BOUNDED via `silhouette_sample_size` (default 5000, config-driven).
  Never remove this bound or call `silhouette_score` unbounded on setting_cols —
  FD002's ~53,700-row training set would attempt a ~23GB pairwise distance
  matrix otherwise. This is a hard requirement, not a style preference.
- `RegimeScaler.inverse_transform_df(df, sensor_cols)` — required two positional
  args, both mandatory. Used to recover physical sensor units from scaled
  values for display (dashboard sensor panel, offline sensor cache). Requires
  original `setting_cols` still present in `df` (transform_df drops them —
  re-join by unit+cycle from raw data first if calling this downstream of
  transform_df).

## dataset_id propagation — audit before adding any new component/endpoint

Every function/component that takes `dataset_id` must actually receive it from
its caller — never rely on the `="FD001"` default silently. Three separate
instances of this exact bug shipped in one day: `api/main.py`'s lifespan
(hardcoded to `"FD001"` only), `app/components/model_evaluation.py` (called
with empty parens), `render_narration_panel` (same). Before merging any new
component or endpoint, grep for empty-parens calls to anything with a
`dataset_id` parameter:
`grep -rn "some_function(\s*)" --include="*.py" app/ api/`

## RUL artifact loading — fail loud, no fallback chain

`app/utils/rul_artifacts.py::_load_rul_artifacts_uncached` loads ONLY from
`models/{dataset_id}/rul_artifacts.joblib`. There is no fallback to a flat
legacy path or a notebooks/ path — that fallback chain used to silently serve
a stale May-14 model from `notebooks/models/` whenever the running Python's
sklearn version didn't match the pickled artifact's version, with zero visible
error. If this file exists but fails to unpickle, that is a version-mismatch
correctness bug (check `.venvs/project-2` is actually activated) — it must
raise, never silently try another path.

## Environment discipline — enforced at process start, not just documented

`app/dashboard.py` now raises at import time if `sys.executable` isn't
`.venvs/project-2`'s interpreter. Do not remove this guard. Base miniforge
(currently sklearn 1.8.0 vs. the project's pinned 1.4.2) has caused two
separate incidents: silently-mispickled training artifacts, and a stale model
being served with no error. Never train, run the dashboard, or run the API
under any interpreter but `.venvs/project-2`.

## OPEN — artifact file sizes exceed GitHub's 100MB limit (unresolved, blocking push)

`models/{FD002,FD004}/rul_artifacts.joblib` and `rul_random_forest.joblib` are
~300MB each (RandomForest with many/deep trees). GitHub hard-rejects any file
over 100MB. Current state: local commits are AHEAD of origin/main and cannot
be pushed until this is resolved. Do NOT retry `git push` as-is — it will keep
failing identically regardless of network conditions. See task below.

## Known duplication needing verification, not yet resolved

- `models/{dataset_id}/rul_random_forest.joblib` may be pure duplication of
  `artifacts.all_models["random_forest"]` already inside `rul_artifacts.joblib`
  — unconfirmed whether anything loads the standalone file independently.
- `models/FD001/rul_artifacts_full.joblib` (June 20 timestamp) looks like a
  stale naming-scheme leftover — unconfirmed whether anything still reads it.
