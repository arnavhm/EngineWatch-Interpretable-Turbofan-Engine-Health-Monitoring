## 1. What This System Does
This system monitors turbofan engine health and predicts Remaining Useful Life (RUL) for NASA CMAPSS FD001 engines.

In plain English:
- Input: per-cycle engine telemetry (unit id, cycle, 3 operating settings, 21 sensors), plus test-set RUL offsets.
- Processing: sensor filtering, scaling, physics-aligned health features (Health Index, velocity, variability), clustering into health states, continuous risk scoring, and supervised RUL regression.
- Output:
  - Engine-level health trajectory over time.
  - State labels: Healthy, Degrading, Critical.
  - Continuous risk score in [0, 1].
  - Last-cycle RUL prediction per test engine.
- Primary users: reliability engineers, maintenance planners, and operations analysts who need interpretable condition monitoring and failure-horizon estimates.

## 2. The Dataset
CMAPSS FD001 is a run-to-failure turbofan degradation benchmark:
- 100 training engines, 100 test engines.
- Single operating condition and single fault mode (HPC degradation).
- 26 columns total:
  - unit, cycle
  - op_setting_1, op_setting_2, op_setting_3
  - sensor_1 through sensor_21

What RUL means:
- RUL is how many cycles remain until failure.
- In training data, each trajectory runs to failure, so final-cycle RUL is 0.
- The per-cycle train formula is:
  - RUL(unit, cycle) = max_cycle(unit) - cycle

Why test RUL is in a separate file:
- Test trajectories are truncated before failure.
- Because they do not run to failure in the test file itself, true remaining life must be provided as external end-of-trajectory offsets from RUL_FD001.txt.

Why train mean is about 108 and test mean is about 76:
- The notebook summary table shows:
  - Train (all cycles) mean RUL = 107.81
  - Test (cutoff) mean RUL = 75.52
- Rounded, these are 108 and 76.
- They differ because train mean is computed over all per-cycle rows (including long early-life spans), while test mean is one cutoff value per truncated trajectory.

## 3. Preprocessing
What compute_rul does:
- Exact formula implemented in data/preprocess.py:
  - RUL = max_cycle(unit) - cycle

Why these six sensors were removed:
- sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19
- In notebook analysis, these were flagged as flat sensors (near-zero variance threshold 1e-06), so they contribute little or no degradation signal.

Why StandardScaler is fit on train only:
- The code fits scaler statistics (mean and std) only on training sensor columns, then applies that fixed scaler to test.
- This preserves evaluation integrity and keeps test strictly unseen during fit.

What data leakage means and concrete failure example:
- Leakage is when information from test data influences training-time transformations or model fitting.
- Example here: if scaler were fit on train+test together, test distribution would shift normalization parameters. Then train features and model boundaries are indirectly tuned using test information, producing optimistic metrics (lower RMSE/NASA than true out-of-sample behavior).

## 4. Health Index
Why averaging raw sensors fails:
- Raw sensors do not all move in the same direction with degradation.
- In the project output, top PC1 loadings include opposite signs, for example:
  - sensor_11: +0.309274
  - sensor_4: +0.301180
  - sensor_12: -0.304419
  - sensor_7: -0.298388
- A naive average can cancel opposing trends and hide degradation.

What PCA does mathematically (one paragraph):
- PCA finds orthogonal directions (principal components) that maximize variance in standardized sensor space. If X is the standardized sensor matrix, PCA computes eigenvectors of X^T X. The first component (PC1) is the direction w maximizing variance of projections z = Xw. This converts many correlated sensors into one dominant latent trend. In this pipeline, PC1 is min-max normalized and direction-corrected to produce a bounded health index.

What PC1 explained variance of 64.3% means:
- One single latent direction (PC1) captures 64.3% of total variance in selected sensors.
- Plainly: most of the sensor movement can be summarized by one health trend instead of many separate channels.

Top sensor loadings and interpretation:
- sensor_11: +0.309274
- sensor_12: -0.304419
- sensor_4: +0.301180
- sensor_7: -0.298388
- sensor_15: +0.287067
- Interpretation: these are the strongest contributors to the health axis; positive-loading sensors increase PC1 when they rise, while negative-loading sensors decrease PC1 when they rise.

Why health index was inverted:
- The fitted artifact reports Inversion applied = True.
- The code checks start-vs-end PC1 direction and flips when needed so higher HI corresponds to healthier early life and lower HI to late degraded life.

What HI 0.75 at cycle 1 dropping to 0.18 at last cycle tells us:
- Mean HI at cycle 1 = 0.7528.
- Mean HI at last cycle = 0.1812.
- This is a strong monotonic decline consistent with progressive degradation toward failure.

## 5. Velocity and Variability
What HI_velocity measures:
- Rolling linear-regression slope of HI within each engine.
- For window size w, with x = [0, 1, ..., w-1] and y = HI values in that window:
  - HI_velocity = slope from first-degree fit, equivalent to polyfit(x, y, 1)[0]

Why velocity is 100% negative in late life:
- Verification output reports late-life negative share = 100.0%.
- This means every sampled late-life window shows declining HI, matching end-of-life deterioration.

What HI_variability measures:
- Rolling standard deviation of HI (windowed per engine), then min-max normalized to [0, 1] using train-fitted bounds.

Why variability increases 63% from early to late life:
- Early mean variability = 0.2045.
- Late mean variability = 0.3340.
- Relative increase = (0.3340 - 0.2045) / 0.2045 = 63.3%.

Why these two features add information beyond HI alone:
- HI is state level (how healthy now).
- HI_velocity is direction/rate (how fast health is changing).
- HI_variability is stability/noise (how erratic trajectory is).
- Together, they separate similar HI levels with different risk dynamics.

## 6. Clustering
Why KMeans with k=3 and not k=2 or k=4:
- The implementation enforces exactly 3 clusters for the intended operational states (Healthy, Degrading, Critical).
- If n_clusters is not 3, the clusterer raises an error by design.
- Quantitative k=2 or k=4 comparison metrics are not available in project outputs.

What silhouette score of 0.40 means:
- Reported silhouette = 0.4005.
- This indicates moderate cluster separation: meaningful structure, not perfectly separated.

How cluster labels were assigned and why justified:
- Raw KMeans labels are arbitrary.
- Code maps clusters by mean health_index ordering:
  - highest mean HI -> Healthy
  - middle mean HI -> Degrading
  - lowest mean HI -> Critical
- This is consistent with the health-index interpretation.

What centroids tell us physically:
- Healthy centroid: HI 0.722986, velocity -0.000675, variability 0.210129
- Degrading centroid: HI 0.514162, velocity -0.003546, variability 0.272143
- Critical centroid: HI 0.310073, velocity -0.008325, variability 0.426241
- Physical reading: health level drops, decline rate becomes more negative, and instability increases as engines move toward critical condition.

## 7. Risk Score
What distance metric is used and why:
- Euclidean distance in scaled feature space to the Critical centroid.
- Chosen because clustering and feature geometry are Euclidean in normalized space.

Why the distance is inverted:
- Smaller distance to Critical means higher danger.
- So after normalization, score is inverted so higher score means higher risk.

How it is normalized to 0 to 1:
- For distance d:
  - normalized = (d - d_min) / (d_max - d_min)
  - risk = 1 - normalized
  - then clipped to [0, 1]
- Fitted bounds from outputs:
  - d_min = 0.0453
  - d_max = 8.7041

Risk values per cluster:
- Healthy: 0.4615
- Degrading: 0.6605
- Critical: 0.8711

Why continuous risk is more useful than discrete labels:
- Labels give only coarse bins.
- Continuous score supports threshold policies, prioritization, ranking, and trend-based escalation before a hard class transition.

## 8. Validation
What Spearman correlation measures and why -0.925 is strong:
- Spearman rho measures monotonic rank relationship between cycle and HI per engine.
- Mean rho = -0.9250 indicates very strong monotonic decrease of HI over lifecycle.

What tercile-based progression check does and why it helps:
- Validation converts risk states to ordinal values and compares medians over early/mid/late thirds of each trajectory.
- This is more robust than strict cycle-by-cycle monotonic checks because it tolerates local noise/oscillation while preserving lifecycle progression.

What Pearson r = -0.7683 between risk and RUL means operationally:
- Strong negative association: higher risk generally corresponds to lower remaining life.
- Operationally, risk score is aligned enough with failure horizon for maintenance triage.

## 9. RUL Model
Why RUL is clipped at 125 cycles:
- Training target uses max_rul_clip = 125.
- Early-life very-high RUL region is less actionable and often noisier relative to failure-focused decisions.
- Clipping emphasizes the regime where maintenance decisions are critical.

What each model assumes:
- Linear Regression: approximately linear additive relation between engineered features and RUL.
- Random Forest: non-linear interactions captured via bagged decision trees.
- Gradient Boosting: stage-wise additive trees minimizing residual error, typically stronger for structured tabular non-linearity.

Why gradient boosting won:
- Metrics from verification:
  - linear_regression: RMSE 19.49, NASA 608.2
  - random_forest: RMSE 19.39, NASA 832.0
  - gradient_boosting: RMSE 18.60, NASA 698.0
- Best RMSE is gradient_boosting at 18.60.

What risk_score 68% importance means:
- Gradient-boosting feature importance:
  - risk_score = 0.681477 (68.15%)
- Most split gain came from risk_score, so the model relies heavily on the continuous risk embedding.

What RMSE 18.60 means operationally:
- Typical prediction error magnitude is about 18.6 cycles at evaluation points (one final observed cycle per test engine).

Why NASA score penalizes late predictions more:
- The implemented asymmetric NASA function uses:
  - early error e < 0: exp(-e/13) - 1
  - late error e >= 0: exp(e/10) - 1
- Smaller denominator for late errors (10 vs 13) produces faster penalty growth, reflecting higher safety/cost risk of overestimating remaining life.

## 10. What Would Break
- If StandardScaler is fit on test data:
  - You leak test distribution into preprocessing, inflate apparent generalization, and invalidate model/metric credibility.
- If you average raw sensors instead of PCA:
  - Opposite-direction sensor trends can cancel, producing a weak or misleading health trajectory.
- If RUL is not clipped at 125:
  - The model over-focuses long early-life horizons, reducing sensitivity in the near-failure region where actionable accuracy is most needed.
- If you evaluate on all test cycles instead of last cycle only:
  - You violate the project’s CMAPSS evaluation protocol and create metrics not comparable to standard benchmark reporting used by this codebase.

## 11. Numbers to Know
| Metric | Value | Source/Note |
|---|---:|---|
| PC1 explained variance | 64.3% | 02_health_index_verification |
| HI mean at cycle 1 | 0.7528 | 02_health_index_verification |
| HI mean at last cycle | 0.1812 | 02_health_index_verification |
| HI inversion applied | True | 02_health_index_verification |
| Top loading sensor_11 | 0.309274 | 02_health_index_verification |
| Top loading sensor_12 | -0.304419 | 02_health_index_verification |
| Top loading sensor_4 | 0.301180 | 02_health_index_verification |
| Top loading sensor_7 | -0.298388 | 02_health_index_verification |
| Top loading sensor_15 | 0.287067 | 02_health_index_verification |
| HI velocity mean (late life) | -0.00568 | 03_velocity_variability_verification |
| HI velocity negative share (late life) | 100.0% | 03_velocity_variability_verification |
| HI variability mean (early life) | 0.2045 | 03_velocity_variability_verification |
| HI variability mean (late life) | 0.3340 | 03_velocity_variability_verification |
| HI variability relative increase | 63.3% | Computed from early/late means |
| Silhouette score | 0.4005 | clustering_checklist |
| Cluster count Healthy | 12595 | clustering_checklist |
| Cluster count Degrading | 6105 | clustering_checklist |
| Cluster count Critical | 1931 | clustering_checklist |
| Cluster pct Healthy | 61.05% | Derived from 20631 train rows |
| Cluster pct Degrading | 29.59% | Derived from 20631 train rows |
| Cluster pct Critical | 9.36% | Derived from 20631 train rows |
| Healthy centroid HI / vel / var | 0.722986 / -0.000675 / 0.210129 | clustering_checklist |
| Degrading centroid HI / vel / var | 0.514162 / -0.003546 / 0.272143 | clustering_checklist |
| Critical centroid HI / vel / var | 0.310073 / -0.008325 / 0.426241 | clustering_checklist |
| Risk mean Healthy | 0.4615 | risk_checklist |
| Risk mean Degrading | 0.6605 | risk_checklist |
| Risk mean Critical | 0.8711 | risk_checklist |
| Risk normalization d_min | 0.0453 | risk_checklist |
| Risk normalization d_max | 8.7041 | risk_checklist |
| Mean Spearman rho (HI vs cycle) | -0.9250 | validation_checklist |
| Monotonic engines (|rho| >= 0.7) | 100.0% | validation_checklist |
| Valid progression | 100.0% | validation_checklist |
| Pearson r (risk vs RUL) | -0.7683 | validation_checklist |
| Train RUL mean (all cycles) | 107.81 | 01_data_exploration comparison table |
| Test RUL mean (cutoff) | 75.52 | 01_data_exploration comparison table |
| Test RUL mean (rounded) | 76 | Rounded from 75.52 |
| RMSE linear_regression | 19.49 | rul_verification_checklist |
| NASA linear_regression | 608.2 | rul_verification_checklist |
| RMSE random_forest | 19.39 | rul_verification_checklist |
| NASA random_forest | 832.0 | rul_verification_checklist |
| RMSE gradient_boosting | 18.60 | rul_verification_checklist |
| NASA gradient_boosting | 698.0 | rul_verification_checklist |
| Best model | gradient_boosting | rul_verification_checklist |
| Feature importance risk_score | 0.681477 | rul_verification_checklist |
| Feature importance health_index | 0.203469 | rul_verification_checklist |
| Feature importance HI_velocity | 0.111475 | rul_verification_checklist |
| Feature importance HI_variability | 0.003578 | rul_verification_checklist |
| 04 evaluation RMSE (GB) | 18.60 cycles | 04_rul_evaluation |
| 04 evaluation NASA (GB) | 698.0 | 04_rul_evaluation |
| Late predictions count | 53 engines | 04_rul_evaluation |
| Early predictions count | 47 engines | 04_rul_evaluation |
| k=2 silhouette | Not available | Not evaluated in project outputs |
| k=4 silhouette | Not available | Not evaluated in project outputs |