"""
features/health_index.py

Purpose:
    Construct a PCA - based Health Index (HI) from the preprocessed CMAPPS sensor data ].
    PC1 captures the dominant source of variance in the selected sensor space, which - in a run - to -failure dataset - is often correlated with degradation.

Mathematical Basis :
     Let X ∈ R^(n × p) be the standardised sensor matrix (n cycles, p sensors).
    PCA finds W ∈ R^p (first eigenvector of X^T X) such that z = XW has maximum variance.
    HI = normalise(z) to [0,1], then invert if HI decreases with health (i.e. increases with cycle).

Input : PreProcessed DataFrame with standardised sensor columns + unit + cycle columns.
Output : DataFrame with an added 'health_index' column in [0, 1]. PCAHealthIndex object (fitted) for reuse on test data.

Assumptions:
    - Sensors are already standardised (mean≈0, std≈1) by preprocesss.py.
    - selected_sensors list comes from config - never hardcoded here.
    - PCA is fitted only on training data to avoid data leakage, then applied to test data using the same PCAHealthIndex object.

Failure conditions:
    - If PC1 explains < 20% of variance, emit a warning (degradation signal may be weak).
    - If majority of engines show non - monotonic HI trends, emit a warning (HI may not correlate well with health).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HealthIndexArtifacts:
    """

    Container for all ouputs and metadata from PCA Health Index construction.

    Carrying everything in a single structured object avoids returning bare tuples and makes the pipeline state inspectable and serializable for later analysis.
    """

    explained_variance_ratio: float  # Fraction of variance explained by PC1
    sensor_loadings: pd.Series  # PC1 loadings for each sensor
    loadings: np.ndarray  # PC1 loadings, shape (n_sensors,)
    invert: bool  # Whether HI was inverted after normalisation
    hi_min: float  # Min of raw PC1 scores (for normalisation)
    hi_max: float  # Max of raw PC1 scores (for normalisation)
    pca: PCA  # Fitted sklearn PCA object
    monotonicity_score: dict[str, float] = field(
        default_factory=dict
    )  # Spearman correlation per engine for monotonicity check


def compute_sensor_contributions(
    engine_df: pd.DataFrame,
    hi_artifacts: "HealthIndexArtifacts",
    sensor_cols: list,
) -> pd.DataFrame:
    """
    Compute per-cycle sensor contributions to the Health Index for one engine.

    Purpose:
        The health index is a linear combination of standardised sensor values
        weighted by PC1 loadings. Decomposing this combination shows which
        sensors are responsible for the current health state - enabling fault
        localisation beyond the aggregate HI value.

    Mathematical definition:
        For each cycle:
            raw_contribution_i = scaled_sensor_i * pc1_loading_i
        The signed contribution shows whether sensor i is pushing HI toward
        degradation (positive) or health (negative), weighted by its loading.

        Absolute contribution normalised to percentage:
            pct_i = |raw_i| / sum(|raw_j|) * 100

    Input:
        engine_df    - DataFrame for ONE engine, must contain scaled sensor columns
        hi_artifacts - HealthIndexArtifacts with pca, loadings, scaler
        sensor_cols  - list of sensor column names (must match artifacts)

    Output:
        DataFrame with columns:
            cycle, {sensor}_contribution (signed) for each sensor,
            top_sensor (name of dominant sensor per cycle),
            top_contribution_pct (percentage contribution of top sensor)

    Assumptions:
        - engine_df contains scaled sensor values (StandardScaler already applied)
        - sensor_cols matches the order used during PCA fitting
        - Single engine only - group by unit before calling

    Failure conditions:
        - Missing sensor columns -> KeyError
        - Mismatch in sensor_cols vs loadings shape -> ValueError
    """
    missing = [c for c in sensor_cols if c not in engine_df.columns]
    if missing:
        raise KeyError(f"Sensor columns missing from engine_df: {missing}")

    loadings = hi_artifacts.loadings

    if len(loadings) != len(sensor_cols):
        raise ValueError(
            f"Loadings length {len(loadings)} does not match "
            f"sensor_cols length {len(sensor_cols)}"
        )

    sensor_values = engine_df[sensor_cols].values
    contributions = sensor_values * loadings

    contrib_df = pd.DataFrame(
        contributions,
        columns=[f"{s}_contribution" for s in sensor_cols],
        index=engine_df.index,
    )
    contrib_df.insert(0, "cycle", engine_df["cycle"].values)

    abs_contributions = np.abs(contributions)
    top_sensor_idx = abs_contributions.argmax(axis=1)
    contrib_df["top_sensor"] = [sensor_cols[i] for i in top_sensor_idx]

    row_totals = abs_contributions.sum(axis=1)
    contrib_df["top_contribution_pct"] = np.round(
        abs_contributions[np.arange(len(top_sensor_idx)), top_sensor_idx]
        / np.where(row_totals > 0, row_totals, 1)
        * 100,
        1,
    )

    return contrib_df


class PCAHealthIndex:
    """
    Constructs and applies a PCA-based Health Index (HI).

    Purpose:
        Fit PCA on training data and transform both training and test data.

    Usage pattern: (mirroring the scaler in preprocess.py)
        hi = PCAHealthIndex(config)
        train_df = hi.fit_transform(train_df).   #Fits PCA + normalisation params
        test_df = hi.transform(test_df)  #Applies same PCA + normalisation to test data, no refitting.
    """

    def __init__(self, config: dict) -> None:
        """
        Input: config dict loaded from config/config.yaml
        Output: initialised (unfitted) PCAHealthIndex
        """
        # Read all parameters from config - never harcode here
        if "health_index" not in config:
            raise KeyError("Missing required config key: 'health_index'")
        if "selected_sensors" not in config:
            raise KeyError("Missing required config key: 'selected_sensors'")

        hi_cfg = config["health_index"]
        self.n_components: int = hi_cfg["n_components"]  # always 1 for HI
        self.normalize: bool = hi_cfg["normalize"]  # should always be True
        self.invert: bool = hi_cfg["invert"]  # default True; validated below
        self.selected_sensors: list[str] = config["selected_sensors"]
        self.random_state: int = config.get(
            "random_state", 42
        )  # for PCA reproducibility

        # Placeholders - populated only aftet fit()
        self._pca: Optional[PCA] = None
        self._hi_min: Optional[float] = None
        self._hi_max: Optional[float] = None
        self._artifacts: Optional[HealthIndexArtifacts] = None

        # ----------------------------------------------------------
        # Private Helpers
        # ----------------------------------------------------------

    def _extract_sensor_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the sensor columns as a numpy matrix for PCA.

        Purpose:  Isolate sensor values; ignore unit, cycle, RUL, HI columns.
        Input:    DataFrame (n_rows × many columns)
        Output:   numpy array (n_rows × len(selected_sensors))
        Failure:  Raises KeyError if any selected sensor is absent.
        """
        missing = [s for s in self.selected_sensors if s not in df.columns]
        if missing:
            raise KeyError(
                f"Selected sensors missing from DataFrame: {missing}. "
                "Ensure preprocess.py ran successfully before calling health_index.py."
            )
        return df[self.selected_sensors].values

    def _determine_inversion(self, scores: np.ndarray, df: pd.DataFrame) -> bool:
        """
        Dtermine whether PC1 scores should be inverted to align with health degradation.

        Purpose: After normalisation, HI should be HIGH at early life and LOW near failure. We check whether the mean PC1 score at the final cycle across all engines. If not, invert.

        Input:  scores - raw PC1 scores (n_rows,)
                df     - DataFrame with 'unit' and 'cycle'
                columns
        Output: bool - True means we should invert (1 - normalised_score)

        Assumption: 'cycle' column counts from 1; last cycle = max cycle per engine.
        """
        temp = df[["unit", "cycle"]].copy()
        temp["score"] = scores

        # Mean score at first observed cycle of each engine, then average across engines
        start_cycles = temp.groupby("unit")["cycle"].min().reset_index()
        start_cycles.columns = ["unit", "start_cycle"]
        temp_start = temp.merge(start_cycles, on="unit")
        mean_at_start = temp_start[temp_start["cycle"] == temp_start["start_cycle"]][
            "score"
        ].mean()

        # Mean score at final cycle of each engine, then average across engines
        last_cycles = temp.groupby("unit")["cycle"].max().reset_index()
        last_cycles.columns = ["unit", "last_cycle"]
        temp2 = temp.merge(last_cycles, on="unit")
        mean_at_end = temp2[temp2["cycle"] == temp2["last_cycle"]]["score"].mean()

        # If start score > end score: PC1 is alread "high = healthy", no inversion needed.
        # If start score < end score: PC1 is "high = degraded", we should invert.
        should_invert = mean_at_start < mean_at_end
        return should_invert

    def _normalise(self, scores: np.ndarray) -> np.ndarray:
        """
        Min-max normalisation to [0, 1] using fitted parameters.

           Purpose:  Make HI dimensionless and bounded regardless of PCA scaling.
           Input:    raw PC1 scores (n_rows,)
           Output:   normalised scores in [0, 1]
           Failure:  If hi_max == hi_min (constant scores), raises ValueError.

           Mathematical definition:
               HI_norm = (z - z_min) / (z_max - z_min)
        """
        if self._hi_max is None or self._hi_min is None:
            raise RuntimeError("PCAHealthIndex must be fitted before normalisation.")
        denom = self._hi_max - self._hi_min
        if denom == 0:
            raise ValueError(
                "PC1 scores are constant — cannot normalise. "
                "Check that sensor columns have non-zero variance after standardisation."
            )
        return (scores - self._hi_min) / denom

    def _validate_monotonicity(
        self, df: pd.DataFrame, hi_col: str = "health_index"
    ) -> dict[str, float]:
        """
        Validate that HI trends downward over each engine's lifecycle.

        Purpose:
            A well-constructed health index should show monotonic decline per engine.
            We use Spearman rank correlation (ρ) between cycle number and HI.
            Expected: ρ << 0 (strong negative).

        Input:  DataFrame with 'unit', 'cycle', and hi_col columns
        Output: dict mapping unit_id → Spearman ρ

        Emits a warning if more than 20% of engines have |ρ| < 0.7.
        """
        scores: dict[str, float] = {}
        for unit_id, group in df.groupby("unit"):
            result = spearmanr(group["cycle"].values, group[hi_col].values)
            rho_value = float(np.asarray(result[0]).item())
            if np.isnan(rho_value):
                rho_value = 0.0
            scores[str(unit_id)] = round(rho_value, 4)

        weak = [uid for uid, rho in scores.items() if rho > -0.7]
        if len(weak) / len(scores) > 0.2:
            warnings.warn(
                f"Monotonicity warning: {len(weak)}/{len(scores)} engines have "
                f"Spearman ρ > -0.7. HI may not reliably track degradation. "
                f"Affected units: {weak[:10]}{'...' if len(weak) > 10 else ''}",
                UserWarning,
            )
        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA on training data and return DataFrame with 'health_index' column.

        Purpose:
            1. Extract sensor matrix from training DataFrame.
            2. Fit PCA (n_components=1) — finds direction of max variance.
            3. Project all training rows onto PC1.
            4. Determine inversion direction from data (override config default if needed).
            5. Fit normalisation parameters (min, max) on training PC1 scores.
            6. Normalise and optionally invert to produce HI ∈ [0,1].
            7. Validate monotonicity across engines.
            8. Store all fitting parameters for later use in transform().

        Input:  Preprocessed training DataFrame
                Shape: (20631, n_cols) for FD001
        Output: Same DataFrame with added 'health_index' column (float, [0,1])

        Failure conditions:
            - Missing sensor columns → KeyError
            - Constant PC1 scores   → ValueError
            - PC1 variance < 20%    → UserWarning (not an error)
        """
        X = self._extract_sensor_matrix(df)

        # Fit PCA — random_state ensures reproducibility of sign conventions
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        scores_raw = self._pca.fit_transform(X).ravel()  # shape: (n_rows,)

        # Warn if degradation signal is weak
        ev = self._pca.explained_variance_ratio_[0]
        if ev < 0.20:
            warnings.warn(
                f"PC1 explains only {ev:.1%} of variance. "
                "Degradation signal may be weak relative to noise.",
                UserWarning,
            )

        # Fit normalisation bounds on raw training scores
        self._hi_min = float(scores_raw.min())
        self._hi_max = float(scores_raw.max())

        # Determine inversion direction from the data itself
        data_driven_invert = self._determine_inversion(scores_raw, df)
        # Use data-driven result; config value is a hint, not a mandate
        self.invert = data_driven_invert

        # Normalise to [0, 1]
        hi_normalised = self._normalise(scores_raw)

        # Invert if PC1 increases with degradation
        if self.invert:
            hi_normalised = 1.0 - hi_normalised

        # Attach to DataFrame
        result = df.copy()
        result["health_index"] = hi_normalised

        # Validate monotonic degradation trend
        mono_scores = self._validate_monotonicity(result)

        # Assemble artifacts for inspection and logging
        loadings = pd.Series(
            self._pca.components_[0],
            index=self.selected_sensors,
            name="PC1_loading",
        ).sort_values(key=abs, ascending=False)

        self._artifacts = HealthIndexArtifacts(
            explained_variance_ratio=float(ev),
            sensor_loadings=loadings,
            loadings=self._pca.components_[0],
            invert=self.invert,
            hi_min=self._hi_min,
            hi_max=self._hi_max,
            pca=self._pca,
            monotonicity_score=mono_scores,
        )

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted PCA and normalisation to test (or validation) data.

        Purpose:
            Uses parameters from fit_transform() — no re-fitting.
            Ensures train/test comparability of HI values.

        Input:  Preprocessed test DataFrame
                Shape: (13096, n_cols) for FD001
        Output: Same DataFrame with added 'health_index' column

        Failure: Raises RuntimeError if called before fit_transform().
        """
        if self._pca is None:
            raise RuntimeError(
                "PCAHealthIndex.transform() called before fit_transform(). "
                "Always fit on training data first."
            )

        X = self._extract_sensor_matrix(df)
        scores_raw = self._pca.transform(X).ravel()

        # Apply training-set normalisation bounds (no re-fitting)
        hi_normalised = self._normalise(scores_raw)

        if self.invert:
            hi_normalised = 1.0 - hi_normalised

        # Clip to [0, 1] — test set may have slightly out-of-range values
        # due to different degradation trajectory endpoints
        hi_normalised = np.clip(hi_normalised, 0.0, 1.0)

        result = df.copy()
        result["health_index"] = hi_normalised
        return result

    def get_artifacts(self) -> HealthIndexArtifacts:
        """
        Return fitted artifacts for logging, visualisation, and inspection.

        Failure: Raises RuntimeError if called before fit_transform().
        """
        if self._artifacts is None:
            raise RuntimeError("No artifacts available — call fit_transform() first.")
        return self._artifacts


# ------------------------------------------------------------------
# Module-level convenience function (mirrors preprocess.py's public API)
# ------------------------------------------------------------------


def build_health_index(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, HealthIndexArtifacts]:
    """
    Top-level entry point for the health index pipeline.

    Purpose:
        Wraps PCAHealthIndex fit/transform into a single call,
        matching the interface pattern of preprocess_train / preprocess_test.

    Input:
        train_df — output of preprocess_train()  (shape: 20631 × n_cols for FD001)
        test_df  — output of preprocess_test()   (shape: 13096 × n_cols for FD001)
        config   — loaded config dict

    Output:
        train_with_hi  — training DataFrame with 'health_index' column
        test_with_hi   — test DataFrame with 'health_index' column
        artifacts      — HealthIndexArtifacts for logging and downstream use

    Assumptions:
        Sensors are already standardised. PCA is sensitive to scale.
    """
    hi = PCAHealthIndex(config)
    train_with_hi = hi.fit_transform(train_df)
    test_with_hi = hi.transform(test_df)
    artifacts = hi.get_artifacts()
    return train_with_hi, test_with_hi, artifacts
