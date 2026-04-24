"""
Module: data/regime.py

Purpose    : Regime-aware sensor normalisation for multi-condition CMAPSS datasets.
Assumptions:
  - FD001/FD003 have 1 operating condition (n_regimes=1); RegimeScaler degenerates
    to a single global StandardScaler.
  - FD002/FD004 have 6 operating conditions; KMeans clusters on op_setting columns
    before fitting per-regime StandardScalers.
  - Scaler fitted on training data only — no leakage.
  - All parameters sourced from config["regimes"]; nothing hardcoded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings


class RegimeScaler:
    """
    Duck-type replacement for StandardScaler that dispatches per operating-condition regime.

    For n_regimes=1 (FD001/FD003), behaves identically to a global StandardScaler.
    For n_regimes>1 (FD002/FD004), fits one StandardScaler per KMeans regime cluster
    and dispatches during transform_df() using the same KMeans model.

    Exposes n_features_in_, mean_, and scale_ for sklearn/pipeline compatibility.
    Use transform_df() for pipeline use; transform() is reserved for n_regimes=1
    and satisfies the duck-type contract expected by apply_scaler().
    """

    def __init__(
        self,
        n_regimes: int,
        setting_cols: list[str],
        random_state: int = 42,
        silhouette_min_threshold: float = 0.30,
        enforce_silhouette_gate: bool = True,
    ) -> None:
        """
        Purpose    : Initialise with regime count and operating condition column names.
        Input      : n_regimes    — number of flight regimes (1 = single-condition dataset).
                     setting_cols — operational setting column names used for KMeans clustering.
                     random_state — KMeans random seed for reproducibility.
        Output     : Unfitted RegimeScaler instance.
        Assumptions: n_regimes matches the dataset's actual operating condition count.
        Failure    : ValueError if n_regimes < 1.
        """
        if n_regimes < 1:
            raise ValueError(f"n_regimes must be >= 1, got {n_regimes}")
        self._n_regimes = n_regimes
        self._setting_cols = setting_cols
        self._random_state = random_state
        self._kmeans: KMeans | None = None
        self._scalers: dict[int, StandardScaler] = {}
        self._sensor_cols: list[str] | None = None
        self._silhouette_min_threshold = float(silhouette_min_threshold)
        self._enforce_silhouette_gate = bool(enforce_silhouette_gate)

    # ------------------------------------------------------------------
    # Sklearn-compatible properties
    # ------------------------------------------------------------------

    @property
    def n_features_in_(self) -> int:
        """
        Purpose    : Return number of sensor features seen during fit.
        Output     : int
        Failure    : RuntimeError if called before fit().
        """
        if not self._scalers:
            raise RuntimeError("RegimeScaler has not been fitted yet.")
        return next(iter(self._scalers.values())).n_features_in_

    @property
    def mean_(self) -> np.ndarray:
        """
        Purpose    : Fleet-average sensor means across all regimes.
        Output     : ndarray of shape (n_sensors,).
        Failure    : RuntimeError if called before fit().
        """
        if not self._scalers:
            raise RuntimeError("RegimeScaler has not been fitted yet.")
        return np.mean([s.mean_ for s in self._scalers.values()], axis=0)

    @property
    def scale_(self) -> np.ndarray:
        """
        Purpose    : Fleet-average sensor standard deviations across all regimes.
        Output     : ndarray of shape (n_sensors,).
        Failure    : RuntimeError if called before fit().
        """
        if not self._scalers:
            raise RuntimeError("RegimeScaler has not been fitted yet.")
        return np.mean([s.scale_ for s in self._scalers.values()], axis=0)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, sensor_cols: list[str]) -> RegimeScaler:
        """
        Purpose    : Fit regime clusters and per-regime StandardScalers on training data.
        Input      : df          — training DataFrame containing setting_cols and sensor_cols.
                     sensor_cols — sensor feature column names to normalise.
        Output     : self (fitted RegimeScaler).
        Assumptions: df is training data only. Must not be called on test data.
                     For n_regimes>1, df must contain all setting_cols.
        Failure    : KeyError if setting_cols or sensor_cols are absent from df.
                     ValueError if any regime cluster receives zero training samples.
        """
        missing_sensors = [c for c in sensor_cols if c not in df.columns]
        if missing_sensors:
            raise KeyError(f"sensor_cols not found in DataFrame: {missing_sensors}")

        self._sensor_cols = list(sensor_cols)

        if self._n_regimes == 1:
            scaler = StandardScaler()
            scaler.fit(df[sensor_cols])
            self._scalers[0] = scaler
        else:
            missing_settings = [c for c in self._setting_cols if c not in df.columns]
            if missing_settings:
                raise KeyError(
                    f"setting_cols required for regime clustering but missing: {missing_settings}"
                )
            self._kmeans = KMeans(
                n_clusters=self._n_regimes,
                random_state=self._random_state,
                n_init=10,
            )
            regime_labels = self._kmeans.fit_predict(df[self._setting_cols])

            label_counts = np.bincount(regime_labels, minlength=self._n_regimes)
            label_pct = (label_counts / label_counts.sum()) * 100.0
            silhouette = silhouette_score(df[self._setting_cols], regime_labels)
            print(
                "[REGIME] KMeans silhouette: "
                f"{silhouette:.4f} (min required: {self._silhouette_min_threshold:.2f})"
            )
            print(
                "[REGIME] Sample distribution (%): "
                + ", ".join(
                    [
                        f"regime_{idx}={pct:.2f}%"
                        for idx, pct in enumerate(label_pct.tolist())
                    ]
                )
            )

            if silhouette < self._silhouette_min_threshold:
                message = (
                    f"Regime silhouette gate failed: {silhouette:.4f} < "
                    f"{self._silhouette_min_threshold:.2f}."
                )
                if self._enforce_silhouette_gate:
                    raise ValueError(message)
                warnings.warn(message, UserWarning)

            for regime_id in range(self._n_regimes):
                mask = regime_labels == regime_id
                if not mask.any():
                    raise ValueError(
                        f"Regime {regime_id} received zero training samples. "
                        "Check n_regimes against the actual number of operating conditions."
                    )
                scaler = StandardScaler()
                scaler.fit(df.loc[mask, sensor_cols])
                self._scalers[regime_id] = scaler

        return self

    def transform_df(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        """
        Purpose    : Apply per-regime scaling and return a clean output DataFrame.
                     Op_setting columns are used for regime dispatch then dropped from output,
                     preserving the same column contract as apply_scaler() for downstream modules.
        Input      : df          — DataFrame with setting_cols (required for n_regimes>1)
                                   and sensor_cols.
                     sensor_cols — column names to scale.
        Output     : DataFrame with sensor_cols scaled, op_setting columns removed.
                     All other columns (unit, cycle, RUL) are preserved unchanged.
        Assumptions: RegimeScaler is fitted. df contains the same setting_cols used at fit time.
        Failure    : RuntimeError if not fitted.
                     KeyError if required columns are missing.
        """
        if not self._scalers:
            raise RuntimeError("RegimeScaler has not been fitted yet.")

        missing_sensors = [c for c in sensor_cols if c not in df.columns]
        if missing_sensors:
            raise KeyError(f"sensor_cols missing from DataFrame: {missing_sensors}")

        scaled_df = df.copy()
        # Ensure float dtype before writing standardized values back into sensor columns.
        scaled_df[sensor_cols] = scaled_df[sensor_cols].astype(float)

        if self._n_regimes == 1:
            scaled_df[sensor_cols] = self._scalers[0].transform(df[sensor_cols])
        else:
            if self._kmeans is None:
                raise RuntimeError(
                    "KMeans model missing — RegimeScaler not fitted correctly."
                )
            missing_settings = [c for c in self._setting_cols if c not in df.columns]
            if missing_settings:
                raise KeyError(
                    f"setting_cols required for regime dispatch but missing: {missing_settings}"
                )
            regime_labels = self._kmeans.predict(df[self._setting_cols])
            for regime_id, regime_scaler in self._scalers.items():
                mask = regime_labels == regime_id
                if mask.any():
                    scaled_df.loc[mask, sensor_cols] = regime_scaler.transform(
                        df.loc[mask, sensor_cols]
                    )

        setting_cols_in_df = [c for c in scaled_df.columns if c in self._setting_cols]
        scaled_df = scaled_df.drop(columns=setting_cols_in_df)

        return scaled_df

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Purpose    : Sklearn-compatible transform interface for n_regimes=1 only.
                     Satisfies the duck-type contract expected by apply_scaler(),
                     which calls scaler.transform(df[sensor_cols]).
        Input      : X — array-like of shape (n_samples, n_sensors), sensor columns only.
        Output     : Scaled array of shape (n_samples, n_sensors).
        Assumptions: Valid only when n_regimes=1. Caller must use transform_df() for
                     multi-regime datasets where op_setting columns are required for dispatch.
        Failure    : ValueError if called with n_regimes > 1 (no setting columns available).
                     RuntimeError if called before fit().
        """
        if not self._scalers:
            raise RuntimeError("RegimeScaler has not been fitted yet.")
        if self._n_regimes != 1:
            raise ValueError(
                "RegimeScaler.transform() is not available for n_regimes > 1. "
                "Regime dispatch requires op_setting columns. Use transform_df(df, sensor_cols)."
            )
        return self._scalers[0].transform(X)


def fit_regime_scaler(
    df: pd.DataFrame,
    sensor_cols: list[str],
    config: dict,
) -> RegimeScaler:
    """
    Purpose    : Validate config and fit a RegimeScaler on training data.
    Input      : df          — training DataFrame with op_setting and sensor columns.
                 sensor_cols — sensor column names to normalise.
                 config      — loaded YAML config containing a 'regimes' block.
    Output     : Fitted RegimeScaler.
    Assumptions: df is training data only — no leakage.
                 config["regimes"] contains n_regimes, setting_cols, and optionally random_state.
    Failure    : KeyError if required config keys are missing.
                 Propagates KeyError/ValueError from RegimeScaler.fit().
    """
    if "regimes" not in config:
        raise KeyError("Missing config key: regimes")
    regime_cfg = config["regimes"]
    for key in ("n_regimes", "setting_cols"):
        if key not in regime_cfg:
            raise KeyError(f"Missing config key: regimes.{key}")

    n_regimes: int = regime_cfg["n_regimes"]
    setting_cols: list[str] = regime_cfg["setting_cols"]
    random_state: int = regime_cfg.get("random_state", 42)
    silhouette_min_threshold: float = regime_cfg.get("silhouette_min_threshold", 0.30)
    enforce_silhouette_gate: bool = regime_cfg.get("enforce_silhouette_gate", True)

    scaler = RegimeScaler(
        n_regimes=n_regimes,
        setting_cols=setting_cols,
        random_state=random_state,
        silhouette_min_threshold=silhouette_min_threshold,
        enforce_silhouette_gate=enforce_silhouette_gate,
    )
    scaler.fit(df, sensor_cols)
    return scaler
