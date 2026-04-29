"""Regression backends with robustness helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings


@dataclass(frozen=True)
class FitResult:
    model: str  # 'nb' or 'ols'
    method: str  # e.g. 'nb', 'nb_w99', 'nb_w98', 'ols_log1p'
    params: Mapping[str, float]
    pvalues: Mapping[str, float]


def _as_numeric(y: pd.Series) -> np.ndarray:
    arr = np.array(y, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def winsorize(y: np.ndarray, p: float = 0.99) -> np.ndarray:
    """Clip to [min, percentile(p)]."""
    y2 = y.copy()
    finite = np.isfinite(y2)
    if not finite.any():
        return y2
    hi = np.nanpercentile(y2[finite], p * 100)
    lo = np.nanmin(y2[finite])
    y2[finite] = np.clip(y2[finite], lo, hi)
    return y2


class RegressionEngine:
    def __init__(
        self,
        *,
        max_retries: int = 2,
        scale_covariates: bool = True,
        covariate_cols: Tuple[str, ...] = ("lp", "sl"),
        drop_zero_variance: bool = True,
    ):
        self.max_retries = max_retries
        self.scale_covariates = scale_covariates
        self.covariate_cols = covariate_cols
        self.drop_zero_variance = drop_zero_variance

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        mode: str = "nb",
    ) -> FitResult:
        """Fit a model. Tries to keep going with safe fallbacks.

        mode:
          - 'nb' (NegativeBinomial GLM)
          - 'ols' (OLS on y directly — expects log-space residuals from eubar intensities)

        If NB fails with the common weights/estimation error, we retry with
        winsorized y and tiny jitter; if it still fails, we fall back to OLS.
        """
        if mode not in {"nb", "ols"}:
            raise ValueError(f"Unknown mode: {mode}")

        # basic sanitation
        y_arr = _as_numeric(y)
        if mode == "nb" and np.nanmin(y_arr) < 0:
            # NB expects counts-like nonnegative response; clip negatives.
            y_arr = np.clip(y_arr, 0, None)

        X_work = X.copy()

        # Drop any strictly constant columns (common when covariates are missing/constant).
        if self.drop_zero_variance:
            keep_cols = []
            for c in X_work.columns:
                v = X_work[c].astype(float)
                if np.nanstd(v.values) == 0:
                    continue
                keep_cols.append(c)
            X_work = X_work[keep_cols].copy()

        # Scale continuous covariates to improve numeric conditioning.
        if self.scale_covariates:
            for c in self.covariate_cols:
                if c not in X_work.columns:
                    continue
                v = X_work[c].astype(float).values
                mu = np.nanmean(v)
                sd = np.nanstd(v)
                if not np.isfinite(mu) or not np.isfinite(sd) or sd == 0:
                    continue
                X_work[c] = (X_work[c].astype(float) - mu) / sd

        Xc = sm.add_constant(X_work, has_constant="add")

        def _fit_nb(y_vec: np.ndarray, method: str) -> FitResult:
            m = sm.GLM(y_vec, Xc, family=sm.families.NegativeBinomial(alpha=1.0))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                res = m.fit()
            return FitResult(
                model="nb",
                method=method,
                params=dict(res.params),
                pvalues=dict(res.pvalues),
            )

        def _fit_ols(y_vec: np.ndarray, method: str = "ols") -> FitResult:
            m = sm.OLS(y_vec, Xc)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                res = m.fit()
            return FitResult(
                model="ols",
                method=method,
                params=dict(res.params),
                pvalues=dict(res.pvalues),
            )

        if mode == "ols":
            return _fit_ols(y_arr, method="ols")

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                y_try = y_arr
                method = (
                    "nb" if attempt == 0 else ("nb_w99" if attempt == 1 else "nb_w98")
                )
                if attempt == 1:
                    y_try = winsorize(y_try, 0.99)
                elif attempt >= 2:
                    y_try = winsorize(y_try, 0.98)
                # jitter zeros a tiny bit (helps IRLS when there are many zeros)
                if attempt > 0:
                    y_try = y_try + 1e-8
                return _fit_nb(y_try, method=method)
            except Exception as e:
                last_err = e

        # final fallback: OLS
        try:
            return _fit_ols(y_arr, method="ols")
        except Exception:
            # if even OLS fails, re-raise original error
            raise last_err if last_err is not None else RuntimeError("Model fit failed")