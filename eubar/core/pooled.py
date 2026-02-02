"""eubar.core.pooled

Pooled (across-window) SNV regression helpers.

This module is the "core" implementation behind :mod:`eubar.snv_pooled`.

Design goals
------------
* Keep pooled behavior identical to the script version.
* Remain robust across slightly different helper signatures in older branches.

Model overview
--------------
AFF pooled stacks the 8 overlapping SNV windows into one long table and fits:

  log1p(y) ~ allele + window + (optional covariates)

Where:
* allele uses treatment coding with the SNV REF allele as the baseline.
* window is a fixed-effect factor for the (motif_pos, snv_index) key.
* cluster-robust SEs are computed by probe/region because probes can repeat.

RAND pooled fits:

  log1p(y) ~ allele_A + allele_C + allele_G + allele_T + window + covariates

Background rows are all-zero for allele_*; the intercept is the BG baseline.
"""

from __future__ import annotations

import inspect
from typing import Dict, Iterable, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm


if TYPE_CHECKING:  # pragma: no cover
    from eubar.core.sequence import SnvWindow


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------


def call_with_accepted_kwargs(fn, *args, **kwargs):
    """Call ``fn`` with only the kwargs it accepts.

    This keeps pooled tools compatible across slightly different helper
    signatures (e.g., include_covariates / fold_half / min_n changes).
    """
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        filt = {k: v for k, v in kwargs.items() if k in accepted}
        return fn(*args, **filt)
    except Exception:
        # If signature introspection fails, fall back.
        return fn(*args, **kwargs)


def parse_snv_str(snv_str: str) -> Tuple[str, int, str, List[str]]:
    """Parse 'chr:pos:REF>ALT' (ALT can be comma-separated)."""
    chrom, pos, change = snv_str.strip().split(":")
    ref, alt = change.split(">")
    alts = [a.strip().upper() for a in alt.split(",") if a.strip()]
    return chrom, int(pos), ref.strip().upper(), alts


# ---------------------------------------------------------------------------
# Pooled table builders
# ---------------------------------------------------------------------------


def label_alleles_from_payload_df(df: pd.DataFrame, ref_allele: str) -> pd.Series:
    """Map dummy-coded allele columns to a single allele label.

    Payload dataframes typically have one-hot A/C/G/T columns. Rows with
    all-zeros correspond to the baseline allele (the window's ref allele).
    """
    cols = [c for c in ["A", "C", "G", "T"] if c in df.columns]
    if not cols:
        raise ValueError("No allele dummy columns found in payload dataframe.")

    conditions = [(df[c].astype(float) == 1.0) for c in cols]
    allele = np.select(conditions, cols, default=ref_allele)
    return pd.Series(allele, index=df.index, dtype="string")


def build_pooled_table(
    payloads: Dict[Tuple[int, int], object],
    *,
    kind: str,
) -> pd.DataFrame:
    """Stack the 8 window payloads (AFF) into one long dataframe."""
    frames: list[pd.DataFrame] = []
    for key, payload in payloads.items():
        df = payload.as_dataframe().copy()

        ref = payload.meta.get("ref_allele")
        if ref is None:
            raise ValueError(f"Payload for window {key} missing meta['ref_allele']")

        df["allele"] = label_alleles_from_payload_df(df, ref)

        # Drop any allele dummy columns if present; pooled AFF uses categorical label.
        drop_cols = [
            c
            for c in ["A", "C", "G", "T", "allele_A", "allele_C", "allele_G", "allele_T"]
            if c in df.columns
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        df["window"] = str(key)
        df["probe"] = df.index.astype(str)
        df["logy"] = np.log1p(df["y"].astype(float))
        df["type"] = kind
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=False)
    out["window"] = out["window"].astype("category")
    out["allele"] = out["allele"].astype("category")
    return out


def build_pooled_table_rand(
    payloads: Dict[int, object],
    *,
    snv: "SnvWindow",
    kind: str,
) -> pd.DataFrame:
    """Stack RAND payloads into one long dataframe.

    RAND payloads are keyed by motif_pos (j). We map motif_pos -> snv_index
    using the SNV's overlapping windows, then create a pooled window factor
    consistent with AFF: (motif_pos, snv_index).
    """

    mp_to_si: Dict[int, int] = {}
    for motif_pos, snv_index in snv.iter_overlapping_windows():
        mp_to_si[int(motif_pos)] = int(snv_index)

    frames: list[pd.DataFrame] = []
    for motif_pos, payload in payloads.items():
        df = payload.as_dataframe().copy()

        # Ensure allele dummy columns exist.
        for c in ["A", "C", "G", "T"]:
            if c not in df.columns:
                df[c] = 0.0

        # Standardize dummy column names (also avoids shadowing any future formula use).
        rename_map = {"A": "allele_A", "C": "allele_C", "G": "allele_G", "T": "allele_T"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        snv_index = mp_to_si.get(int(motif_pos), -1)
        df["window"] = str((int(motif_pos), int(snv_index)))
        df["probe"] = df.index.astype(str)
        df["logy"] = np.log1p(df["y"].astype(float))
        df["type"] = kind
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=False) if frames else pd.DataFrame()
    if not out.empty:
        out["window"] = out["window"].astype("category")
    return out


# ---------------------------------------------------------------------------
# Design matrix builders 
# ---------------------------------------------------------------------------


_ALLELES = ["A", "C", "G", "T"]


def _window_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode window and drop one level for identifiability."""
    w = pd.get_dummies(df["window"].astype("category"), prefix="window", dtype=float)
    w = w.reindex(sorted(w.columns), axis=1)
    if w.shape[1] > 0:
        w = w.iloc[:, 1:]
    return w


def _allele_dummies_treatment(df: pd.DataFrame, ref_allele: str) -> pd.DataFrame:
    """One-hot encode allele and drop ref allele (treatment coding)."""
    a = pd.get_dummies(df["allele"].astype("category"), prefix="allele", dtype=float)

    for al in _ALLELES:
        col = f"allele_{al}"
        if col not in a.columns:
            a[col] = 0.0

    a = a[[f"allele_{al}" for al in _ALLELES]]
    ref_col = f"allele_{ref_allele}"
    if ref_col in a.columns:
        a = a.drop(columns=[ref_col])
    return a


def _covariate_block(df: pd.DataFrame, include_covariates: bool) -> pd.DataFrame:
    if not include_covariates:
        return pd.DataFrame(index=df.index)
    cols = [c for c in ["lp", "sl"] if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].astype(float)


def _fit_cluster_robust(res, groups):
    """Return result with cluster-robust covariance (if available)."""
    try:
        return res.get_robustcov_results(cov_type="cluster", groups=groups)
    except Exception:
        return res


def fit_pooled_model(
    df: pd.DataFrame,
    *,
    ref_allele: str,
    include_covariates: bool = True,
    mode: str = "ols",
):
    """Fit pooled AFF model (allele treatment coding + window fixed effects)."""

    if mode == "ols":
        y = df["logy"].astype(float).to_numpy()
    elif mode == "nb":
        y = np.clip(np.round(df["y"].astype(float).to_numpy()), 0, None)
    else:
        raise ValueError("mode must be 'ols' or 'nb'")

    X_parts = [
        _allele_dummies_treatment(df, ref_allele),
        _window_dummies(df),
        _covariate_block(df, include_covariates),
    ]
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")

    if mode == "ols":
        res0 = sm.OLS(y, X).fit()
        res = _fit_cluster_robust(res0, groups=df["probe"])
        res.model_kind = "ols_log1p"
        return res

    res0 = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    res = _fit_cluster_robust(res0, groups=df["probe"])
    res.model_kind = "nb"
    return res


def fit_pooled_model_rand(
    df: pd.DataFrame,
    *,
    include_covariates: bool = True,
    mode: str = "ols",
):
    """Fit pooled RAND model (A/C/G/T dummies + window fixed effects).

    Background rows have all allele_* = 0, so intercept is the BG baseline.
    """

    if df.empty:
        raise ValueError("RAND pooled table is empty.")

    for c in ["allele_A", "allele_C", "allele_G", "allele_T"]:
        if c not in df.columns:
            df[c] = 0.0

    if mode == "ols":
        y = df["logy"].astype(float).to_numpy()
    elif mode == "nb":
        y = np.clip(np.round(df["y"].astype(float).to_numpy()), 0, None)
    else:
        raise ValueError("mode must be 'ols' or 'nb'")

    X_parts = [
        df[["allele_A", "allele_C", "allele_G", "allele_T"]].astype(float),
        _window_dummies(df),
        _covariate_block(df, include_covariates),
    ]
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")

    if mode == "ols":
        res0 = sm.OLS(y, X).fit()
        res = _fit_cluster_robust(res0, groups=df["probe"])
        res.model_kind = "ols_log1p_rand"
        return res

    res0 = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
    res = _fit_cluster_robust(res0, groups=df["probe"])
    res.model_kind = "nb_rand"
    return res


# ---------------------------------------------------------------------------
# Effect extraction + formatting
# ---------------------------------------------------------------------------


def _named_params_pvals(res):
    """Return (params_dict, pvals_dict) keyed by exog name.

    statsmodels sometimes returns params/pvalues as numpy arrays (especially
    after robust covariance). We key them using ``res.model.exog_names`` to keep
    extraction stable.
    """

    names: list[str] = []
    try:
        names = list(getattr(res.model, "exog_names", None) or [])
    except Exception:
        names = []

    params = getattr(res, "params", None)
    pvals = getattr(res, "pvalues", None)

    if hasattr(params, "to_dict") and getattr(params, "index", None) is not None:
        p_dict = params.to_dict()
    else:
        p_arr = np.asarray(params, dtype=float) if params is not None else np.asarray([])
        p_dict = {n: float(v) for n, v in zip(names, p_arr)} if names else {}

    if hasattr(pvals, "to_dict") and getattr(pvals, "index", None) is not None:
        pv_dict = pvals.to_dict()
    else:
        pv_arr = np.asarray(pvals, dtype=float) if pvals is not None else np.asarray([])
        pv_dict = {n: float(v) for n, v in zip(names, pv_arr)} if names else {}

    return p_dict, pv_dict


def effect_table_from_result(
    res,
    *,
    ref_allele: str,
    alleles_to_report: Iterable[str],
    include_fold_change: bool = False,
) -> pd.DataFrame:
    """Extract per-allele effects vs REF from pooled AFF result."""

    params, pvals = _named_params_pvals(res)

    rows = []
    for a in alleles_to_report:
        if a == ref_allele:
            rows.append(
                {
                    "allele": a,
                    "effect": np.nan,
                    **({"fold_change_(y+1)": np.nan} if include_fold_change else {}),
                    "pval": np.nan,
                }
            )
            continue

        term = f"allele_{a}"
        if term not in params:
            rows.append(
                {
                    "allele": a,
                    "effect": np.nan,
                    **({"fold_change_(y+1)": np.nan} if include_fold_change else {}),
                    "pval": np.nan,
                }
            )
            continue

        eff = float(params[term])
        pv = float(pvals.get(term, float("nan")))
        fc = float(np.exp(eff)) if include_fold_change else None

        rows.append(
            {
                "allele": a,
                "effect": eff,
                **({"fold_change_(y+1)": float(fc)} if include_fold_change else {}),
                "pval": pv,
            }
        )

    return pd.DataFrame(rows)


def effect_table_from_result_rand(
    res,
    *,
    alleles_to_report: Iterable[str],
    include_fold_change: bool = False,
) -> pd.DataFrame:
    """Extract per-allele effects vs BG from pooled RAND result."""

    params, pvals = _named_params_pvals(res)

    param_name = {"A": "allele_A", "C": "allele_C", "G": "allele_G", "T": "allele_T"}

    rows = []
    for a in alleles_to_report:
        p = param_name.get(a, a)
        if p not in params:
            rows.append(
                {
                    "allele": a,
                    "effect": np.nan,
                    **({"fold_change_(y+1)": np.nan} if include_fold_change else {}),
                    "pval": np.nan,
                }
            )
            continue

        eff = float(params[p])
        pv = float(pvals.get(p, float("nan")))
        fc = float(np.exp(eff)) if include_fold_change else None

        rows.append(
            {
                "allele": a,
                "effect": eff,
                **({"fold_change_(y+1)": float(fc)} if include_fold_change else {}),
                "pval": pv,
            }
        )

    return pd.DataFrame(rows)


def fmt_pval(p):
    """Render p-values like the script output (scientific notation, NaNs)."""
    if p is None:
        return "nan"
    try:
        p = float(p)
    except Exception:
        return str(p)

    if np.isnan(p):
        return "nan"

    # statsmodels can underflow extremely small p-values to 0.0
    if p == 0.0:
        return f"<{np.finfo(float).tiny:.1e}"

    return f"{p:.3e}"
