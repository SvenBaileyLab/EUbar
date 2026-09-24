"""Calibration helpers for EUbar.

The first supported calibration is ``--max-probes``.  It measures how stable
allele-effect estimates are when representative wildcard k-mer families are
subsampled at candidate probe caps.  P-values are reported diagnostically but
never enter the recommendation rule.

This module deliberately reuses :func:`eubar.core.sampling.subsample_probes`,
so calibration exercises the same proportional per-allele probe sampling used
by production EUbar analyses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .design import fold_lp_half
from .matching import wildcard_pos_from_offset
from .masking import MaskHit, _signature_codes
from .patterns import SequenceMask
from .sampling import subsample_probes
from .sequence import _open_fasta

BASES = "ACGT"


def rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt.", "TGCAtgca."))[::-1]


def canonical_wildcard(wildcard: str) -> str:
    rev = rc(wildcard)
    return wildcard if wildcard <= rev else rev


def parse_caps(text: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    if not vals or any(x < 10 for x in vals):
        raise ValueError("caps must be comma-separated integers >= 10")
    return vals


def parse_region(region: str) -> Tuple[str, int, int]:
    chrom, coords = region.rsplit(":", 1)
    start_s, end_s = coords.split("-", 1)
    return chrom, int(start_s), int(end_s)


def family_candidates(
    kmer_index: Mapping[str, Mapping[str, int]],
    k: int,
) -> List[str]:
    """Return canonical one-wildcard families represented by the array."""
    out = set()
    for kmer in kmer_index.keys():
        if len(kmer) != k or any(c not in BASES for c in kmer):
            continue
        for j in range(k):
            out.add(canonical_wildcard(kmer[:j] + "." + kmer[j + 1 :]))
    return sorted(out)


def wildcard_position(wildcard: str) -> int:
    if wildcard.count(".") != 1:
        raise ValueError(f"Expected one '.', got {wildcard!r}")
    return wildcard.index(".")


def build_region_lookup(
    wildcard: str,
    kmer_index: Mapping[str, Mapping[str, int]],
) -> Dict[str, Dict[str, int]]:
    """Build EUbar-style allele -> region -> wildcard-position lookup."""
    j = wildcard_position(wildcard)
    k = len(wildcard)
    out: Dict[str, Dict[str, int]] = {b: {} for b in BASES}

    for base in BASES:
        filled = wildcard.replace(".", base)
        rev = rc(filled)
        match_kmers = [(filled, False)] if rev == filled else [(filled, False), (rev, True)]
        for match_kmer, is_reverse in match_kmers:
            region_map = kmer_index.get(match_kmer)
            if not region_map:
                continue
            dest = out[base]
            for region, offset in region_map.items():
                if region in dest:
                    continue
                dest[region] = wildcard_pos_from_offset(
                    offset=int(offset),
                    j=j,
                    kmer_size=k,
                    is_reverse=is_reverse,
                )
    return {allele: regions for allele, regions in out.items() if regions}


def lookup_to_dataframe(
    region_lookup: Mapping[str, Mapping[str, int]],
    intensities: Mapping[str, float],
    *,
    fold_half: bool = True,
) -> pd.DataFrame:
    """Build the unambiguous probe rows used by EUbar's design stage."""
    region_to_alleles: Dict[str, set] = defaultdict(set)
    region_to_pos: Dict[str, int] = {}
    for allele, regmap in region_lookup.items():
        for region, pos in regmap.items():
            region_to_alleles[region].add(allele)
            region_to_pos.setdefault(region, int(pos))

    rows: List[Dict[str, object]] = []
    for region, alleles in region_to_alleles.items():
        if len(alleles) != 1 or region not in intensities:
            continue
        try:
            _, start, end = parse_region(region)
        except Exception:
            continue
        allele = next(iter(alleles))
        length = max(1, end - start)
        kmer_pos = int(region_to_pos[region])
        lp = float(kmer_pos) / float(length)
        if fold_half:
            lp = fold_lp_half(lp)
        rows.append(
            {
                "region": region,
                "allele": allele,
                "y": float(intensities[region]),
                "lp": float(lp),
                "sl": float(length),
                "kmer_pos": kmer_pos,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Convert accepted unambiguous rows back to a production sampling lookup."""
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in df.itertuples(index=False):
        out[str(row.allele)][str(row.region)] = int(row.kmer_pos)
    return {allele: dict(regions) for allele, regions in out.items()}


def choose_reference(df: pd.DataFrame) -> str:
    counts = df["allele"].value_counts().to_dict()
    present = [a for a in BASES if counts.get(a, 0) > 0]
    return sorted(present, key=lambda a: (-counts.get(a, 0), a))[0]


@dataclass(frozen=True)
class FastDesign:
    X: np.ndarray
    y: np.ndarray
    alleles: np.ndarray
    regions: np.ndarray
    ref: str
    allele_cols: Dict[str, Optional[int]]
    y_sd: float


def make_design(
    df: pd.DataFrame,
    *,
    ref: str,
    include_covariates: bool = True,
) -> FastDesign:
    """Build a reusable numeric OLS design for one wildcard family.

    Continuous covariates are standardized once on the full family.  Because
    this is an affine reparameterization in a model with an intercept, the
    allele-effect estimates are unchanged when rows are subsequently subset.
    """
    n = len(df)
    cols: List[np.ndarray] = [np.ones(n, dtype=float)]
    allele_cols: Dict[str, Optional[int]] = {ref: None}
    allele_vec = df["allele"].to_numpy(dtype=str)

    for allele in BASES:
        if allele == ref or not np.any(allele_vec == allele):
            continue
        allele_cols[allele] = len(cols)
        cols.append((allele_vec == allele).astype(float))

    if include_covariates:
        for cov in ("lp", "sl"):
            values = df[cov].to_numpy(dtype=float)
            sd = float(np.std(values))
            if not np.isfinite(sd) or sd <= 0:
                continue
            cols.append((values - float(np.mean(values))) / sd)

    X = np.column_stack(cols)
    y = df["y"].to_numpy(dtype=float)
    y_sd = float(np.std(y))
    if not np.isfinite(y_sd) or y_sd <= 1e-12:
        y_sd = 1.0

    for allele in BASES:
        allele_cols.setdefault(allele, None)

    return FastDesign(
        X=X,
        y=y,
        alleles=allele_vec,
        regions=df["region"].to_numpy(dtype=str),
        ref=ref,
        allele_cols=allele_cols,
        y_sd=y_sd,
    )


@dataclass(frozen=True)
class OLSFit:
    beta: np.ndarray
    cov: np.ndarray
    df_resid: int


def fast_ols(X: np.ndarray, y: np.ndarray) -> Optional[OLSFit]:
    n, p = X.shape
    if n <= p + 1:
        return None
    try:
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < 2:
            return None
        resid = y - X @ beta
        df_resid = int(n - rank)
        if df_resid <= 0:
            return None
        sigma2 = float(np.dot(resid, resid) / df_resid)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        return OLSFit(beta=beta, cov=cov, df_resid=df_resid)
    except Exception:
        return None


def contrast_vector(design: FastDesign, a: str, b: str) -> np.ndarray:
    vec = np.zeros(design.X.shape[1], dtype=float)
    ia = design.allele_cols.get(a)
    ib = design.allele_cols.get(b)
    if a != design.ref and ia is not None:
        vec[ia] += 1.0
    if b != design.ref and ib is not None:
        vec[ib] -= 1.0
    return vec


def contrast_stats(fit: OLSFit, vec: np.ndarray) -> Tuple[float, float]:
    est = float(vec @ fit.beta)
    var = float(vec @ fit.cov @ vec)
    if not np.isfinite(var) or var <= 0:
        return est, np.nan
    se = math.sqrt(var)
    tval = abs(est / se)
    logp = math.log(2.0) + float(student_t.logsf(tval, fit.df_resid))
    if not np.isfinite(logp):
        neglog10p = 320.0
    else:
        neglog10p = max(0.0, -logp / math.log(10.0))
    return est, neglog10p


def available_contrasts(design: FastDesign) -> List[Tuple[str, str]]:
    present = [a for a in BASES if np.any(design.alleles == a)]
    return [(a, b) for i, a in enumerate(present) for b in present[i + 1 :]]


def quantile(values: Iterable[float], q: float) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.quantile(vals, q)) if len(vals) else np.nan


def median(values: Iterable[float]) -> float:
    return quantile(values, 0.5)


def _sampled_indices(
    family_lookup: Mapping[str, Mapping[str, int]],
    region_to_index: Mapping[str, int],
    *,
    cap: int,
    seed: int,
    wildcard: str,
    repeat: int,
) -> np.ndarray:
    sampled = subsample_probes(
        family_lookup,
        cap,
        seed=seed,
        context=("CALIBRATE_MAX_PROBES", wildcard, cap, repeat),
    )
    selected = {
        region
        for regmap in sampled.values()
        for region in regmap.keys()
        if region in region_to_index
    }
    return np.asarray(sorted(region_to_index[r] for r in selected), dtype=int)


def evaluate_cap(
    design: FastDesign,
    family_lookup: Mapping[str, Mapping[str, int]],
    full_stats: Dict[str, Tuple[float, float]],
    contrasts: Sequence[Tuple[str, str]],
    *,
    wildcard: str,
    cap: int,
    repeats: int,
    seed: int,
) -> Optional[Tuple[Dict[str, object], List[Dict[str, object]]]]:
    """Evaluate one cap using the exact production ``subsample_probes`` path."""
    if len(design.y) < cap:
        return None

    region_to_index = {str(region): i for i, region in enumerate(design.regions)}
    estimates: Dict[str, List[float]] = defaultdict(list)
    pvals: Dict[str, List[float]] = defaultdict(list)
    realized_n: List[int] = []
    successes = 0

    for rep in range(repeats):
        idx = _sampled_indices(
            family_lookup,
            region_to_index,
            cap=cap,
            seed=seed,
            wildcard=wildcard,
            repeat=rep,
        )
        if len(idx) < 3:
            continue
        fit = fast_ols(design.X[idx], design.y[idx])
        if fit is None:
            continue

        any_ok = False
        for a, b in contrasts:
            key = f"{a}-{b}"
            est, nlp = contrast_stats(fit, contrast_vector(design, a, b))
            if np.isfinite(est):
                estimates[key].append(est)
                pvals[key].append(nlp)
                any_ok = True
        if any_ok:
            successes += 1
            realized_n.append(len(idx))

    if successes == 0:
        return None

    sd_norm: List[float] = []
    bias_norm: List[float] = []
    rmse_norm: List[float] = []
    median_neglogp: List[float] = []
    contrast_rows: List[Dict[str, object]] = []

    for a, b in contrasts:
        key = f"{a}-{b}"
        vals = np.asarray(estimates.get(key, []), dtype=float)
        nlps = np.asarray(pvals.get(key, []), dtype=float)
        if len(vals) == 0 or key not in full_stats:
            continue
        target, full_nlp = full_stats[key]
        sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        bias = abs(float(np.mean(vals)) - target)
        rmse = math.sqrt(float(np.mean((vals - target) ** 2)))
        sdn = sd / design.y_sd
        bn = bias / design.y_sd
        rn = rmse / design.y_sd
        sd_norm.append(sdn)
        bias_norm.append(bn)
        rmse_norm.append(rn)
        median_neglogp.append(median(nlps))
        contrast_rows.append(
            {
                "wildcard_kmer": wildcard,
                "cap_probes": cap,
                "contrast": key,
                "full_effect": target,
                "full_effect_abs_norm": abs(target) / design.y_sd,
                "full_neglog10p": full_nlp,
                "sample_effect_mean": float(np.mean(vals)),
                "sample_effect_sd_norm": sdn,
                "sample_effect_bias_norm": bn,
                "sample_effect_rmse_norm": rn,
                "sample_neglog10p_median": median(nlps),
                "sample_neglog10p_p90": quantile(nlps, 0.90),
                "repeats_successful": len(vals),
            }
        )

    if not sd_norm:
        return None

    return (
        {
            "wildcard_kmer": wildcard,
            "cap_probes": cap,
            "n_probes_available": len(design.y),
            "n_probes_kept_median": median(realized_n),
            "repeats_requested": repeats,
            "repeats_successful": successes,
            "success_fraction": successes / repeats,
            "effect_sd_norm": median(sd_norm),
            "effect_bias_norm": median(bias_norm),
            "effect_rmse_norm": median(rmse_norm),
            "neglog10p_median": median(median_neglogp),
        },
        contrast_rows,
    )


def summarize(
    family_curves: pd.DataFrame,
    family_info: pd.DataFrame,
    caps: Sequence[int],
    *,
    max_effect_sd: float,
    max_effect_bias: float,
    min_eligible_fraction: float,
    borderline_relative_tolerance: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize caps and apply the established recommendation rule."""
    rows: List[Dict[str, object]] = []
    n_total = len(family_info)

    for cap in caps:
        eligible = int((family_info["n_probes_available"] >= cap).sum())
        eligible_frac = eligible / n_total if n_total else 0.0
        sub = family_curves[family_curves["cap_probes"] == cap]
        rows.append(
            {
                "cap_probes": cap,
                "n_families_total": n_total,
                "n_families_eligible": eligible,
                "eligible_fraction": eligible_frac,
                "n_families_evaluated": len(sub),
                "effect_sd_norm_median": median(sub["effect_sd_norm"] if len(sub) else []),
                "effect_sd_norm_p90": quantile(sub["effect_sd_norm"] if len(sub) else [], 0.90),
                "effect_bias_norm_median": median(sub["effect_bias_norm"] if len(sub) else []),
                "effect_bias_norm_p90": quantile(sub["effect_bias_norm"] if len(sub) else [], 0.90),
                "effect_rmse_norm_median": median(sub["effect_rmse_norm"] if len(sub) else []),
                "neglog10p_median": median(sub["neglog10p_median"] if len(sub) else []),
                "n_probes_kept_median": median(sub["n_probes_kept_median"] if len(sub) else []),
            }
        )

    summary = pd.DataFrame(rows)
    recommended: Optional[int] = None
    for _, row in summary.iterrows():
        if (
            row["eligible_fraction"] >= min_eligible_fraction
            and np.isfinite(row["effect_sd_norm_p90"])
            and np.isfinite(row["effect_bias_norm_p90"])
            and row["effect_sd_norm_p90"] <= max_effect_sd
            and row["effect_bias_norm_p90"] <= max_effect_bias
        ):
            recommended = int(row["cap_probes"])
            break

    # Recommendation remains strict.  The optional "borderline" annotation below
    # is informational only and NEVER changes which cap is selected.
    if recommended is None:
        status = "NO_TESTED_CAP_MET_STABILITY_CRITERIA"
        rec_value: object = "NA"
        if (summary["n_families_evaluated"] > 0).any():
            best_tested: object = int(
                summary.loc[summary["n_families_evaluated"] > 0, "cap_probes"].max()
            )
        else:
            best_tested = "NA"
        rec_row = None
        lower_row = None
    else:
        status = "RECOMMENDED"
        rec_value = recommended
        best_tested = recommended
        rec_row = summary.loc[summary["cap_probes"] == recommended].iloc[0]
        lower = summary.loc[summary["cap_probes"] < recommended]
        lower_row = lower.iloc[-1] if len(lower) else None

    def _finite_or_na(value):
        try:
            return float(value) if np.isfinite(value) else "NA"
        except Exception:
            return "NA"

    borderline = False
    lower_failed: List[str] = []
    lower_sd_excess: object = "NA"
    lower_bias_excess: object = "NA"
    lower_eligible_deficit: object = "NA"

    if lower_row is not None:
        sd = float(lower_row["effect_sd_norm_p90"])
        bias = float(lower_row["effect_bias_norm_p90"])
        eligible = float(lower_row["eligible_fraction"])
        lower_sd_excess = max(0.0, sd - max_effect_sd) if np.isfinite(sd) else "NA"
        lower_bias_excess = max(0.0, bias - max_effect_bias) if np.isfinite(bias) else "NA"
        lower_eligible_deficit = max(0.0, min_eligible_fraction - eligible)
        if eligible < min_eligible_fraction:
            lower_failed.append("eligible_fraction")
        if not np.isfinite(sd) or sd > max_effect_sd:
            lower_failed.append("effect_sd_norm_p90")
        if not np.isfinite(bias) or bias > max_effect_bias:
            lower_failed.append("effect_bias_norm_p90")

        # "Borderline" means the immediately lower tested cap fails only by a
        # small relative amount.  This is a reporting aid, not a relaxed rule.
        def _within_relative(observed: float, threshold: float) -> bool:
            if not np.isfinite(observed) or threshold <= 0:
                return False
            return observed <= threshold * (1.0 + borderline_relative_tolerance)

        borderline = (
            eligible >= min_eligible_fraction
            and _within_relative(sd, max_effect_sd)
            and _within_relative(bias, max_effect_bias)
            and bool(lower_failed)
        )

    recommendation = pd.DataFrame(
        [
            {
                "recommended_max_probes": rec_value,
                "status": status,
                "best_tested_cap": best_tested,
                "recommended_eligible_fraction": (
                    _finite_or_na(rec_row["eligible_fraction"]) if rec_row is not None else "NA"
                ),
                "recommended_effect_sd_norm_p90": (
                    _finite_or_na(rec_row["effect_sd_norm_p90"]) if rec_row is not None else "NA"
                ),
                "recommended_effect_bias_norm_p90": (
                    _finite_or_na(rec_row["effect_bias_norm_p90"]) if rec_row is not None else "NA"
                ),
                "previous_tested_cap": (
                    int(lower_row["cap_probes"]) if lower_row is not None else "NA"
                ),
                "previous_eligible_fraction": (
                    _finite_or_na(lower_row["eligible_fraction"]) if lower_row is not None else "NA"
                ),
                "previous_effect_sd_norm_p90": (
                    _finite_or_na(lower_row["effect_sd_norm_p90"]) if lower_row is not None else "NA"
                ),
                "previous_effect_bias_norm_p90": (
                    _finite_or_na(lower_row["effect_bias_norm_p90"]) if lower_row is not None else "NA"
                ),
                "previous_failed_criteria": (
                    ",".join(lower_failed) if lower_failed else "NA"
                ),
                "previous_effect_sd_excess": lower_sd_excess,
                "previous_effect_bias_excess": lower_bias_excess,
                "previous_eligible_deficit": lower_eligible_deficit,
                "previous_cap_borderline": bool(borderline),
                "borderline_relative_tolerance": float(borderline_relative_tolerance),
                "rule": (
                    "smallest cap with p90 effect SD and p90 effect bias <= thresholds; "
                    "p-values not used; borderline annotation does not alter recommendation"
                ),
                "max_effect_sd_norm_p90": max_effect_sd,
                "max_effect_bias_norm_p90": max_effect_bias,
                "min_eligible_fraction": min_eligible_fraction,
            }
        ]
    )
    return summary, recommendation


def build_pvalue_quartiles(contrast_curves: pd.DataFrame) -> pd.DataFrame:
    if contrast_curves.empty:
        return pd.DataFrame()

    key = contrast_curves[
        ["wildcard_kmer", "contrast", "full_effect_abs_norm"]
    ].drop_duplicates()
    try:
        key["effect_quartile"] = pd.qcut(
            key["full_effect_abs_norm"],
            q=4,
            labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"],
            duplicates="drop",
        )
    except Exception:
        return pd.DataFrame()

    merged = contrast_curves.merge(
        key[["wildcard_kmer", "contrast", "effect_quartile"]],
        on=["wildcard_kmer", "contrast"],
        how="left",
    )
    rows: List[Dict[str, object]] = []
    for (quartile, cap), sub in merged.groupby(
        ["effect_quartile", "cap_probes"], observed=True
    ):
        rows.append(
            {
                "effect_quartile": str(quartile),
                "cap_probes": int(cap),
                "n_contrasts": int(len(sub)),
                "neglog10p_median": median(sub["sample_neglog10p_median"]),
                "neglog10p_p90": quantile(sub["sample_neglog10p_median"], 0.90),
                "full_effect_abs_norm_median": median(sub["full_effect_abs_norm"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["effect_quartile", "cap_probes"])


@dataclass(frozen=True)
class CalibrationResult:
    recommendation: pd.DataFrame
    summary: pd.DataFrame
    family_curves: pd.DataFrame
    contrasts: pd.DataFrame
    contrast_curves: pd.DataFrame
    pvalue_quartiles: pd.DataFrame
    families_examined: int


def run_max_probes_calibration(
    *,
    kmer_index: Mapping[str, Mapping[str, int]],
    intensities: Mapping[str, float],
    caps: Sequence[int],
    kmer_size: int = 8,
    n_families: int = 100,
    repeats: int = 50,
    seed: int = 1,
    min_probes: int = 50,
    include_covariates: bool = True,
    fold_half: bool = True,
    max_effect_sd: float = 0.10,
    max_effect_bias: float = 0.10,
    min_eligible_fraction: float = 0.50,
    borderline_relative_tolerance: float = 0.05,
    progress_callback=None,
) -> CalibrationResult:
    """Run the supported EUbar max-probes calibration."""
    if n_families <= 0:
        raise ValueError("n_families must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    caps = sorted(set(int(x) for x in caps))
    if not caps or any(x < 10 for x in caps):
        raise ValueError("caps must contain integers >= 10")

    candidates = family_candidates(kmer_index, kmer_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    family_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    contrast_curve_rows: List[Dict[str, object]] = []
    full_contrast_rows: List[Dict[str, object]] = []
    examined = 0

    for wildcard in candidates:
        if len(family_rows) >= n_families:
            break
        examined += 1

        lookup = build_region_lookup(wildcard, kmer_index)
        df = lookup_to_dataframe(lookup, intensities, fold_half=fold_half)
        if len(df) < min_probes or df["allele"].nunique() < 2:
            continue

        ref = choose_reference(df)
        design = make_design(df, ref=ref, include_covariates=include_covariates)
        full_fit = fast_ols(design.X, design.y)
        if full_fit is None:
            continue
        contrasts = available_contrasts(design)
        if not contrasts:
            continue

        full_stats: Dict[str, Tuple[float, float]] = {}
        for a, b in contrasts:
            key = f"{a}-{b}"
            est, nlp = contrast_stats(full_fit, contrast_vector(design, a, b))
            if np.isfinite(est):
                full_stats[key] = (est, nlp)
                full_contrast_rows.append(
                    {
                        "wildcard_kmer": wildcard,
                        "contrast": key,
                        "n_probes_available": len(design.y),
                        "full_effect": est,
                        "full_effect_abs_norm": abs(est) / design.y_sd,
                        "full_neglog10p": nlp,
                    }
                )
        if not full_stats:
            continue

        family_lookup = dataframe_to_lookup(df)
        family_rows.append(
            {
                "wildcard_kmer": wildcard,
                "ref_for_design": ref,
                "n_probes_available": len(design.y),
                "n_alleles": int(df["allele"].nunique()),
                "y_sd": design.y_sd,
            }
        )

        family_no = len(family_rows)
        if progress_callback is not None:
            progress_callback(
                "family",
                {
                    "accepted": family_no,
                    "requested": n_families,
                    "examined": examined,
                    "wildcard": wildcard,
                    "n_probes": len(design.y),
                    "eligible_caps": sum(len(design.y) >= cap for cap in caps),
                },
            )

        for cap in caps:
            if len(design.y) < cap:
                continue
            result = evaluate_cap(
                design,
                family_lookup,
                full_stats,
                contrasts,
                wildcard=wildcard,
                cap=cap,
                repeats=repeats,
                seed=seed,
            )
            if result is None:
                continue
            row, contrast_rows = result
            curve_rows.append(row)
            contrast_curve_rows.extend(contrast_rows)

    if not family_rows:
        raise ValueError("No wildcard families passed the calibration filters")

    family_info = pd.DataFrame(family_rows)
    family_curves = pd.DataFrame(curve_rows)
    contrast_curves = pd.DataFrame(contrast_curve_rows)
    contrasts_df = pd.DataFrame(full_contrast_rows)
    summary, recommendation = summarize(
        family_curves,
        family_info,
        caps,
        max_effect_sd=max_effect_sd,
        max_effect_bias=max_effect_bias,
        min_eligible_fraction=min_eligible_fraction,
        borderline_relative_tolerance=borderline_relative_tolerance,
    )
    pvalue_quartiles = build_pvalue_quartiles(contrast_curves)

    return CalibrationResult(
        recommendation=recommendation,
        summary=summary,
        family_curves=family_curves,
        contrasts=contrasts_df,
        contrast_curves=contrast_curves,
        pvalue_quartiles=pvalue_quartiles,
        families_examined=examined,
    )


# ---------------------------------------------------------------------------
# Explicit mask calibration

@dataclass(frozen=True)
class MaskCandidateFamily:
    mask: str
    tested_position: int
    display_pattern: str
    allele_codes: Dict[str, int]


def parse_masks(text: str) -> List[str]:
    """Parse comma-separated explicit 0/1 masks, preserving first occurrence."""
    out: List[str] = []
    seen = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        mask = SequenceMask.parse(item)
        if mask.pattern not in seen:
            seen.add(mask.pattern)
            out.append(mask.pattern)
    if not out:
        raise ValueError("at least one mask is required")
    return out




def generate_matched_random_masks(
    reference_mask: str,
    n: int,
    *,
    seed: int = 1,
    exclude: Sequence[str] = (),
    require_informative_ends: bool = True,
) -> List[str]:
    """Generate unique random masks matched for span and information content.

    Random controls preserve the reference mask's total span and number of
    informative (1) positions.  By default the first and last positions remain
    informative so the nominal span is also the informative span; this avoids
    controls that are effectively shorter masks padded with ignored bases.

    Reverse-equivalent masks are sampled only once because masked matching
    evaluates both DNA strands.
    """
    if n < 0:
        raise ValueError("n random controls must be >= 0")
    if n == 0:
        return []

    ref = SequenceMask.parse(reference_mask)
    span = int(ref.span)
    n_info = int(ref.n_informative)
    if require_informative_ends and span >= 2:
        if n_info < 2:
            raise ValueError("matched random controls require at least two informative bases")
        movable = np.arange(1, span - 1, dtype=int)
        choose_n = n_info - 2
    else:
        movable = np.arange(0, span, dtype=int)
        choose_n = n_info

    if choose_n < 0 or choose_n > len(movable):
        raise ValueError("cannot generate matched random controls for this mask")

    # Treat a mask and its reversed coordinate pattern as the same architecture
    # because the matcher already scans both DNA strands.
    def canon(pattern: str) -> str:
        rev = pattern[::-1]
        return pattern if pattern <= rev else rev

    blocked = {canon(SequenceMask.parse(x).pattern) for x in exclude}
    blocked.add(canon(ref.pattern))
    rng = np.random.default_rng(int(seed))
    out: List[str] = []
    seen = set(blocked)

    # Rejection sampling is cheap for the mask sizes EUbar currently uses.
    # Bound attempts so impossible requests fail clearly rather than hanging.
    max_attempts = max(1000, int(n) * 200)
    attempts = 0
    while len(out) < int(n) and attempts < max_attempts:
        attempts += 1
        bits = ["0"] * span
        if require_informative_ends and span >= 2:
            bits[0] = "1"
            bits[-1] = "1"
        if choose_n:
            selected = rng.choice(movable, size=choose_n, replace=False)
            for pos in selected:
                bits[int(pos)] = "1"
        pattern = "".join(bits)
        key = canon(pattern)
        if key in seen:
            continue
        seen.add(key)
        # Emit a canonical orientation to make runs easy to compare.
        out.append(key)

    if len(out) < int(n):
        raise ValueError(
            f"requested {n} unique matched random controls for {reference_mask}, "
            f"but only generated {len(out)} after excluding equivalent masks"
        )
    return out


def _parse_bed_region(region: str) -> Tuple[str, int, int]:
    chrom, coords = str(region).split(":", 1)
    start_s, end_s = coords.split("-", 1)
    return chrom, int(start_s), int(end_s)


def sample_mask_candidate_families(
    *,
    mask: SequenceMask,
    probe_regions: Sequence[str],
    genome_fasta: str,
    n_candidates: int,
    seed: int,
) -> List[MaskCandidateFamily]:
    """Sample candidate wildcard families from real probe windows.

    Candidate families are drawn from the same probe-region universe used by
    masked SNV mode.  No motif labels or external truth set are used.
    """
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    regions = list(probe_regions)
    if not regions:
        raise ValueError("intensity table contains no probe regions")

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(regions))
    fasta = _open_fasta(genome_fasta)
    out: List[MaskCandidateFamily] = []
    seen = set()

    # One random window per shuffled probe region is enough to obtain a broad,
    # deterministic candidate set without scanning the entire array twice.
    for idx in order:
        region = regions[int(idx)]
        try:
            chrom, start0, end0 = _parse_bed_region(region)
            seq = fasta[chrom][start0:end0].seq.upper()
        except Exception:
            continue
        if len(seq) < mask.span:
            continue

        max_off = len(seq) - mask.span
        off0 = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
        tested_position = int(mask.informative[int(rng.integers(0, mask.n_informative))])
        context = seq[off0 : off0 + mask.span]
        if len(context) != mask.span:
            continue

        codes: Dict[str, int] = {}
        valid = True
        for allele in BASES:
            filled = list(context)
            filled[tested_position] = allele
            code = mask.signature_code("".join(filled))
            if code is None:
                valid = False
                break
            codes[allele] = int(code)
        if not valid:
            continue

        # Include the tested position because two different informative
        # positions can, in principle, generate the same four full signatures.
        key = (tested_position, tuple(sorted(codes.values())))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            MaskCandidateFamily(
                mask=mask.pattern,
                tested_position=tested_position,
                display_pattern=mask.display_pattern(context, tested_position),
                allele_codes=codes,
            )
        )
        if len(out) >= n_candidates:
            break

    if not out:
        raise ValueError(f"could not sample any candidate families for mask {mask.pattern}")
    return out


def scan_mask_target_codes(
    *,
    mask: SequenceMask,
    target_codes: Sequence[int],
    probe_regions: Sequence[str],
    genome_fasta: str,
) -> Tuple[Dict[int, Dict[str, MaskHit]], Dict[str, int]]:
    """Scan the probe universe once and retain hits only for target signatures."""
    targets = set(int(x) for x in target_codes)
    hits: Dict[int, Dict[str, MaskHit]] = {code: {} for code in targets}
    ambiguous: Dict[int, set] = {code: set() for code in targets}
    fasta = _open_fasta(genome_fasta)
    n_valid_regions = 0
    n_windows = 0

    for region in probe_regions:
        try:
            chrom, start0, end0 = _parse_bed_region(region)
            seq = fasta[chrom][start0:end0].seq.upper()
        except Exception:
            continue
        if len(seq) < mask.span:
            continue
        n_valid_regions += 1
        f_codes, r_codes, valid_f, valid_r = _signature_codes(seq, mask)
        n = len(f_codes)
        n_windows += int(n)
        local: Dict[int, List[MaskHit]] = defaultdict(list)

        for off0 in range(n):
            if bool(valid_f[off0]):
                code = int(f_codes[off0])
                if code in targets:
                    local[code].append(MaskHit(offset_1based=off0 + 1, is_reverse=False))
            if bool(valid_r[off0]):
                code = int(r_codes[off0])
                if code in targets:
                    local[code].append(MaskHit(offset_1based=off0 + 1, is_reverse=True))

        for code, local_hits in local.items():
            uniq = {(h.offset_1based, h.is_reverse): h for h in local_hits}
            if len(uniq) == 1:
                if region not in ambiguous[code]:
                    hits[code][region] = next(iter(uniq.values()))
            else:
                ambiguous[code].add(region)
                hits[code].pop(region, None)

    return hits, {
        "n_probe_regions": len(probe_regions),
        "n_valid_regions": n_valid_regions,
        "n_windows": n_windows,
        "n_target_codes": len(targets),
        "n_target_region_hits": sum(len(v) for v in hits.values()),
        "n_ambiguous_region_hits": sum(len(v) for v in ambiguous.values()),
    }


def mask_family_lookup(
    family: MaskCandidateFamily,
    mask: SequenceMask,
    target_hits: Mapping[int, Mapping[str, MaskHit]],
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    j = int(family.tested_position)
    for allele, code in family.allele_codes.items():
        regmap: Dict[str, int] = {}
        for region, hit in target_hits.get(int(code), {}).items():
            if hit.is_reverse:
                wildcard_pos = int(hit.offset_1based) + (mask.span - 1 - j)
            else:
                wildcard_pos = int(hit.offset_1based) + j
            regmap[str(region)] = int(wildcard_pos)
        if regmap:
            out[str(allele)] = regmap
    return out


def _residual_ratio(design: FastDesign, fit: OLSFit) -> float:
    resid = design.y - design.X @ fit.beta
    resid_sd = float(np.std(resid))
    if not np.isfinite(resid_sd) or design.y_sd <= 0:
        return np.nan
    return resid_sd / design.y_sd


def _minor_allele_fraction(df: pd.DataFrame) -> float:
    counts = df["allele"].value_counts().to_numpy(dtype=float)
    if len(counts) < 2 or counts.sum() <= 0:
        return np.nan
    return float(counts.min() / counts.sum())


@dataclass(frozen=True)
class MaskCalibrationResult:
    recommendation: pd.DataFrame
    summary: pd.DataFrame
    families: pd.DataFrame
    family_curves: pd.DataFrame


def build_mask_recommendation(
    summary: pd.DataFrame,
    *,
    stability_probes: int,
    min_eligible_fraction: float,
    near_optimal_relative_tolerance: float = 0.05,
    random_control_reference_mask: Optional[str] = None,
    n_random_controls_requested: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate mask metrics and recommend among candidate masks only.

    Candidate masks are eligible for recommendation. Explicit and random control
    masks are evaluated identically but can never become the recommendation.
    Random controls are matched to the recommended candidate's span and number
    of informative positions and provide a descriptive null distribution for
    asking whether that architecture is unusually stable relative to arbitrary
    masks with the same information content.

    ``near_optimal`` remains an operational tolerance band, not a formal
    statistical-equivalence test. The random-control empirical tail fraction is
    also descriptive because each mask is evaluated on its own sampled probe
    families.
    """
    if near_optimal_relative_tolerance < 0:
        raise ValueError("near_optimal_relative_tolerance must be >= 0")

    summary = summary.copy()
    if "mask_role" not in summary.columns:
        summary["mask_role"] = "candidate"
    summary["mask_role"] = summary["mask_role"].fillna("candidate").astype(str)
    if "control_source" not in summary.columns:
        summary["control_source"] = np.where(
            summary["mask_role"].eq("candidate"), "candidate", "explicit"
        )
    summary["control_source"] = summary["control_source"].fillna("candidate").astype(str)

    summary["recommendation_eligible"] = summary["mask_role"].eq("candidate")
    summary["meets_representation_criteria"] = (
        (summary["eligible_fraction"] >= float(min_eligible_fraction))
        & (summary["n_families_evaluated"] > 0)
        & np.isfinite(summary["effect_sd_norm_p90"])
    )
    summary["stability_rank"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    summary["candidate_stability_rank"] = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    summary["stability_ratio_to_best_candidate"] = np.nan
    summary["stability_pct_above_best_candidate"] = np.nan
    summary["stability_ratio_to_best"] = np.nan
    summary["stability_pct_above_best"] = np.nan
    summary["near_optimal"] = False
    summary["within_candidate_near_optimal_band"] = False
    summary["beats_recommended_candidate"] = False

    analyzable = summary[summary["meets_representation_criteria"]].copy()
    candidates = analyzable[analyzable["mask_role"] == "candidate"].copy()
    controls = analyzable[analyzable["mask_role"] == "control"].copy()
    random_controls = controls[controls["control_source"] == "random"].copy()

    rule_text = (
        "among candidate masks meeting min eligible-family fraction, choose lowest p90 "
        "normalized coefficient SD at a fixed probe count; residual noise then representation "
        "break exact ties; diagnostic controls are evaluated identically but excluded from "
        "recommendation; random controls preserve the recommended candidate's span and number "
        "of informative positions; near-optimal candidates are within the configured relative "
        "stability band; no external labels used"
    )

    if candidates.empty:
        recommendation = pd.DataFrame([{
            "recommended_mask": "NA",
            "status": "NO_CANDIDATE_MASK_MET_REPRESENTATION_CRITERIA",
            "near_optimal_masks": "NA",
            "n_near_optimal": 0,
            "near_optimal_relative_tolerance": float(near_optimal_relative_tolerance),
            "near_optimal_threshold_effect_sd_norm_p90": "NA",
            "best_control_mask": "NA",
            "best_control_effect_sd_norm_p90": "NA",
            "best_control_pct_vs_recommended": "NA",
            "control_status": "CONTROLS_PRESENT_BUT_NO_CANDIDATE_RECOMMENDATION" if len(controls) else "NO_CONTROLS",
            "random_control_reference_mask": random_control_reference_mask or "NA",
            "n_random_controls_requested": int(n_random_controls_requested),
            "n_random_controls_evaluated": int(len(random_controls)),
            "random_control_fraction_better_or_equal": "NA",
            "random_control_candidate_percentile": "NA",
            "random_control_empirical_p": "NA",
            "random_control_effect_sd_norm_median": "NA",
            "random_control_effect_sd_norm_p10": "NA",
            "random_control_effect_sd_norm_p90": "NA",
            "rule": rule_text,
            "stability_probes": int(stability_probes),
            "min_eligible_fraction": float(min_eligible_fraction),
        }])
        return summary, recommendation

    analyzable = analyzable.sort_values(
        ["effect_sd_norm_p90", "residual_sd_ratio_p90", "eligible_fraction", "mask"],
        ascending=[True, True, False, True], kind="mergesort",
    )
    for rank, idx in enumerate(analyzable.index, start=1):
        summary.loc[idx, "stability_rank"] = rank

    candidates = candidates.sort_values(
        ["effect_sd_norm_p90", "residual_sd_ratio_p90", "eligible_fraction", "mask"],
        ascending=[True, True, False, True], kind="mergesort",
    )
    best = candidates.iloc[0]
    best_sd = float(best["effect_sd_norm_p90"])
    near_threshold = best_sd * (1.0 + float(near_optimal_relative_tolerance))
    near = candidates[candidates["effect_sd_norm_p90"] <= near_threshold].copy()
    near_masks = near["mask"].astype(str).tolist()

    for rank, (idx, _) in enumerate(candidates.iterrows(), start=1):
        summary.loc[idx, "candidate_stability_rank"] = rank

    for idx, row in analyzable.iterrows():
        observed = float(row["effect_sd_norm_p90"])
        ratio = observed / best_sd if best_sd > 0 else 1.0
        summary.loc[idx, "stability_ratio_to_best_candidate"] = ratio
        summary.loc[idx, "stability_pct_above_best_candidate"] = (ratio - 1.0) * 100.0
        summary.loc[idx, "stability_ratio_to_best"] = ratio
        summary.loc[idx, "stability_pct_above_best"] = (ratio - 1.0) * 100.0
        within_band = observed <= near_threshold
        summary.loc[idx, "within_candidate_near_optimal_band"] = bool(within_band)
        if str(row["mask_role"]) == "candidate":
            summary.loc[idx, "near_optimal"] = bool(within_band)
        else:
            summary.loc[idx, "beats_recommended_candidate"] = bool(observed < best_sd)

    best_control = None
    if not controls.empty:
        controls = controls.sort_values(
            ["effect_sd_norm_p90", "residual_sd_ratio_p90", "eligible_fraction", "mask"],
            ascending=[True, True, False, True], kind="mergesort",
        )
        best_control = controls.iloc[0]

    if best_control is None:
        control_fields = {
            "best_control_mask": "NA",
            "best_control_effect_sd_norm_p90": "NA",
            "best_control_pct_vs_recommended": "NA",
            "control_status": "NO_CONTROLS",
        }
    else:
        control_sd = float(best_control["effect_sd_norm_p90"])
        pct_vs = ((control_sd / best_sd) - 1.0) * 100.0 if best_sd > 0 else 0.0
        if control_sd < best_sd:
            control_status = "CONTROL_OUTPERFORMS_RECOMMENDED_CANDIDATE"
        elif control_sd <= near_threshold:
            control_status = "CONTROL_WITHIN_CANDIDATE_NEAR_OPTIMAL_BAND"
        else:
            control_status = "CONTROLS_WORSE_THAN_CANDIDATE_BAND"
        control_fields = {
            "best_control_mask": str(best_control["mask"]),
            "best_control_effect_sd_norm_p90": control_sd,
            "best_control_pct_vs_recommended": pct_vs,
            "control_status": control_status,
        }

    if random_controls.empty:
        random_fields = {
            "random_control_reference_mask": random_control_reference_mask or "NA",
            "n_random_controls_requested": int(n_random_controls_requested),
            "n_random_controls_evaluated": 0,
            "random_control_fraction_better_or_equal": "NA",
            "random_control_candidate_percentile": "NA",
            "random_control_empirical_p": "NA",
            "random_control_effect_sd_norm_median": "NA",
            "random_control_effect_sd_norm_p10": "NA",
            "random_control_effect_sd_norm_p90": "NA",
        }
    else:
        vals = random_controls["effect_sd_norm_p90"].to_numpy(dtype=float)
        n_rand = int(len(vals))
        n_better_equal = int(np.sum(vals <= best_sd))
        fraction_better_equal = n_better_equal / n_rand
        candidate_percentile = 100.0 * float(np.mean(vals >= best_sd))
        empirical_p = (1.0 + n_better_equal) / (1.0 + n_rand)
        random_fields = {
            "random_control_reference_mask": random_control_reference_mask or str(best["mask"]),
            "n_random_controls_requested": int(n_random_controls_requested),
            "n_random_controls_evaluated": n_rand,
            "random_control_fraction_better_or_equal": fraction_better_equal,
            "random_control_candidate_percentile": candidate_percentile,
            "random_control_empirical_p": empirical_p,
            "random_control_effect_sd_norm_median": median(vals),
            "random_control_effect_sd_norm_p10": quantile(vals, 0.10),
            "random_control_effect_sd_norm_p90": quantile(vals, 0.90),
        }

    recommendation = pd.DataFrame([{
        "recommended_mask": str(best["mask"]),
        "status": "RECOMMENDED_UNIQUE" if len(near_masks) == 1 else "RECOMMENDED_WITH_NEAR_OPTIMAL_ALTERNATIVES",
        "eligible_fraction": float(best["eligible_fraction"]),
        "effect_sd_norm_p90": best_sd,
        "effect_bias_norm_p90": float(best["effect_bias_norm_p90"]),
        "residual_sd_ratio_p90": float(best["residual_sd_ratio_p90"]),
        "minor_allele_fraction_median": float(best["minor_allele_fraction_median"]),
        "near_optimal_masks": ",".join(near_masks),
        "n_near_optimal": int(len(near_masks)),
        "near_optimal_relative_tolerance": float(near_optimal_relative_tolerance),
        "near_optimal_threshold_effect_sd_norm_p90": float(near_threshold),
        **control_fields,
        **random_fields,
        "rule": rule_text,
        "stability_probes": int(stability_probes),
        "min_eligible_fraction": float(min_eligible_fraction),
    }])
    return summary, recommendation


def run_mask_calibration(
    *,
    intensities: Mapping[str, float],
    genome_fasta: str,
    masks: Sequence[str],
    control_masks: Sequence[str] = (),
    random_controls: int = 0,
    n_families: int = 500,
    repeats: int = 50,
    stability_probes: int = 100,
    min_probes: int = 50,
    min_eligible_fraction: float = 0.50,
    near_optimal_relative_tolerance: float = 0.05,
    candidate_multiplier: int = 5,
    seed: int = 1,
    include_covariates: bool = True,
    fold_half: bool = True,
    progress_callback=None,
) -> MaskCalibrationResult:
    """Compare explicit masks and optional matched random controls.

    Candidate masks are evaluated first. If ``random_controls`` is positive,
    EUbar identifies the best candidate using the normal rule, then generates
    that many unique random masks matched to the best candidate's span and
    number of informative bases. First/last positions remain informative so a
    random control cannot masquerade as a shorter effective span. Random masks
    are evaluated identically but remain diagnostic controls only.
    """
    if n_families <= 0 or repeats <= 0:
        raise ValueError("n_families and repeats must be positive")
    if stability_probes < 10:
        raise ValueError("stability_probes must be >= 10")
    if candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be >= 1")
    if random_controls < 0:
        raise ValueError("random_controls must be >= 0")
    if not (0.0 <= min_eligible_fraction <= 1.0):
        raise ValueError("min_eligible_fraction must be between 0 and 1")
    if near_optimal_relative_tolerance < 0:
        raise ValueError("near_optimal_relative_tolerance must be >= 0")

    candidate_patterns: List[str] = []
    seen_patterns = set()
    for m in masks:
        obj = SequenceMask.parse(m)
        if obj.pattern not in seen_patterns:
            seen_patterns.add(obj.pattern)
            candidate_patterns.append(obj.pattern)
    if not candidate_patterns:
        raise ValueError("at least one candidate mask is required")

    explicit_control_patterns: List[str] = []
    for m in control_masks:
        obj = SequenceMask.parse(m)
        if obj.pattern not in seen_patterns:
            seen_patterns.add(obj.pattern)
            explicit_control_patterns.append(obj.pattern)

    probe_regions = tuple(str(r) for r in intensities.keys())
    required_n = max(int(min_probes), int(stability_probes) + 1)
    summary_rows: List[Dict[str, object]] = []
    family_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    def evaluate_patterns(patterns: Sequence[str], *, role: str, control_source: str, start_no: int, total_masks: int) -> None:
        for local_no, pattern in enumerate(patterns, start=1):
            mask_no = int(start_no) + local_no - 1
            mask = SequenceMask.parse(pattern)
            n_candidates_target = max(int(n_families) * int(candidate_multiplier), int(n_families) + 50)
            candidates = sample_mask_candidate_families(
                mask=mask,
                probe_regions=probe_regions,
                genome_fasta=genome_fasta,
                n_candidates=n_candidates_target,
                seed=int(seed) + mask_no * 100003,
            )
            target_codes = sorted({code for fam in candidates for code in fam.allele_codes.values()})
            if progress_callback is not None:
                progress_callback("mask_scan_start", {
                    "mask": mask.pattern, "mask_no": mask_no, "n_masks": total_masks,
                    "n_candidates": len(candidates), "n_target_codes": len(target_codes),
                    "mask_role": role, "control_source": control_source,
                })
            target_hits, scan_stats = scan_mask_target_codes(
                mask=mask, target_codes=target_codes, probe_regions=probe_regions, genome_fasta=genome_fasta,
            )
            if progress_callback is not None:
                progress_callback("mask_scan_done", {
                    "mask": mask.pattern, "mask_role": role, "control_source": control_source, **scan_stats,
                })

            eligible_payloads = []
            mask_family_rows: List[Dict[str, object]] = []
            for family_no, family in enumerate(candidates, start=1):
                lookup = mask_family_lookup(family, mask, target_hits)
                df = lookup_to_dataframe(lookup, intensities, fold_half=fold_half)
                n_probes = int(len(df))
                n_alleles = int(df["allele"].nunique()) if n_probes else 0
                counts = df["allele"].value_counts().to_dict() if n_probes else {}
                row: Dict[str, object] = {
                    "mask": mask.pattern, "mask_role": role, "control_source": control_source,
                    "mask_span": mask.span, "n_informative": mask.n_informative,
                    "family_no": family_no, "tested_position": int(family.tested_position),
                    "wildcard_pattern": family.display_pattern, "n_probes_available": n_probes,
                    "n_alleles": n_alleles,
                    "minor_allele_fraction": _minor_allele_fraction(df) if n_probes else np.nan,
                    "n_A": int(counts.get("A", 0)), "n_C": int(counts.get("C", 0)),
                    "n_G": int(counts.get("G", 0)), "n_T": int(counts.get("T", 0)),
                    "eligible": False, "residual_sd_ratio": np.nan,
                }
                if n_probes < required_n or n_alleles < 2:
                    mask_family_rows.append(row); continue
                ref = choose_reference(df)
                design = make_design(df, ref=ref, include_covariates=include_covariates)
                full_fit = fast_ols(design.X, design.y)
                if full_fit is None:
                    mask_family_rows.append(row); continue
                contrasts = available_contrasts(design)
                if not contrasts:
                    mask_family_rows.append(row); continue
                full_stats: Dict[str, Tuple[float, float]] = {}
                for a, b in contrasts:
                    est, nlp = contrast_stats(full_fit, contrast_vector(design, a, b))
                    if np.isfinite(est):
                        full_stats[f"{a}-{b}"] = (est, nlp)
                if not full_stats:
                    mask_family_rows.append(row); continue
                row["eligible"] = True
                row["residual_sd_ratio"] = _residual_ratio(design, full_fit)
                mask_family_rows.append(row)
                eligible_payloads.append((family, design, dataframe_to_lookup(df), full_stats, contrasts))

            family_rows.extend(mask_family_rows)
            eligible_count = sum(bool(r["eligible"]) for r in mask_family_rows)
            eligible_fraction = eligible_count / len(candidates) if candidates else 0.0
            selected = eligible_payloads[: int(n_families)]
            mask_curve_rows: List[Dict[str, object]] = []
            for eval_no, (family, design, lookup, full_stats, contrasts) in enumerate(selected, start=1):
                result = evaluate_cap(
                    design, lookup, full_stats, contrasts,
                    wildcard=f"{mask.pattern}|{family.display_pattern}", cap=int(stability_probes),
                    repeats=int(repeats), seed=int(seed),
                )
                if result is None:
                    continue
                curve, _ = result
                curve = dict(curve)
                curve.update({
                    "mask": mask.pattern, "mask_role": role, "control_source": control_source,
                    "mask_span": mask.span, "n_informative": mask.n_informative,
                    "tested_position": int(family.tested_position),
                    "wildcard_pattern": family.display_pattern, "stability_probes": int(stability_probes),
                })
                curve_rows.append(curve); mask_curve_rows.append(curve)
                if progress_callback is not None:
                    progress_callback("mask_family", {
                        "mask": mask.pattern, "evaluated": eval_no,
                        "requested": min(int(n_families), len(eligible_payloads)),
                        "mask_role": role, "control_source": control_source,
                    })

            curve_df = pd.DataFrame(mask_curve_rows)
            fam_df = pd.DataFrame(mask_family_rows)
            eligible_df = fam_df[fam_df["eligible"] == True]  # noqa: E712
            summary_rows.append({
                "mask": mask.pattern, "mask_role": role, "control_source": control_source,
                "mask_span": mask.span, "n_informative": mask.n_informative,
                "n_candidate_families": len(candidates), "n_families_eligible": eligible_count,
                "eligible_fraction": eligible_fraction, "n_families_evaluated": len(curve_df),
                "stability_probes": int(stability_probes),
                "n_probes_median": median(eligible_df["n_probes_available"] if len(eligible_df) else []),
                "n_probes_p10": quantile(eligible_df["n_probes_available"] if len(eligible_df) else [], 0.10),
                "n_probes_p90": quantile(eligible_df["n_probes_available"] if len(eligible_df) else [], 0.90),
                "minor_allele_fraction_median": median(eligible_df["minor_allele_fraction"] if len(eligible_df) else []),
                "residual_sd_ratio_median": median(eligible_df["residual_sd_ratio"] if len(eligible_df) else []),
                "residual_sd_ratio_p90": quantile(eligible_df["residual_sd_ratio"] if len(eligible_df) else [], 0.90),
                "effect_sd_norm_median": median(curve_df["effect_sd_norm"] if len(curve_df) else []),
                "effect_sd_norm_p90": quantile(curve_df["effect_sd_norm"] if len(curve_df) else [], 0.90),
                "effect_bias_norm_median": median(curve_df["effect_bias_norm"] if len(curve_df) else []),
                "effect_bias_norm_p90": quantile(curve_df["effect_bias_norm"] if len(curve_df) else [], 0.90),
                "n_probe_regions": int(scan_stats["n_probe_regions"]),
                "n_windows_scanned": int(scan_stats["n_windows"]),
                "n_target_codes": int(scan_stats["n_target_codes"]),
                "n_target_region_hits": int(scan_stats["n_target_region_hits"]),
            })

    initial_patterns = candidate_patterns + explicit_control_patterns
    total_initial = len(initial_patterns)
    evaluate_patterns(candidate_patterns, role="candidate", control_source="candidate", start_no=1, total_masks=total_initial)
    if explicit_control_patterns:
        evaluate_patterns(
            explicit_control_patterns, role="control", control_source="explicit",
            start_no=1 + len(candidate_patterns), total_masks=total_initial,
        )

    # First pass chooses the candidate architecture only. Controls never change it.
    initial_summary, initial_recommendation = build_mask_recommendation(
        pd.DataFrame(summary_rows), stability_probes=int(stability_probes),
        min_eligible_fraction=float(min_eligible_fraction),
        near_optimal_relative_tolerance=float(near_optimal_relative_tolerance),
    )
    rec_mask = str(initial_recommendation.iloc[0].get("recommended_mask", "NA"))
    random_patterns: List[str] = []
    if int(random_controls) > 0 and rec_mask != "NA":
        random_patterns = generate_matched_random_masks(
            rec_mask, int(random_controls), seed=int(seed) + 700001,
            exclude=candidate_patterns + explicit_control_patterns,
            require_informative_ends=True,
        )
        total_final = total_initial + len(random_patterns)
        evaluate_patterns(
            random_patterns, role="control", control_source="random",
            start_no=1 + total_initial, total_masks=total_final,
        )

    summary, recommendation = build_mask_recommendation(
        pd.DataFrame(summary_rows), stability_probes=int(stability_probes),
        min_eligible_fraction=float(min_eligible_fraction),
        near_optimal_relative_tolerance=float(near_optimal_relative_tolerance),
        random_control_reference_mask=(rec_mask if random_patterns else None),
        n_random_controls_requested=int(random_controls),
    )
    return MaskCalibrationResult(
        recommendation=recommendation,
        summary=summary,
        families=pd.DataFrame(family_rows),
        family_curves=pd.DataFrame(curve_rows),
    )
