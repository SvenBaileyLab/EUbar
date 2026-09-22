"""Notebook-friendly inspection helpers.

These helpers expose the *exact* regression inputs used by the SNV + scan
pipelines, without requiring you to run the CLI drivers.

They are intentionally thin wrappers around existing core components:
  MotifMatcher -> DesignBuilder -> (optional) RandSampler/RandRegressor.

The main idea is that in a Jupyter notebook you can do:

    I = IntensityTable.from_file(...)
    K = KmerIndex.from_file(...)
    matcher = MotifMatcher(K.kmers)
    design = DesignBuilder(I.values)
    snv = SnvWindow.from_snv(...)

    payloads = build_aff_payloads_for_snv(snv, matcher=matcher, design=design, k=8)
    payloads[(motif_pos, snv_index)].X
    payloads[(motif_pos, snv_index)].y

And inspect exactly what is sent into the regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .design import DesignBuilder, WindowDesign
from .matching import MotifMatcher
from .rand import RandSampler
from .sampling import subsample_probes
from .sequence import RegionWindow, SnvWindow


@dataclass(frozen=True)
class RegressionPayload:
    """A fully inspectable regression input bundle."""

    kind: str  # "AFF" or "RAND"
    motif_pos: int
    snv_index: Optional[int]  # None for RAND payloads (RAND is per motif_pos)
    ref_allele: Optional[str]
    local_kmer: str
    wildcard_kmer: Optional[str]
    X: pd.DataFrame
    y: pd.Series
    meta: Dict[str, Any]

    def as_dataframe(self, y_name: str = "y") -> pd.DataFrame:
        """Return a single dataframe with y + X side-by-side."""
        df = self.X.copy()
        df.insert(0, y_name, self.y)
        return df

    def summary(self) -> Dict[str, Any]:
        """Lightweight summary intended for quick notebook sanity checks."""
        out = {
            "kind": self.kind,
            "motif_pos": self.motif_pos,
            "snv_index": self.snv_index,
            "ref_allele": self.ref_allele,
            "n": int(self.X.shape[0]),
            "p": int(self.X.shape[1]),
            "columns": list(self.X.columns),
            "y_min": float(np.min(self.y)) if len(self.y) else np.nan,
            "y_max": float(np.max(self.y)) if len(self.y) else np.nan,
            "y_mean": float(np.mean(self.y)) if len(self.y) else np.nan,
        }
        # Prefer "used" counts (what actually went into regression), but keep backward compatibility.
        if "allele_counts" in self.meta:
            out["allele_counts"] = self.meta.get("allele_counts")
        if "allele_counts_used" in self.meta:
            out["allele_counts_used"] = self.meta.get("allele_counts_used")
        if "allele_counts_raw" in self.meta:
            out["allele_counts_raw"] = self.meta.get("allele_counts_raw")
        out.update({k: self.meta.get(k) for k in ("n_bg", "n_hit") if k in self.meta})
        return out

    def checks(self) -> Dict[str, Any]:
        """Return a dict of common pitfalls (NA/inf, all-zero cols, etc.)."""
        x = self.X
        y = self.y

        x_has_na = bool(x.isna().any().any())
        y_has_na = bool(pd.isna(y).any())
        x_has_inf = bool(np.isinf(x.to_numpy(dtype=float, copy=False)).any())
        y_has_inf = bool(np.isinf(np.asarray(y, dtype=float)).any())

        # All-zero columns are often a sign of empty matches or a bug in encoding.
        all_zero_cols = [c for c in x.columns if float(x[c].sum()) == 0.0]

        return {
            "x_has_na": x_has_na,
            "y_has_na": y_has_na,
            "x_has_inf": x_has_inf,
            "y_has_inf": y_has_inf,
            "all_zero_cols": all_zero_cols,
            "n": int(x.shape[0]),
            "p": int(x.shape[1]),
        }


def _ref_allele(region_seq: str, motif_pos: int, snv_index: int) -> str:
    idx = motif_pos + snv_index
    if idx < 0 or idx >= len(region_seq):
        raise ValueError("motif_pos/snv_index out of bounds")
    return region_seq[idx]


def _wildcard_kmer(local_kmer: str, snv_index: int) -> str:
    w = list(local_kmer)
    w[snv_index] = "."
    return "".join(w)


def _allele_counts_used_from_design(X: pd.DataFrame, ref_allele: str) -> Dict[str, int]:
    """Compute allele counts from the *actual* design matrix used in regression.

    DesignBuilder dummy-codes alleles relative to ref_allele, so the reference allele
    corresponds to rows where all non-reference allele dummy columns are zero.
    """
    alleles = "ACGT"
    allele_cols = [a for a in alleles if a in X.columns]
    if allele_cols:
        d = X[allele_cols].sum(axis=1)
    else:
        # No allele columns (shouldn't happen for normal runs), treat all as reference.
        d = pd.Series(0.0, index=X.index)

    counts: Dict[str, int] = {}
    for a in alleles:
        if a == ref_allele:
            counts[a] = int((d == 0).sum())
        else:
            counts[a] = int((X[a] == 1).sum()) if a in X.columns else 0
    return counts


def build_aff_payload(
    *,
    region_seq: str,
    motif_pos: int,
    snv_index: int,
    region_lookup: Mapping[str, Mapping[str, int]],
    design: DesignBuilder,
    k: int = 8,
    include_covariates: bool = True,
    fold_half: bool = True,
    mask_pattern: Optional[str] = None,
) -> Optional[RegressionPayload]:
    """Build the exact AFF WindowDesign payload used for regression.

    This is the lowest-level helper and is useful when you already have
    region_lookup (e.g. from MotifMatcher.scan).
    """
    if not region_lookup:
        return None

    ref = _ref_allele(region_seq, motif_pos, snv_index)
    wd: Optional[WindowDesign] = design.build(
        region_lookup=region_lookup,
        ref_allele=ref,
        include_covariates=include_covariates,
        fold_half=fold_half,
    )
    if wd is None:
        return None

    local_kmer = region_seq[motif_pos : motif_pos + k]
    if len(local_kmer) != k:
        return None

    # Raw counts: directly from the matcher lookup (may include ambiguous or filtered-out rows)
    allele_counts_raw = {a: int(len(region_lookup.get(a, {}) or {})) for a in "ACGT"}
    # Used counts: computed from the actual X that went into regression
    allele_counts_used = _allele_counts_used_from_design(wd.X, ref_allele=ref)

    meta: Dict[str, Any] = {
        "ref_allele": ref,
        "alt_alleles": list(wd.alt_alleles),
        "all_alleles_in_lookup": sorted(region_lookup.keys()),
        # Backward-compatible key: default to the used counts (truth for regression)
        "allele_counts": allele_counts_used,
        "allele_counts_used": allele_counts_used,
        "allele_counts_raw": allele_counts_raw,
        "n_unambiguous_used": int(len(wd.regions)),
    }

    return RegressionPayload(
        kind="AFF",
        motif_pos=motif_pos,
        snv_index=snv_index,
        ref_allele=ref,
        local_kmer=local_kmer,
        wildcard_kmer=(
            "".join(
                "." if i == snv_index else (b if mask_pattern[i] == "1" else "x")
                for i, b in enumerate(local_kmer)
            )
            if mask_pattern is not None
            else _wildcard_kmer(local_kmer, snv_index)
        ),
        X=wd.X,
        y=wd.y,
        meta=meta,
    )


def build_aff_payloads_for_snv(
    snv: SnvWindow,
    *,
    matcher: MotifMatcher,
    design: DesignBuilder,
    k: int = 8,
    include_covariates: bool = True,
    fold_half: bool = True,
    cache_matches: bool = True,
    max_probes: Optional[int] = None,
    seed: int = 0,
) -> Dict[Tuple[int, int], RegressionPayload]:
    """Return {(motif_pos, snv_index): payload} for all windows overlapping the SNV."""
    # Fast path: in SNV mode we only need the single wildcard position that
    # corresponds to the SNV for each overlapping window (8× less work than
    # the full scan).
    windows = list(snv.iter_overlapping_windows())
    if hasattr(matcher, "filter_windows"):
        windows = list(matcher.filter_windows(windows))
    if hasattr(matcher, "scan_snv"):
        matches = matcher.scan_snv(snv.seq, k, windows)
    else:
        matches = matcher.scan(snv.seq, k)
    mask_pattern = getattr(matcher, "mask_pattern", None)
    out: Dict[Tuple[int, int], RegressionPayload] = {}
    for motif_pos, snv_index_in_kmer in windows:
        region_lookup = matches.allele_region_offsets.get(motif_pos, {}).get(
            snv_index_in_kmer, {}
        )
        region_lookup = subsample_probes(
            region_lookup, max_probes, seed,
            ("AFF", snv.seq, motif_pos, snv_index_in_kmer))
        payload = build_aff_payload(
            region_seq=snv.seq,
            motif_pos=motif_pos,
            snv_index=snv_index_in_kmer,
            region_lookup=region_lookup,
            design=design,
            k=k,
            include_covariates=include_covariates,
            fold_half=fold_half,
            mask_pattern=mask_pattern,
        )
        if payload is not None:
            out[(motif_pos, snv_index_in_kmer)] = payload
    return out


def build_aff_payloads_for_region_scan(
    region: RegionWindow,
    *,
    matcher: MotifMatcher,
    design: DesignBuilder,
    k: int = 8,
    include_covariates: bool = True,
    fold_half: bool = True,
) -> Dict[Tuple[int, int], RegressionPayload]:
    """Return {(motif_pos, snv_index): payload} for every window in a scan region."""
    matches = matcher.scan(region.seq, k)
    out: Dict[Tuple[int, int], RegressionPayload] = {}
    for motif_pos in sorted(matches.allele_region_offsets.keys()):
        for snv_index in sorted(matches.allele_region_offsets[motif_pos].keys()):
            region_lookup = matches.allele_region_offsets[motif_pos][snv_index]
            payload = build_aff_payload(
                region_seq=region.seq,
                motif_pos=motif_pos,
                snv_index=snv_index,
                region_lookup=region_lookup,
                design=design,
                k=k,
                include_covariates=include_covariates,
                fold_half=fold_half,
            )
            if payload is not None:
                out[(motif_pos, snv_index)] = payload
    return out


def build_rand_payloads_for_snv(
    *,
    snv: SnvWindow,
    snv_str: str,
    matcher: MotifMatcher,
    design: DesignBuilder,
    k: int = 8,
    rand_n: int = 500,
    fold_half: bool = True,
    min_n: int = 10,
    max_probes: Optional[int] = None,
    seed: int = 0,
) -> Dict[int, RegressionPayload]:
    """Build RAND regression inputs for each motif position (0..k-1).

    This mirrors the data path used by analyze_snv() (sampling excluded regions from AFF).
    It returns payloads *without fitting*.
    """
    windows = list(snv.iter_overlapping_windows())
    if hasattr(matcher, "filter_windows"):
        windows = list(matcher.filter_windows(windows))
    if hasattr(matcher, "scan_snv"):
        matches = matcher.scan_snv(snv.seq, k, windows)
    else:
        matches = matcher.scan(snv.seq, k)

    wildcard_index_by_pos = {int(m): int(j) for m, j in windows}
    aff_regions_per_allele: Dict[int, Dict[str, Dict[str, int]]] = {
        j: {} for j in range(k)
    }
    matched_regions_all: set[str] = set()
    for motif_pos, snv_index_in_kmer in windows:
        region_lookup = matches.allele_region_offsets.get(motif_pos, {}).get(
            snv_index_in_kmer, {}
        )
        if not region_lookup:
            continue
        aff_regions_per_allele[motif_pos] = {
            a: dict(m) for a, m in region_lookup.items()
        }
        for m in region_lookup.values():
            matched_regions_all.update(m.keys())

    sampler = (
        matcher.make_rand_sampler()
        if hasattr(matcher, "make_rand_sampler")
        else RandSampler(matcher.kmer_positions)
    )
    sample_kwargs = dict(
        matched_regions=matched_regions_all, snv_str=snv_str, k=k, rand_n=rand_n
    )
    if getattr(matcher, "is_masked", False):
        sample_kwargs["positions"] = wildcard_index_by_pos
    sample = sampler.sample(**sample_kwargs)

    payloads: Dict[int, RegressionPayload] = {}
    alleles = ["A", "C", "G", "T"]

    for j in range(k):
        hit_sets = aff_regions_per_allele.get(j, {}) or {}

        hit_sets = subsample_probes(
            hit_sets, max_probes, seed, ("RAND", snv_str, j))

        union_counts = {}
        for a in alleles:
            amap = hit_sets.get(a) or {}
            for r in amap.keys():
                union_counts[r] = union_counts.get(r, 0) + 1

        ambiguous = {r for r, c in union_counts.items() if c > 1}
        if ambiguous:
            hit_sets = {
                a: {
                    r: off
                    for r, off in (hit_sets.get(a) or {}).items()
                    if r not in ambiguous
                }
                for a in alleles
            }
        bg_map = sample.by_pos.get(j, {}) or {}

        hit_regions = set()
        for a in alleles:
            hit_regions.update(set((hit_sets.get(a) or {}).keys()))

        all_regions = list(hit_regions | set(bg_map.keys()))
        # stable order helps notebook debugging
        all_regions.sort()

        rows = []
        y = []
        row_meta = []
        for region in all_regions:
            if region not in design.intensities:
                continue
            val = design.intensities[region]
            if val is None or not np.isfinite(val):
                continue

            # region length (cached)
            from .design import region_length

            length = int(region_length(region))

            # Determine if this is a hit row and which allele (if hit).
            hit_allele: Optional[str] = None
            wildcard_pos: Optional[int] = None
            for a in alleles:
                amap = hit_sets.get(a, {}) or {}
                if region in amap:
                    hit_allele = a
                    wildcard_pos = int(amap[region])
                    break

            if wildcard_pos is None:
                # background row: we only know offset -> wildcard_pos for lp
                offset = bg_map.get(region, None)
                if offset is None:
                    continue
                if getattr(sample, "values_are_wildcard_pos", False):
                    wildcard_pos = int(offset)
                else:
                    # Legacy contiguous RAND stores k-mer offsets.
                    from .matching import wildcard_pos_from_offset

                    wildcard_pos = wildcard_pos_from_offset(
                        offset=int(offset), j=j, kmer_size=k, is_reverse=False
                    )

            lp_val = (float(wildcard_pos) / float(length)) if length > 0 else 0.5
            if fold_half:
                from .design import fold_lp_half

                lp_val = fold_lp_half(lp_val)

            row = {
                a: (1.0 if region in (hit_sets.get(a, {}) or {}) else 0.0)
                for a in alleles
            }
            row["lp"] = float(lp_val)
            row["sl"] = int(length)
            rows.append(row)
            y.append(float(np.round(val)))
            row_meta.append(
                {
                    "region": region,
                    "group": "HIT" if hit_allele is not None else "BG",
                    "allele": hit_allele,
                    "wildcard_pos": int(wildcard_pos),
                    "length": int(length),
                }
            )

        if len(y) < min_n:
            continue

        X = pd.DataFrame(rows, index=[m["region"] for m in row_meta])
        y_s = pd.Series(y, index=X.index, dtype=float)

        local_kmer = snv.seq[j : j + k]
        if len(local_kmer) != k:
            local_kmer = ""

        meta: Dict[str, Any] = {
            "n_total": int(len(y_s)),
            "n_bg": int(sum(1 for m in row_meta if m["group"] == "BG")),
            "n_hit": int(sum(1 for m in row_meta if m["group"] == "HIT")),
            "allele_counts": {
                a: int(float(X[a].sum())) if a in X.columns else 0 for a in alleles
            },
            "row_meta": pd.DataFrame(row_meta).set_index("region")
            if row_meta
            else pd.DataFrame(),
        }

        payloads[j] = RegressionPayload(
            kind="RAND",
            motif_pos=j,
            snv_index=None,
            ref_allele=None,
            local_kmer=local_kmer,
            wildcard_kmer=None,
            X=X,
            y=y_s,
            meta=meta,
        )

    return payloads


def probe_diagnostics(payload: Optional[RegressionPayload], allele: str) -> Dict[str, Any]:
    """Compute diagnostic metrics for one payload + allele combination.

    Works for both AFF and RAND payloads. Returns a dict with three keys:

    n_probes   -- total rows in the model for this window.

    n_allele   -- number of probes for this specific allele.
                  For AFF: probes whose sequence matched this allele at the
                  wildcard position (the allele dummy == 1 rows).
                  For RAND: hit probes for this allele specifically
                  (from meta['allele_counts']).

    outlier_infl -- (max(y_allele) - median(y_other)) / (IQR(y_other) + 1e-6).
                    Measures how far the single highest probe in the allele
                    group sits above the other group's distribution. Large
                    values suggest one outlier probe is driving the effect
                    rather than a consistent shift across the group.
                    For AFF, 'other' is all non-allele rows.
                    For RAND, 'other' is the background (BG) probes only.

    Any metric that cannot be computed (e.g. empty groups, missing metadata)
    is returned as the string 'NA'.
    """
    if payload is None:
        return {"n_probes": "NA", "n_allele": "NA", "outlier_infl": "NA"}

    X = payload.X
    y = payload.y.values
    n_probes = len(y)

    if payload.kind == "RAND":
        # For RAND, use allele_counts from meta for n_allele and row_meta for
        # the BG mask — this gives a cleaner "hit allele vs background" comparison
        # than treating all other allele dummies as the reference group.
        n_allele = int((payload.meta.get("allele_counts") or {}).get(allele, 0))

        row_meta = payload.meta.get("row_meta")
        if row_meta is None or row_meta.empty or n_allele == 0:
            return {"n_probes": n_probes, "n_allele": n_allele, "outlier_infl": "NA"}

        allele_mask = (X[allele].values == 1.0) if allele in X.columns \
                      else np.zeros(n_probes, dtype=bool)
        bg_mask = (row_meta.reindex(X.index)["group"] == "BG").values

    else:
        # AFF: allele dummy == 1 is the allele group; all-dummies-zero is the ref group.
        allele_cols = [c for c in X.columns if c in ("A", "C", "G", "T")]
        allele_mask = (X[allele].values == 1.0) if allele in X.columns \
                      else np.zeros(n_probes, dtype=bool)
        bg_mask = (X[allele_cols].sum(axis=1) == 0).values if allele_cols \
                  else ~allele_mask
        n_allele = int(allele_mask.sum())

    n_other = int(bg_mask.sum())
    if n_allele == 0 or n_other == 0:
        return {"n_probes": n_probes, "n_allele": n_allele, "outlier_infl": "NA"}

    y_allele = y[allele_mask]
    y_other  = y[bg_mask]

    iqr_other    = float(np.percentile(y_other, 75) - np.percentile(y_other, 25))
    outlier_infl = round(
        (float(np.max(y_allele)) - float(np.median(y_other))) / (iqr_other + 1e-6),
        3,
    )

    return {"n_probes": n_probes, "n_allele": n_allele, "outlier_infl": outlier_infl}