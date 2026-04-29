"""High-level analysis orchestrators for scan and snv modes.

These functions stitch together:
  - MotifMatcher (scan-style matching)
  - DesignBuilder (unambiguous regions + covariates)
  - RegressionEngine (NB with retries/winsorize + fallback OLS)
  - Optional RAND sampling/regression

The point is that *scan* and *snv* share the same matching + fitting code.
SNV mode simply filters to the k windows overlapping the SNV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import math

from .design import DesignBuilder
from .matching import MatchResults, MotifMatcher
from .rand import RandRegressor, RandSampler
from .regression import RegressionEngine
from .sequence import RegionWindow, SnvWindow


@dataclass(frozen=True)
class AffResultRow:
    snv_str: str
    motif_pos: int
    snv_index: int
    allele: str
    coef: float
    pval: float
    ref: str
    label: str = "AFF"
    model: str = "nb"


def _ref_allele(region_seq: str, motif_pos: int, snv_index: int) -> str:
    idx = motif_pos + snv_index
    if idx < 0 or idx >= len(region_seq):
        raise ValueError("motif_pos/snv_index out of bounds")
    return region_seq[idx]


def run_aff_window(
    *,
    snv_str: str,
    region_seq: str,
    motif_pos: int,
    snv_index: int,
    region_lookup: Mapping[str, Mapping[str, int]],
    design: DesignBuilder,
    engine: RegressionEngine,
    mode: str = "nb",
    include_covariates: bool = True,
    fold_half: bool = True,
    k: int = 8,
    max_probes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fit AFF for one (motif_pos, snv_index) window and return Perl-table rows."""

    # Subsample probes proportionally across alleles if max_probes is set
    if max_probes is not None:
        total = sum(len(v) for v in region_lookup.values())
        if total > max_probes:
            rng = np.random.default_rng(abs(hash(str(sorted(region_lookup.keys())))) % 2**32)
            new_lookup = {}
            for a, regions in region_lookup.items():
                n_keep = max(1, round(max_probes * len(regions) / total))
                items = list(regions.items())
                sampled = rng.choice(len(items), size=min(n_keep, len(items)), replace=False)
                new_lookup[a] = dict(items[i] for i in sampled)
            region_lookup = new_lookup

    ref = _ref_allele(region_seq, motif_pos, snv_index)
    wd = design.build(
        region_lookup=region_lookup,
        ref_allele=ref,
        include_covariates=include_covariates,
        fold_half=fold_half,
    )
    if wd is None:
        return []
    fit = engine.fit(wd.X, wd.y, mode=mode)
    all_alleles = sorted(region_lookup.keys())
    alt_alleles = [a for a in all_alleles if a != ref]
    local_kmer = region_seq[motif_pos : motif_pos + k]
    wildcard = list(local_kmer)
    wildcard[snv_index] = "."
    wildcard_kmer = "".join(wildcard)
    rows: List[Dict[str, Any]] = []
    for a in ["A", "C", "G", "T"]:
        filled = list(local_kmer)
        filled[snv_index] = a
        filled_kmer = "".join(filled)
        if a == ref:
            rows.append(
                {
                    "snv_str": snv_str,
                    "wildcard_kmer": wildcard_kmer,
                    "filled_kmer": filled_kmer,
                    "motif_pos": motif_pos,
                    "snv_index": snv_index,
                    "absolute_pos": motif_pos + snv_index,
                    "allele": a,
                    "coef": "NA",
                    "pval": "NA",
                    "label": "AFF",
                    "model": fit.model,
                    "method": "na_ref",
                    "ref": ref,
                }
            )
            continue
        if a in all_alleles:
            rows.append(
                {
                    "snv_str": snv_str,
                    "wildcard_kmer": wildcard_kmer,
                    "filled_kmer": filled_kmer,
                    "motif_pos": motif_pos,
                    "snv_index": snv_index,
                    "absolute_pos": motif_pos + snv_index,
                    "allele": a,
                    "coef": float(fit.params.get(a, math.nan)),
                    "pval": float(fit.pvalues.get(a, math.nan)),
                    "label": "AFF",
                    "model": fit.model,
                    "method": fit.method,
                    "ref": ref,
                }
            )
        else:
            rows.append(
                {
                    "snv_str": snv_str,
                    "wildcard_kmer": wildcard_kmer,
                    "filled_kmer": filled_kmer,
                    "motif_pos": motif_pos,
                    "snv_index": snv_index,
                    "absolute_pos": motif_pos + snv_index,
                    "allele": a,
                    "coef": "NA",
                    "pval": "NA",
                    "label": "AFF",
                    "model": fit.model,
                    "method": "na_missing",
                    "ref": ref,
                }
            )
    return rows


def analyze_region_scan(
    *,
    region: RegionWindow,
    matcher: MotifMatcher,
    design: DesignBuilder,
    engine: RegressionEngine,
    k: int = 8,
    mode: str = "nb",
    include_covariates: bool = True,
    fold_half: bool = True,
    max_probes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Scan-mode analysis: run AFF for every (motif_pos, snv_index) in the region."""

    matches = matcher.scan(region.seq, k)
    out: List[Dict[str, Any]] = []
    for motif_pos in sorted(matches.allele_region_offsets.keys()):
        for snv_index in sorted(matches.allele_region_offsets[motif_pos].keys()):
            region_lookup = matches.allele_region_offsets[motif_pos][snv_index]
            rows = run_aff_window(
                snv_str=f"{region.chrom}:{region.start}-{region.end}",
                region_seq=region.seq,
                motif_pos=motif_pos,
                snv_index=snv_index,
                region_lookup=region_lookup,
                design=design,
                engine=engine,
                mode=mode,
                include_covariates=include_covariates,
                fold_half=fold_half,
                k=k,
                max_probes=max_probes,
            )
            out.extend(rows)
    return out


def analyze_snv(
    *,
    snv: SnvWindow,
    snv_str: str,
    matcher: MotifMatcher,
    design: DesignBuilder,
    engine: RegressionEngine,
    k: int = 8,
    mode: str = "nb",
    include_covariates: bool = True,
    fold_half: bool = True,
    rand_n: int = 500,
    max_probes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """SNV-mode analysis: AFF for overlapping windows + RAND for each motif_pos."""

    matches = matcher.scan(snv.seq, k)

    # AFF for the k overlapping windows
    aff_rows: List[Dict[str, Any]] = []

    # For RAND we want: aff_regions_per_allele[motif_pos][allele] = {region: wildcard_pos}
    aff_regions_per_allele: Dict[int, Dict[str, Dict[str, int]]] = {
        j: {} for j in range(k)
    }
    matched_regions_all: set[str] = set()

    for motif_pos, snv_index_in_kmer in snv.iter_overlapping_windows():
        region_lookup = matches.allele_region_offsets.get(motif_pos, {}).get(
            snv_index_in_kmer, {}
        )
        if not region_lookup:
            continue

        # stash for RAND
        aff_regions_per_allele[motif_pos] = {
            a: dict(m) for a, m in region_lookup.items()
        }
        for a, m in region_lookup.items():
            matched_regions_all.update(m.keys())

        aff_rows.extend(
            run_aff_window(
                snv_str=snv_str,
                region_seq=snv.seq,
                motif_pos=motif_pos,
                snv_index=snv_index_in_kmer,
                region_lookup=region_lookup,
                design=design,
                engine=engine,
                mode=mode,
                include_covariates=include_covariates,
                fold_half=fold_half,
                k=k,
                max_probes=max_probes,
            )
        )

    # Optionally skip RAND entirely (used by --no-rand).
    if rand_n <= 0:
        return aff_rows

    # RAND sampling/regression
    if rand_n <= 0:
        return aff_rows

    sampler = RandSampler(matcher.kmer_positions)
    sample = sampler.sample(
        matched_regions=matched_regions_all,
        snv_str=snv_str,
        k=k,
        rand_n=rand_n,
    )
    rand_reg = RandRegressor(design.intensities, engine=engine)
    rand_rows = rand_reg.fit(
        aff_regions_per_allele=aff_regions_per_allele,
        rand_regions_by_pos=sample.by_pos,
        snv_str=snv_str,
        k=k,
        mode=mode,
        fold_half=fold_half,
        max_probes=max_probes,
    )

    return aff_rows + rand_rows