"""Design matrix construction (unambiguous regions + covariates)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd


@lru_cache(maxsize=500_000)
def region_length(region: str) -> int:
    """Parse region string "chr:start-end" and return length (end-start).

    Cached because the same probe/region IDs recur across many windows/SNVs.
    """
    try:
        _, coords = region.split(":")
        start_s, end_s = coords.split("-")
        start, end = int(start_s), int(end_s)
        return max(1, end - start)
    except Exception:
        return 200


def fold_lp_half(lp_val: float) -> float:
    """Fold lp into [0,0.5] symmetric around 0.5 (Perl-style)."""
    return 1.0 - lp_val if lp_val > 0.5 else lp_val


def extract_covariates(
    regions: List[str],
    region_to_kmer_pos: Mapping[str, int],
    *,
    fold_half: bool = True,
) -> Tuple[List[float], List[int]]:
    """Compute lp (location proportion) and sl (sequence length) per region."""
    lp: List[float] = []
    sl: List[int] = []
    for region in regions:
        size = region_length(region)
        kmer_pos = int(region_to_kmer_pos.get(region, 0))
        lp_val = (kmer_pos / size) if size > 0 else 0.5
        if fold_half:
            lp_val = fold_lp_half(lp_val)
        lp.append(float(lp_val))
        sl.append(int(size))
    return lp, sl


@dataclass(frozen=True)
class WindowDesign:
    regions: List[str]
    ref_allele: str
    alt_alleles: List[str]
    X: pd.DataFrame
    y: pd.Series


class DesignBuilder:
    """Build a per-window design matrix from scan-style matches."""

    def __init__(self, intensities: Mapping[str, float]):
        self.intensities = intensities

    def build(
        self,
        *,
        region_lookup: Mapping[str, Mapping[str, int]],
        ref_allele: str,
        include_covariates: bool = True,
        fold_half: bool = True,
    ) -> Optional[WindowDesign]:
        """Build X/y.

        region_lookup: dict[allele] -> dict[region] = wildcard_pos
        """
        # region -> alleles present
        region_to_alleles: Dict[str, set] = {}
        region_to_pos: Dict[str, int] = {}
        for allele, regmap in region_lookup.items():
            for region, pos in regmap.items():
                region_to_alleles.setdefault(region, set()).add(allele)
                region_to_pos.setdefault(region, int(pos))

        regions = [r for r, a in region_to_alleles.items() if len(a) == 1 and r in self.intensities]
        if len(regions) < 10:
            return None

        region_allele = {r: next(iter(region_to_alleles[r])) for r in regions}
        all_alleles = sorted(set(region_allele.values()))
        alt_alleles = [a for a in all_alleles if a != ref_allele]
        if not alt_alleles:
            return None

        X = pd.DataFrame(0.0, index=regions, columns=alt_alleles, dtype=float)
        for r in regions:
            a = region_allele[r]
            if a in alt_alleles:
                X.at[r, a] = 1.0

        y = pd.Series([self.intensities[r] for r in regions], index=regions, dtype=float)

        if include_covariates:
            region_to_kmer_pos = {r: region_to_pos.get(r, 0) for r in regions}
            lp, sl = extract_covariates(regions, region_to_kmer_pos, fold_half=fold_half)
            X["lp"] = lp
            X["sl"] = sl

        return WindowDesign(regions=regions, ref_allele=ref_allele, alt_alleles=alt_alleles, X=X, y=y)
