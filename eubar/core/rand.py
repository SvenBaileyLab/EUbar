"""RAND sampling + regression.

Design:
- For each motif position j (0..k-1), build rows consisting of:
  * motif-hit regions (one-hot A/C/G/T)
  * random/background regions (all allele dummies = 0)
- Covariates:
  lp = wildcard_pos / region_length (folded to [0,0.5])
  sl = region_length

The main reason RAND explodes is fit stability; we route fitting through
RegressionEngine (winsorize/retry/fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set

import hashlib
import numpy as np
import pandas as pd

from .design import fold_lp_half
from .matching import wildcard_pos_from_offset
from .regression import RegressionEngine


def deterministic_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


@dataclass(frozen=True)
class RandSample:
    by_pos: Mapping[int, Mapping[str, int]]  # motif_pos -> {region: offset}
    used_all: Set[str]


class RandSampler:
    def __init__(self, kmers: Mapping[str, Mapping[str, int]]):
        self.kmers = kmers
        self._kmer_keys = list(kmers.keys())

    def sample(
        self,
        *,
        matched_regions: Set[str],
        snv_str: str,
        k: int,
        rand_n: int = 500,
    ) -> RandSample:
        import random

        rng = random.Random(deterministic_hash(snv_str) + 1337)
        # Cache kmer -> list(region) so we don't rebuild lists on every attempt.
        # This is a big win because RAND sampling may perform many thousands of attempts.
        region_key_cache: Dict[str, List[str]] = {}
        by_pos: Dict[int, Dict[str, int]] = {j: {} for j in range(k)}
        used_all: Set[str] = set()

        for j in range(k):
            used_j: Set[str] = set()
            attempts = 0
            max_attempts = max(10000, rand_n * 200)

            while len(used_j) < rand_n and attempts < max_attempts:
                attempts += 1
                kmer = rng.choice(self._kmer_keys)
                region_map = self.kmers.get(kmer)
                if not region_map:
                    continue

                keys = region_key_cache.get(kmer)
                if keys is None:
                    # Convert once; keep as list for random.choice
                    keys = list(region_map.keys())
                    region_key_cache[kmer] = keys
                if not keys:
                    continue

                region = rng.choice(keys)
                if region in matched_regions or region in used_all or region in used_j:
                    continue

                by_pos[j][region] = int(region_map[region])
                used_j.add(region)
                used_all.add(region)

        return RandSample(by_pos=by_pos, used_all=used_all)


class RandRegressor:
    def __init__(
        self,
        intensities: Mapping[str, float],
        *,
        engine: Optional[RegressionEngine] = None,
    ):
        self.intensities = intensities
        self.engine = engine or RegressionEngine()

    def fit(
        self,
        *,
        aff_regions_per_allele: Mapping[int, Mapping[str, Mapping[str, int]]],
        rand_regions_by_pos: Mapping[int, Mapping[str, int]],
        snv_str: str,
        k: int,
        mode: str = "nb",
        fold_half: bool = True,
        min_n: int = 10,
        max_probes: Optional[int] = None,
    ) -> List[Dict]:
        results: List[Dict] = []
        alleles = ["A", "C", "G", "T"]

        for j in range(k):
            hit_sets = aff_regions_per_allele.get(j, {}) or {}
            bg_map = rand_regions_by_pos.get(j, {}) or {}

            # Subsample hit probes proportionally across alleles if max_probes set
            if max_probes is not None:
                total_hits = sum(len(v) for v in hit_sets.values())
                if total_hits > max_probes:
                    rng = np.random.default_rng(abs(hash(snv_str + str(j))) % 2**32)
                    new_hit_sets = {}
                    for a, regions in hit_sets.items():
                        n_keep = max(1, round(max_probes * len(regions) / total_hits))
                        items = list(regions.items())
                        sampled = rng.choice(len(items), size=min(n_keep, len(items)), replace=False)
                        new_hit_sets[a] = dict(items[i] for i in sampled)
                    hit_sets = new_hit_sets

            hit_regions: Set[str] = set()
            for a in alleles:
                hit_regions.update(set((hit_sets.get(a) or {}).keys()))

            all_regions = hit_regions | set(bg_map.keys())

            rows = []
            y = []
            for region in all_regions:
                if region not in self.intensities:
                    continue
                val = self.intensities[region]
                if val is None or not np.isfinite(val):
                    continue

                # length
                try:
                    _, coords = region.split(":")
                    start_s, end_s = coords.split("-")
                    start, end = int(start_s), int(end_s)
                    length = max(1, end - start)
                except Exception:
                    length = 200

                # wildcard position
                wildcard_pos = None
                for a in alleles:
                    amap = hit_sets.get(a, {}) or {}
                    if region in amap:
                        wildcard_pos = int(amap[region])
                        break

                if wildcard_pos is None:
                    offset = bg_map.get(region, None)
                    if offset is None:
                        continue
                    wildcard_pos = wildcard_pos_from_offset(
                        offset=int(offset), j=j, kmer_size=k, is_reverse=False
                    )

                lp_val = (wildcard_pos / length) if length > 0 else 0.5
                if fold_half:
                    lp_val = fold_lp_half(lp_val)

                row = {
                    a: (1.0 if region in (hit_sets.get(a, {}) or {}) else 0.0)
                    for a in alleles
                }
                row["lp"] = float(lp_val)
                row["sl"] = int(length)
                rows.append(row)
                y.append(float(np.round(val)))

            if len(y) < min_n:
                continue

            X = pd.DataFrame(rows)
            y_s = pd.Series(y)

            fit = self.engine.fit(X, y_s, mode=mode)

            for a in alleles:
                results.append(
                    {
                        "snv_str": snv_str,
                        "motif_pos": j,
                        "allele": a,
                        "coef": float(fit.params.get(a, 0.0)),
                        "pval": float(fit.pvalues.get(a, np.nan))
                        if a in fit.pvalues
                        else float("nan"),
                        "label": "RAND",
                        "model": fit.model,
                        "method": fit.method,
                        "n_total": int(len(y)),
                        "n_hit": int(X[a].sum()) if a in X.columns else 0,
                        "n_bg": int(len(bg_map)),
                    }
                )

        return results