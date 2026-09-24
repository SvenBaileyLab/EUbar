"""Probe input, ranks and k-mer lookup for motif discovery."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from .sequence import reverse_complement as reverse_complement

BASES = ("A", "C", "G", "T")


def canonical_kmer(seq: str) -> str:
    rc = reverse_complement(seq)
    return seq if seq <= rc else rc


def kmer_matches_pattern(kmer: str, pattern: str) -> bool:
    """Return True if `kmer` matches `pattern`, where '.' is a wildcard."""
    if len(kmer) != len(pattern):
        return False
    for kb, pb in zip(kmer, pattern):
        if pb == ".":
            continue
        if kb != pb:
            return False
    return True


def fg_indices_for_pattern(
    pattern: str,
    kmer_to_idx: Dict[str, np.ndarray],
    cache: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Union of probe indices for all k-mers matching `pattern` ('.' wildcard).

    If `cache` is provided, memoize results by pattern string.
    """
    if cache is not None:
        hit = cache.get(pattern)
        if hit is not None:
            return hit

    idx_list: List[np.ndarray] = []
    for kmer, idx in kmer_to_idx.items():
        # IMPORTANT:
        # When kmers are stored in canonical (combine_revcomp) form, a query
        # pattern may match either the stored kmer *or* its reverse complement.
        # If we only test the stored string, revcomp-enabled runs can miss
        # legitimate matches and fail to extend motifs.
        if kmer_matches_pattern(kmer, pattern) or kmer_matches_pattern(
            reverse_complement(kmer), pattern
        ):
            idx_list.append(idx)

    if not idx_list:
        out = np.array([], dtype=int)
    elif len(idx_list) == 1:
        out = idx_list[0]
    else:
        out = np.unique(np.concatenate(idx_list))

    if cache is not None:
        cache[pattern] = out
    return out


def read_intensities(path: str) -> Dict[str, float]:
    """Reads intensities into {region: float}.

    Accepts either:
      region value
    or:
      chrom start end value [extra]
    """
    intensities: Dict[str, float] = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split()
            if len(parts) < 2:
                continue

            if ":" in parts[0] and "-" in parts[0] and len(parts) >= 2:
                region = parts[0]
                try:
                    val = float(parts[1])
                except ValueError:
                    continue
            elif len(parts) >= 4:
                chrom, start, end = parts[0], parts[1], parts[2]
                region = f"{chrom}:{start}-{end}"
                try:
                    val = float(parts[3])
                except ValueError:
                    continue
            else:
                continue

            intensities[region] = val

    return intensities


def read_unique_kmer_positions(path: str) -> Dict[str, Dict[str, int]]:
    """Reads k-mer positions; retains only regions with a single occurrence (count==1).

    Input line format:
        kmer<TAB>region1;count1;offsets1,region2;count2;offsets2,...

    Output:
        {kmer: {region: offset_int}}
    """
    out: Dict[str, Dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                kmer, entries = line.split("\t")
            except ValueError:
                continue
            region_map: Dict[str, int] = {}
            for entry in entries.strip(",").split(","):
                try:
                    region, count, offsets = entry.split(";")
                except ValueError:
                    continue
                try:
                    if int(count) != 1:
                        continue
                except ValueError:
                    continue
                # offsets can be space-separated, but count==1 so it should be a single int
                off = offsets.strip().split()
                if len(off) != 1:
                    continue
                try:
                    region_map[region] = int(off[0])
                except ValueError:
                    continue
            if region_map:
                out[kmer] = region_map
    return out


@dataclass
class ProbeRanks:
    regions: List[str]  # index -> region
    scores: np.ndarray  # index -> intensity
    order: np.ndarray  # indices sorted by score desc
    region_to_i: Dict[str, int]  # region -> index
    i_to_rank: np.ndarray  # index -> rank where 0 is best/highest intensity


def build_probe_ranks(intensities: Dict[str, float]) -> ProbeRanks:
    regions = list(intensities.keys())
    scores = np.array([float(intensities[r]) for r in regions], dtype=float)

    # rank: 0 = best/highest intensity
    order = np.argsort(-scores)
    i_to_rank = np.empty(len(scores), dtype=int)
    i_to_rank[order] = np.arange(len(scores), dtype=int)

    region_to_i = {r: i for i, r in enumerate(regions)}

    return ProbeRanks(
        regions=regions,
        scores=scores,
        order=order,
        region_to_i=region_to_i,
        i_to_rank=i_to_rank,
    )


def build_kmer_to_probe_idx(
    kmer_positions: Dict[str, Dict[str, int]],
    region_to_i: Dict[str, int],
    combine_revcomp: bool,
) -> Dict[str, np.ndarray]:
    """Return {kmer_key: np.ndarray(probe_indices)}.

    If combine_revcomp=True, use canonical(kmer) as the key and union probe sets
    across kmer and its reverse complement.

    If False, keep kmers as-is.
    """
    tmp: Dict[str, set] = {}

    for kmer, region_map in kmer_positions.items():
        key = canonical_kmer(kmer) if combine_revcomp else kmer
        s = tmp.setdefault(key, set())
        for region in region_map.keys():
            i = region_to_i.get(region)
            if i is not None:
                s.add(i)

    return {k: np.fromiter(sorted(v), dtype=int) for k, v in tmp.items() if v}
