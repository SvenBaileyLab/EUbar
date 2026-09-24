from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core.masking import SequenceMask, _signature_codes
from .core.sequence import _open_fasta

try:
    import logomaker
except Exception:
    logomaker = None


BASES = ("A", "C", "G", "T")
STANDARD_DNA_COLORS = {"A": "#008000", "C": "#0000ff", "G": "#ffa600", "T": "#ff0000"}
PRETTY_DNA_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}


def _dna_colors(pretty_logo: bool) -> dict:
    return PRETTY_DNA_COLORS if pretty_logo else STANDARD_DNA_COLORS


def reverse_complement(seq: str) -> str:
    tbl = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(tbl)[::-1]


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


def _mask_code_to_bases(code: int, n_bases: int) -> str:
    """Decode a 2-bit masked-signature code back to A/C/G/T bases."""
    chars = ["A"] * int(n_bases)
    x = int(code)
    for i in range(int(n_bases) - 1, -1, -1):
        chars[i] = BASES[x & 3]
        x >>= 2
    return "".join(chars)


def _mask_pattern_from_code(mask: SequenceMask, code: int) -> str:
    """Render a masked signature as a full-span pattern using '.' for spacers."""
    informative = _mask_code_to_bases(code, mask.n_informative)
    out = ["."] * mask.span
    for base, pos in zip(informative, mask.informative):
        out[int(pos)] = base
    return "".join(out)


def _mask_seed_to_code(seed: str, mask: SequenceMask) -> int:
    """Parse a forced masked seed into a signature code.

    Accepts either:
      * a full-span string, e.g. ``TTGGC....GCCAA`` for a 14-bp mask, or
      * only the informative bases, e.g. ``TTGGCGCCAA`` for a 10-one mask.

    Characters at ignored mask positions are not used. Informative positions
    must contain A/C/G/T.
    """
    raw = str(seed).strip().upper()
    if len(raw) == mask.n_informative:
        informative = raw
    elif len(raw) == mask.span:
        informative = "".join(raw[int(pos)] for pos in mask.informative)
    else:
        raise ValueError(
            f"masked --seed must have length {mask.n_informative} (informative only) "
            f"or {mask.span} (full mask span); got {len(raw)}"
        )
    if any(base not in BASES for base in informative):
        raise ValueError("masked --seed must use A/C/G/T at every informative mask position")
    code = 0
    base_to_code = {b: i for i, b in enumerate(BASES)}
    for base in informative:
        code = (code << 2) | base_to_code[base]
    return int(code)


def _region_coords(region: str) -> Tuple[str, int, int]:
    chrom, coords = str(region).split(":", 1)
    start_s, end_s = coords.split("-", 1)
    return chrom, int(start_s), int(end_s)


def collect_mask_seed_candidates(
    *,
    mask: SequenceMask,
    ranks: ProbeRanks,
    genome_fasta: str,
    top_probe_count: int,
    max_candidates: int,
) -> List[int]:
    """Collect masked signature candidates from the highest-intensity probes.

    This is deliberately a *candidate-generation* step, not the final score.
    Candidate signatures are subsequently rescored against the entire probe
    universe with the same rank-based E-score used by ordinary ``eubar motifs``.
    """
    n_top = min(max(1, int(top_probe_count)), len(ranks.regions))
    max_candidates = max(1, int(max_candidates))
    fasta = _open_fasta(genome_fasta)
    counts: Counter = Counter()

    for probe_i in ranks.order[:n_top]:
        region = ranks.regions[int(probe_i)]
        try:
            chrom, start0, end0 = _region_coords(region)
            seq = fasta[chrom][start0:end0].seq.upper()
        except Exception:
            continue
        if len(seq) < mask.span:
            continue
        f_codes, r_codes, valid_f, valid_r = _signature_codes(seq, mask)
        # Count presence per probe, rather than repeated windows within one probe.
        local = set()
        if len(f_codes):
            local.update(int(x) for x in f_codes[valid_f])
            local.update(int(x) for x in r_codes[valid_r])
        counts.update(local)

    if not counts:
        return []
    return [int(code) for code, _ in counts.most_common(max_candidates)]


def _hits_to_probe_idx(
    hits: Dict[str, object], region_to_i: Dict[str, int]
) -> np.ndarray:
    vals = sorted(
        {
            int(region_to_i[r])
            for r in hits.keys()
            if r in region_to_i
        }
    )
    return np.asarray(vals, dtype=int)


def scan_mask_codes_for_motifs(
    *,
    mask: SequenceMask,
    target_codes: Sequence[int],
    probe_regions: Sequence[str],
    genome_fasta: str,
) -> Tuple[Dict[int, Dict[str, int]], Dict[str, int]]:
    """Scan masked signatures for motif discovery.

    A signature is retained when it occurs at one unique *physical window* in a
    probe region. Forward and reverse matches at the same offset count as the
    same occurrence. This matters for palindromic/bipartite motifs such as NFIC:
    a self-reverse-complement signature should not be discarded merely because
    both strand orientations describe the same genomic window.
    """
    targets = {int(x) for x in target_codes}
    hits: Dict[int, Dict[str, int]] = {code: {} for code in targets}
    ambiguous: Dict[int, set] = {code: set() for code in targets}
    fasta = _open_fasta(genome_fasta)
    n_valid_regions = 0
    n_windows = 0

    for region in probe_regions:
        try:
            chrom, start0, end0 = _region_coords(region)
            seq = fasta[chrom][start0:end0].seq.upper()
        except Exception:
            continue
        if len(seq) < mask.span:
            continue
        n_valid_regions += 1
        f_codes, r_codes, valid_f, valid_r = _signature_codes(seq, mask)
        n = len(f_codes)
        n_windows += int(n)
        local: Dict[int, set] = defaultdict(set)
        for off0 in range(n):
            if bool(valid_f[off0]):
                code = int(f_codes[off0])
                if code in targets:
                    local[code].add(int(off0))
            if bool(valid_r[off0]):
                code = int(r_codes[off0])
                if code in targets:
                    local[code].add(int(off0))
        for code, offsets in local.items():
            if len(offsets) == 1:
                if region not in ambiguous[code]:
                    hits[code][region] = next(iter(offsets)) + 1
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


def choose_mask_seed(
    *,
    mask: SequenceMask,
    candidate_codes: Sequence[int],
    target_hits: Dict[int, Dict[str, object]],
    ranks: ProbeRanks,
    min_F: int,
) -> pd.DataFrame:
    """Score masked signature candidates and return them best-first."""
    rows: List[dict] = []
    N_total = len(ranks.scores)
    for code in candidate_codes:
        idx = _hits_to_probe_idx(target_hits.get(int(code), {}), ranks.region_to_i)
        F = int(len(idx))
        if F < int(min_F):
            continue
        E = escore_auc_minus_half_all_bg(idx, ranks.i_to_rank, N_total)
        rows.append(
            {
                "kmer": _mask_pattern_from_code(mask, int(code)),
                "code": int(code),
                "E": float(E),
                "F": F,
                "gaps": int(mask.span - mask.n_informative),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values(["E", "F", "kmer"], ascending=[False, False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _variant_code_for_informative_base(
    seed_bases: str, informative_index: int, base: str
) -> int:
    chars = list(seed_bases)
    chars[int(informative_index)] = str(base)
    base_to_code = {b: i for i, b in enumerate(BASES)}
    code = 0
    for ch in chars:
        code = (code << 2) | base_to_code[ch]
    return int(code)


def ppm_from_mask_seed_wobble(
    *,
    mask: SequenceMask,
    seed_code: int,
    variant_hits: Dict[int, Dict[str, object]],
    ranks: ProbeRanks,
    min_per_base: int,
    beta: float,
    min_support: int = 1,
    pseudocount: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a full-span PPM by wobbling only informative mask positions.

    Ignored spacer positions are represented as uniform 0.25 probabilities,
    which gives zero information content in the bits logo.
    """
    seed_bases = _mask_code_to_bases(seed_code, mask.n_informative)
    reduced_rows: List[Dict] = []
    ppm_rows: List[pd.Series] = []
    informative_to_rank = {int(pos): i for i, pos in enumerate(mask.informative)}

    for pos in range(mask.span):
        if pos not in informative_to_rank:
            ppm_rows.append(
                pd.Series({b: 0.25 for b in BASES}, name=pos, dtype=float)
            )
            for base in BASES:
                reduced_rows.append(
                    {
                        "pos": pos,
                        "base": base,
                        "variant": mask.pattern,
                        "E_reduced": 0.0,
                        "p": np.nan,
                        "F": 0,
                        "B": 0,
                        "masked_spacer": True,
                    }
                )
            continue

        j = int(informative_to_rank[pos])
        variants: Dict[str, np.ndarray] = {}
        codes: Dict[str, int] = {}
        for base in BASES:
            code = _variant_code_for_informative_base(seed_bases, j, base)
            codes[base] = code
            variants[base] = _hits_to_probe_idx(
                variant_hits.get(code, {}), ranks.region_to_i
            )

        _, clean_sets = reduced_test_four_variants(
            variants, min_per_base=int(min_per_base)
        )
        support = {b: int(len(clean_sets[b])) for b in BASES}
        good_bases = [b for b in BASES if support[b] >= int(min_per_base)]

        if len(good_bases) < 2:
            for base in BASES:
                reduced_rows.append(
                    {
                        "pos": pos,
                        "base": base,
                        "variant": _mask_pattern_from_code(mask, codes[base]),
                        "E_reduced": np.nan,
                        "p": np.nan,
                        "F": support[base],
                        "B": 0,
                        "masked_spacer": False,
                    }
                )
            supported = [b for b in BASES if support[b] >= int(min_support)]
            if len(supported) == 1:
                probs = pd.Series({b: 0.0 for b in BASES}, dtype=float)
                probs[supported[0]] = 1.0
            else:
                seed_base = seed_bases[j]
                probs = pd.Series({b: 0.0 for b in BASES}, dtype=float)
                probs[seed_base] = 1.0
            if pseudocount > 0:
                probs = probs + float(pseudocount)
                probs = probs / float(probs.sum())
            ppm_rows.append(pd.Series(probs, name=pos))
            continue

        pos_es: Dict[str, float] = {}
        for base in BASES:
            fg = clean_sets[base]
            if int(len(fg)) < int(min_per_base):
                pos_es[base] = np.nan
                reduced_rows.append(
                    {
                        "pos": pos,
                        "base": base,
                        "variant": _mask_pattern_from_code(mask, codes[base]),
                        "E_reduced": np.nan,
                        "p": np.nan,
                        "F": int(len(fg)),
                        "B": 0,
                        "masked_spacer": False,
                    }
                )
                continue
            bg_parts = [
                clean_sets[x]
                for x in BASES
                if x != base and int(len(clean_sets[x])) > 0
            ]
            bg = np.concatenate(bg_parts) if bg_parts else np.array([], dtype=int)
            E_red, p = reduced_escore_and_p(fg, bg, ranks.i_to_rank)
            pos_es[base] = E_red
            reduced_rows.append(
                {
                    "pos": pos,
                    "base": base,
                    "variant": _mask_pattern_from_code(mask, codes[base]),
                    "E_reduced": E_red,
                    "p": p,
                    "F": int(len(fg)),
                    "B": int(len(bg)),
                    "masked_spacer": False,
                }
            )

        es_series = pd.Series(pos_es, dtype=float)
        es_filled = es_series.reindex(BASES).copy()
        es_filled[~np.isfinite(es_filled)] = -0.5
        probs = softmax_from_escores(es_filled, beta=float(beta))
        for base in BASES:
            if base not in es_series or not np.isfinite(es_series.get(base, np.nan)):
                probs[base] = 0.0
        if float(probs.sum()) <= 0:
            probs = pd.Series({b: 0.0 for b in BASES}, dtype=float)
            probs[seed_bases[j]] = 1.0
        else:
            probs = probs / float(probs.sum())
        if pseudocount > 0:
            probs = probs + float(pseudocount)
            probs = probs / float(probs.sum())
        ppm_rows.append(pd.Series({b: float(probs[b]) for b in BASES}, name=pos))

    reduced_df = pd.DataFrame(reduced_rows)
    ppm = pd.DataFrame(ppm_rows)
    ppm.index = list(range(mask.span))
    ppm.index.name = "pos"
    ppm = ppm[list(BASES)]
    return reduced_df, ppm


def escore_auc_minus_half_all_bg(
    fg_idx: np.ndarray,
    i_to_rank: np.ndarray,
    N_total: int,
) -> float:
    """Compute E = AUC - 0.5 for fg vs all other probes.

    Uses i_to_rank (0=best) and a rank-combinatoric formula that avoids scanning bg.
    Assumes unique ranks.
    """
    F = int(len(fg_idx))
    if F == 0 or F == N_total:
        return float("nan")
    B = N_total - F

    r = np.sort(i_to_rank[fg_idx])  # smaller is better
    # For each fg rank r[i], count bg ranks worse than it:
    # total_worse = (N_total - 1 - r[i])
    # fg_worse = F - (i+1)
    # bg_worse = total_worse - fg_worse
    i = np.arange(1, F + 1, dtype=int)
    bg_worse = (N_total - 1 - r) - (F - i)
    auc = bg_worse.sum() / (F * B)
    return float(auc - 0.5)


def reduced_escore_and_p(
    fg_idx: np.ndarray,
    bg_idx: np.ndarray,
    i_to_rank: np.ndarray,
) -> Tuple[float, float]:
    """Reduced E-score comparing fg vs bg (both are explicit sets).

    Returns:
      (E_reduced, approx_two_sided_p)

    p-value uses a normal approximation to Mann-Whitney U (no scipy).
    """
    F = int(len(fg_idx))
    B = int(len(bg_idx))
    if F == 0 or B == 0:
        return float("nan"), float("nan")

    # Higher score = better intensity
    scores_fg = -(i_to_rank[fg_idx]).astype(float)
    scores_bg = -(i_to_rank[bg_idx]).astype(float)

    scores = np.concatenate([scores_fg, scores_bg])
    labels = np.concatenate([np.ones(F, dtype=int), np.zeros(B, dtype=int)])

    # ranks for unique scores
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    sum_ranks_fg = ranks[labels == 1].sum()
    U = sum_ranks_fg - F * (F + 1) / 2.0
    auc = U / (F * B)
    E = float(auc - 0.5)

    # Normal approximation p-value (two-sided)
    mu = F * B / 2.0
    var = F * B * (F + B + 1) / 12.0
    if var <= 0:
        p = float("nan")
    else:
        z = (U - mu) / math.sqrt(var)
        # two-sided using erfc
        p = float(math.erfc(abs(z) / math.sqrt(2.0)))

    return E, p


def reduced_test_four_variants(
    variants: Dict[str, np.ndarray],
    min_per_base: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Prepare disjoint fg sets for 4 variants by dropping ambiguous probes.

    variants:
      base -> probe_idx array

    Returns:
      df_counts with columns base,F
      clean_sets: base -> cleaned probe_idx array

    Ambiguous probes (present in >1 base set) are removed from all sets.
    """
    # probe -> count of variant membership
    counts: Dict[int, int] = {}
    for idx in variants.values():
        for i in idx:
            counts[int(i)] = counts.get(int(i), 0) + 1

    clean_sets: Dict[str, np.ndarray] = {}
    for b, idx in variants.items():
        keep = [int(i) for i in idx if counts.get(int(i), 0) == 1]
        clean_sets[b] = np.array(keep, dtype=int)

    rows = [{"base": b, "F": int(len(clean_sets[b]))} for b in BASES]
    df_counts = pd.DataFrame(rows)

    # enforce minimum coverage
    if (df_counts["F"] < min_per_base).any():
        # still return counts, but caller can stop
        return df_counts, clean_sets

    return df_counts, clean_sets


def softmax_from_escores(es: pd.Series, beta: float) -> pd.Series:
    x = (beta * es.astype(float)).to_numpy()
    x = x - np.nanmax(x)  # stability
    w = np.exp(x)
    w = w / w.sum()
    return pd.Series(w, index=es.index, dtype=float)


def ppm_from_seed_wobble(
    seed: str,
    kmer_to_idx: Dict[str, np.ndarray],
    i_to_rank: np.ndarray,
    min_per_base: int,
    beta: float,
    min_support: int = 1,
    pseudocount: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute an k-position PPM from reduced E-scores at each position.

    Returns:
      reduced_table (long)
      ppm (index=pos, cols=A,C,G,T)
    """
    k = len(seed)
    reduced_rows: List[Dict] = []

    ppm_rows: List[pd.Series] = []

    use_patterns = "." in seed
    pattern_cache: Dict[str, np.ndarray] = {}

    for pos in range(k):
        variants: Dict[str, np.ndarray] = {}
        for b in BASES:
            var = seed[:pos] + b + seed[pos + 1 :]
            if use_patterns:
                idx = fg_indices_for_pattern(var, kmer_to_idx, cache=pattern_cache)
            else:
                idx = kmer_to_idx.get(var)
                if idx is None:
                    idx = np.array([], dtype=int)
            variants[b] = idx

        counts_df, clean_sets = reduced_test_four_variants(
            variants, min_per_base=min_per_base
        )

        # Allow wobble even if one base is rare/absent (e.g. F=0 for T at pos=7).
        good_bases = [b for b in BASES if int(len(clean_sets[b])) >= min_per_base]

        # If fewer than 2 bases have enough support, fall back intelligently
        if len(good_bases) < 2:
            support = {b: int(len(clean_sets[b])) for b in BASES}
            for b in BASES:
                reduced_rows.append(
                    dict(
                        pos=pos,
                        base=b,
                        variant=seed[:pos] + b + seed[pos + 1 :],
                        E_reduced=np.nan,
                        p=np.nan,
                        F=support[b],
                        B=0,
                    )
                )

            # single-base support fallback (only one base has meaningful support)
            supported = [b for b in BASES if support[b] >= int(min_support)]
            if len(supported) == 1:
                dom = supported[0]
                probs = pd.Series({b: 0.0 for b in BASES}, dtype=float)
                probs[dom] = 1.0
            # seed-prior fallback if any seed base exists at this position
            elif pos < len(seed) and seed[pos] in BASES:
                dom = seed[pos]
                probs = pd.Series({b: 0.0 for b in BASES}, dtype=float)
                probs[dom] = 1.0
            else:
                probs = pd.Series({b: 0.25 for b in BASES}, dtype=float)

            if pseudocount > 0:
                probs = probs + float(pseudocount)
                probs = probs / float(probs.sum())

            ppm_rows.append(pd.Series({b: float(probs[b]) for b in BASES}, name=pos))
            continue

        # Otherwise compute E_reduced for supported bases; set unsupported bases to NaN and prob=0.
        pos_es: Dict[str, float] = {}
        pos_p: Dict[str, float] = {}

        for b in BASES:
            fg = clean_sets[b]
            if int(len(fg)) < min_per_base:
                # not enough FG probes for this base
                pos_es[b] = np.nan
                pos_p[b] = np.nan
                reduced_rows.append(
                    dict(
                        pos=pos,
                        base=b,
                        variant=seed[:pos] + b + seed[pos + 1 :],
                        E_reduced=np.nan,
                        p=np.nan,
                        F=int(len(fg)),
                        B=0,
                    )
                )
                continue

            bg_parts = [
                clean_sets[x] for x in BASES if x != b and int(len(clean_sets[x])) > 0
            ]
            bg = np.concatenate(bg_parts) if bg_parts else np.array([], dtype=int)

            E_red, p = reduced_escore_and_p(fg, bg, i_to_rank)
            pos_es[b] = E_red
            pos_p[b] = p
            reduced_rows.append(
                dict(
                    pos=pos,
                    base=b,
                    variant=seed[:pos] + b + seed[pos + 1 :],
                    E_reduced=E_red,
                    p=p,
                    F=int(len(fg)),
                    B=int(len(bg)),
                )
            )

        # Probabilities: softmax over available bases; unsupported bases get 0 weight.
        es_series = pd.Series(pos_es, dtype=float)
        es_filled = es_series.copy()
        es_filled[~np.isfinite(es_filled)] = -0.5  # worst possible E
        probs = softmax_from_escores(es_filled, beta=beta)

        for b in BASES:
            if not np.isfinite(es_series[b]):  # unsupported base
                probs[b] = 0.0

        s = float(probs.sum())
        if s <= 0:
            # fallback hierarchy: single-base support -> seed prior -> uniform
            support = {b: int(len(clean_sets[b])) for b in BASES}
            supported = [b for b in BASES if support[b] >= int(min_support)]
            if len(supported) == 1:
                dom = supported[0]
                probs = pd.Series({b: 0.0 for b in BASES}, dtype=float)
                probs[dom] = 1.0
            elif pos < len(seed) and seed[pos] in BASES:
                dom = seed[pos]
                probs = pd.Series({b: 0.0 for b in BASES}, dtype=float)
                probs[dom] = 1.0
            else:
                probs = pd.Series({b: 0.25 for b in BASES}, dtype=float)
        else:
            probs = probs / s

        if pseudocount > 0:
            probs = probs + float(pseudocount)
            probs = probs / float(probs.sum())

        ppm_rows.append(pd.Series({b: float(probs[b]) for b in BASES}, name=pos))

    reduced_df = pd.DataFrame(reduced_rows)
    ppm = pd.DataFrame(ppm_rows)
    ppm.index = list(range(k))
    ppm.index.name = "pos"

    ppm = ppm.loc[:, ~ppm.columns.duplicated()].copy()
    ppm = ppm[list(BASES)]
    return reduced_df, ppm


def choose_seed(
    kmer_to_idx: Dict[str, np.ndarray],
    i_to_rank: np.ndarray,
    min_F: int,
    max_gaps: int = 0,
) -> pd.DataFrame:
    """Compute E-scores for seed candidates.

    If max_gaps == 0: evaluates each exact k-mer.
    If max_gaps > 0: additionally evaluates '.'-gapped patterns of the same length
    (up to `max_gaps` wildcards), as described in Berger et al. Fig 3a.

    Returns a dataframe sorted by E desc, then fewer gaps, then support (F) desc.
    """

    N_total = len(i_to_rank)
    rows: List[dict] = []

    kmers = list(kmer_to_idx.keys())
    if not kmers:
        return pd.DataFrame(columns=["kmer", "E", "F", "gaps"])

    k = len(kmers[0])

    # --- exact k-mers ---
    for kmer, idx in kmer_to_idx.items():
        F = int(len(idx))
        if F < min_F:
            continue
        E = escore_auc_minus_half_all_bg(idx, i_to_rank, N_total)
        rows.append({"kmer": kmer, "E": E, "F": F, "gaps": 0})

    # --- gapped patterns ('.' wildcards) ---
    if max_gaps and max_gaps > 0:
        max_gaps = min(int(max_gaps), k)
        kmer_items = list(kmer_to_idx.items())

        for gaps in range(1, max_gaps + 1):
            for gap_pos in combinations(range(k), gaps):
                fixed_pos = [i for i in range(k) if i not in gap_pos]

                # Group exact kmers by the bases at fixed positions
                groups: Dict[str, List[np.ndarray]] = defaultdict(list)
                for kmer, idx in kmer_items:
                    key = "".join(kmer[i] for i in fixed_pos)
                    groups[key].append(idx)

                for key, idx_list in groups.items():
                    if len(idx_list) == 1:
                        fg_idx = idx_list[0]
                    else:
                        fg_idx = np.unique(np.concatenate(idx_list))

                    F = int(len(fg_idx))
                    if F < min_F:
                        continue

                    E = escore_auc_minus_half_all_bg(fg_idx, i_to_rank, N_total)

                    pat = ["."] * k
                    for pos, base in zip(fixed_pos, key):
                        pat[pos] = base
                    pat = "".join(pat)
                    rows.append({"kmer": pat, "E": E, "F": F, "gaps": gaps})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df.drop_duplicates(subset=["kmer"], inplace=True)
    df.sort_values(["E", "gaps", "F"], ascending=[False, True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def extend_side_greedy(
    seed: str,
    kmer_to_idx: Dict[str, np.ndarray],
    i_to_rank: np.ndarray,
    beta: float,
    min_per_base: int,
    side: str,
    max_steps: int,
    combine_revcomp: bool = False,
) -> Tuple[List[pd.Series], str]:
    """Greedy flank extension using reduced tests on shifted k-mers.

    side:
      - "left": variants are (b + window[:-1])
      - "right": variants are (window[1:] + b)

    Returns:
      flank_ppm: list of pd.Series of probabilities (each has index A,C,G,T)
                ordered from nearest-to-seed outward (so caller can reverse for left)
      final_window: the last k-mer window after extension
    """
    window = seed
    flank_ppm: List[pd.Series] = []

    for _ in range(max_steps):
        variants: Dict[str, np.ndarray] = {}
        if side == "left":
            anchor = window[:-1]
            for b in BASES:
                var = b + anchor
                key = canonical_kmer(var) if combine_revcomp else var
                variants[b] = kmer_to_idx.get(key, np.array([], dtype=int))
        elif side == "right":
            anchor = window[1:]
            for b in BASES:
                var = anchor + b
                key = canonical_kmer(var) if combine_revcomp else var
                variants[b] = kmer_to_idx.get(key, np.array([], dtype=int))
        else:
            raise ValueError("side must be 'left' or 'right'")

        counts_df, clean_sets = reduced_test_four_variants(
            variants, min_per_base=min_per_base
        )
        if (counts_df["F"] < min_per_base).any():
            break

        es: Dict[str, float] = {}
        for b in BASES:
            fg = clean_sets[b]
            bg = np.concatenate([clean_sets[x] for x in BASES if x != b])
            E_red, _ = reduced_escore_and_p(fg, bg, i_to_rank)
            es[b] = E_red

        probs = softmax_from_escores(pd.Series(es), beta=beta)
        flank_ppm.append(probs)

        best_base = str(probs.idxmax())
        if side == "left":
            window = best_base + window[:-1]
        else:
            window = window[1:] + best_base

    return flank_ppm, window


def _info_content(p: pd.Series, eps: float = 1e-12) -> float:
    """Information content proxy in [0,1]. 0 = uniform, 1 = deterministic."""
    v = np.array([float(p.get(b, 0.0)) for b in BASES], dtype=float)
    v = np.clip(v, eps, 1.0)
    v = v / v.sum()
    ent = -np.sum(v * np.log(v))
    ent_max = np.log(4.0)
    return float(1.0 - ent / ent_max)


def extend_side(
    window: str,
    window_probs: List[pd.Series],
    kmer_to_idx: Dict[str, np.ndarray],
    i_to_rank: np.ndarray,
    beta: float,
    min_per_base: int,
    side: str,
    max_steps: int,
    max_gaps: int,
    ic_stop_threshold: float,
    ic_stop_consecutive: int,
) -> Tuple[List[pd.Series], str, List[pd.Series], List[Dict]]:
    """Extend motif flanks while allowing gaps by dropping least-informative positions.

    This mirrors the intuition from Berger et al. Fig 3a: when shifting a k-mer window,
    you can maintain probe support by treating a low-information position as a wildcard ('.').

    Practical implementation here:
      - Extend one base at a time (left or right).
      - At each step, build 4 candidate k-mer *patterns* (with '.' wildcards allowed)
        that differ only in the newly added base.
      - If support is insufficient, progressively add wildcards at the least-informative
        anchor positions (up to max_gaps) and retry.
      - Stop when we still can't get enough support.

    Returns:
      flank_ppm    : list of probability vectors for the newly added positions (nearest->outward)
      final_window : final k-mer consensus window after extension
      final_probs  : updated list of probability vectors corresponding to final_window
      reduced_rows : list of dict rows (one per base per step) for a "reduced_full" table
                     (pos is filled in by the caller).
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    if len(window_probs) != len(window):
        raise ValueError("window_probs length must match window")

    flank_ppm: List[pd.Series] = []
    low_streak = 0

    reduced_rows: List[Dict] = []
    pattern_cache: Dict[str, np.ndarray] = {}

    for step in range(max_steps):
        if side == "right":
            anchor = list(window[1:])
            anchor_probs = window_probs[1:]
            anchor_len = len(anchor)
            order = sorted(
                range(anchor_len), key=lambda i: _info_content(anchor_probs[i])
            )

            def make_pattern(new_base: str, gaps: int) -> str:
                a = anchor.copy()
                for gi in range(min(gaps, anchor_len)):
                    a[order[gi]] = "."
                return "".join(a) + new_base

        else:  # left
            anchor = list(window[:-1])
            anchor_probs = window_probs[:-1]
            anchor_len = len(anchor)
            order = sorted(
                range(anchor_len), key=lambda i: _info_content(anchor_probs[i])
            )

            def make_pattern(new_base: str, gaps: int) -> str:
                a = anchor.copy()
                for gi in range(min(gaps, anchor_len)):
                    a[order[gi]] = "."
                return new_base + "".join(a)

        chosen_gaps: Optional[int] = None
        clean_sets: Optional[Dict[str, np.ndarray]] = None

        for gaps in range(0, min(max_gaps, anchor_len) + 1):
            variants: Dict[str, np.ndarray] = {}
            for b in BASES:
                pat = make_pattern(b, gaps)
                variants[b] = fg_indices_for_pattern(
                    pat, kmer_to_idx, cache=pattern_cache
                )

            counts_df, tmp_sets = reduced_test_four_variants(
                variants, min_per_base=min_per_base
            )
            if not (counts_df["F"] < min_per_base).any():
                chosen_gaps = gaps
                clean_sets = tmp_sets
                break

        if clean_sets is None or chosen_gaps is None:
            break  # can't extend further

        # Compute reduced E-scores for each base and build probs.
        es: Dict[str, float] = {}
        pvals: Dict[str, float] = {}
        F_sizes: Dict[str, int] = {}
        B_sizes: Dict[str, int] = {}

        for b in BASES:
            fg = clean_sets[b]
            bg = np.concatenate([clean_sets[x] for x in BASES if x != b])
            E_red, p = reduced_escore_and_p(fg, bg, i_to_rank)
            es[b] = E_red
            pvals[b] = p
            F_sizes[b] = int(len(fg))
            B_sizes[b] = int(len(bg))

        probs = softmax_from_escores(pd.Series(es), beta=beta)
        ic = _info_content(probs)

        # IC-based stop: prevent runaway extension when added positions are nearly uniform
        if ic_stop_threshold is not None and ic < ic_stop_threshold:
            if low_streak + 1 >= ic_stop_consecutive:
                break
            low_streak += 1
        else:
            low_streak = 0

        flank_ppm.append(probs)

        # Save rows (caller assigns the final motif position index)
        for b in BASES:
            reduced_rows.append(
                dict(
                    step=step,
                    side=side,
                    gaps_used=int(chosen_gaps),
                    base=b,
                    variant=make_pattern(b, int(chosen_gaps)),
                    E_reduced=float(es[b]),
                    p=float(pvals[b]),
                    F=int(F_sizes[b]),
                    B=int(B_sizes[b]),
                    IC=float(ic),
                )
            )

        best_base = str(probs.idxmax())

        # Update window and window_probs by shifting 1 and appending the new probs
        if side == "right":
            window = window[1:] + best_base
            window_probs = window_probs[1:] + [probs]
        else:
            window = best_base + window[:-1]
            window_probs = [probs] + window_probs[:-1]

    return flank_ppm, window, window_probs, reduced_rows


def stitch_ppm(
    left_flank: List[pd.Series],
    core_ppm: pd.DataFrame,
    right_flank: List[pd.Series],
) -> pd.DataFrame:
    """Stitch flank probability vectors + core PPM into a single PPM with integer positions."""
    rows: List[pd.Series] = []
    # left flank is nearest->outward; to stitch left-to-right, reverse it
    for s in reversed(left_flank):
        rows.append(pd.Series({b: float(s[b]) for b in BASES}))

    for _, r in core_ppm.iterrows():
        rows.append(pd.Series({b: float(r[b]) for b in BASES}))

    for s in right_flank:
        rows.append(pd.Series({b: float(s[b]) for b in BASES}))

    ppm = pd.DataFrame(rows)
    ppm.columns = list(BASES)
    # positions: negative for left flank, then 0..k-1, then k..
    L = len(left_flank)
    k = core_ppm.shape[0]
    idx = list(range(-L, 0)) + list(range(0, k)) + list(range(k, k + len(right_flank)))
    ppm.index = idx
    ppm.index.name = "pos"
    return ppm


def consensus_from_ppm(ppm: pd.DataFrame) -> str:
    # Robust to accidental duplicate indices/columns
    ppm = ppm.loc[:, ~ppm.columns.duplicated()].copy()
    if set(BASES).issubset(ppm.columns):
        ppm = ppm[list(BASES)]
    out = []
    for pos in ppm.index:
        row = ppm.loc[pos]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        out.append(str(row.astype(float).idxmax()))
    return "".join(out)


def write_meme(ppm: pd.DataFrame, out_path: str, motif_name: str) -> None:
    with open(out_path, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        w = ppm.shape[0]
        f.write(f"MOTIF {motif_name}\n")
        f.write(f"letter-probability matrix: alength= 4 w= {w} nsites= {w} E= 0\n")

        for pos in ppm.index:
            row = ppm.loc[pos]
            f.write(f"{row['A']:.6f} {row['C']:.6f} {row['G']:.6f} {row['T']:.6f}\n")


def ppm_to_bits_matrix(ppm: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Convert a PPM (probabilities) to a per-letter height matrix in *bits*.

    For each position i:
      IC_bits(i) = log2(4) - H(i), where H(i) = -Σ_b p(b,i) log2 p(b,i)
      height(b,i) = p(b,i) * IC_bits(i)

    This is the standard "information content" sequence-logo height convention.
    """
    mat = ppm.copy()
    # Ensure columns are A,C,G,T and numeric
    mat = mat.loc[:, ~mat.columns.duplicated()].copy()
    mat = mat[list(BASES)].astype(float)

    # Normalize each row defensively
    row_sums = mat.sum(axis=1).replace(0.0, np.nan)
    mat = mat.div(row_sums, axis=0)

    p = np.clip(mat.to_numpy(dtype=float), eps, 1.0)
    H = -(p * np.log2(p)).sum(axis=1)  # entropy in bits
    IC = np.log2(4.0) - H  # max 2 bits for DNA
    heights = p * IC[:, None]  # per-letter heights

    out = pd.DataFrame(heights, columns=list(BASES), index=ppm.index)
    out.index.name = ppm.index.name
    return out


def reverse_complement_ppm(ppm: pd.DataFrame) -> pd.DataFrame:
    """Return a reverse-complemented copy of a PPM/height matrix."""
    rc = ppm.copy().iloc[::-1].reset_index(drop=True)
    col_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    rc = rc.rename(columns=col_map)
    rc = rc[[b for b in BASES if b in rc.columns]]
    return rc


def reverse_complement_reduced_df(reduced_full_df: pd.DataFrame) -> pd.DataFrame:
    """Reverse-complement reduced enrichment table for plotting."""
    if reduced_full_df is None or reduced_full_df.empty:
        return reduced_full_df.copy()

    rc = reduced_full_df.copy()
    pos_vals = sorted(rc["pos"].dropna().astype(int).unique())
    pos_map = {old: new for old, new in zip(pos_vals, reversed(pos_vals))}
    rc["pos"] = rc["pos"].astype(int).map(pos_map)

    base_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rc["base"] = rc["base"].map(lambda b: base_map.get(b, b))

    sort_cols = [c for c in ["pos", "side", "base"] if c in rc.columns]
    if sort_cols:
        rc = rc.sort_values(sort_cols).reset_index(drop=True)
    return rc


def plot_logo(
    ppm: pd.DataFrame,
    out_png: str,
    title: str,
    pretty_logo: bool = False,
    mode: str = "prob",
) -> None:
    """Save a sequence logo.

    mode:
      - "prob": y-axis is probability (0..1)
      - "bits": y-axis is information content in bits (0..2), using height(b)=p(b)*IC_bits
    """
    mode = str(mode).lower().strip()
    if mode not in ("prob", "bits"):
        raise ValueError("mode must be 'prob' or 'bits'")

    dna_colors = _dna_colors(pretty_logo)

    # Choose the matrix to plot
    if mode == "bits":
        logo_mat = ppm_to_bits_matrix(ppm)
        y_label = "bits"
        y_max = 2.0
    else:
        logo_mat = ppm.copy()
        y_label = "prob"
        y_max = 1.0

    fig, ax = plt.subplots(figsize=(max(6, ppm.shape[0] * 0.6), 2.8))

    L = ppm.shape[0]

    if pretty_logo:
        bg = "#e6e6e6"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    if logomaker is not None:
        logo_df = logo_mat.copy()
        logo_df.index = range(ppm.shape[0])
        logomaker.Logo(logo_df, ax=ax, color_scheme=dna_colors)
        ax.set_ylabel(y_label, fontsize=18, fontfamily="Carlito")
        ax.set_xlabel("pos", fontsize=18, fontfamily="Carlito")
        ax.set_xticks(np.arange(L))
        ax.set_xticklabels(
            [str(i) for i in range(1, L + 1)],
            rotation=90,
            fontsize=18,
            fontfamily="Carlito",
        )
        ax.tick_params(axis="y", labelsize=18)
        for label in ax.get_yticklabels():
            label.set_fontfamily("Carlito")
        ax.set_ylim(0, y_max)
    else:
        # fallback: draw a simple sequence logo using stacked letters
        from matplotlib.textpath import TextPath
        from matplotlib.patches import PathPatch
        from matplotlib.transforms import Affine2D
        from matplotlib.font_manager import FontProperties

        fp = FontProperties(family="Carlito", weight="bold")

        def add_letter(letter: str, x: float, y: float, height: float) -> None:
            if height <= 0:
                return
            tp = TextPath((0, 0), letter, size=1, prop=fp)
            bb = tp.get_extents()
            sx = 0.9 / max(bb.width, 1e-6)
            sy = height / max(bb.height, 1e-6)
            trans = Affine2D().scale(sx, sy).translate(x + 0.05, y)
            patch = PathPatch(
                tp,
                lw=0,
                facecolor=dna_colors.get(letter, "black"),
                transform=trans + ax.transData,
            )
            ax.add_patch(patch)

        L = logo_mat.shape[0]
        for i in range(L):
            heights = {b: float(logo_mat.iloc[i][b]) for b in BASES}
            y0 = 0.0
            for letter, h in sorted(heights.items(), key=lambda kv: kv[1]):
                add_letter(letter, x=float(i), y=y0, height=h)
                y0 += h

        ax.set_xlim(0, L)
        ax.set_ylim(0, y_max)
        ax.set_xticks(np.arange(L))
        ax.set_xticklabels(
            [str(i) for i in range(1, L + 1)],
            rotation=90,
            fontsize=18,
            fontfamily="Carlito",
        )
        ax.set_ylabel(y_label, fontsize=18, fontfamily="Carlito")
        ax.set_xlabel("pos", fontsize=18, fontfamily="Carlito")
        ax.tick_params(axis="y", labelsize=18)
        for label in ax.get_yticklabels():
            label.set_fontfamily("Carlito")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_enrichment_bars(
    reduced_full_df: pd.DataFrame,
    out_path: str,
    pretty_logo: bool = False,
    title: Optional[str] = None,
) -> None:
    """Plot grouped bars of reduced enrichment (E_reduced) for A/C/G/T at each motif position.

    Notes:
      - We remap the 'pos' values from reduced_full_df to 0..L-1 so the x-axis matches the logo.
      - If pretty_logo is True, use the same DNA colors + light gray background as the logo.
    """
    if reduced_full_df is None or len(reduced_full_df) == 0:
        return

    # Expect columns: pos, base, E_reduced
    df = reduced_full_df.copy()
    if (
        "pos" not in df.columns
        or "base" not in df.columns
        or "E_reduced" not in df.columns
    ):
        return

    # Keep only valid bases
    df = df[df["base"].isin(BASES)].copy()
    if len(df) == 0:
        return

    # Aggregate across steps/sides if present (take mean per pos/base)
    df_agg = df.groupby(["pos", "base"], as_index=False)["E_reduced"].mean()

    # Build pivot: rows=pos (original), cols=base
    pivot = df_agg.pivot(index="pos", columns="base", values="E_reduced").sort_index()

    # Map original positions to 0..L-1 for display to match logo
    orig_positions = list(pivot.index)
    pos_map = {p: i for i, p in enumerate(orig_positions)}
    pivot = (
        pivot.reset_index()
        .assign(pos0=lambda d: d["pos"].map(pos_map))
        .set_index("pos0")
    )
    pivot.index.name = "pos"

    L = pivot.shape[0]
    if L == 0:
        return

    # Ensure all bases present as columns
    for b in BASES:
        if b not in pivot.columns:
            pivot[b] = 0.0
        pivot = pivot.loc[:, list(BASES)]

    # Colors
    dna_colors = _dna_colors(pretty_logo)

    import matplotlib.pyplot as plt
    import numpy as np

    fig_w = max(8.0, 0.8 * L)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6), dpi=150)

    if pretty_logo:
        ax.set_facecolor("#e6e6e6")
        fig.patch.set_facecolor("#e6e6e6")

    x = np.arange(L, dtype=float)
    width = 0.18
    offsets = {"A": -1.5 * width, "C": -0.5 * width, "G": 0.5 * width, "T": 1.5 * width}

    for b in BASES:
        ax.bar(
            x + offsets[b],
            pivot[b].to_numpy(dtype=float),
            width=width,
            label=b,
            color=dna_colors.get(b),
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlim(-0.6, L - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(i) for i in range(1, L + 1)],
        rotation=90,
        fontsize=18,
        fontfamily="Carlito",
    )
    ax.set_xlabel("pos", fontsize=18, fontfamily="Carlito")
    ax.set_ylabel("Enrichment score")

    if title:
        ax.set_title(title)

    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_seed_enrichment_curve(
    seed: str,
    fg_idx: np.ndarray,
    ranks: ProbeRanks,
    out_png: str,
) -> None:
    """A simple seed enrichment plot: FG and BG detection curves across ranked probes."""
    N = len(ranks.scores)
    fg_mask = np.zeros(N, dtype=bool)
    fg_mask[fg_idx] = True

    ordered_mask = fg_mask[ranks.order]  # probes sorted high->low

    # cumulative detection rates
    x = np.arange(1, N + 1)
    fg_cum = np.cumsum(ordered_mask)
    bg_cum = np.cumsum(~ordered_mask)

    F = fg_mask.sum()
    B = N - F
    fg_rate = fg_cum / max(F, 1)
    bg_rate = bg_cum / max(B, 1)

    # E-score
    E = escore_auc_minus_half_all_bg(fg_idx, ranks.i_to_rank, N)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x / N, fg_rate, label=f"Foreground (F={F})")
    ax.plot(x / N, bg_rate, label=f"Background (B={B})")
    ax.set_xlabel("fraction of probes scanned (high → low intensity)")
    ax.set_ylabel("detection rate")
    ax.set_title(f"Seed enrichment curves: {seed}  (E={E:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_seed_enrichment_roc(
    seed: str,
    fg_idx: np.ndarray,
    ranks: ProbeRanks,
    out_png: str,
) -> None:
    """ROC-style enrichment plot; AUC maps directly to the E-score (E = AUC - 0.5)."""
    N = len(ranks.scores)
    fg_mask = np.zeros(N, dtype=bool)
    fg_mask[fg_idx] = True

    ordered = fg_mask[ranks.order]  # high→low intensity
    tp = np.cumsum(ordered)
    fp = np.cumsum(~ordered)
    F = int(tp[-1])
    B = int(fp[-1])

    if F == 0 or B == 0:
        return

    tpr = tp / F
    fpr = fp / B

    E = escore_auc_minus_half_all_bg(fg_idx, ranks.i_to_rank, N)
    auc = E + 0.5

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(fpr, tpr, label=f"seed={seed} (AUC={auc:.3f}, E={E:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Seed enrichment (ROC)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_escore_histogram(es_df: pd.DataFrame, out_png: str, title: str) -> None:
    """Histogram of E-scores for seed candidates (useful sanity check)."""
    if es_df.empty:
        return
    vals = es_df["E"].astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(vals, bins=60)
    ax.set_xlabel("E-score")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run_masked_motifs(args, intensities: Dict[str, float], ranks: ProbeRanks) -> int:
    """Masked motif discovery path.

    The mask defines the motif span and which positions are informative. Seed
    discovery is performed on masked signatures sampled from the highest-
    intensity probes, then rescored against the full probe universe. Wobble is
    restricted to informative positions; spacer positions remain uniform in the
    PPM and therefore contribute zero information in the bits logo.
    """
    mask = SequenceMask.parse(args.mask)
    if not args.genome:
        sys.stderr.write("[error] --genome is required when --mask is used.\n")
        return 2

    probe_regions = tuple(ranks.regions)
    if not probe_regions:
        sys.stderr.write("[error] Intensity file contains no probe regions.\n")
        return 2

    sys.stderr.write(
        f"[MASK-MOTIF] mask={mask.pattern} span={mask.span} "
        f"informative={mask.n_informative}\n"
    )
    if args.array:
        sys.stderr.write(
            "[MASK-MOTIF] note: --array is not used in masked motif mode; "
            "probe regions come from the intensity file and sequence comes from --genome.\n"
        )

    if args.seed is None:
        candidate_codes = collect_mask_seed_candidates(
            mask=mask,
            ranks=ranks,
            genome_fasta=args.genome,
            top_probe_count=int(args.mask_candidate_probes),
            max_candidates=int(args.mask_max_candidates),
        )
        if not candidate_codes:
            sys.stderr.write("[error] No masked seed candidates could be generated.\n")
            return 2
        sys.stderr.write(
            f"[MASK-MOTIF] generated {len(candidate_codes):,} candidate signatures "
            f"from top {min(int(args.mask_candidate_probes), len(ranks.regions)):,} probes; "
            "scanning full probe universe\n"
        )
        candidate_hits, scan_stats = scan_mask_codes_for_motifs(
            mask=mask,
            target_codes=candidate_codes,
            probe_regions=probe_regions,
            genome_fasta=args.genome,
        )
        sys.stderr.write(
            f"[MASK-MOTIF] indexed {scan_stats['n_windows']:,} probe windows; "
            f"candidate hits={scan_stats['n_target_region_hits']:,}\n"
        )
        es_df = choose_mask_seed(
            mask=mask,
            candidate_codes=candidate_codes,
            target_hits=candidate_hits,
            ranks=ranks,
            min_F=int(args.min_F),
        )
        if es_df.empty:
            sys.stderr.write("[error] No masked candidates passed the --min-F filter.\n")
            return 2
        seed_code = int(es_df.loc[0, "code"])
        seed_hits = candidate_hits.get(seed_code, {})
        top_path = os.path.join(args.outdir, f"{args.prefix}.top_escores.tsv")
        es_df.head(int(args.top_n_report)).to_csv(top_path, sep="\t", index=False)
        sys.stderr.write(f"[ok] Wrote top E-score table: {top_path}\n")
    else:
        try:
            seed_code = _mask_seed_to_code(args.seed, mask)
        except ValueError as exc:
            sys.stderr.write(f"[error] {exc}\n")
            return 2
        candidate_hits, scan_stats = scan_mask_codes_for_motifs(
            mask=mask,
            target_codes=[seed_code],
            probe_regions=probe_regions,
            genome_fasta=args.genome,
        )
        seed_hits = candidate_hits.get(seed_code, {})
        seed_idx = _hits_to_probe_idx(seed_hits, ranks.region_to_i)
        F = int(len(seed_idx))
        E = (
            escore_auc_minus_half_all_bg(seed_idx, ranks.i_to_rank, len(ranks.scores))
            if F > 0
            else np.nan
        )
        es_df = pd.DataFrame(
            [
                {
                    "kmer": _mask_pattern_from_code(mask, seed_code),
                    "code": seed_code,
                    "E": E,
                    "F": F,
                    "gaps": int(mask.span - mask.n_informative),
                }
            ]
        )

    seed = _mask_pattern_from_code(mask, seed_code)
    sys.stderr.write(f"[seed] {seed}\n")

    seed_bases = _mask_code_to_bases(seed_code, mask.n_informative)
    wobble_codes = sorted(
        {
            _variant_code_for_informative_base(seed_bases, j, base)
            for j in range(mask.n_informative)
            for base in BASES
        }
    )
    sys.stderr.write(
        f"[MASK-MOTIF] wobbling {mask.n_informative} informative positions "
        f"({len(wobble_codes)} unique signatures); scanning probe universe\n"
    )
    wobble_hits, wobble_stats = scan_mask_codes_for_motifs(
        mask=mask,
        target_codes=wobble_codes,
        probe_regions=probe_regions,
        genome_fasta=args.genome,
    )
    sys.stderr.write(
        f"[MASK-MOTIF] wobble scan indexed {wobble_stats['n_windows']:,} probe windows; "
        f"target hits={wobble_stats['n_target_region_hits']:,}\n"
    )

    reduced_df, ppm = ppm_from_mask_seed_wobble(
        mask=mask,
        seed_code=seed_code,
        variant_hits=wobble_hits,
        ranks=ranks,
        min_per_base=int(args.min_per_base),
        beta=float(args.beta),
        min_support=int(args.min_support),
        pseudocount=float(args.pseudocount),
    )

    consensus_chars = []
    for pos in range(mask.span):
        if mask.pattern[pos] == "0":
            consensus_chars.append(".")
        else:
            consensus_chars.append(str(ppm.loc[pos, list(BASES)].idxmax()))
    consensus = "".join(consensus_chars)
    sys.stderr.write(f"[consensus] {consensus}\n")

    ppm_path = os.path.join(args.outdir, f"{args.prefix}.ppm.tsv")
    meme_path = os.path.join(args.outdir, f"{args.prefix}.meme")
    logo_prob_path = os.path.join(args.outdir, f"{args.prefix}.logo_prob.png")
    logo_bits_path = os.path.join(args.outdir, f"{args.prefix}.logo_bits.png")
    logo_prob_rc_path = os.path.join(args.outdir, f"{args.prefix}.logo_prob_rc.png")
    logo_bits_rc_path = os.path.join(args.outdir, f"{args.prefix}.logo_bits_rc.png")
    seed_curve_path = os.path.join(args.outdir, f"{args.prefix}.seed_enrichment_curve.png")
    seed_roc_path = os.path.join(args.outdir, f"{args.prefix}.seed_roc.png")
    seed_hist_path = os.path.join(args.outdir, f"{args.prefix}.seed_escore_hist.png")
    enrich_bar_path = os.path.join(args.outdir, f"{args.prefix}.reduced_enrichment.png")
    enrich_bar_rc_path = os.path.join(args.outdir, f"{args.prefix}.reduced_enrichment_rc.png")
    reduced_path = os.path.join(args.outdir, f"{args.prefix}.reduced.tsv")
    reduced_full_path = os.path.join(args.outdir, f"{args.prefix}.reduced_full.tsv")

    reduced_full_df = reduced_df.copy()
    if not reduced_full_df.empty:
        reduced_full_df["side"] = "core"
        reduced_full_df["step"] = reduced_full_df["pos"].astype(int)
        reduced_full_df["gaps_used"] = int(mask.span - mask.n_informative)
        reduced_full_df.sort_values(["pos", "base"], inplace=True)

    ppm.to_csv(ppm_path, sep="\t")
    reduced_df.to_csv(reduced_path, sep="\t", index=False)
    reduced_full_df.to_csv(reduced_full_path, sep="\t", index=False)

    ppm_normal = ppm.reset_index(drop=True)
    ppm_rc = reverse_complement_ppm(ppm_normal)
    write_meme(ppm_normal, meme_path, motif_name=consensus)
    title = f"{consensus} (mask={mask.pattern})"
    plot_logo(ppm_normal, logo_prob_path, title=title, pretty_logo=bool(args.pretty_logo), mode="prob")
    plot_logo(ppm_normal, logo_bits_path, title=title, pretty_logo=bool(args.pretty_logo), mode="bits")
    plot_logo(ppm_rc, logo_prob_rc_path, title=title + " [RC]", pretty_logo=bool(args.pretty_logo), mode="prob")
    plot_logo(ppm_rc, logo_bits_rc_path, title=title + " [RC]", pretty_logo=bool(args.pretty_logo), mode="bits")
    plot_enrichment_bars(reduced_full_df, enrich_bar_path, title=None, pretty_logo=bool(args.pretty_logo))
    plot_enrichment_bars(
        reverse_complement_reduced_df(reduced_full_df),
        enrich_bar_rc_path,
        title=None,
        pretty_logo=bool(args.pretty_logo),
    )

    seed_idx = _hits_to_probe_idx(seed_hits, ranks.region_to_i)
    if seed_idx.size > 0:
        plot_seed_enrichment_curve(seed, seed_idx, ranks, seed_curve_path)
        plot_seed_enrichment_roc(seed, seed_idx, ranks, seed_roc_path)
    plot_escore_histogram(es_df, seed_hist_path, title="Masked seed candidate E-score distribution")

    sys.stderr.write(f"[ok] PPM:   {ppm_path}\n")
    sys.stderr.write(f"[ok] MEME:  {meme_path}\n")
    sys.stderr.write(f"[ok] logo:  {logo_prob_path}\n")
    sys.stderr.write(f"[ok] reduced enrichment plot: {enrich_bar_path}\n")
    sys.stderr.write(f"[ok] seed curve: {seed_curve_path}\n")
    sys.stderr.write(f"[ok] seed ROC:   {seed_roc_path}\n")
    sys.stderr.write(f"[ok] E-score hist: {seed_hist_path}\n")
    sys.stderr.write(f"[ok] reduced table: {reduced_path}\n")
    sys.stderr.write(f"[ok] reduced_full: {reduced_full_path}\n")
    return 0


def motif_logscore(kmer: str, ppm: pd.DataFrame, eps: float = 1e-12) -> float:
    """Sum log(prob) under the motif for a k-mer aligned to ppm.

    Assumes k-mer length equals motif length. If motif length differs, caller should slice.
    """
    if len(kmer) != ppm.shape[0]:
        raise ValueError("kmer length must equal motif length for motif_logscore")
    s = 0.0
    for pos, base in zip(ppm.index, kmer):
        p = float(ppm.loc[pos, base]) if base in BASES else 0.0
        s += math.log(max(p, eps))
    return s


def plot_motif_vs_escore(
    es_df: pd.DataFrame,
    ppm: pd.DataFrame,
    out_png: str,
    q: int = 20,
) -> None:
    """QC plot: bin motif log-scores and plot mean E-score per bin."""
    if es_df.empty:
        return

    L = int(ppm.shape[0])

    # Find any representative k-mer length (avoid assuming row 0 exists / is a string)
    k_example = None
    for x in es_df["kmer"].astype(str).tolist():
        if x:
            k_example = len(x)
            break

    if k_example is None or L != k_example:
        return

    df = es_df.copy()
    df["motif_logscore"] = df["kmer"].apply(lambda k: motif_logscore(str(k), ppm))

    # bin and aggregate
    qc = df.dropna(subset=["motif_logscore", "E"]).copy()
    if qc.empty:
        return

    qc["bin"] = pd.qcut(qc["motif_logscore"], q=q, duplicates="drop")
    trend = (
        qc.groupby("bin", observed=False)
        .agg(
            mean_motif=("motif_logscore", "mean"),
            mean_E=("E", "mean"),
            n=("E", "size"),
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(trend["mean_motif"], trend["mean_E"], marker="o")
    ax.set_xlabel("mean motif log-score (binned)")
    ax.set_ylabel("mean E-score")
    ax.set_title("Motif score vs E-score (QC)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Seed-and-wobble motif discovery from probe intensities + kmer occurrence index."
    )
    ap.add_argument("--intensities", required=True, help="Path to probe intensity file")
    ap.add_argument(
        "--array",
        dest="array",
        required=False,
        default=None,
        help="Path to k-mer array file mapping k-mers to genomic regions",
    )
    ap.add_argument(
        "--kmers", dest="array", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    ap.add_argument(
        "--mask",
        default=None,
        help=(
            "Optional explicit 0/1 sequence mask for masked motif discovery, e.g. "
            "11111000011111. When supplied, the mask defines the motif span and "
            "--genome is required; the ordinary k-mer seed/extension path is left unchanged."
        ),
    )
    ap.add_argument(
        "--genome",
        default=None,
        help="Genome FASTA used for masked motif discovery (--mask mode)",
    )
    ap.add_argument(
        "--mask-candidate-probes",
        type=int,
        default=5000,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--mask-max-candidates",
        type=int,
        default=10000,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--kmer-size", type=int, default=8, help="k-mer size (default: 8)")
    ap.add_argument(
        "--no-combine-revcomp",
        dest="combine_revcomp",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    ap.set_defaults(combine_revcomp=True)

    ap.add_argument(
        "--min-F",
        type=int,
        default=20,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--max-gaps",
        type=int,
        default=3,
        help="Max number of wildcard ('.') positions when searching seed patterns (default: 3; use 0 for exact k-mers only)",
    )
    ap.add_argument(
        "--min-per-base",
        type=int,
        default=20,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--min-support",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--pseudocount",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=10.0,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--auto-max-steps",
        type=int,
        default=20,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--extend-left",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--extend-right",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--ic-stop-threshold",
        type=float,
        default=0.20,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--ic-stop-consecutive",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--outdir",
        default=".",
        help="Output directory (default: current directory)",
    )
    ap.add_argument(
        "--prefix",
        default="affinity_motif",
        help="Output prefix (default: affinity_motif)",
    )

    ap.add_argument(
        "--pretty-logo",
        action="store_true",
        help="Render a prettier motif logo (light gray background + fixed DNA colors)",
    )
    ap.add_argument(
        "--seed",
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--top-n-report",
        type=int,
        default=50,
        help=argparse.SUPPRESS,
    )

    args = ap.parse_args(argv)

    # Cap max gaps used during EXTENSION to what is possible for the chosen k (anchor length = k-1).
    # (Seed search can still use up to args.max_gaps wildcards within length-k patterns.)
    max_gaps_ext = min(int(args.max_gaps), int(args.kmer_size) - 1)
    if max_gaps_ext < 0:
        max_gaps_ext = 0

    if args.kmer_size != 8:
        sys.stderr.write(
            f"[note] Using k={args.kmer_size}. Seed/wobble/extension support this, "
            "but probe support and runtime may change with k.\n"
        )

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    intensities = read_intensities(args.intensities)
    ranks = build_probe_ranks(intensities)

    if args.mask is not None:
        return run_masked_motifs(args, intensities, ranks)

    if not args.array:
        sys.stderr.write("[error] --array is required unless --mask is supplied.\n")
        return 2

    kmer_positions = read_unique_kmer_positions(args.array)

    # basic k-mer length sanity
    kmer_positions = {
        k: v for k, v in kmer_positions.items() if len(k) == args.kmer_size
    }

    kmer_to_idx = build_kmer_to_probe_idx(
        kmer_positions=kmer_positions,
        region_to_i=ranks.region_to_i,
        combine_revcomp=args.combine_revcomp,
    )

    if args.seed is None:
        es_df = choose_seed(
            kmer_to_idx, ranks.i_to_rank, min_F=args.min_F, max_gaps=args.max_gaps
        )
        if es_df.empty:
            sys.stderr.write("[error] No kmers passed min-F filter.\n")
            return 2
        seed = str(es_df.loc[0, "kmer"])

        top_path = os.path.join(args.outdir, f"{args.prefix}.top_escores.tsv")
        es_df.head(args.top_n_report).to_csv(top_path, sep="\t", index=False)
        sys.stderr.write(f"[ok] Wrote top E-score table: {top_path}\n")
    else:
        seed = args.seed.strip().upper()
        es_df = choose_seed(
            kmer_to_idx, ranks.i_to_rank, min_F=args.min_F, max_gaps=args.max_gaps
        )

    sys.stderr.write(f"[seed] {seed}\n")

    # Core seed-and-wobble
    reduced_df, core_ppm = ppm_from_seed_wobble(
        seed=seed,
        kmer_to_idx=kmer_to_idx,
        i_to_rank=ranks.i_to_rank,
        min_per_base=args.min_per_base,
        beta=args.beta,
        min_support=args.min_support,
        pseudocount=args.pseudocount,
    )

    # If the seed search used wildcards, derive a concrete core consensus to use for optional extension.
    core_seed = consensus_from_ppm(core_ppm)

    # Optional flanks
    left_flank: List[pd.Series] = []
    right_flank: List[pd.Series] = []
    left_rows: List[Dict] = []
    right_rows: List[Dict] = []

    # Extension works by shifting a k-mer window and (optionally)
    # turning low-information anchor positions into '.' wildcards to preserve support.
    #
    # By default we auto-extend (args.extend_left/right == -1) until support fails.

    window = core_seed
    window_probs = [core_ppm.loc[i] for i in core_ppm.index]

    # left
    if args.extend_left != 0:
        steps = args.auto_max_steps if args.extend_left < 0 else int(args.extend_left)
        left_flank, window, window_probs, left_rows = extend_side(
            window=window,
            window_probs=window_probs,
            kmer_to_idx=kmer_to_idx,
            i_to_rank=ranks.i_to_rank,
            beta=args.beta,
            min_per_base=args.min_per_base,
            side="left",
            max_steps=steps,
            max_gaps=int(max_gaps_ext),
            ic_stop_threshold=float(args.ic_stop_threshold),
            ic_stop_consecutive=int(args.ic_stop_consecutive),
        )

    # right (start from core again so left and right are symmetric around the core)
    window = core_seed
    window_probs = [core_ppm.loc[i] for i in core_ppm.index]

    if args.extend_right != 0:
        steps = args.auto_max_steps if args.extend_right < 0 else int(args.extend_right)
        right_flank, window, window_probs, right_rows = extend_side(
            window=window,
            window_probs=window_probs,
            kmer_to_idx=kmer_to_idx,
            i_to_rank=ranks.i_to_rank,
            beta=args.beta,
            min_per_base=args.min_per_base,
            side="right",
            max_steps=steps,
            max_gaps=int(max_gaps_ext),
            ic_stop_threshold=float(args.ic_stop_threshold),
            ic_stop_consecutive=int(args.ic_stop_consecutive),
        )
    ppm = stitch_ppm(left_flank, core_ppm, right_flank)
    consensus = consensus_from_ppm(ppm)
    sys.stderr.write(f"[consensus] {consensus}\n")

    # Write outputs
    ppm_path = os.path.join(args.outdir, f"{args.prefix}.ppm.tsv")
    meme_path = os.path.join(args.outdir, f"{args.prefix}.meme")
    logo_prob_path = os.path.join(args.outdir, f"{args.prefix}.logo_prob.png")
    logo_bits_path = os.path.join(args.outdir, f"{args.prefix}.logo_bits.png")
    logo_prob_rc_path = os.path.join(args.outdir, f"{args.prefix}.logo_prob_rc.png")
    logo_bits_rc_path = os.path.join(args.outdir, f"{args.prefix}.logo_bits_rc.png")
    # Back-compat: keep the old name as the probability logo
    logo_path = logo_prob_path
    seed_curve_path = os.path.join(
        args.outdir, f"{args.prefix}.seed_enrichment_curve.png"
    )
    seed_roc_path = os.path.join(args.outdir, f"{args.prefix}.seed_roc.png")
    seed_hist_path = os.path.join(args.outdir, f"{args.prefix}.seed_escore_hist.png")
    qc_path = os.path.join(args.outdir, f"{args.prefix}.motif_vs_E.png")
    enrich_bar_path = os.path.join(args.outdir, f"{args.prefix}.reduced_enrichment.png")
    enrich_bar_rc_path = os.path.join(
        args.outdir, f"{args.prefix}.reduced_enrichment_rc.png"
    )

    reduced_path = os.path.join(args.outdir, f"{args.prefix}.reduced.tsv")

    # Full reduced table (core wobble + flank extensions) for recreating Fig 3a-style panels.
    reduced_full_path = os.path.join(args.outdir, f"{args.prefix}.reduced_full.tsv")

    core_len = int(core_ppm.shape[0])

    core_df = reduced_df.copy()
    core_df["side"] = "core"
    core_df["step"] = core_df["pos"].astype(int)
    core_df["gaps_used"] = 0

    left_df = pd.DataFrame(left_rows)
    if not left_df.empty:
        left_df["pos"] = -(left_df["step"].astype(int) + 1)

    right_df = pd.DataFrame(right_rows)
    if not right_df.empty:
        right_df["pos"] = core_len + right_df["step"].astype(int)

    reduced_full_df = pd.concat(
        [left_df, core_df, right_df], ignore_index=True, sort=False
    )
    # stable ordering
    if not reduced_full_df.empty:
        reduced_full_df["pos"] = reduced_full_df["pos"].astype(int)
        reduced_full_df.sort_values(["pos", "side", "base"], inplace=True)

    ppm.to_csv(ppm_path, sep="\t")
    reduced_df.to_csv(reduced_path, sep="\t", index=False)
    reduced_full_df.to_csv(reduced_full_path, sep="\t", index=False)

    ppm_normal = ppm.reset_index(drop=True)
    ppm_rc = reverse_complement_ppm(ppm_normal)
    write_meme(ppm_normal, meme_path, motif_name=consensus)
    plot_logo(
        ppm_normal,
        logo_prob_path,
        title=f"{consensus} (seed={seed})",
        pretty_logo=bool(args.pretty_logo),
        mode="prob",
    )
    plot_logo(
        ppm_normal,
        logo_bits_path,
        title=f"{consensus} (seed={seed})",
        pretty_logo=bool(args.pretty_logo),
        mode="bits",
    )
    plot_logo(
        ppm_rc,
        logo_prob_rc_path,
        title=f"{consensus} (seed={seed}) [RC]",
        pretty_logo=bool(args.pretty_logo),
        mode="prob",
    )
    plot_logo(
        ppm_rc,
        logo_bits_rc_path,
        title=f"{consensus} (seed={seed}) [RC]",
        pretty_logo=bool(args.pretty_logo),
        mode="bits",
    )
    plot_enrichment_bars(
        reduced_full_df, enrich_bar_path, title=None, pretty_logo=bool(args.pretty_logo)
    )
    reduced_full_df_rc = reverse_complement_reduced_df(reduced_full_df)
    plot_enrichment_bars(
        reduced_full_df_rc,
        enrich_bar_rc_path,
        title=None,
        pretty_logo=bool(args.pretty_logo),
    )

    # Seed enrichment plots + histogram of candidate E-scores
    fg_seed = fg_indices_for_pattern(seed, kmer_to_idx)
    if fg_seed.size > 0:
        plot_seed_enrichment_curve(seed, fg_seed, ranks, seed_curve_path)
        plot_seed_enrichment_roc(seed, fg_seed, ranks, seed_roc_path)
    plot_escore_histogram(
        es_df, seed_hist_path, title="Seed candidate E-score distribution"
    )

    # QC plot (only if motif length == kmer_size)
    try:
        es_df_exact = es_df[es_df.get("gaps", 0) == 0].copy()
        plot_motif_vs_escore(es_df_exact, ppm.reset_index(drop=True), qc_path)
    except Exception:
        pass

    sys.stderr.write(f"[ok] PPM:   {ppm_path}\n")
    sys.stderr.write(f"[ok] MEME:  {meme_path}\n")
    sys.stderr.write(f"[ok] logo:  {logo_path}\n")
    sys.stderr.write(f"[ok] reduced enrichment plot: {enrich_bar_path}\n")
    sys.stderr.write(f"[ok] seed curve: {seed_curve_path}\n")
    sys.stderr.write(f"[ok] seed ROC:   {seed_roc_path}\n")
    sys.stderr.write(f"[ok] E-score hist: {seed_hist_path}\n")
    sys.stderr.write(f"[ok] reduced table: {reduced_path}\n")
    sys.stderr.write(f"[ok] reduced_full: {reduced_full_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
