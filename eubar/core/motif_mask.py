"""Candidate signatures and wobble for explicit masked motifs."""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from .motif_data import BASES, ProbeRanks
from .motif_search import escore_auc_minus_half_all_bg, reduced_escore_and_p, reduced_test_four_variants, softmax_from_escores
from .patterns import SequenceMask
from .masking import _signature_codes
from .sequence import _open_fasta


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
