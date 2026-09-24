"""Seed scoring, wobble and extension of contiguous motifs."""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from .motif_data import BASES, canonical_kmer, fg_indices_for_pattern


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
