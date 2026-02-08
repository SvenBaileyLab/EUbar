#!/usr/bin/env python3
"""eubar.snv_indel

Prototype indel support using the "Option 2" REF-vs-ALT sequence-state model:

1) Take a reference-coordinate window: [pos-flank, pos+len(REF)-1+flank]
2) Build REF sequence from genome FASTA, then build ALT sequence by applying the indel.
3) Enumerate k-mers in REF and ALT, then keep only *changed* k-mers:
      lost   = REF_kmers - ALT_kmers
      gained = ALT_kmers - REF_kmers
4) Map lost/gained k-mers to probe regions via the kmer index.
5) Build a probe-level table (deduplicated by region) and fit:

      log1p(y) ~ is_alt + (optional covariates)

Output is one effect + one p-value per variant (and some diagnostics if --debug).

Notes
-----
- This is intentionally simple and self-contained so you can iterate quickly.
- It does NOT try to mimic SNV per-window regression; it produces a single ALT-vs-REF effect.
- It uses only *changed* k-mers to avoid dilution by shared k-mers.

"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from eubar.core.data import IntensityTable, KmerIndex
from eubar.core.sequence import fetch_sequence, parse_snv, reverse_complement
from eubar.core.design import extract_covariates


try:
    from scipy.stats import norm  # type: ignore
except Exception:
    norm = None


def _format_pval(p: float, z: float = math.nan) -> str:
    """Format p-values robustly; avoid printing 0.0."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "nan"
    try:
        p = float(p)
    except Exception:
        return "nan"

    if p > 0.0:
        return f"{p:.3e}"

    # underflow
    if norm is not None and isinstance(z, float) and math.isfinite(z):
        logp = math.log(2.0) + float(norm.logsf(abs(z)))
        log10p = logp / math.log(10.0)
        exp10 = int(math.floor(log10p))
        mant = 10 ** (log10p - exp10)
        return f"{mant:.2f}e{exp10}"

    return "<1e-300"


def _sliding_kmers(seq: str, k: int) -> List[str]:
    return [seq[i:i+k] for i in range(0, max(0, len(seq) - k + 1))]


def _apply_indel_to_window(ref_window: str, offset: int, ref: str, alt: str) -> str:
    """Return ALT window string after replacing ref at offset with alt."""
    if offset < 0 or offset + len(ref) > len(ref_window):
        raise ValueError("indel offset out of bounds for window")
    got = ref_window[offset:offset+len(ref)]
    if got != ref:
        raise ValueError(f"reference allele mismatch in window: got={got} expected={ref}")
    return ref_window[:offset] + alt + ref_window[offset+len(ref):]


def _build_ref_alt_windows(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    genome_fasta: str,
    *,
    flank: int,
    debug: bool = False,
) -> Tuple[str, str, bool, int, int]:
    """Fetch a reference-coordinate window and return (ref_seq, alt_seq, flipped, start, end).

    Window definition (1-based, inclusive):
        start = pos - flank
        end   = pos + len(ref) - 1 + flank

    The edit offset within the fetched REF sequence is:
        offset = pos - start

    If the reference allele doesn't match, we try reverse-complement logic:
      - reverse-complement the entire fetched window
      - reverse-complement ref and alt
      - compute the equivalent offset in the RC string
    """
    start = pos - flank
    end = pos + len(ref) - 1 + flank
    if start < 1:
        raise ValueError(f"window start < 1 (start={start}); decrease flank or check variant position")

    ref_seq = fetch_sequence(genome_fasta, chrom, start, end)
    offset = pos - start  # 0-based within ref_seq

    flipped = False
    try:
        alt_seq = _apply_indel_to_window(ref_seq, offset, ref, alt)
    except ValueError:
        # try reverse complement orientation
        ref_seq_rc = reverse_complement(ref_seq)
        ref_rc = reverse_complement(ref)
        alt_rc = reverse_complement(alt)

        # compute offset in rc string:
        # original allele spans [offset, offset+len(ref)-1] in ref_seq (0-based)
        # in rc, that span maps to [len(ref_seq)- (offset+len(ref)), len(ref_seq) - offset)
        offset_rc = len(ref_seq) - (offset + len(ref))
        try:
            alt_seq_rc = _apply_indel_to_window(ref_seq_rc, offset_rc, ref_rc, alt_rc)
        except ValueError as e:
            raise ValueError(
                f"Reference allele mismatch (forward and reverse-complement). "
                f"Check your REF/ALT and genome build. Details: {e}"
            )
        ref_seq = ref_seq_rc
        alt_seq = alt_seq_rc
        flipped = True

        if debug:
            sys.stderr.write(
                f"[INDELWIN] flipped=True ref={ref}->{ref_rc} alt={alt}->{alt_rc} "
                f"start={start} end={end} offset_rc={offset_rc}\n"
            )

    if debug:
        sys.stderr.write(
            f"[INDELWIN] {chrom}:{pos}:{ref}>{alt} start={start} end={end} "
            f"len_refwin={len(ref_seq)} len_altwin={len(alt_seq)} flipped={flipped}\n"
        )

    return ref_seq, alt_seq, flipped, start, end


def _regions_for_kmers(
    kmers: Sequence[str],
    kmer_index: KmerIndex,
) -> Tuple[Set[str], Dict[str, int]]:
    """Return (regions_set, region_to_offset).

    region_to_offset records one offset per region (minimum offset seen).
    """
    regions: Set[str] = set()
    region_to_offset: Dict[str, int] = {}
    for k in kmers:
        m = kmer_index.lookup(k)
        if not m:
            continue
        for region, off in m.items():
            regions.add(region)
            try:
                off_i = int(off)
            except Exception:
                off_i = 0
            if region not in region_to_offset or off_i < region_to_offset[region]:
                region_to_offset[region] = off_i
    return regions, region_to_offset


def _build_probe_table(
    ref_regions: Set[str],
    alt_regions: Set[str],
    intens: IntensityTable,
    ref_off: Dict[str, int],
    alt_off: Dict[str, int],
    *,
    fold_half: bool = True,
    include_covariates: bool = True,
    drop_overlap: bool = True,
) -> pd.DataFrame:
    """Construct a probe-level dataframe with logy/y and is_alt.

    If drop_overlap=True, probes present in both REF and ALT groups are removed.
    """
    if drop_overlap:
        overlap = ref_regions & alt_regions
        ref_regions = set(ref_regions) - overlap
        alt_regions = set(alt_regions) - overlap
    else:
        overlap = set()

    ref_list = sorted(ref_regions)
    alt_list = sorted(alt_regions)

    regions = ref_list + alt_list
    is_alt = [0] * len(ref_list) + [1] * len(alt_list)

    y = []
    region_to_kmer_pos: Dict[str, int] = {}
    for r in ref_list:
        v = intens.get(r)
        if v is None:
            continue
        y.append(float(v))
        region_to_kmer_pos[r] = int(ref_off.get(r, 0))
    # ALT
    y_alt = []
    for r in alt_list:
        v = intens.get(r)
        if v is None:
            continue
        y_alt.append(float(v))
        region_to_kmer_pos[r] = int(alt_off.get(r, 0))

    # IMPORTANT: handle missing intensities by filtering the parallel is_alt list
    kept_regions = []
    kept_is_alt = []
    kept_y = []
    for r, ia in zip(regions, is_alt):
        v = intens.get(r)
        if v is None:
            continue
        kept_regions.append(r)
        kept_is_alt.append(int(ia))
        kept_y.append(float(v))

    df = pd.DataFrame({
        "probe": kept_regions,
        "y": kept_y,
        "logy": np.log1p(np.asarray(kept_y, dtype=float)),
        "is_alt": kept_is_alt,
    })

    if include_covariates and len(df) > 0:
        lp, sl = extract_covariates(df["probe"].tolist(), region_to_kmer_pos, fold_half=fold_half)
        df["lp"] = lp
        df["sl"] = sl

    df.attrs["n_overlap_dropped"] = len(overlap)
    df.attrs["n_ref_regions"] = len(ref_regions)
    df.attrs["n_alt_regions"] = len(alt_regions)
    df.attrs["n_kept"] = len(df)
    return df


def _fit_ref_alt_model(df: pd.DataFrame, *, mode: str = "ols", include_covariates: bool = True) -> Tuple[float, float, float]:
    """Fit logy ~ is_alt (+ covariates). Return (coef, pval, z/t stat) for is_alt."""
    if len(df) == 0:
        return math.nan, math.nan, math.nan

    y = df["logy"].astype(float).to_numpy() if mode == "ols" else np.clip(np.round(df["y"].astype(float).to_numpy()), 0, None)

    X_cols = ["is_alt"]
    if include_covariates:
        for c in ("lp", "sl"):
            if c in df.columns:
                X_cols.append(c)

    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")

    if mode == "ols":
        res = sm.OLS(y, X).fit()
    else:
        res = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()

    coef = float(res.params.get("is_alt", math.nan))
    pval = float(res.pvalues.get("is_alt", math.nan))
    stat = float(res.tvalues.get("is_alt", math.nan))
    return coef, pval, stat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prototype indel effect model (REF vs ALT sequence state)")
    grp = ap.add_mutually_exclusive_group(required=True)
    ap.add_argument("--intensities", required=True, help="Path to probe intensity file")
    ap.add_argument("--kmerPositions", required=True, help="Path to k-mer array file mapping kmers to genomic regions")
    ap.add_argument("--genome", required=True, help="Path to reference genome in FASTA format")
    grp.add_argument("--snv-list", help="Comma-separated variants in chr:pos:REF>ALT format (SNVs or indels)")
    grp.add_argument("--snv-list-file", help="File with variants, one per line in chr:pos:REF>ALT format")
    ap.add_argument("--kmer_size", type=int, default=8, help="k-mer size (default: 8)")
    ap.add_argument("--flank", type=int, default=20, help="Reference-coordinate flank (bp) on each side of the variant (default: 20)")
    ap.add_argument("--mode", choices=["ols", "nb"], default="ols", help="Model: OLS on log1p(y) or Negative Binomial on rounded counts")
    ap.add_argument("--no-covariates", action="store_true", help="Disable lp/sl covariates")
    ap.add_argument("--raw-lp", action="store_true", help="Use raw lp (no fold to [0,0.5])")
    ap.add_argument("--keep-overlap", action="store_true", help="Keep probes that appear in both REF and ALT probe sets (default drops them)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    intens = IntensityTable.from_file(args.intensities)
    kidx = KmerIndex.from_file(args.kmerPositions)

    # parse variant list
    if args.snv_list_file:
        with open(args.snv_list_file) as f:
            var_list = [ln.strip() for ln in f if ln.strip()]
    else:
        var_list = [s.strip() for s in (args.snv_list or "").split(",") if s.strip()]

    print("variant	effect_is_alt	pval	ref_kmers_lost	alt_kmers_gained	ref_regions	alt_regions	overlap_dropped	probes_used	comparison	ref_kmers_used	alt_kmers_used")
    for var in var_list:
        try:
            chrom, pos, ref, alt = parse_snv(var)
            if len(ref) == 1 and len(alt) == 1:
                # allow SNVs too, but this script is aimed at indels
                pass

            ref_seq, alt_seq, flipped, start, end = _build_ref_alt_windows(
                chrom, pos, ref, alt, args.genome, flank=args.flank, debug=args.debug
            )

            ref_kmers = _sliding_kmers(ref_seq, args.kmer_size)
            alt_kmers = _sliding_kmers(alt_seq, args.kmer_size)

            ref_set = set(ref_kmers)
            alt_set = set(alt_kmers)
            lost = sorted(ref_set - alt_set)
            gained = sorted(alt_set - ref_set)

            shared = sorted(ref_set & alt_set)

            # Choose which k-mer sets define the REF vs ALT probe groups.
            # - If both sides have unique k-mers: compare gained (ALT-only) vs lost (REF-only).
            # - If only gained exists (dup/ins): compare gained vs shared (baseline).
            # - If only lost exists (pure deletion): compare shared (ALT) vs lost.
            if len(lost) > 0 and len(gained) > 0:
                ref_kmers_used = lost
                alt_kmers_used = gained
                comparison = "gained_vs_lost"
            elif len(gained) > 0 and len(lost) == 0:
                ref_kmers_used = shared
                alt_kmers_used = gained
                comparison = "gained_vs_shared"
            elif len(lost) > 0 and len(gained) == 0:
                ref_kmers_used = lost
                alt_kmers_used = shared
                comparison = "shared_vs_lost"
            else:
                ref_kmers_used = shared
                alt_kmers_used = shared
                comparison = "shared_only"

            ref_regions, ref_off = _regions_for_kmers(ref_kmers_used, kidx)
            alt_regions, alt_off = _regions_for_kmers(alt_kmers_used, kidx)
            df = _build_probe_table(
                ref_regions,
                alt_regions,
                intens,
                ref_off,
                alt_off,
                fold_half=(not args.raw_lp),
                include_covariates=(not args.no_covariates),
                drop_overlap=(not args.keep_overlap),
            )

            coef, pval, stat = _fit_ref_alt_model(df, mode=args.mode, include_covariates=(not args.no_covariates))
            pval_str = _format_pval(pval, z=stat)

            print(
                f"{var}\t{coef}\t{pval_str}\t{len(lost)}\t{len(gained)}\t"
                f"{df.attrs.get('n_ref_regions',0)}\t{df.attrs.get('n_alt_regions',0)}\t"
                f"{df.attrs.get('n_overlap_dropped',0)}\t{df.attrs.get('n_kept',0)}\t{comparison}\t{len(ref_kmers_used)}\t{len(alt_kmers_used)}"
            )

            if args.debug:
                sys.stderr.write(
                    f"[DEBUG] {var} lost_kmers={len(lost)} gained_kmers={len(gained)} "
                    f"ref_regions={df.attrs.get('n_ref_regions',0)} alt_regions={df.attrs.get('n_alt_regions',0)} "
                    f"overlap_dropped={df.attrs.get('n_overlap_dropped',0)} probes_used={df.attrs.get('n_kept',0)} comparison={comparison} ref_kmers_used={len(ref_kmers_used)} alt_kmers_used={len(alt_kmers_used)}\n"
                )

        except Exception as e:
            print(f"{var}\t{math.nan}\t{math.nan}\t0\t0\t0\t0\t0\t0", file=sys.stdout)
            if args.debug:
                print(f"[ERROR] {var}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
