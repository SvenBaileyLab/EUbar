#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from collections import defaultdict

from eubar.core.data import IntensityTable, KmerIndex
from eubar.core.sequence import RegionWindow
from eubar.core.matching import MotifMatcher
from eubar.core.design import DesignBuilder
from eubar.core.regression import RegressionEngine
from eubar.core.analyze import analyze_region_scan
from eubar.core.reporting import print_rows_as_tsv, plot_aff_motif_effects

from eubar.core.inspect import build_aff_payloads_for_region_scan
from eubar.core.pooled import (
    build_pooled_table,
    fit_pooled_model,
    effect_table_from_result,
)


def _to_legacy_rows(dict_rows: list[dict]) -> list[list]:
    """Convert internal dict rows to the legacy scan TSV row shape."""
    out: list[list] = []
    for r in dict_rows:
        out.append(
            [
                r.get("wildcard_kmer", ""),
                r.get("filled_kmer", ""),
                int(r.get("motif_pos", 0)),
                int(r.get("snv_index", 0)),
                r.get("label", "AFF"),
                r.get("allele", ""),
                r.get("coef", float("nan")),
                r.get("pval", float("nan")),
                int(
                    r.get(
                        "absolute_pos",
                        int(r.get("motif_pos", 0)) + int(r.get("snv_index", 0)),
                    )
                ),
                r.get("method", "NA"),
            ]
        )
    return out


def _nan(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _best_pval_rows(rows: list[list]) -> list[list]:
    """Reduce to one (coef,pval) per (absolute_position, allele).

    We pick the single row with the *smallest p-value* among all overlapping
    windows for the same allele at the same genomic position.

    The genomic position is defined consistently with plotting/reporting:
        absolute_position = window_index + snp_index + 1
    """

    best: dict[tuple[int, str], list] = {}

    def _as_float(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    for r in rows:
        if len(r) < 8:
            continue
        window_index = int(r[2])
        snp_index = int(r[3])
        allele = str(r[5])
        pval = _as_float(r[7])
        coef = _as_float(r[6])
        abspos = window_index + snp_index + 1
        key = (abspos, allele)

        # Skip unusable rows
        if math.isnan(pval) or pval <= 0:
            continue

        prev = best.get(key)
        if prev is None:
            best[key] = r
            continue

        prev_p = _as_float(prev[7])
        if (pval < prev_p) or (pval == prev_p and abs(coef) > abs(_as_float(prev[6]))):
            best[key] = r

    # Return in genomic order, keeping allele grouping stable
    out = list(best.values())
    out.sort(key=lambda r: (int(r[2]) + int(r[3]) + 1, str(r[5])))

    # Tag method/label for traceability if there's a method column
    for r in out:
        if len(r) >= 10:
            r[9] = (str(r[9]) if r[9] is not None else "") + "|best_pval"
    return out


def _pooled_scan_legacy_rows(
    *,
    region: RegionWindow,
    matcher: MotifMatcher,
    design: DesignBuilder,
    k: int,
    mode: str,
    include_covariates: bool,
    fold_half: bool,
) -> list[list]:
    """Pooled scan: one pooled fit per absolute position in the region.

    Groups all (motif_pos, snv_index) windows that correspond to the same absolute
    position (motif_pos + snv_index) and fits:

      log1p(y) ~ allele + window + (optional covariates)
    """

    payloads = build_aff_payloads_for_region_scan(
        region,
        matcher=matcher,
        design=design,
        k=k,
        include_covariates=include_covariates,
        fold_half=fold_half,
    )

    by_abs: dict[int, dict[tuple[int, int], object]] = defaultdict(dict)
    for (motif_pos, snv_index), payload in payloads.items():
        abs_pos = int(motif_pos) + int(snv_index)
        by_abs[abs_pos][(motif_pos, snv_index)] = payload

    alleles = ["A", "C", "G", "T"]
    out: list[list] = []

    for abs_pos in sorted(by_abs.keys()):
        group = by_abs[abs_pos]
        rep_key = sorted(group.keys())[0]
        rep = group[rep_key]

        # All windows in this group should share the same ref allele (the base at abs_pos).
        ref = getattr(rep, "ref_allele", None) or (
            region.seq[abs_pos] if 0 <= abs_pos < len(region.seq) else None
        )
        if ref not in {"A", "C", "G", "T"}:
            # Skip weird bases (N) in the reference sequence.
            continue

        try:
            pooled_df = build_pooled_table(group, kind="AFF")
            res = fit_pooled_model(
                pooled_df,
                ref_allele=ref,
                include_covariates=include_covariates,
                mode=mode,
            )
            eff = effect_table_from_result(
                res,
                ref_allele=ref,
                alleles_to_report=alleles,
                include_fold_change=False,
            )
        except Exception:
            # If a position can't be fit (too few rows, singular design, etc.), skip it.
            continue

        wildcard_kmer = getattr(rep, "wildcard_kmer", "") or ""
        filled_kmer = getattr(rep, "local_kmer", "") or ""

        for _, r in eff.iterrows():
            out.append(
                [
                    wildcard_kmer,
                    filled_kmer,
                    int(abs_pos),  # window_index (repurposed to absolute position)
                    0,  # snp_index (fixed)
                    "AFF",
                    r.get("allele", ""),
                    _nan(r.get("effect")),
                    _nan(r.get("pval")),
                    int(abs_pos),
                    f"pooled_{mode}",
                ]
            )

    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Refactored scan-mode motif regression")
    p.add_argument("--intensities", required=True, help="Path to probe intensity file")
    p.add_argument(
        "--array",
        "--kmerPositions",
        dest="array",
        required=True,
        help="Path to k-mer array file mapping k-mers to genomic regions",
    )
    p.add_argument("--genome", required=True, help="Path to reference genome in FASTA format")
    p.add_argument("--region", required=True, help="chr:start-end (1-based inclusive)")
    p.add_argument(
        "--kmer-size",
        "--kmer_size",
        dest="kmer_size",
        type=int,
        default=8,
        help="K-mer size (default: 8)",
    )

    p.add_argument(
        "--mode",
        choices=["ols", "nb"],
        default="ols",
        help="Regression mode (ols or nb)",
    )

    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--pooled",
        action="store_true",
        help="Pool overlapping windows per absolute position and fit one pooled model per position",
    )
    grp.add_argument(
        "--best-pval", dest="best_pval",
        action="store_true",
        help="Summarize non-pooled scan by choosing, for each (position,allele), the overlapping window with the smallest p-value",
    )
    p.add_argument(
        "--no-covariates", action="store_true", help="Disable lp and sl covariates"
    )
    p.add_argument(
        "--reverse", action="store_true", help="Use reverse complement of the sequence"
    )
    p.add_argument(
        "--raw-lp",
        action="store_true",
        help="Use raw lp in [0,1] (no folding to [0,0.5])",
    )
    p.add_argument(
        "--max-probes", type=int, default=None, dest="max_probes",
        help="Subsample to at most this many probes per window before fitting."
    )
    p.add_argument(
        "--save-figure", type=str, help="Filename to save figure (e.g. motif_plot.png)"
    )
    args = p.parse_args(argv)

    intens = IntensityTable.from_file(args.intensities)
    kmers = KmerIndex.from_file(args.array)

    region = RegionWindow.from_region_string(
        args.region, args.genome, reverse=args.reverse
    )
    matcher = MotifMatcher(kmers.kmers)
    design = DesignBuilder(intens.values)
    engine = RegressionEngine()

    if args.pooled:
        legacy_rows = _pooled_scan_legacy_rows(
            region=region,
            matcher=matcher,
            design=design,
            k=args.kmer_size,
            mode=args.mode,
            include_covariates=(not args.no_covariates),
            fold_half=(not args.raw_lp),
        )
    else:
        rows = analyze_region_scan(
            region=region,
            matcher=matcher,
            design=design,
            engine=engine,
            k=args.kmer_size,
            mode=args.mode,
            include_covariates=(not args.no_covariates),
            fold_half=(not args.raw_lp),
            max_probes=args.max_probes,
        )
        legacy_rows = _to_legacy_rows(rows)
        if args.best_pval:
            legacy_rows = _best_pval_rows(legacy_rows)

    print_rows_as_tsv(legacy_rows)
    if args.save_figure:
        plot_aff_motif_effects(legacy_rows, args.save_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
