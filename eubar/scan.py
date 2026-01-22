#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math

from eubar.core.data import IntensityTable, KmerIndex
from eubar.core.sequence import RegionWindow
from eubar.core.matching import MotifMatcher
from eubar.core.design import DesignBuilder
from eubar.core.regression import RegressionEngine
from eubar.core.analyze import analyze_region_scan
from eubar.core.reporting import print_rows_as_tsv, plot_aff_motif_effects


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
                int(r.get("absolute_pos", int(r.get("motif_pos", 0)) + int(r.get("snv_index", 0)))),
                r.get("method", "NA"),
            ]
        )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Refactored scan-mode motif regression")
    p.add_argument("--intensities", required=True, help="Path to probe intensity file")
    p.add_argument("--kmerPositions", required=True, help="Path to k-mer array file mapping kmers to genomic regions")
    p.add_argument("--genome", required=True, help="FASTA genome file")
    p.add_argument("--region", required=True, help="chr:start-end (1-based inclusive)")
    p.add_argument("--kmer_size", type=int, default=8, help="K-mer size (default: 8)")
    p.add_argument("--mode", choices=["nb", "ols"], default="nb",  help="Regression mode (nb or ols)")
    p.add_argument("--no-covariates", action="store_true", help="Disable lp and sl covariates")
    p.add_argument("--reverse", action="store_true", help="Use reverse complement of the sequence")
    p.add_argument("--raw-lp", action="store_true", help="Use raw lp in [0,1] (no folding to [0,0.5])")
    p.add_argument("--save-figure", type=str, help="Filename to save figure (e.g. motif_plot.png)")
    args = p.parse_args(argv)

    intens = IntensityTable.from_file(args.intensities)
    kmers = KmerIndex.from_file(args.kmerPositions)

    region = RegionWindow.from_region_string(args.region, args.genome, reverse=args.reverse)
    matcher = MotifMatcher(kmers.kmers)
    design = DesignBuilder(intens.values)
    engine = RegressionEngine()

    rows = analyze_region_scan(
        region=region,
        matcher=matcher,
        design=design,
        engine=engine,
        k=args.kmer_size,
        mode=args.mode,
        include_covariates=(not args.no_covariates),
        fold_half=(not args.raw_lp),
    )
    legacy_rows = _to_legacy_rows(rows)
    print_rows_as_tsv(legacy_rows)
    if args.save_figure:
        plot_aff_motif_effects(legacy_rows, args.save_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
