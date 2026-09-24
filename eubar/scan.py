#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict
from eubar.task_config import ScanConfig, TaskConfigError

import argparse
import math
import sys
from contextlib import redirect_stdout
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


def _holm_scan(args, region, matcher, design, engine):
    """Saturation mutagenesis using the exact SNV fits and within-SNV families.

    Compact output follows ``snv --best-pval --holm``. Genome positions are
    1-based; reverse-mode alleles are on the reverse-complement strand.
    Windows include flanking sequence outside the requested scan interval.
    """
    from eubar.core.sequence import SnvWindow
    from eubar.core.analyze import analyze_snv
    from eubar.core.snv_results import _apply_holm_within_snv, _best_pval_summary_rows
    from eubar.core.reporting import _print_best_pval_table

    header = "snv\ttype\tallele\teffect\tpval\traw_pval"
    if args.diagnostics:
        header += "\tmotif_pos\twildcard_kmer\tn_probes\tn_allele"
    print(header)
    plot_rows = []
    for offset, ref in enumerate(region.seq):
        pos = region.end - offset if args.reverse else region.start + offset
        if ref not in "ACGT":
            print(f"[WARNING] Skipping {region.chrom}:{pos}: reference {ref}", file=sys.stderr)
            continue
        for alt in "ACGT":
            if alt == ref:
                continue
            snv_str = f"{region.chrom}:{pos}:{ref}>{alt}"
            snv = SnvWindow.from_snv(snv_str, args.genome, k=args.kmer_size)
            # Avoid silently truncated windows at contig boundaries.
            if len(snv.seq) != 2 * args.kmer_size - 1:
                raise ValueError(f"{snv_str}: insufficient reference flanking sequence")
            rows = analyze_snv(
                snv=snv, snv_str=snv_str, matcher=matcher, design=design,
                engine=engine, k=args.kmer_size, mode=args.mode,
                include_covariates=not args.no_covariates,
                fold_half=not args.raw_lp,
                rand_n=0 if args.no_rand else args.rand_n,
                max_probes=args.max_probes, seed=args.seed,
            )
            rows = _apply_holm_within_snv(rows, snv_str)
            _print_best_pval_table(
                snv_str, rows, include_rand=not args.no_rand,
                diagnostics=args.diagnostics, snv=snv, matcher=matcher,
                design=design, k=args.kmer_size, rand_n=args.rand_n,
                fold_half=not args.raw_lp, max_probes=args.max_probes,
                seed=args.seed, holm=True,
            )
            if args.save_figure:
                aff = _best_pval_summary_rows(rows, snv_str, use_holm=True)[0]
                plot_rows.append([
                    "", "", offset, 0, "AFF", alt, aff["effect"],
                    aff["pval"], offset, "snv_holm_best_pval",
                ])
    if args.save_figure:
        if any(math.isfinite(r[6]) and math.isfinite(r[7]) for r in plot_rows):
            # Offsets and alleles already follow the requested strand.
            with redirect_stdout(sys.stderr):
                plot_aff_motif_effects(plot_rows, args.save_figure, reverse=False)
        else:
            print("[WARNING] No finite AFF results; figure not written", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    defaults = ScanConfig()
    p = argparse.ArgumentParser(description="Refactored scan-mode motif regression")
    p.add_argument("--intensities", required=True, help="Path to probe intensity file")
    array_group = p.add_mutually_exclusive_group(required=True)
    array_group.add_argument(
        "--array",
        dest="array",
        help="Path to k-mer array file mapping k-mers to genomic regions",
    )
    array_group.add_argument(
        "--kmerPositions", dest="array", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("--genome", required=True, help="Path to reference genome in FASTA format")
    p.add_argument("--region", required=True, help="chr:start-end (1-based inclusive)")
    p.add_argument(
        "--kmer-size",
        dest="kmer_size",
        type=int,
        default=defaults.kmer_size,
        help="K-mer size (default: 8)",
    )
    p.add_argument(
        "--kmer_size", dest="kmer_size", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    p.add_argument(
        "--mode", choices=["ols", "nb"], default=defaults.mode, help=argparse.SUPPRESS,
    )

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--pooled", action="store_true", help=argparse.SUPPRESS)
    grp.add_argument(
        "--best-pval", dest="best_pval",
        action="store_true",
        help="Summarize non-pooled scan by minimum p-value; with --holm, use SNV fits and same-window RAND support instead",
    )
    p.add_argument("--no-covariates", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--reverse", action="store_true", help="Use reverse complement of the sequence"
    )
    p.add_argument("--raw-lp", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=defaults.seed, help="Seed for max-probes subsampling (default: 0); RAND background retains its existing deterministic sampling.")
    p.add_argument(
        "--max-probes", type=int, default=defaults.max_probes, dest="max_probes",
        help="Subsample to at most this many probes per window before fitting."
    )
    p.add_argument(
        "--save-figure", type=str, help="Filename to save figure (e.g. motif_plot.png)"
    )
    p.add_argument(
        "--holm", action="store_true",
        help="With --best-pval, use SNV fits and within-SNV Holm correction "
             "(ALT AFF and combined REF/ALT RAND families). Outputs compact SNV "
             "TSV with adjusted pval and raw_pval; includes flanking windows. "
             "This does not correct across the whole region.",
    )
    p.add_argument("--rand-n", type=int, default=defaults.rand_n,
                   help="RAND background size for --holm (default: 500)")
    p.add_argument("--no-rand", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--diagnostics", action="store_true",
                   help="Append SNV diagnostics in --best-pval --holm mode")
    p.set_defaults(**asdict(defaults))
    return p


def run_scan(config: ScanConfig) -> int:
    """Run scan with validated task options; no command-line parsing."""
    config.validate()
    args = config
    intens = IntensityTable.from_file(args.intensities)
    kmers = KmerIndex.from_file(args.array)

    region = RegionWindow.from_region_string(
        args.region, args.genome, reverse=args.reverse
    )
    matcher = MotifMatcher(kmers.kmers)
    design = DesignBuilder(intens.values)
    engine = RegressionEngine()

    if args.holm:
        return _holm_scan(args, region, matcher, design, engine)

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
            seed=args.seed,
        )
        legacy_rows = _to_legacy_rows(rows)
        if args.best_pval:
            legacy_rows = _best_pval_rows(legacy_rows)

    print_rows_as_tsv(legacy_rows)
    if args.save_figure:
        plot_aff_motif_effects(legacy_rows, args.save_figure, reverse=args.reverse)
    return 0



def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = ScanConfig.from_values(vars(args))
        return run_scan(config)
    except TaskConfigError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
