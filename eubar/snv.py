#!/usr/bin/env python3

from __future__ import annotations

import csv
import argparse
import sys
import math

import numpy as np

from eubar.core.data import IntensityTable, KmerIndex
from eubar.core.sequence import SnvWindow
from eubar.core.matching import MotifMatcher
from eubar.core.design import DesignBuilder
from eubar.core.regression import RegressionEngine
from eubar.core.analyze import analyze_snv
from eubar.core.reporting import print_motif_effect_table


# Optional: for log-space p-value recovery (prevents 0.0 underflow)
try:
    from scipy.stats import norm  # type: ignore
except Exception:
    norm = None


def _extract_stat(r: dict) -> float:
    """Try to extract a test statistic from a row dict (z/t)."""
    for k in ("stat", "z", "zval", "zvalue", "t", "tval", "tvalue"):
        v = r.get(k)
        if v is None:
            continue
        try:
            x = float(v)
            if math.isfinite(x):
                return x
        except Exception:
            continue
    return math.nan


def _format_pval(p: float, stat: float = math.nan) -> str:
    """Format p-values robustly (never print 0.0)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "nan"

    # Normal case
    if p > 0.0:
        return f"{p:.3e}"

    # Underflow case: p == 0.0
    # If we have a test statistic and scipy is available, recover p in log-space.
    if norm is not None and isinstance(stat, float) and math.isfinite(stat):
        # two-sided p = 2*sf(|z|)
        logp = math.log(2.0) + float(norm.logsf(abs(stat)))
        log10p = logp / math.log(10.0)

        # Represent as mantissa*10^exp string, even when too small for float
        exp10 = int(math.floor(log10p))
        mant = 10 ** (log10p - exp10)
        return f"{mant:.2f}e{exp10}"

    # Fallback when we can't recover: avoid printing 0.0
    return "<1e-300"


def _best_pval_summary_rows(rows):
    """Return one summary row per (label,type) x allele by taking the window with smallest p-value.

    This is meant to mimic snv_pooled's output shape, but using the per-window SNV results from snv.py.
    """

    # collect by (label, allele)
    # store: (pval, coef, stat)
    best = {}  # (label, allele) -> (pval, coef, stat)

    for r in rows:
        label = r.get("label", "AFF")
        allele = r.get("allele")
        if allele not in ("A", "C", "G", "T"):
            continue

        # parse pval/coef (rows can contain 'NA' strings)
        p = r.get("pval")
        c = r.get("coef")
        try:
            p = float(p)
        except Exception:
            p = math.nan
        try:
            c = float(c)
        except Exception:
            c = math.nan

        stat = _extract_stat(r)

        key = (label, allele)

        if math.isnan(p):
            # keep NaN only if we have nothing yet
            if key not in best:
                best[key] = (math.nan, c, stat)
            continue

        if key not in best:
            best[key] = (p, c, stat)
            continue

        prev_p, prev_c, prev_stat = best[key]
        # prefer any numeric p over NaN, and smaller p wins
        if math.isnan(prev_p) or p < prev_p:
            best[key] = (p, c, stat)
        else:
            # keep previous best
            best[key] = (prev_p, prev_c, prev_stat)

    # emit in stable order
    out = []
    for label in ("AFF", "RAND"):
        for allele in ("A", "C", "G", "T"):
            p, c, stat = best.get((label, allele), (math.nan, math.nan, math.nan))
            out.append({"type": label, "allele": allele, "effect": c, "pval": p, "stat": stat})
    return out


def _print_best_pval_table(snv_str, rows, *, include_rand=True):
    """Print a snv_pooled-like table: snv, type, allele, effect, pval."""
    summary = _best_pval_summary_rows(rows)
    for r in summary:
        if (not include_rand) and r["type"] == "RAND":
            continue
        effect = r["effect"]
        pval = r["pval"]
        stat = r.get("stat", math.nan)
        pval_str = _format_pval(pval, stat=stat)
        # Match snv_pooled style: 'nan' for missing
        print(f"{snv_str}\t{r['type']}\t{r['allele']}\t{effect}\t{pval_str}")


def _format_cell(v):
    """Keep legacy 'NA' strings; render NaN/None as 'NA'."""
    if v is None:
        return "NA"
    if isinstance(v, float) and (v != v):
        return "NA"
    return v


def _append_long_tsv(out_path, snv_str, seq, k, dict_rows):
    """Append scan-style rows to a TSV file, with SNV as first column."""
    header = [
        "snv",
        "wildcard_kmer",
        "filled_kmer",
        "window_index",
        "snv_index",
        "type",
        "allele",
        "coef",
        "pval",
        "absolute_pos",
        "method",
    ]
    # compute snv_index per motif_pos from sequence length: snv is centered at k-1
    center = k - 1
    # motif_pos ranges over windows that cover center
    snv_index_by_motif = {j: center - j for j in range(k)}

    rows = []
    for r in dict_rows:
        motif_pos = int(r.get("motif_pos", 0))
        snv_index = int(r.get("snv_index", snv_index_by_motif.get(motif_pos, 0)))
        local_kmer = seq[motif_pos : motif_pos + k]
        if len(local_kmer) != k:
            continue
        wildcard_kmer = r.get("wildcard_kmer")
        filled_kmer = r.get("filled_kmer")
        allele = r.get("allele", "")
        if not wildcard_kmer:
            w = list(local_kmer)
            w[snv_index] = "."
            wildcard_kmer = "".join(w)
        if not filled_kmer:
            f = list(local_kmer)
            if allele:
                f[snv_index] = allele
            filled_kmer = "".join(f)

        absolute_pos = int(r.get("absolute_pos", motif_pos + snv_index))
        method = r.get("method", "NA")
        row = [
            snv_str,
            wildcard_kmer,
            filled_kmer,
            motif_pos,
            snv_index,
            r.get("label", ""),
            allele,
            _format_cell(r.get("coef")),
            _format_cell(r.get("pval")),
            absolute_pos,
            method,
        ]
        rows.append(row)

    # write header if file doesn't exist or empty
    need_header = True
    try:
        import os

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            need_header = False
    except Exception:
        pass

    with open(out_path, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if need_header:
            w.writerow(header)
        for row in rows:
            w.writerow(row)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run motif regression on SNV(s)")
    group = p.add_mutually_exclusive_group(required=True)
    p.add_argument("--intensities", required=True, help="Path to probe intensity file")
    p.add_argument("--kmerPositions", required=True, help="Path to k-mer array file mapping kmers to genomic regions")
    p.add_argument("--genome", required=True, help="Path to reference genome in FASTA format")
    group.add_argument("--snv-list", help="Comma-separated list of SNVs in chr:pos:ref>alt format")
    group.add_argument("--snv-list-file", help="Optional file with SNVs, one per line in chr:pos:ref>alt format")
    p.add_argument("--kmer_size", type=int, default=8, help="K-mer size (default: 8)")
    p.add_argument("--rand-n", type=int, default=500, help="Number of random probes to use for RAND regression (default: 500)")
    p.add_argument("--no-rand", action="store_true", help="Skip RAND regression and only output AFF (still prints the motif-effect table)")
    p.add_argument(
        "--best_pval",
        action="store_true",
        help="Summarize results by choosing, for each (type, allele), the motif_pos with the smallest p-value and printing one effect+pval per allele in snv_pooled-like TSV format",
    )
    p.add_argument("--output_long", type=str, help="Write a scan-style long TSV (with SNV as first column) to this path")
    p.add_argument("--mode", choices=["nb", "ols"], default="ols", help="Regression type: negative binomial ('nb') or ordinary least squares ('ols')")
    p.add_argument("--no-covariates", action="store_true")
    p.add_argument("--raw-lp", action="store_true", help="Use raw lp in [0,1] (no folding to [0,0.5])")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    if args.best_pval:
        print("snv\ttype\tallele\teffect\tpval")

    intens = IntensityTable.from_file(args.intensities)
    kmers = KmerIndex.from_file(args.kmerPositions)
    matcher = MotifMatcher(kmers.kmers)
    design = DesignBuilder(intens.values)
    engine = RegressionEngine()

    if args.snv_list_file:
        with open(args.snv_list_file) as f:
            snvs = [line.strip() for line in f if line.strip()]
    else:
        snvs = [s.strip() for s in (args.snv_list or "").split(",") if s.strip()]

    for snv_str in snvs:
        try:
            snv = SnvWindow.from_snv(snv_str, args.genome, k=args.kmer_size, debug=args.debug)
        except Exception as e:
            print(f"[ERROR] {snv_str}: {e}", file=sys.stderr)
            continue

        rows = analyze_snv(
            snv=snv,
            snv_str=snv_str,
            matcher=matcher,
            design=design,
            engine=engine,
            k=args.kmer_size,
            mode=args.mode,
            include_covariates=(not args.no_covariates),
            fold_half=(not args.raw_lp),
            rand_n=(0 if args.no_rand else args.rand_n),
        )

        if args.best_pval:
            _print_best_pval_table(snv_str, rows, include_rand=(not args.no_rand))
        else:
            print_motif_effect_table(snv_str, snv.chrom, snv.pos, snv.seq, rows)

        if args.output_long:
            _append_long_tsv(args.output_long, snv_str, snv.seq, args.kmer_size, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
