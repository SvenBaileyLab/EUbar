#!/usr/bin/env python3

from __future__ import annotations

import csv
import argparse
import sys
from eubar.core.data import IntensityTable, KmerIndex
from eubar.core.sequence import SnvWindow
from eubar.core.matching import MotifMatcher
from eubar.core.design import DesignBuilder
from eubar.core.regression import RegressionEngine
from eubar.core.analyze import analyze_snv
from eubar.core.reporting import print_motif_effect_table

def _format_cell(v):
    """Keep legacy 'NA' strings; render NaN/None as 'NA'."""
    if v is None:
        return 'NA'
    if isinstance(v, float) and (v != v):
        return 'NA'
    return v


def _append_long_tsv(out_path, snv_str, seq, k, dict_rows):
    """Append scan-style rows to a TSV file, with SNV as first column."""
    header = ['snv','wildcard_kmer','filled_kmer','window_index','snv_index','type','allele','coef','pval','absolute_pos','method']
    # compute snv_index per motif_pos from sequence length: snv is centered at k-1
    center = k - 1
    # motif_pos ranges over windows that cover center
    snv_index_by_motif = {j: center - j for j in range(k)}

    rows = []
    for r in dict_rows:
        motif_pos = int(r.get('motif_pos', 0))
        snv_index = int(r.get('snv_index', snv_index_by_motif.get(motif_pos, 0)))
        local_kmer = seq[motif_pos:motif_pos+k]
        if len(local_kmer) != k:
            continue
        wildcard_kmer = r.get('wildcard_kmer')
        filled_kmer = r.get('filled_kmer')
        allele = r.get('allele','')
        if not wildcard_kmer:
            w = list(local_kmer)
            w[snv_index] = '.'
            wildcard_kmer = ''.join(w)
        if not filled_kmer:
            f = list(local_kmer)
            if allele:
                f[snv_index] = allele
            filled_kmer = ''.join(f)

        absolute_pos = int(r.get('absolute_pos', motif_pos + snv_index))
        method = r.get('method', 'NA')
        row = [snv_str, wildcard_kmer, filled_kmer, motif_pos, snv_index, r.get('label',''), allele, _format_cell(r.get('coef')), _format_cell(r.get('pval')), absolute_pos, method]
        rows.append(row)

    # write header if file doesn't exist or empty
    need_header = True
    try:
        import os
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            need_header = False
    except Exception:
        pass

    with open(out_path, 'a', newline='') as f:
        w = csv.writer(f, delimiter='	')
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
    p.add_argument("--output_long", type=str, help="Write a scan-style long TSV (with SNV as first column) to this path")
    p.add_argument("--mode", choices=["nb", "ols"], default="nb", help="Regression type: negative binomial ('nb') or ordinary least squares ('ols')")
    p.add_argument("--no-covariates", action="store_true")
    p.add_argument("--raw-lp", action="store_true", help="Use raw lp in [0,1] (no folding to [0,0.5])")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

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

        print_motif_effect_table(snv_str, snv.chrom, snv.pos, snv.seq, rows)
        if args.output_long:
            _append_long_tsv(args.output_long, snv_str, snv.seq, args.kmer_size, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
