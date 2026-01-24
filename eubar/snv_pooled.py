#!/usr/bin/env python3
"""snv_pooled.py

Pooled (across-window) allele effects for a single SNV, using the same AFF
"sliding-window" mechanics as eubar, but reporting ONE effect/p-value per allele.

Key idea
--------
Instead of fitting 8 separate regressions (one per window), we *pool* all probes
from all windows into one long table and fit:

  log1p(y) ~ allele + window + (optional covariates)

- `allele` is a categorical factor with the SNV's REF allele as baseline.
- `window` absorbs systematic differences across the 8 windows.
- We cluster SEs by probe/region because the same probe can appear in multiple
  windows (so rows are not independent).

Interpretation
--------------
- effect (coef) is on log(y+1) scale. Positive => higher (y+1) for that allele.
- fold_change_(y+1) = exp(effect). e.g. 1.15 ≈ +15% on (y+1).

By default this script reports BOTH:
  - AFF pooled effects (union of allele-specific k-mer hits around the SNV)
  - RAND pooled effects (random probes + the AFF union, matching the legacy Perl rNegBinom idea)

Output is long-form TSV by default:
  snv\ttype\tallele\teffect\tpval

That is 8 lines per SNV (4 alleles × 2 types).
"""

from __future__ import annotations

import argparse
import csv
import inspect
import sys

from eubar.core.data import IntensityTable, KmerIndex
from eubar.core.design import DesignBuilder
from eubar.core.matching import MotifMatcher
from eubar.core.sequence import SnvWindow

# Pooled helpers live in core.
from eubar.core.pooled import (
    parse_snv_str,
    call_with_accepted_kwargs,
    build_pooled_table,
    build_pooled_table_rand,
    fit_pooled_model,
    fit_pooled_model_rand,
    effect_table_from_result,
    effect_table_from_result_rand,
    fmt_pval,
)


# ---- robust import for payload builders -------------------------------------

def _import_build_aff_payloads():
    """Try a few module locations; keeps the CLI compatible across repo layouts."""
    candidates = [
        ("eubar.core.analyze", "build_aff_payloads_for_snv"),
        ("eubar.core.inspect", "build_aff_payloads_for_snv"),
        ("eubar.core.reporting", "build_aff_payloads_for_snv"),
    ]
    last = None
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name)
        except Exception as e:
            last = e
    raise ImportError(
        "Could not import build_aff_payloads_for_snv from expected modules. "
        "Try adding it to eubar.core.analyze (or adjust the import list in snv_pooled.py)."
    ) from last


build_aff_payloads_for_snv = _import_build_aff_payloads()


def _import_build_rand_payloads():
    """Try a few module locations for the RAND payload builder."""
    candidates = [
        ("eubar.core.analyze", "build_rand_payloads_for_snv"),
        ("eubar.core.inspect", "build_rand_payloads_for_snv"),
        ("eubar.core.reporting", "build_rand_payloads_for_snv"),
    ]
    last = None
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name)
        except Exception as e:
            last = e
    raise ImportError(
        "Could not import build_rand_payloads_for_snv from expected modules. "
        "Try adding it to eubar.core.analyze (or adjust the import list in snv_pooled.py)."
    ) from last


build_rand_payloads_for_snv = _import_build_rand_payloads()


# ---- CLI -------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Pooled allele effects across the 8 sliding windows for SNV(s). "
            "Outputs long-form TSV: snv, type, allele, effect, pval (optionally fold_change)."
        )
    )

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--snv-list", help="Comma-separated SNVs like chr:pos:REF>ALT")
    grp.add_argument("--snv-list-file", help="File with one SNV per line")

    ap.add_argument("--intensities", required=True)
    ap.add_argument("--kmerPositions", required=True)
    ap.add_argument("--genome", required=True)

    ap.add_argument("--k", type=int, default=8)
    ap.add_argument(
        "--mode",
        choices=["ols", "nb"],
        default="ols",
        help="Model family fit on log1p(y)",
    )
    ap.add_argument("--no-covariates", action="store_true")
    ap.add_argument("--raw-lp", action="store_true", help="Do not fold lp to [0,0.5]")

    ap.add_argument(
        "--rand-n",
        type=int,
        default=500,
        help="Number of random probes per window (RAND)",
    )
    ap.add_argument(
        "--min-n",
        type=int,
        default=10,
        help="Minimum number of rows required to fit a RAND window (after filtering)",
    )
    ap.add_argument("--no-rand", action="store_true", help="Skip RAND pooled fits")

    ap.add_argument("--with-fc", action="store_true", help="Include fold_change_(y+1) in output")
    ap.add_argument("--output", type=str, default=None, help="Write TSV here (default: stdout)")
    ap.add_argument("--print-summary", action="store_true", help="Print the full statsmodels summary")

    args = ap.parse_args(argv)

    intens = IntensityTable.from_file(args.intensities)
    kmers = KmerIndex.from_file(args.kmerPositions)

    matcher = MotifMatcher(kmers.kmers)

    # Compatibility shim: some versions expose match(...) vs match_window(...)
    if not hasattr(matcher, "match_window") and hasattr(matcher, "match"):
        setattr(matcher, "match_window", getattr(matcher, "match"))

    design = DesignBuilder(intens.values)

    if args.snv_list_file:
        with open(args.snv_list_file) as f:
            snvs = [ln.strip() for ln in f if ln.strip()]
    else:
        snvs = [s.strip() for s in (args.snv_list or "").split(",") if s.strip()]

    alleles = ["A", "C", "G", "T"]

    out_fh = sys.stdout if args.output is None else open(args.output, "w", newline="")
    try:
        w = csv.writer(out_fh, delimiter="\t")
        header = ["snv", "type", "allele", "effect", "pval"]
        if args.with_fc:
            header.insert(4, "fold_change_(y+1)")
        w.writerow(header)

        for snv_str in snvs:
            try:
                _, _, ref, _alts = parse_snv_str(snv_str)
                if ref not in {"A", "C", "G", "T"}:
                    raise ValueError(f"Bad ref allele in SNV: {ref}")

                snv = SnvWindow.from_snv(snv_str, args.genome, k=args.k)

                # -------------------------
                # AFF pooled
                # -------------------------
                aff = call_with_accepted_kwargs(
                    build_aff_payloads_for_snv,
                    snv,
                    matcher=matcher,
                    design=design,
                    k=args.k,
                    include_covariates=(not args.no_covariates),
                    fold_half=(not args.raw_lp),
                )
                pooled_aff = build_pooled_table(aff, kind="AFF")
                res_aff = fit_pooled_model(
                    pooled_aff,
                    ref_allele=ref,
                    include_covariates=(not args.no_covariates),
                    mode=args.mode,
                )
                out_aff = effect_table_from_result(
                    res_aff,
                    ref_allele=ref,
                    alleles_to_report=alleles,
                    include_fold_change=args.with_fc,
                )
                for _, r in out_aff.iterrows():
                    row = [snv_str, "AFF", r["allele"], r.get("effect"), fmt_pval(r.get("pval"))]
                    if args.with_fc:
                        row.insert(4, r.get("fold_change_(y+1)"))
                    w.writerow(row)

                if args.print_summary:
                    print(str(res_aff.summary()), file=sys.stderr)

                # -------------------------
                # RAND pooled (optional)
                # -------------------------
                if not args.no_rand:
                    rand = call_with_accepted_kwargs(
                        build_rand_payloads_for_snv,
                        snv=snv,
                        snv_str=snv_str,
                        matcher=matcher,
                        design=design,
                        k=args.k,
                        rand_n=args.rand_n,
                        fold_half=(not args.raw_lp),
                        min_n=getattr(args, "min_n", 10),
                    )
                    pooled_rand = build_pooled_table_rand(rand, snv=snv, kind="RAND")
                    res_rand = fit_pooled_model_rand(
                        pooled_rand,
                        include_covariates=(not args.no_covariates),
                        mode=args.mode,
                    )
                    out_rand = effect_table_from_result_rand(
                        res_rand,
                        alleles_to_report=alleles,
                        include_fold_change=args.with_fc,
                    )
                    for _, r in out_rand.iterrows():
                        row = [snv_str, "RAND", r["allele"], r.get("effect"), fmt_pval(r.get("pval"))]
                        if args.with_fc:
                            row.insert(4, r.get("fold_change_(y+1)"))
                        w.writerow(row)

            except Exception as e:
                print(f"[ERROR] {snv_str}: failed to build pooled table: {e}", file=sys.stderr)

    finally:
        if out_fh is not sys.stdout:
            out_fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
