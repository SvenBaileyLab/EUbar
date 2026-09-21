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
from eubar.core.inspect import (
    build_aff_payloads_for_snv,
    build_rand_payloads_for_snv,
    probe_diagnostics,
)


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

    if p > 0.0:
        return f"{p:.3e}"

    # Underflow: recover p in log-space if we have a test statistic.
    if norm is not None and isinstance(stat, float) and math.isfinite(stat):
        logp = math.log(2.0) + float(norm.logsf(abs(stat)))
        log10p = logp / math.log(10.0)
        exp10 = int(math.floor(log10p))
        mant = 10 ** (log10p - exp10)
        return f"{mant:.2f}e{exp10}"

    return "<1e-300"


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def _parse_ref_alt_from_snv(snv_str: str):
    try:
        allele_part = snv_str.split(":", 2)[2]
        ref, alt = allele_part.split(">", 1)
        ref = ref.strip().upper()
        alt = alt.strip().upper()
        if ref in {"A", "C", "G", "T"} and alt in {"A", "C", "G", "T"}:
            return ref, alt
    except Exception:
        pass
    return None, None


def _holm_adjust(pvals):
    """Return Holm-adjusted p-values, preserving NaNs.

    Holm controls the family-wise error rate under arbitrary dependence.
    This implementation is dependency-free and equivalent to the standard
    step-down Holm procedure.
    """
    out = [math.nan] * len(pvals)
    valid = []
    for i, p in enumerate(pvals):
        p = _safe_float(p)
        if math.isfinite(p):
            valid.append((i, min(max(p, 0.0), 1.0)))

    m = len(valid)
    if m == 0:
        return out

    valid.sort(key=lambda x: x[1])
    running = 0.0
    for rank, (idx, p) in enumerate(valid, start=1):
        adjusted = (m - rank + 1) * p
        running = max(running, adjusted)
        out[idx] = min(running, 1.0)
    return out


def _apply_holm_within_snv(rows, snv_str: str):
    """Annotate rows with ``pval_holm`` for the tests used by --best-pval.

    Families are corrected separately:
      * AFF: ALT-allele AFF tests across motif positions (normally k tests).
      * RAND: REF + ALT RAND tests across motif positions (normally 2*k tests).

    Other rows are left with pval_holm=NaN because they are not part of the
    --best-pval decision for the requested SNV. Raw ``pval`` is never changed.
    """
    ref, alt = _parse_ref_alt_from_snv(snv_str)
    out = [dict(r) for r in rows]
    for r in out:
        r["pval_holm"] = math.nan

    if alt is None:
        return out

    aff_idx = [
        i for i, r in enumerate(out)
        if r.get("label", "AFF") == "AFF" and r.get("allele") == alt
        and math.isfinite(_safe_float(r.get("pval")))
    ]
    aff_adj = _holm_adjust([out[i].get("pval") for i in aff_idx])
    for i, q in zip(aff_idx, aff_adj):
        out[i]["pval_holm"] = q

    rand_alleles = {a for a in (ref, alt) if a is not None}
    rand_idx = [
        i for i, r in enumerate(out)
        if r.get("label", "AFF") == "RAND" and r.get("allele") in rand_alleles
        and math.isfinite(_safe_float(r.get("pval")))
    ]
    rand_adj = _holm_adjust([out[i].get("pval") for i in rand_idx])
    for i, q in zip(rand_idx, rand_adj):
        out[i]["pval_holm"] = q

    return out


def _best_supported_motif_pos(rows, snv_str: str, *, use_holm: bool = False):
    """Choose one motif position per SNP using ALT AFF first, then RAND support.

    AFF candidates are ranked by their raw p-value (Holm preserves this order).
    With ``use_holm=True``, RAND support requires the Holm-adjusted RAND p-value
    to be < 0.05; otherwise the legacy raw RAND p-value is used.
    """
    ref, alt = _parse_ref_alt_from_snv(snv_str)
    if alt is None:
        return None

    by_key = {}
    candidates = []

    for r in rows:
        label = r.get("label", "AFF")
        allele = r.get("allele")
        if label not in ("AFF", "RAND") or allele not in ("A", "C", "G", "T"):
            continue

        motif_pos = r.get("motif_pos")
        try:
            motif_pos = int(motif_pos)
        except Exception:
            continue

        p_raw = _safe_float(r.get("pval"))
        p_holm = _safe_float(r.get("pval_holm"))
        c = _safe_float(r.get("coef"))

        by_key[(motif_pos, label, allele)] = {
            "effect": c,
            "pval": p_raw,
            "pval_holm": p_holm,
        }

        if label == "AFF" and allele == alt and not math.isnan(p_raw):
            candidates.append((p_raw, -abs(c) if not math.isnan(c) else 0.0, motif_pos))

    if not candidates:
        return None

    candidates.sort()

    def _rand_passes(r):
        if r is None:
            return False
        p = r["pval_holm"] if use_holm else r["pval"]
        c = r["effect"]
        return not math.isnan(p) and not math.isnan(c) and p < 0.05 and c > 0.0

    def _passes_rand_support(motif_pos):
        if ref is None:
            return False
        return (
            _rand_passes(by_key.get((motif_pos, "RAND", alt)))
            or _rand_passes(by_key.get((motif_pos, "RAND", ref)))
        )

    for _, _, motif_pos in candidates:
        if _passes_rand_support(motif_pos):
            return motif_pos

    return candidates[0][2]


def _best_pval_summary_rows(rows, snv_str: str, *, use_holm: bool = False):
    """Return three rows for the chosen motif position: AFF alt, RAND ref, RAND alt."""
    ref, alt = _parse_ref_alt_from_snv(snv_str)
    chosen_pos = _best_supported_motif_pos(rows, snv_str, use_holm=use_holm)

    def _find(label, allele):
        if chosen_pos is None:
            return None
        for r in rows:
            if (
                r.get("label", "AFF") == label
                and r.get("allele") == allele
                and _safe_float(r.get("motif_pos")) == float(chosen_pos)
            ):
                raw_p = _safe_float(r.get("pval"))
                holm_p = _safe_float(r.get("pval_holm"))
                return {
                    "type":      label,
                    "allele":    allele,
                    "effect":    _safe_float(r.get("coef")),
                    "pval":      holm_p if use_holm else raw_p,
                    "raw_pval":  raw_p,
                    "holm_pval": holm_p,
                    "stat":      _extract_stat(r),
                    "motif_pos": chosen_pos,
                }
        return None

    def _empty(label, allele):
        return {
            "type":      label,
            "allele":    allele,
            "effect":    math.nan,
            "pval":      math.nan,
            "raw_pval":  math.nan,
            "holm_pval": math.nan,
            "stat":      math.nan,
            "motif_pos": chosen_pos,
        }

    return [
        _find("AFF",  alt) or _empty("AFF",  alt),
        _find("RAND", ref) or _empty("RAND", ref),
        _find("RAND", alt) or _empty("RAND", alt),
    ]


def _print_best_pval_table(
    snv_str,
    rows,
    *,
    include_rand=True,
    diagnostics=False,
    snv=None,
    matcher=None,
    design=None,
    k=8,
    rand_n=500,
    fold_half=True,
    max_probes=None,
    seed=0,
    holm=False,
):
    """Print the compact best_pval TSV rows for one SNV."""
    summary = _best_pval_summary_rows(rows, snv_str, use_holm=holm)

    # Build payloads once up front if diagnostics are requested.
    aff_payloads = rand_payloads = {}
    if diagnostics and snv is not None:
        try:
            aff_payloads = build_aff_payloads_for_snv(
                snv, matcher=matcher, design=design, k=k,
                include_covariates=True, fold_half=fold_half,
                max_probes=max_probes,
                seed=seed,
            )
        except Exception:
            pass
        try:
            rand_payloads = build_rand_payloads_for_snv(
                snv=snv, snv_str=snv_str, matcher=matcher, design=design,
                k=k, rand_n=rand_n, fold_half=fold_half, min_n=10,
                max_probes=max_probes,
                seed=seed,
            )
        except Exception:
            pass

    for r in summary:
            if (not include_rand) and r["type"] == "RAND":
                continue
            pval_str = _format_pval(r["pval"], stat=r.get("stat", math.nan))
            line = f"{snv_str}\t{r['type']}\t{r['allele']}\t{r['effect']}\t{pval_str}"
            if holm:
                raw_pval_str = _format_pval(r.get("raw_pval", math.nan), stat=r.get("stat", math.nan))
                line += f"\t{raw_pval_str}"
            if diagnostics:
                chosen_pos = r["motif_pos"]
                aff_key = next((k for k in aff_payloads if k[0] == chosen_pos), None)
                aff_payload = aff_payloads.get(aff_key)
                wildcard_kmer = aff_payload.wildcard_kmer if aff_payload is not None else "NA"
                if r["type"] == "AFF":
                    diag = probe_diagnostics(aff_payload, r["allele"])
                else:
                    diag = probe_diagnostics(rand_payloads.get(chosen_pos), r["allele"])
                line += f"\t{chosen_pos}\t{wildcard_kmer}\t{diag['n_probes']}\t{diag['n_allele']}"
            print(line)


def _print_motif_effect_table_holm(snv_str, rows):
    """Print the normal SNV motif table with Holm p in pval and raw_pval appended."""
    ref, alt = _parse_ref_alt_from_snv(snv_str)
    for label in ("AFF", "RAND"):
        alleles = [a for a in (ref, alt) if a in ("A", "C", "G", "T")]
        for allele in alleles:
            if label == "AFF" and allele == ref:
                continue
            selected = [
                r for r in rows
                if r.get("label", "AFF") == label and r.get("allele") == allele
            ]
            for r in sorted(selected, key=lambda x: x.get("motif_pos", 0)):
                raw_p = _safe_float(r.get("pval"))
                holm_p = _safe_float(r.get("pval_holm"))
                coef = _safe_float(r.get("coef"))
                p_out = _format_pval(holm_p, stat=_extract_stat(r))
                p_raw = _format_pval(raw_p, stat=_extract_stat(r))
                motif_pos = r.get("motif_pos", "NA")
                wildcard = r.get("wildcard_kmer", "NA") if label == "AFF" else "NA"
                coef_str = f"{coef:.7g}" if math.isfinite(coef) else "NA"
                print(
                    f"{snv_str}\t{label}\t{allele}\t{motif_pos}\t{wildcard}\t"
                    f"{coef_str}\t{p_out}\t{p_raw}"
                )


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
        "snv", "wildcard_kmer", "filled_kmer", "window_index", "snv_index",
        "type", "allele", "coef", "pval", "absolute_pos", "method",
    ]
    center = k - 1
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
        rows.append([
            snv_str, wildcard_kmer, filled_kmer, motif_pos, snv_index,
            r.get("label", ""), allele,
            _format_cell(r.get("coef")), _format_cell(r.get("pval")),
            int(r.get("absolute_pos", motif_pos + snv_index)),
            r.get("method", "NA"),
        ])

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
    p.add_argument(
        "--array", "--kmerPositions", dest="array", required=True,
        help="Path to k-mer array file mapping k-mers to genomic regions",
    )
    p.add_argument("--genome", required=True, help="Path to reference genome in FASTA format")
    group.add_argument("--snv-list", help="Comma-separated list of SNVs in chr:pos:ref>alt format")
    group.add_argument(
        "--snv-list-file",
        help="Optional file with SNVs, one per line in chr:pos:ref>alt format",
    )
    p.add_argument(
        "--kmer-size", "--kmer_size", dest="kmer_size", type=int, default=8,
        help="K-mer size (default: 8)",
    )
    p.add_argument(
        "--rand-n", type=int, default=500,
        help="Number of random probes to use for RAND regression (default: 500)",
    )
    p.add_argument(
        "--no-rand", action="store_true",
        help="Skip RAND regression and only output AFF (still prints the motif-effect table)",
    )
    p.add_argument(
        "--best-pval", action="store_true", dest="best_pval",
        help=(
            "Summarize results by choosing one shared motif position per SNV using "
            "the ALT AFF signal and same-position RAND support, then print a compact "
            "TSV with one row each for AFF alt, RAND ref, and RAND alt."
        ),
    )
    p.add_argument(
        "--holm", action="store_true",
        help=(
            "Apply within-SNV Holm correction before --best-pval selection: "
            "ALT AFF p-values are corrected across motif positions, and REF+ALT "
            "RAND p-values are corrected together across motif positions. RAND "
            "support then uses Holm p < 0.05. Output pval is Holm-adjusted and "
            "raw_pval is appended. Without --best-pval, corrected p-values are "
            "also shown in the full motif table."
        ),
    )
    p.add_argument(
        "--diagnostics", action="store_true",
        help=(
            "Append diagnostic columns (motif_pos, wildcard_kmer, n_probes, n_allele) to the "
            "--best_pval output. Requires --best_pval."
        ),
    )
    # p.add_argument(
    #     "--output-long", type=str, dest="output_long",
    #     help="Write a scan-style long TSV (with SNV as first column) to this path",
    # )
    p.add_argument(
        "--mode", choices=["nb", "ols"], default="ols",
        help="Regression type: negative binomial ('nb') or ordinary least squares ('ols')",
    )
    p.add_argument("--no-covariates", action="store_true", help="Disable lp and sl covariates")
    p.add_argument(
        "--raw-lp", action="store_true",
        help="Use raw lp in [0,1] (no folding to [0,0.5])",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for max-probes subsampling (default: 0); RAND background retains its existing deterministic sampling.")
    p.add_argument(
        "--max-probes", type=int, default=None, dest="max_probes",
        help="Subsample to at most this many probes per window before fitting."
    )
    p.add_argument("--debug", action="store_true", help="Print additional debugging information")
    args = p.parse_args(argv)

    if args.diagnostics and not args.best_pval:
        p.error("--diagnostics requires --best_pval")

    if args.best_pval:
        header = "snv\ttype\tallele\teffect\tpval"
        if args.holm:
            header += "\traw_pval"
        if args.diagnostics:
            header += "\tmotif_pos\twildcard_kmer\tn_probes\tn_allele"
        print(header)
    else:
        header = "snv\ttype\tallele\tmotif_pos\twildcard_kmer\teffect\tpval"
        if args.holm:
            header += "\traw_pval"
        print(header)

    intens = IntensityTable.from_file(args.intensities)

    # Warn if intensities appear to be on an intensity_like scale rather than
    # resid_log. resid_log values are centered near 0 with negative values;
    # intensity_like values are strictly non-negative. If the minimum value
    # is >= 0, the file is likely intensity_like which will give wrong results.
    _vals = list(intens.values.values())
    if _vals and min(_vals) >= 0:
        print(
            "[WARNING] Intensity values appear to be non-negative (min="
            f"{min(_vals):.3f}). EUbar expects resid_log intensities "
            "(centered near 0, with negative values). If you used "
            "'--resid-output intensity_like' when generating intensities, "
            "please regenerate with '--resid-output resid_log'.",
            file=sys.stderr,
        )
    kmers = KmerIndex.from_file(args.array)
    matcher = MotifMatcher(kmers.kmers)
    design = DesignBuilder(intens.values)
    engine = RegressionEngine()

    fold_half = not args.raw_lp

    if args.snv_list_file:
        with open(args.snv_list_file) as f:
            snvs = [line.strip() for line in f if line.strip()]
    else:
        snvs = [s.strip() for s in (args.snv_list or "").split(",") if s.strip()]

    for snv_str in snvs:
        try:
            snv = SnvWindow.from_snv(
                snv_str, args.genome, k=args.kmer_size, debug=args.debug,
            )
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
            fold_half=fold_half,
            rand_n=(0 if args.no_rand else args.rand_n),
            max_probes=args.max_probes,
            seed=args.seed,
        )

        if args.holm:
            rows = _apply_holm_within_snv(rows, snv_str)

        if args.best_pval:
            _print_best_pval_table(
                snv_str,
                rows,
                include_rand=(not args.no_rand),
                diagnostics=args.diagnostics,
                snv=snv,
                matcher=matcher,
                design=design,
                k=args.kmer_size,
                rand_n=args.rand_n,
                fold_half=fold_half,
                max_probes=args.max_probes,
                seed=args.seed,
                holm=args.holm,
            )
        else:
            if args.holm:
                _print_motif_effect_table_holm(snv_str, rows)
            else:
                print_motif_effect_table(snv_str, snv.chrom, snv.pos, snv.seq, rows)

        # if args.output_long:
        #     _append_long_tsv(args.output_long, snv_str, snv.seq, args.kmer_size, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())