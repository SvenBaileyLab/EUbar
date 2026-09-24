"""Output formatting functions for reporting regression results.
"""

from __future__ import annotations

import math
import csv
from .snv_results import (
    _extract_stat, _safe_float, _parse_ref_alt_from_snv, _best_pval_summary_rows,
)
from .inspect import (
    build_aff_payloads_for_snv, build_rand_payloads_for_snv, probe_diagnostics,
)

# Retain log-space recovery for underflowed p-values.
try:
    from scipy.stats import norm
except Exception:
    norm = None

from typing import Any, Dict, Sequence

import pandas as pd
import numpy as np


def _normalize_row_len(row, n):
    """Pad or trim a row to exactly n elements."""
    r = list(row)
    if len(r) < n:
        r = r + ["NA"] * (n - len(r))
    elif len(r) > n:
        r = r[:n]
    return r


def print_motif_effect_table(
    snv_str: str,
    chrom: str,
    pos: int,
    region_seq: str,
    results: Sequence[Dict[str, Any]],
) -> None:
    """Print one row per (type, allele, motif_pos) for ref and alt alleles only."""
    try:
        allele_part = snv_str.split(":", 2)[2]
        ref_allele = allele_part.split(">")[0].strip().upper()
        alt_allele = allele_part.split(">")[1].strip().upper()
    except Exception:
        ref_allele = alt_allele = None

    for group in ["AFF", "RAND"]:
        for allele in [a for a in [ref_allele, alt_allele] if a in ("A", "C", "G", "T")]:
            if group == "AFF" and allele == ref_allele:
                continue
            for r in sorted(results, key=lambda x: x.get("motif_pos", 0)):
                if r.get("label", "AFF") != group:
                    continue
                if r.get("allele") != allele:
                    continue

                motif_pos = r.get("motif_pos", "NA")
                wildcard_kmer = r.get("wildcard_kmer", "NA")
                coef = r.get("coef", "NA")
                pval = r.get("pval", "NA")

                if isinstance(coef, float) and not math.isfinite(coef):
                    coef = "NA"
                if isinstance(pval, float):
                    if not math.isfinite(pval):
                        pval_str = "NA"
                    elif pval <= 0:
                        pval_str = "<1e-300"
                    else:
                        pval_str = f"{pval:.3e}"
                else:
                    pval_str = str(pval)

                coef_str = f"{coef:.7g}" if isinstance(coef, float) else str(coef)
                print(f"{snv_str}\t{group}\t{allele}\t{motif_pos}\t{wildcard_kmer}\t{coef_str}\t{pval_str}")


def print_rows_as_scan_tsv_with_snv(
    rows: Sequence[Sequence[Any]],
    snv_str: str,
    header=(
        "snv",
        "wildcard_kmer",
        "filled_kmer",
        "window_index",
        "snv_index",
        "type",
        "allele",
        "coef",
        "pval",
        "method",
    ),
) -> None:
    """
    Print regression results in scan-style TSV format with SNV string as first column.
    Replaces NaNs with the string "NaN".
    """
    df_rows = []
    ncols = len(header) - 1
    for row in rows:
        row = _normalize_row_len(row, ncols)
        formatted = [snv_str]  # Add SNV column first
        for item in row[:9]:  # scan-style fields + method
            if isinstance(item, float) and math.isnan(item):
                formatted.append("NaN")
            elif item == "":
                formatted.append("NaN")
            else:
                formatted.append(item)
        df_rows.append(formatted)

    df = pd.DataFrame(df_rows, columns=header)
    print(df.to_csv(sep="\t", index=False))


def print_rows_as_tsv(
    rows,
    header=(
        "wildcard_kmer",
        "filled_kmer",
        "window_index",
        "snp_index",
        "type",
        "allele",
        "coef",
        "pval",
        "absolute_pos",
        "method",
    ),
):
    """Print rows as TSV (stdout), matching legacy formatting."""

    df_rows = []
    ncols = len(header)
    for row in rows:
        row = _normalize_row_len(row, ncols)
        formatted = []
        for item in row:
            if isinstance(item, float) and math.isnan(item):
                formatted.append("NaN")
            elif item == "":
                formatted.append("NaN")
            else:
                formatted.append(item)
        df_rows.append(formatted)

    df = pd.DataFrame(
        df_rows,
        columns=list(header),
    )
    print(df.to_csv(sep="\t", index=False))


def _plot_aff_motif_effects(rows, save_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        [list(r)[:9] for r in rows],
        columns=[
            "wildcard_kmer",
            "filled_kmer",
            "window_index",
            "snp_index",
            "type",
            "allele",
            "coef",
            "pval",
            "absolute_pos",
        ],
    )

    df = df[df["type"] == "AFF"].copy()

    df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
    df = df.dropna(subset=["coef", "pval"])
    df["-log10(pval)"] = -np.log10(df["pval"])
    df["absolute_position"] = df["window_index"] + df["snp_index"] + 1
    region_length = df["absolute_position"].max()

    allele_colors = {"A": "#008000", "C": "#0000ff", "G": "#ffa600", "T": "#ff0000"}

    fig_width = max(12, region_length * 0.15)
    fig, axs = plt.subplots(2, 1, figsize=(fig_width, 6), sharex=True)

    for metric, ax in zip(["coef", "-log10(pval)"], axs):
        for _, row in df.iterrows():
            x = int(row["absolute_position"])
            y = row[metric]
            allele = row["allele"]
            color = allele_colors.get(allele, "gray")
            if pd.notna(y):
                ax.text(
                    x,
                    y,
                    allele,
                    color=color,
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        ymin = df[metric].min()
        ymax = df[metric].max()
        yrange = ymax - ymin if ymax != ymin else 1
        padding = yrange * 0.1
        ax.set_ylim(ymin - padding, ymax + padding)
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.4)

    axs[1].set_xlabel("Genomic Position")
    axs[0].set_title("Motif Scan: Coefficients and -log10(p-values)")

    step = 10 if region_length > 80 else 5 if region_length > 40 else 1
    axs[1].set_xticks(range(1, int(region_length) + 1, step))
    axs[1].set_xlim(0.5, region_length + 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Info] Figure saved to: {save_path}")


def plot_aff_motif_effects(rows, save_path, reverse=False):
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        [list(r)[:9] for r in rows],
        columns=[
            "wildcard_kmer",
            "filled_kmer",
            "window_index",
            "snp_index",
            "type",
            "allele",
            "coef",
            "pval",
            "absolute_pos",
        ],
    )

    df = df[df["type"] == "AFF"].copy()

    df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
    df = df.dropna(subset=["coef", "pval"]).copy()

    # --- FIX: avoid -log10(0) / NaN / inf ---
    # Clamp p-values to a tiny positive floor and 1.0 ceiling, and drop non-finite.
    p = df["pval"].to_numpy(dtype=float)
    p = np.where(np.isfinite(p), p, np.nan)
    p = np.clip(p, 1e-300, 1.0)  # floor prevents inf; ceiling keeps sanity
    df["pval_clamped"] = p
    df["-log10(pval)"] = -np.log10(df["pval_clamped"])
    # ----------------------------------------

    df["absolute_position"] = (
        df["window_index"].astype(int) + df["snp_index"].astype(int) + 1
    )
    region_length = int(df["absolute_position"].max()) if len(df) else 1

    allele_colors = {"A": "#008000", "C": "#0000ff", "G": "#ffa600", "T": "#ff0000"}

    fig_width = max(12, region_length * 0.15)
    fig, axs = plt.subplots(2, 1, figsize=(fig_width, 6), sharex=True)

    for metric, ax in zip(["coef", "-log10(pval)"], axs):
        for _, row in df.iterrows():
            x = int(row["absolute_position"])
            y = row[metric]
            allele = row["allele"]
            color = allele_colors.get(allele, "gray")
            if pd.notna(y) and np.isfinite(float(y)):
                ax.text(
                    x,
                    y,
                    allele,
                    color=color,
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        # --- FIX: compute y-lims using finite values only ---
        vals = df[metric].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            ymin, ymax = 0.0, 1.0
        else:
            ymin = float(np.min(vals[finite]))
            ymax = float(np.max(vals[finite]))
        yrange = ymax - ymin if ymax != ymin else 1.0
        padding = yrange * 0.1
        ax.set_ylim(ymin - padding, ymax + padding)
        # ----------------------------------------------------

        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.4)

    axs[1].set_xlabel("Genomic Position")
    axs[0].set_title("Motif Scan: Coefficients and -log10(p-values)")

    step = 10 if region_length > 80 else 5 if region_length > 40 else 1
    axs[1].set_xticks(range(1, int(region_length) + 1, step))
    axs[1].set_xlim(
        (region_length + 0.5, 0.5) if reverse else (0.5, region_length + 0.5)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Info] Figure saved to: {save_path}")

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
