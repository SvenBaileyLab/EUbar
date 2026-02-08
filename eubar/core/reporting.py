"""Output formatting functions for reporting regression results.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import pandas as pd
import numpy as np


def _normalize_row_len(row, n):
    """Pad or trim a row to exactly n elements."""
    r = list(row)
    if len(r) < n:
        r = r + ['NA'] * (n - len(r))
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
    ref_allele = snv_str.split(":")[2].split(">")[0]
    motif_positions = sorted(set(r["motif_pos"] for r in results))

    print(f"{snv_str}\t{chrom}\t{pos}\t{region_seq}")

    for group in ["AFF", "RAND"]:
        for allele in ["A", "C", "G", "T"]:
            coefs = []
            pvals = []

            for i in motif_positions:
                match = next(
                    (
                        r
                        for r in results
                        if r["motif_pos"] == i
                        and r["allele"] == allele
                        and r.get("label", "AFF") == group
                    ),
                    None,
                )

                if group == "AFF" and allele == ref_allele:
                    coefs.append(0)
                    pvals.append("NA")
                elif match:
                    coefs.append(match["coef"])
                    pvals.append(match["pval"])
                else:
                    coefs.append(0)
                    pvals.append("NA")

            pos_str = ",".join(map(str, motif_positions))
            coef_str = ",".join(f"{x:.7g}" for x in coefs)
            pval_str = ",".join(
                f"{x:.7g}" if isinstance(x, float) else x for x in pvals
            )

            print(f"{group}\t{snv_str}\t{allele}\t{pos_str}\t{coef_str}\t{pval_str}")


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

    allele_colors = {"A": "black", "C": "red", "G": "green", "T": "blue"}

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

def plot_aff_motif_effects(rows, save_path):
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

    df["absolute_position"] = df["window_index"].astype(int) + df["snp_index"].astype(int) + 1
    region_length = int(df["absolute_position"].max()) if len(df) else 1

    allele_colors = {"A": "black", "C": "red", "G": "green", "T": "blue"}

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
    axs[1].set_xlim(0.5, region_length + 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Info] Figure saved to: {save_path}")
