"""Motif matrices, file formats and diagnostic plots."""
from __future__ import annotations
from typing import List, Optional
import math
import numpy as np
import pandas as pd
from .motif_data import BASES, ProbeRanks
from .motif_search import escore_auc_minus_half_all_bg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import logomaker
except Exception:
    logomaker = None

STANDARD_DNA_COLORS = {"A": "#008000", "C": "#0000ff", "G": "#ffa600", "T": "#ff0000"}
PRETTY_DNA_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}


def _dna_colors(pretty_logo: bool) -> dict:
    return PRETTY_DNA_COLORS if pretty_logo else STANDARD_DNA_COLORS


def stitch_ppm(
    left_flank: List[pd.Series],
    core_ppm: pd.DataFrame,
    right_flank: List[pd.Series],
) -> pd.DataFrame:
    """Stitch flank probability vectors + core PPM into a single PPM with integer positions."""
    rows: List[pd.Series] = []
    # left flank is nearest->outward; to stitch left-to-right, reverse it
    for s in reversed(left_flank):
        rows.append(pd.Series({b: float(s[b]) for b in BASES}))

    for _, r in core_ppm.iterrows():
        rows.append(pd.Series({b: float(r[b]) for b in BASES}))

    for s in right_flank:
        rows.append(pd.Series({b: float(s[b]) for b in BASES}))

    ppm = pd.DataFrame(rows)
    ppm.columns = list(BASES)
    # positions: negative for left flank, then 0..k-1, then k..
    L = len(left_flank)
    k = core_ppm.shape[0]
    idx = list(range(-L, 0)) + list(range(0, k)) + list(range(k, k + len(right_flank)))
    ppm.index = idx
    ppm.index.name = "pos"
    return ppm


def consensus_from_ppm(ppm: pd.DataFrame) -> str:
    # Robust to accidental duplicate indices/columns
    ppm = ppm.loc[:, ~ppm.columns.duplicated()].copy()
    if set(BASES).issubset(ppm.columns):
        ppm = ppm[list(BASES)]
    out = []
    for pos in ppm.index:
        row = ppm.loc[pos]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        out.append(str(row.astype(float).idxmax()))
    return "".join(out)


def write_meme(ppm: pd.DataFrame, out_path: str, motif_name: str) -> None:
    with open(out_path, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        w = ppm.shape[0]
        f.write(f"MOTIF {motif_name}\n")
        f.write(f"letter-probability matrix: alength= 4 w= {w} nsites= {w} E= 0\n")

        for pos in ppm.index:
            row = ppm.loc[pos]
            f.write(f"{row['A']:.6f} {row['C']:.6f} {row['G']:.6f} {row['T']:.6f}\n")


def ppm_to_bits_matrix(ppm: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Convert a PPM (probabilities) to a per-letter height matrix in *bits*.

    For each position i:
      IC_bits(i) = log2(4) - H(i), where H(i) = -Σ_b p(b,i) log2 p(b,i)
      height(b,i) = p(b,i) * IC_bits(i)

    This is the standard "information content" sequence-logo height convention.
    """
    mat = ppm.copy()
    # Ensure columns are A,C,G,T and numeric
    mat = mat.loc[:, ~mat.columns.duplicated()].copy()
    mat = mat[list(BASES)].astype(float)

    # Normalize each row defensively
    row_sums = mat.sum(axis=1).replace(0.0, np.nan)
    mat = mat.div(row_sums, axis=0)

    p = np.clip(mat.to_numpy(dtype=float), eps, 1.0)
    H = -(p * np.log2(p)).sum(axis=1)  # entropy in bits
    IC = np.log2(4.0) - H  # max 2 bits for DNA
    heights = p * IC[:, None]  # per-letter heights

    out = pd.DataFrame(heights, columns=list(BASES), index=ppm.index)
    out.index.name = ppm.index.name
    return out


def reverse_complement_ppm(ppm: pd.DataFrame) -> pd.DataFrame:
    """Return a reverse-complemented copy of a PPM/height matrix."""
    rc = ppm.copy().iloc[::-1].reset_index(drop=True)
    col_map = {"A": "T", "C": "G", "G": "C", "T": "A"}
    rc = rc.rename(columns=col_map)
    rc = rc[[b for b in BASES if b in rc.columns]]
    return rc


def reverse_complement_reduced_df(reduced_full_df: pd.DataFrame) -> pd.DataFrame:
    """Reverse-complement reduced enrichment table for plotting."""
    if reduced_full_df is None or reduced_full_df.empty:
        return reduced_full_df.copy()

    rc = reduced_full_df.copy()
    pos_vals = sorted(rc["pos"].dropna().astype(int).unique())
    pos_map = {old: new for old, new in zip(pos_vals, reversed(pos_vals))}
    rc["pos"] = rc["pos"].astype(int).map(pos_map)

    base_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rc["base"] = rc["base"].map(lambda b: base_map.get(b, b))

    sort_cols = [c for c in ["pos", "side", "base"] if c in rc.columns]
    if sort_cols:
        rc = rc.sort_values(sort_cols).reset_index(drop=True)
    return rc


def plot_logo(
    ppm: pd.DataFrame,
    out_png: str,
    title: str,
    pretty_logo: bool = False,
    mode: str = "prob",
) -> None:
    """Save a sequence logo.

    mode:
      - "prob": y-axis is probability (0..1)
      - "bits": y-axis is information content in bits (0..2), using height(b)=p(b)*IC_bits
    """
    mode = str(mode).lower().strip()
    if mode not in ("prob", "bits"):
        raise ValueError("mode must be 'prob' or 'bits'")

    dna_colors = _dna_colors(pretty_logo)

    # Choose the matrix to plot
    if mode == "bits":
        logo_mat = ppm_to_bits_matrix(ppm)
        y_label = "bits"
        y_max = 2.0
    else:
        logo_mat = ppm.copy()
        y_label = "prob"
        y_max = 1.0

    fig, ax = plt.subplots(figsize=(max(6, ppm.shape[0] * 0.6), 2.8))

    L = ppm.shape[0]

    if pretty_logo:
        bg = "#e6e6e6"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    if logomaker is not None:
        logo_df = logo_mat.copy()
        logo_df.index = range(ppm.shape[0])
        logomaker.Logo(logo_df, ax=ax, color_scheme=dna_colors)
        ax.set_ylabel(y_label, fontsize=18, fontfamily="Carlito")
        ax.set_xlabel("pos", fontsize=18, fontfamily="Carlito")
        ax.set_xticks(np.arange(L))
        ax.set_xticklabels(
            [str(i) for i in range(1, L + 1)],
            rotation=90,
            fontsize=18,
            fontfamily="Carlito",
        )
        ax.tick_params(axis="y", labelsize=18)
        for label in ax.get_yticklabels():
            label.set_fontfamily("Carlito")
        ax.set_ylim(0, y_max)
    else:
        # fallback: draw a simple sequence logo using stacked letters
        from matplotlib.textpath import TextPath
        from matplotlib.patches import PathPatch
        from matplotlib.transforms import Affine2D
        from matplotlib.font_manager import FontProperties

        fp = FontProperties(family="Carlito", weight="bold")

        def add_letter(letter: str, x: float, y: float, height: float) -> None:
            if height <= 0:
                return
            tp = TextPath((0, 0), letter, size=1, prop=fp)
            bb = tp.get_extents()
            sx = 0.9 / max(bb.width, 1e-6)
            sy = height / max(bb.height, 1e-6)
            trans = Affine2D().scale(sx, sy).translate(x + 0.05, y)
            patch = PathPatch(
                tp,
                lw=0,
                facecolor=dna_colors.get(letter, "black"),
                transform=trans + ax.transData,
            )
            ax.add_patch(patch)

        L = logo_mat.shape[0]
        for i in range(L):
            heights = {b: float(logo_mat.iloc[i][b]) for b in BASES}
            y0 = 0.0
            for letter, h in sorted(heights.items(), key=lambda kv: kv[1]):
                add_letter(letter, x=float(i), y=y0, height=h)
                y0 += h

        ax.set_xlim(0, L)
        ax.set_ylim(0, y_max)
        ax.set_xticks(np.arange(L))
        ax.set_xticklabels(
            [str(i) for i in range(1, L + 1)],
            rotation=90,
            fontsize=18,
            fontfamily="Carlito",
        )
        ax.set_ylabel(y_label, fontsize=18, fontfamily="Carlito")
        ax.set_xlabel("pos", fontsize=18, fontfamily="Carlito")
        ax.tick_params(axis="y", labelsize=18)
        for label in ax.get_yticklabels():
            label.set_fontfamily("Carlito")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_enrichment_bars(
    reduced_full_df: pd.DataFrame,
    out_path: str,
    pretty_logo: bool = False,
    title: Optional[str] = None,
) -> None:
    """Plot grouped bars of reduced enrichment (E_reduced) for A/C/G/T at each motif position.

    Notes:
      - We remap the 'pos' values from reduced_full_df to 0..L-1 so the x-axis matches the logo.
      - If pretty_logo is True, use the same DNA colors + light gray background as the logo.
    """
    if reduced_full_df is None or len(reduced_full_df) == 0:
        return

    # Expect columns: pos, base, E_reduced
    df = reduced_full_df.copy()
    if (
        "pos" not in df.columns
        or "base" not in df.columns
        or "E_reduced" not in df.columns
    ):
        return

    # Keep only valid bases
    df = df[df["base"].isin(BASES)].copy()
    if len(df) == 0:
        return

    # Aggregate across steps/sides if present (take mean per pos/base)
    df_agg = df.groupby(["pos", "base"], as_index=False)["E_reduced"].mean()

    # Build pivot: rows=pos (original), cols=base
    pivot = df_agg.pivot(index="pos", columns="base", values="E_reduced").sort_index()

    # Map original positions to 0..L-1 for display to match logo
    orig_positions = list(pivot.index)
    pos_map = {p: i for i, p in enumerate(orig_positions)}
    pivot = (
        pivot.reset_index()
        .assign(pos0=lambda d: d["pos"].map(pos_map))
        .set_index("pos0")
    )
    pivot.index.name = "pos"

    L = pivot.shape[0]
    if L == 0:
        return

    # Ensure all bases present as columns
    for b in BASES:
        if b not in pivot.columns:
            pivot[b] = 0.0
        pivot = pivot.loc[:, list(BASES)]

    # Colors
    dna_colors = _dna_colors(pretty_logo)

    import matplotlib.pyplot as plt
    import numpy as np

    fig_w = max(8.0, 0.8 * L)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6), dpi=150)

    if pretty_logo:
        ax.set_facecolor("#e6e6e6")
        fig.patch.set_facecolor("#e6e6e6")

    x = np.arange(L, dtype=float)
    width = 0.18
    offsets = {"A": -1.5 * width, "C": -0.5 * width, "G": 0.5 * width, "T": 1.5 * width}

    for b in BASES:
        ax.bar(
            x + offsets[b],
            pivot[b].to_numpy(dtype=float),
            width=width,
            label=b,
            color=dna_colors.get(b),
        )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlim(-0.6, L - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(i) for i in range(1, L + 1)],
        rotation=90,
        fontsize=18,
        fontfamily="Carlito",
    )
    ax.set_xlabel("pos", fontsize=18, fontfamily="Carlito")
    ax.set_ylabel("Enrichment score")

    if title:
        ax.set_title(title)

    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_seed_enrichment_curve(
    seed: str,
    fg_idx: np.ndarray,
    ranks: ProbeRanks,
    out_png: str,
) -> None:
    """A simple seed enrichment plot: FG and BG detection curves across ranked probes."""
    N = len(ranks.scores)
    fg_mask = np.zeros(N, dtype=bool)
    fg_mask[fg_idx] = True

    ordered_mask = fg_mask[ranks.order]  # probes sorted high->low

    # cumulative detection rates
    x = np.arange(1, N + 1)
    fg_cum = np.cumsum(ordered_mask)
    bg_cum = np.cumsum(~ordered_mask)

    F = fg_mask.sum()
    B = N - F
    fg_rate = fg_cum / max(F, 1)
    bg_rate = bg_cum / max(B, 1)

    # E-score
    E = escore_auc_minus_half_all_bg(fg_idx, ranks.i_to_rank, N)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x / N, fg_rate, label=f"Foreground (F={F})")
    ax.plot(x / N, bg_rate, label=f"Background (B={B})")
    ax.set_xlabel("fraction of probes scanned (high → low intensity)")
    ax.set_ylabel("detection rate")
    ax.set_title(f"Seed enrichment curves: {seed}  (E={E:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_seed_enrichment_roc(
    seed: str,
    fg_idx: np.ndarray,
    ranks: ProbeRanks,
    out_png: str,
) -> None:
    """ROC-style enrichment plot; AUC maps directly to the E-score (E = AUC - 0.5)."""
    N = len(ranks.scores)
    fg_mask = np.zeros(N, dtype=bool)
    fg_mask[fg_idx] = True

    ordered = fg_mask[ranks.order]  # high→low intensity
    tp = np.cumsum(ordered)
    fp = np.cumsum(~ordered)
    F = int(tp[-1])
    B = int(fp[-1])

    if F == 0 or B == 0:
        return

    tpr = tp / F
    fpr = fp / B

    E = escore_auc_minus_half_all_bg(fg_idx, ranks.i_to_rank, N)
    auc = E + 0.5

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot(fpr, tpr, label=f"seed={seed} (AUC={auc:.3f}, E={E:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Seed enrichment (ROC)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_escore_histogram(es_df: pd.DataFrame, out_png: str, title: str) -> None:
    """Histogram of E-scores for seed candidates (useful sanity check)."""
    if es_df.empty:
        return
    vals = es_df["E"].astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(vals, bins=60)
    ax.set_xlabel("E-score")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def motif_logscore(kmer: str, ppm: pd.DataFrame, eps: float = 1e-12) -> float:
    """Sum log(prob) under the motif for a k-mer aligned to ppm.

    Assumes k-mer length equals motif length. If motif length differs, caller should slice.
    """
    if len(kmer) != ppm.shape[0]:
        raise ValueError("kmer length must equal motif length for motif_logscore")
    s = 0.0
    for pos, base in zip(ppm.index, kmer):
        p = float(ppm.loc[pos, base]) if base in BASES else 0.0
        s += math.log(max(p, eps))
    return s


def plot_motif_vs_escore(
    es_df: pd.DataFrame,
    ppm: pd.DataFrame,
    out_png: str,
    q: int = 20,
) -> None:
    """QC plot: bin motif log-scores and plot mean E-score per bin."""
    if es_df.empty:
        return

    L = int(ppm.shape[0])

    # Find any representative k-mer length (avoid assuming row 0 exists / is a string)
    k_example = None
    for x in es_df["kmer"].astype(str).tolist():
        if x:
            k_example = len(x)
            break

    if k_example is None or L != k_example:
        return

    df = es_df.copy()
    df["motif_logscore"] = df["kmer"].apply(lambda k: motif_logscore(str(k), ppm))

    # bin and aggregate
    qc = df.dropna(subset=["motif_logscore", "E"]).copy()
    if qc.empty:
        return

    qc["bin"] = pd.qcut(qc["motif_logscore"], q=q, duplicates="drop")
    trend = (
        qc.groupby("bin", observed=False)
        .agg(
            mean_motif=("motif_logscore", "mean"),
            mean_E=("E", "mean"),
            n=("E", "size"),
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(trend["mean_motif"], trend["mean_E"], marker="o")
    ax.set_xlabel("mean motif log-score (binned)")
    ax.set_ylabel("mean E-score")
    ax.set_title("Motif score vs E-score (QC)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
