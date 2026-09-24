#!/usr/bin/env python3
"""EUbar calibration command.

Supported targets:
  * ``max-probes`` — choose a production probe cap from effect stability and
    representation. P-values are diagnostic only.
  * ``mask`` — compare explicit 0/1 masks using only the array's probe-region
    universe, genome sequence, and residualized intensities. No external truth
    labels are used.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

import numpy as np

from eubar.core.calibration import (
    parse_caps,
    parse_masks,
    run_mask_calibration,
    run_max_probes_calibration,
)
from eubar.core.data import IntensityTable, KmerIndex


DEFAULT_CAPS = "100,200,400,600,800,1000,1250,1500,2000,2500"


def _caps_arg(text: str):
    try:
        return parse_caps(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _read_mask_values(inline: Optional[str], path: Optional[str], *, option_name: str) -> list[str]:
    """Read explicit masks from a comma-separated argument and/or text file."""
    values: list[str] = []
    if inline:
        values.extend(parse_masks(inline))
    if path:
        try:
            with open(path) as fh:
                for raw in fh:
                    line = raw.split("#", 1)[0].strip()
                    if line:
                        values.extend(parse_masks(line.replace("\t", ",").replace(" ", ",")))
        except OSError as exc:
            raise ValueError(f"could not read {option_name} file {path!r}: {exc}") from exc

    out: list[str] = []
    seen = set()
    for mask in values:
        if mask not in seen:
            seen.add(mask)
            out.append(mask)
    return out


def _load_masks(args) -> list[str]:
    out = _read_mask_values(args.masks, args.mask_file, option_name="--mask-file")
    if not out:
        raise ValueError("mask calibration requires --masks or --mask-file")
    return out


def _load_control_masks(args, candidate_masks: Sequence[str]) -> list[str]:
    """Load diagnostic control masks, excluding masks already tested as candidates."""
    out = _read_mask_values(
        args.control_masks,
        args.control_mask_file,
        option_name="--control-mask-file",
    )
    candidate_set = set(candidate_masks)
    return [mask for mask in out if mask not in candidate_set]


def _write_max_plot(summary, recommendation, pquart, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        print(f"[warning] --plot requested but matplotlib is unavailable: {exc}", file=sys.stderr)
        return False

    rec = recommendation.iloc[0]["recommended_max_probes"]
    try:
        rec_num = float(rec)
        if not np.isfinite(rec_num):
            rec_num = None
    except Exception:
        rec_num = None

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
        fig.suptitle(f"EUbar max-probes calibration — recommendation: {rec}", fontsize=14)

        ax = axes[0, 0]
        ax.plot(summary["cap_probes"], summary["effect_sd_norm_median"], marker="o", label="median")
        ax.plot(summary["cap_probes"], summary["effect_sd_norm_p90"], marker="o", label="p90")
        ax.axhline(float(recommendation.iloc[0]["max_effect_sd_norm_p90"]), linestyle="--", linewidth=1)
        if rec_num is not None:
            ax.axvline(rec_num, linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("max probes")
        ax.set_ylabel("coefficient SD / intensity SD")
        ax.set_title("Effect stability")
        ax.legend(frameon=False)

        ax = axes[0, 1]
        ax.plot(summary["cap_probes"], summary["effect_bias_norm_median"], marker="o", label="median")
        ax.plot(summary["cap_probes"], summary["effect_bias_norm_p90"], marker="o", label="p90")
        ax.axhline(float(recommendation.iloc[0]["max_effect_bias_norm_p90"]), linestyle="--", linewidth=1)
        if rec_num is not None:
            ax.axvline(rec_num, linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("max probes")
        ax.set_ylabel("|sampled-full effect| / intensity SD")
        ax.set_title("Effect bias")
        ax.legend(frameon=False)

        ax = axes[1, 0]
        if not pquart.empty:
            for quartile, sub in pquart.groupby("effect_quartile", observed=True):
                sub = sub.sort_values("cap_probes")
                ax.plot(sub["cap_probes"], sub["neglog10p_median"], marker="o", label=str(quartile))
        if rec_num is not None:
            ax.axvline(rec_num, linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("max probes")
        ax.set_ylabel("median -log10(OLS p)")
        ax.set_title("P-value trend (diagnostic only)")
        if not pquart.empty:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[1, 1]
        ax.plot(summary["cap_probes"], summary["eligible_fraction"], marker="o")
        if rec_num is not None:
            ax.axvline(rec_num, linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("max probes")
        ax.set_ylabel("eligible family fraction")
        ax.set_title("Representation")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
    return True


def _write_mask_plot(summary, recommendation, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as exc:
        print(f"[warning] --plot requested but matplotlib is unavailable: {exc}", file=sys.stderr)
        return False

    rec = str(recommendation.iloc[0]["recommended_mask"])
    if "control_source" in summary.columns:
        main = summary[summary["control_source"] != "random"].copy()
        random_df = summary[summary["control_source"] == "random"].copy()
    else:
        main = summary.copy()
        random_df = summary.iloc[0:0].copy()

    if "mask_role" in main.columns:
        labels = [
            f"{mask} [control]" if str(role) == "control" else str(mask)
            for mask, role in zip(main["mask"], main["mask_role"])
        ]
    else:
        labels = main["mask"].astype(str).tolist()
    x = np.arange(len(labels))

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
        fig.suptitle(f"EUbar mask calibration — recommendation: {rec}", fontsize=14)

        ax = axes[0, 0]
        ax.plot(x, main["effect_sd_norm_p90"], marker="o")
        try:
            near_threshold = float(recommendation.iloc[0]["near_optimal_threshold_effect_sd_norm_p90"])
            if np.isfinite(near_threshold):
                ax.axhline(near_threshold, linestyle="--", linewidth=1, label="near-optimal bound")
                ax.legend(frameon=False, fontsize=8)
        except Exception:
            pass
        ax.set_ylabel("p90 coefficient SD / intensity SD")
        ax.set_title("Effect stability at fixed probe count")

        ax = axes[0, 1]
        ax.plot(x, main["residual_sd_ratio_p90"], marker="o")
        ax.set_ylabel("p90 residual SD / intensity SD")
        ax.set_title("Within-family intensity coherence")

        ax = axes[1, 0]
        ax.plot(x, main["eligible_fraction"], marker="o")
        ax.axhline(float(recommendation.iloc[0]["min_eligible_fraction"]), linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("eligible family fraction")
        ax.set_title("Representation")

        ax = axes[1, 1]
        ax.plot(x, main["n_probes_median"], marker="o")
        ax.set_ylabel("median probes per eligible family")
        ax.set_title("Probe support")

        for ax in axes.flat:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_xlabel("mask")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Random controls are intentionally kept off the categorical main page;
        # dozens of mask labels would be unreadable. Show their stability null
        # distribution on a dedicated page instead.
        if len(random_df):
            vals = random_df.loc[
                np.isfinite(random_df["effect_sd_norm_p90"]), "effect_sd_norm_p90"
            ].to_numpy(dtype=float)
            if len(vals):
                fig, ax = plt.subplots(figsize=(9.0, 5.5))
                bins = min(30, max(8, int(np.sqrt(len(vals)) * 2)))
                ax.hist(vals, bins=bins)
                try:
                    candidate_sd = float(recommendation.iloc[0]["effect_sd_norm_p90"])
                    ax.axvline(candidate_sd, linestyle="--", linewidth=1.5, label=f"recommended candidate ({candidate_sd:.4f})")
                except Exception:
                    pass
                ax.set_xlabel("p90 coefficient SD / intensity SD (lower is better)")
                ax.set_ylabel("matched random masks")
                ax.set_title("Matched random-control mask stability")
                ax.legend(frameon=False)
                note = (
                    f"reference={recommendation.iloc[0].get('random_control_reference_mask', rec)}; "
                    f"n={int(recommendation.iloc[0].get('n_random_controls_evaluated', len(vals)))}; "
                    f"candidate percentile={float(recommendation.iloc[0].get('random_control_candidate_percentile', np.nan)):.1f}"
                )
                fig.text(0.5, 0.01, note, ha="center", fontsize=9)
                fig.tight_layout(rect=[0, 0.04, 1, 1])
                pdf.savefig(fig)
                plt.close(fig)
    return True

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Calibrate EUbar analysis parameters from probe representation and effect stability. "
            "Use 'max-probes' for the production probe cap or 'mask' to compare explicit masks."
        )
    )
    p.add_argument(
        "calibration",
        nargs="?",
        default="max-probes",
        choices=("max-probes", "mask"),
        help="Calibration target (default: max-probes)",
    )
    p.add_argument(
        "--array",
        dest="array",
        required=False,
        help="EUbar k-mer positions/array file (required for max-probes calibration)",
    )
    p.add_argument(
        "--kmerPositions", dest="array", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("--intensities", required=True, help="EUbar resid_log intensity file")
    out_group = p.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--out-prefix",
        dest="out_prefix",
        help="Output prefix",
    )
    out_group.add_argument(
        "--out", dest="out_prefix", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("--genome", dest="genome", default=None,
                   help="Genome FASTA (required for mask calibration)")
    p.add_argument(
        "--genome-fasta", dest="genome", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("--masks", default=None,
                   help="Comma-separated explicit 0/1 masks for mask calibration")
    p.add_argument("--mask-file", default=None,
                   help="Text file containing one or more candidate 0/1 masks")
    p.add_argument(
        "--control-masks",
        default=None,
        help=(
            "Comma-separated diagnostic control masks. Controls are fully evaluated and "
            "reported but are excluded from the automatic recommendation."
        ),
    )
    p.add_argument(
        "--control-mask-file",
        default=None,
        help=(
            "Text file containing diagnostic control masks. Controls are evaluated but "
            "excluded from the automatic recommendation."
        ),
    )
    p.add_argument(
        "--random-controls",
        type=int,
        default=0,
        help=(
            "Mask calibration only: generate this many random diagnostic masks matched "
            "to the recommended candidate's span and number of informative positions. "
            "First/last positions remain informative. Random controls are excluded from "
            "the recommendation (default: 0)."
        ),
    )
    p.add_argument("--stability-probes", type=int, default=100,
                   help="Fixed probe count used to compare mask effect stability (default: 100)")
    p.add_argument("--candidate-multiplier", type=int, default=5, help=argparse.SUPPRESS)
    p.add_argument("--kmer-size", type=int, default=8)
    p.add_argument(
        "--n-families",
        dest="n_families",
        type=int,
        default=None,
        help=(
            "Representative/evaluated families. Defaults: 100 for max-probes; "
            "500 for mask calibration."
        ),
    )
    p.add_argument(
        "--n-kmers", dest="n_families", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("--repeats", type=int, default=50, help="Subsamples per family and cap")
    p.add_argument("--caps", type=_caps_arg, default=_caps_arg(DEFAULT_CAPS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min-probes", type=int, default=50, help="Minimum usable probes per family")
    # Retained for reproducibility/development, but not part of the normal public interface.
    p.add_argument("--no-covariates", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--raw-lp", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--max-effect-sd", type=float, default=0.10, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--max-effect-bias", type=float, default=0.10, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--min-eligible-fraction", type=float, default=0.50, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--borderline-relative-tolerance", type=float, default=0.05, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--near-optimal-relative-tolerance", type=float, default=0.05, help=argparse.SUPPRESS
    )
    p.add_argument("--plot", action="store_true", help="Also write a diagnostic PDF")
    p.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    return p


def _run_max_probes(args, intens: IntensityTable) -> int:
    if not args.array:
        raise ValueError("max-probes calibration requires --array")
    print(f"[load] array: {args.array}", file=sys.stderr)
    kmers = KmerIndex.from_file(args.array)

    started = time.monotonic()
    last_progress = [started]

    def progress(kind, info):
        if args.quiet or kind != "family":
            return
        now = time.monotonic()
        accepted = int(info["accepted"])
        requested = int(info["requested"])
        if accepted <= 5 or accepted == requested or now - last_progress[0] >= 2.0:
            elapsed = now - started
            eta = (elapsed / accepted) * max(0, requested - accepted) if accepted else float("nan")
            print(
                f"[calibrate] accepted {accepted}/{requested} families "
                f"(examined {info['examined']}; probes={info['n_probes']:,}; "
                f"eligible_caps={info['eligible_caps']}) | "
                f"elapsed {_format_duration(elapsed)} | ETA ~{_format_duration(eta)}",
                file=sys.stderr,
                flush=True,
            )
            last_progress[0] = now

    result = run_max_probes_calibration(
        kmer_index=kmers.kmers,
        intensities=intens.values,
        caps=args.caps,
        kmer_size=args.kmer_size,
        n_families=(100 if args.n_families is None else args.n_families),
        repeats=args.repeats,
        seed=args.seed,
        min_probes=args.min_probes,
        include_covariates=not args.no_covariates,
        fold_half=not args.raw_lp,
        max_effect_sd=args.max_effect_sd,
        max_effect_bias=args.max_effect_bias,
        min_eligible_fraction=args.min_eligible_fraction,
        borderline_relative_tolerance=args.borderline_relative_tolerance,
        progress_callback=progress,
    )

    paths = {
        "recommendation": args.out_prefix + ".recommendation.tsv",
        "summary": args.out_prefix + ".summary.tsv",
        "family_curves": args.out_prefix + ".family_curves.tsv",
        "contrasts": args.out_prefix + ".contrasts.tsv",
        "pvalue_quartiles": args.out_prefix + ".pvalue_quartiles.tsv",
    }
    result.recommendation.to_csv(paths["recommendation"], sep="\t", index=False, na_rep="NA")
    result.summary.to_csv(paths["summary"], sep="\t", index=False, na_rep="NA")
    result.family_curves.to_csv(paths["family_curves"], sep="\t", index=False, na_rep="NA")
    result.contrasts.to_csv(paths["contrasts"], sep="\t", index=False, na_rep="NA")
    result.pvalue_quartiles.to_csv(paths["pvalue_quartiles"], sep="\t", index=False, na_rep="NA")
    if args.plot:
        pdf_path = args.out_prefix + ".pdf"
        if _write_max_plot(result.summary, result.recommendation, result.pvalue_quartiles, pdf_path):
            paths["pdf"] = pdf_path

    print("\n=== Max-probes recommendation ===")
    print(result.recommendation.to_string(index=False))
    rec_row = result.recommendation.iloc[0]
    if bool(rec_row.get("previous_cap_borderline", False)):
        print(
            "\n[info] Previous tested cap "
            f"{rec_row['previous_tested_cap']} is borderline: "
            f"failed {rec_row['previous_failed_criteria']} only slightly. "
            "The strict recommendation is unchanged."
        )
    print("\n=== Stability summary ===")
    display_cols = [
        "cap_probes", "eligible_fraction", "effect_sd_norm_p90",
        "effect_bias_norm_p90", "neglog10p_median",
    ]
    print(result.summary[display_cols].to_string(index=False))
    print("\nWrote:")
    for path in paths.values():
        print(f"  {path}")
    if len(result.family_curves) == 0:
        print("[warning] no cap/family combinations were evaluable", file=sys.stderr)
    return 0


def _run_mask(args, intens: IntensityTable) -> int:
    if not args.genome:
        raise ValueError("mask calibration requires --genome")
    masks = _load_masks(args)
    control_masks = _load_control_masks(args, masks)
    started = time.monotonic()
    if control_masks and not args.quiet:
        print(
            "[mask-calibrate] diagnostic controls (excluded from recommendation): "
            + ",".join(control_masks),
            file=sys.stderr,
            flush=True,
        )

    def progress(kind, info):
        if args.quiet:
            return
        if kind == "mask_scan_start":
            print(
                f"[mask-calibrate] {info['mask_no']}/{info['n_masks']} mask={info['mask']} "
                f"role={info.get('control_source', info.get('mask_role', 'candidate'))} "
                f"candidates={info['n_candidates']} target_codes={info['n_target_codes']} — scanning probes",
                file=sys.stderr, flush=True,
            )
        elif kind == "mask_scan_done":
            print(
                f"[mask-calibrate] mask={info['mask']} indexed {info['n_windows']:,} probe windows; "
                f"target hits={info['n_target_region_hits']:,}",
                file=sys.stderr, flush=True,
            )
        elif kind == "mask_family":
            evaluated = int(info["evaluated"])
            requested = int(info["requested"])
            if evaluated <= 3 or evaluated == requested or evaluated % 20 == 0:
                print(
                    f"[mask-calibrate] mask={info['mask']} stability {evaluated}/{requested} families | "
                    f"elapsed {_format_duration(time.monotonic() - started)}",
                    file=sys.stderr, flush=True,
                )

    result = run_mask_calibration(
        intensities=intens.values,
        genome_fasta=args.genome,
        masks=masks,
        control_masks=control_masks,
        random_controls=args.random_controls,
        n_families=(500 if args.n_families is None else args.n_families),
        repeats=args.repeats,
        stability_probes=args.stability_probes,
        min_probes=args.min_probes,
        min_eligible_fraction=args.min_eligible_fraction,
        near_optimal_relative_tolerance=args.near_optimal_relative_tolerance,
        candidate_multiplier=args.candidate_multiplier,
        seed=args.seed,
        include_covariates=not args.no_covariates,
        fold_half=not args.raw_lp,
        progress_callback=progress,
    )

    paths = {
        "recommendation": args.out_prefix + ".recommendation.tsv",
        "summary": args.out_prefix + ".summary.tsv",
        "families": args.out_prefix + ".families.tsv",
        "family_curves": args.out_prefix + ".family_curves.tsv",
    }
    result.recommendation.to_csv(paths["recommendation"], sep="\t", index=False, na_rep="NA")
    result.summary.to_csv(paths["summary"], sep="\t", index=False, na_rep="NA")
    result.families.to_csv(paths["families"], sep="\t", index=False, na_rep="NA")
    result.family_curves.to_csv(paths["family_curves"], sep="\t", index=False, na_rep="NA")
    if args.plot:
        pdf_path = args.out_prefix + ".pdf"
        if _write_mask_plot(result.summary, result.recommendation, pdf_path):
            paths["pdf"] = pdf_path

    print("\n=== Mask recommendation ===")
    print(result.recommendation.to_string(index=False))
    rec_row = result.recommendation.iloc[0]
    if str(rec_row.get("recommended_mask", "NA")) != "NA":
        n_near = int(rec_row.get("n_near_optimal", 0))
        tol = float(rec_row.get("near_optimal_relative_tolerance", 0.0))
        if n_near > 1:
            print(
                f"\n[info] Near-optimal candidate masks (within {100.0 * tol:.1f}% of the best "
                f"stability metric): {rec_row['near_optimal_masks']}"
            )
            print(
                "[info] This is a practical tolerance band, not a statistical-equivalence claim."
            )
        control_status = str(rec_row.get("control_status", "NO_CONTROLS"))
        if control_status != "NO_CONTROLS":
            print(
                "\n[control] " + control_status
                + f"; best control={rec_row.get('best_control_mask', 'NA')}"
                + f"; p90 stability={rec_row.get('best_control_effect_sd_norm_p90', 'NA')}"
                + f"; % vs recommended={rec_row.get('best_control_pct_vs_recommended', 'NA')}"
            )
            if control_status == "CONTROL_OUTPERFORMS_RECOMMENDED_CANDIDATE":
                print(
                    "[control] WARNING: a diagnostic control mask is more stable than the "
                    "recommended candidate. Treat mask architecture as unresolved."
                )
            elif control_status == "CONTROL_WITHIN_CANDIDATE_NEAR_OPTIMAL_BAND":
                print(
                    "[control] NOTE: a diagnostic control falls within the candidate near-optimal "
                    "band. The exact architecture is not specifically resolved by stability alone."
                )
        n_rand = int(rec_row.get("n_random_controls_evaluated", 0) or 0)
        if n_rand > 0:
            print(
                "\n[random-controls] matched to "
                f"{rec_row.get('random_control_reference_mask', 'NA')}; "
                f"evaluated={n_rand}; "
                f"candidate percentile={float(rec_row['random_control_candidate_percentile']):.1f}; "
                f"fraction random <= candidate={float(rec_row['random_control_fraction_better_or_equal']):.3f}; "
                f"empirical tail p={float(rec_row['random_control_empirical_p']):.4f}"
            )
            print(
                "[random-controls] Lower effect-instability is better. Percentile is the "
                "fraction of matched random masks that the candidate equals or outperforms."
            )

    print("\n=== Mask summary ===")
    display_cols = [
        "mask", "mask_role", "control_source", "mask_span", "n_informative", "eligible_fraction",
        "n_probes_median", "minor_allele_fraction_median", "residual_sd_ratio_p90",
        "effect_sd_norm_p90", "effect_bias_norm_p90",
        "stability_pct_above_best_candidate", "near_optimal",
        "within_candidate_near_optimal_band", "beats_recommended_candidate",
    ]
    print(result.summary[display_cols].to_string(index=False))
    print("\nWrote:")
    for path in paths.values():
        print(f"  {path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[load] intensities: {args.intensities}", file=sys.stderr)
    intens = IntensityTable.from_file(args.intensities)
    try:
        if args.calibration == "max-probes":
            return _run_max_probes(args, intens)
        if args.calibration == "mask":
            return _run_mask(args, intens)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
