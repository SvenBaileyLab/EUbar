"""Calibration tables, console summaries and optional plots."""
import sys
import numpy as np


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



def write_max_probe_report(result, out_prefix: str, *, plot: bool = False) -> int:
    paths = {
        "recommendation": out_prefix + ".recommendation.tsv",
        "summary": out_prefix + ".summary.tsv",
        "family_curves": out_prefix + ".family_curves.tsv",
        "contrasts": out_prefix + ".contrasts.tsv",
        "pvalue_quartiles": out_prefix + ".pvalue_quartiles.tsv",
    }
    result.recommendation.to_csv(paths["recommendation"], sep="\t", index=False, na_rep="NA")
    result.summary.to_csv(paths["summary"], sep="\t", index=False, na_rep="NA")
    result.family_curves.to_csv(paths["family_curves"], sep="\t", index=False, na_rep="NA")
    result.contrasts.to_csv(paths["contrasts"], sep="\t", index=False, na_rep="NA")
    result.pvalue_quartiles.to_csv(paths["pvalue_quartiles"], sep="\t", index=False, na_rep="NA")
    if plot:
        pdf_path = out_prefix + ".pdf"
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



def write_mask_report(result, out_prefix: str, *, plot: bool = False) -> int:
    paths = {
        "recommendation": out_prefix + ".recommendation.tsv",
        "summary": out_prefix + ".summary.tsv",
        "families": out_prefix + ".families.tsv",
        "family_curves": out_prefix + ".family_curves.tsv",
    }
    result.recommendation.to_csv(paths["recommendation"], sep="\t", index=False, na_rep="NA")
    result.summary.to_csv(paths["summary"], sep="\t", index=False, na_rep="NA")
    result.families.to_csv(paths["families"], sep="\t", index=False, na_rep="NA")
    result.family_curves.to_csv(paths["family_curves"], sep="\t", index=False, na_rep="NA")
    if plot:
        pdf_path = out_prefix + ".pdf"
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
