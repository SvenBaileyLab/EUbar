from __future__ import annotations

from dataclasses import asdict
from eubar.task_config import MotifsConfig, TaskConfigError

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from .core.patterns import SequenceMask
# Keep earlier helper imports available to notebooks and existing callers.
from .core.motif_data import BASES
from .core.motif_data import (
    reverse_complement as reverse_complement,
    canonical_kmer as canonical_kmer,
    kmer_matches_pattern as kmer_matches_pattern,
    fg_indices_for_pattern as fg_indices_for_pattern,
    read_intensities as read_intensities,
    read_unique_kmer_positions as read_unique_kmer_positions,
    ProbeRanks as ProbeRanks,
    build_probe_ranks as build_probe_ranks,
    build_kmer_to_probe_idx as build_kmer_to_probe_idx,
)
from .core.motif_search import (
    escore_auc_minus_half_all_bg as escore_auc_minus_half_all_bg,
    reduced_escore_and_p as reduced_escore_and_p,
    reduced_test_four_variants as reduced_test_four_variants,
    softmax_from_escores as softmax_from_escores,
    ppm_from_seed_wobble as ppm_from_seed_wobble,
    choose_seed as choose_seed,
    extend_side_greedy as extend_side_greedy,
    _info_content as _info_content,
    extend_side as extend_side,
)
from .core.motif_mask import (
    _mask_code_to_bases as _mask_code_to_bases,
    _mask_pattern_from_code as _mask_pattern_from_code,
    _mask_seed_to_code as _mask_seed_to_code,
    _region_coords as _region_coords,
    collect_mask_seed_candidates as collect_mask_seed_candidates,
    _hits_to_probe_idx as _hits_to_probe_idx,
    scan_mask_codes_for_motifs as scan_mask_codes_for_motifs,
    choose_mask_seed as choose_mask_seed,
    _variant_code_for_informative_base as _variant_code_for_informative_base,
    ppm_from_mask_seed_wobble as ppm_from_mask_seed_wobble,
)
from .core.motif_output import (
    _dna_colors as _dna_colors,
    stitch_ppm as stitch_ppm,
    consensus_from_ppm as consensus_from_ppm,
    write_meme as write_meme,
    ppm_to_bits_matrix as ppm_to_bits_matrix,
    reverse_complement_ppm as reverse_complement_ppm,
    reverse_complement_reduced_df as reverse_complement_reduced_df,
    plot_logo as plot_logo,
    plot_enrichment_bars as plot_enrichment_bars,
    plot_seed_enrichment_curve as plot_seed_enrichment_curve,
    plot_seed_enrichment_roc as plot_seed_enrichment_roc,
    plot_escore_histogram as plot_escore_histogram,
    motif_logscore as motif_logscore,
    plot_motif_vs_escore as plot_motif_vs_escore,
)


def run_masked_motifs(args, intensities: Dict[str, float], ranks: ProbeRanks) -> int:
    """Masked motif discovery path.

    The mask defines the motif span and which positions are informative. Seed
    discovery is performed on masked signatures sampled from the highest-
    intensity probes, then rescored against the full probe universe. Wobble is
    restricted to informative positions; spacer positions remain uniform in the
    PPM and therefore contribute zero information in the bits logo.
    """
    mask = SequenceMask.parse(args.mask)
    if not args.genome:
        sys.stderr.write("[error] --genome is required when --mask is used.\n")
        return 2

    probe_regions = tuple(ranks.regions)
    if not probe_regions:
        sys.stderr.write("[error] Intensity file contains no probe regions.\n")
        return 2

    sys.stderr.write(
        f"[MASK-MOTIF] mask={mask.pattern} span={mask.span} "
        f"informative={mask.n_informative}\n"
    )
    if args.array:
        sys.stderr.write(
            "[MASK-MOTIF] note: --array is not used in masked motif mode; "
            "probe regions come from the intensity file and sequence comes from --genome.\n"
        )

    if args.seed is None:
        candidate_codes = collect_mask_seed_candidates(
            mask=mask,
            ranks=ranks,
            genome_fasta=args.genome,
            top_probe_count=int(args.mask_candidate_probes),
            max_candidates=int(args.mask_max_candidates),
        )
        if not candidate_codes:
            sys.stderr.write("[error] No masked seed candidates could be generated.\n")
            return 2
        sys.stderr.write(
            f"[MASK-MOTIF] generated {len(candidate_codes):,} candidate signatures "
            f"from top {min(int(args.mask_candidate_probes), len(ranks.regions)):,} probes; "
            "scanning full probe universe\n"
        )
        candidate_hits, scan_stats = scan_mask_codes_for_motifs(
            mask=mask,
            target_codes=candidate_codes,
            probe_regions=probe_regions,
            genome_fasta=args.genome,
        )
        sys.stderr.write(
            f"[MASK-MOTIF] indexed {scan_stats['n_windows']:,} probe windows; "
            f"candidate hits={scan_stats['n_target_region_hits']:,}\n"
        )
        es_df = choose_mask_seed(
            mask=mask,
            candidate_codes=candidate_codes,
            target_hits=candidate_hits,
            ranks=ranks,
            min_F=int(args.min_F),
        )
        if es_df.empty:
            sys.stderr.write("[error] No masked candidates passed the --min-F filter.\n")
            return 2
        seed_code = int(es_df.loc[0, "code"])
        seed_hits = candidate_hits.get(seed_code, {})
        top_path = os.path.join(args.outdir, f"{args.prefix}.top_escores.tsv")
        es_df.head(int(args.top_n_report)).to_csv(top_path, sep="\t", index=False)
        sys.stderr.write(f"[ok] Wrote top E-score table: {top_path}\n")
    else:
        try:
            seed_code = _mask_seed_to_code(args.seed, mask)
        except ValueError as exc:
            sys.stderr.write(f"[error] {exc}\n")
            return 2
        candidate_hits, scan_stats = scan_mask_codes_for_motifs(
            mask=mask,
            target_codes=[seed_code],
            probe_regions=probe_regions,
            genome_fasta=args.genome,
        )
        seed_hits = candidate_hits.get(seed_code, {})
        seed_idx = _hits_to_probe_idx(seed_hits, ranks.region_to_i)
        F = int(len(seed_idx))
        E = (
            escore_auc_minus_half_all_bg(seed_idx, ranks.i_to_rank, len(ranks.scores))
            if F > 0
            else np.nan
        )
        es_df = pd.DataFrame(
            [
                {
                    "kmer": _mask_pattern_from_code(mask, seed_code),
                    "code": seed_code,
                    "E": E,
                    "F": F,
                    "gaps": int(mask.span - mask.n_informative),
                }
            ]
        )

    seed = _mask_pattern_from_code(mask, seed_code)
    sys.stderr.write(f"[seed] {seed}\n")

    seed_bases = _mask_code_to_bases(seed_code, mask.n_informative)
    wobble_codes = sorted(
        {
            _variant_code_for_informative_base(seed_bases, j, base)
            for j in range(mask.n_informative)
            for base in BASES
        }
    )
    sys.stderr.write(
        f"[MASK-MOTIF] wobbling {mask.n_informative} informative positions "
        f"({len(wobble_codes)} unique signatures); scanning probe universe\n"
    )
    wobble_hits, wobble_stats = scan_mask_codes_for_motifs(
        mask=mask,
        target_codes=wobble_codes,
        probe_regions=probe_regions,
        genome_fasta=args.genome,
    )
    sys.stderr.write(
        f"[MASK-MOTIF] wobble scan indexed {wobble_stats['n_windows']:,} probe windows; "
        f"target hits={wobble_stats['n_target_region_hits']:,}\n"
    )

    reduced_df, ppm = ppm_from_mask_seed_wobble(
        mask=mask,
        seed_code=seed_code,
        variant_hits=wobble_hits,
        ranks=ranks,
        min_per_base=int(args.min_per_base),
        beta=float(args.beta),
        min_support=int(args.min_support),
        pseudocount=float(args.pseudocount),
    )

    consensus_chars = []
    for pos in range(mask.span):
        if mask.pattern[pos] == "0":
            consensus_chars.append(".")
        else:
            consensus_chars.append(str(ppm.loc[pos, list(BASES)].idxmax()))
    consensus = "".join(consensus_chars)
    sys.stderr.write(f"[consensus] {consensus}\n")

    ppm_path = os.path.join(args.outdir, f"{args.prefix}.ppm.tsv")
    meme_path = os.path.join(args.outdir, f"{args.prefix}.meme")
    logo_prob_path = os.path.join(args.outdir, f"{args.prefix}.logo_prob.png")
    logo_bits_path = os.path.join(args.outdir, f"{args.prefix}.logo_bits.png")
    logo_prob_rc_path = os.path.join(args.outdir, f"{args.prefix}.logo_prob_rc.png")
    logo_bits_rc_path = os.path.join(args.outdir, f"{args.prefix}.logo_bits_rc.png")
    seed_curve_path = os.path.join(args.outdir, f"{args.prefix}.seed_enrichment_curve.png")
    seed_roc_path = os.path.join(args.outdir, f"{args.prefix}.seed_roc.png")
    seed_hist_path = os.path.join(args.outdir, f"{args.prefix}.seed_escore_hist.png")
    enrich_bar_path = os.path.join(args.outdir, f"{args.prefix}.reduced_enrichment.png")
    enrich_bar_rc_path = os.path.join(args.outdir, f"{args.prefix}.reduced_enrichment_rc.png")
    reduced_path = os.path.join(args.outdir, f"{args.prefix}.reduced.tsv")
    reduced_full_path = os.path.join(args.outdir, f"{args.prefix}.reduced_full.tsv")

    reduced_full_df = reduced_df.copy()
    if not reduced_full_df.empty:
        reduced_full_df["side"] = "core"
        reduced_full_df["step"] = reduced_full_df["pos"].astype(int)
        reduced_full_df["gaps_used"] = int(mask.span - mask.n_informative)
        reduced_full_df.sort_values(["pos", "base"], inplace=True)

    ppm.to_csv(ppm_path, sep="\t")
    reduced_df.to_csv(reduced_path, sep="\t", index=False)
    reduced_full_df.to_csv(reduced_full_path, sep="\t", index=False)

    ppm_normal = ppm.reset_index(drop=True)
    ppm_rc = reverse_complement_ppm(ppm_normal)
    write_meme(ppm_normal, meme_path, motif_name=consensus)
    title = f"{consensus} (mask={mask.pattern})"
    plot_logo(ppm_normal, logo_prob_path, title=title, pretty_logo=bool(args.pretty_logo), mode="prob")
    plot_logo(ppm_normal, logo_bits_path, title=title, pretty_logo=bool(args.pretty_logo), mode="bits")
    plot_logo(ppm_rc, logo_prob_rc_path, title=title + " [RC]", pretty_logo=bool(args.pretty_logo), mode="prob")
    plot_logo(ppm_rc, logo_bits_rc_path, title=title + " [RC]", pretty_logo=bool(args.pretty_logo), mode="bits")
    plot_enrichment_bars(reduced_full_df, enrich_bar_path, title=None, pretty_logo=bool(args.pretty_logo))
    plot_enrichment_bars(
        reverse_complement_reduced_df(reduced_full_df),
        enrich_bar_rc_path,
        title=None,
        pretty_logo=bool(args.pretty_logo),
    )

    seed_idx = _hits_to_probe_idx(seed_hits, ranks.region_to_i)
    if seed_idx.size > 0:
        plot_seed_enrichment_curve(seed, seed_idx, ranks, seed_curve_path)
        plot_seed_enrichment_roc(seed, seed_idx, ranks, seed_roc_path)
    plot_escore_histogram(es_df, seed_hist_path, title="Masked seed candidate E-score distribution")

    sys.stderr.write(f"[ok] PPM:   {ppm_path}\n")
    sys.stderr.write(f"[ok] MEME:  {meme_path}\n")
    sys.stderr.write(f"[ok] logo:  {logo_prob_path}\n")
    sys.stderr.write(f"[ok] reduced enrichment plot: {enrich_bar_path}\n")
    sys.stderr.write(f"[ok] seed curve: {seed_curve_path}\n")
    sys.stderr.write(f"[ok] seed ROC:   {seed_roc_path}\n")
    sys.stderr.write(f"[ok] E-score hist: {seed_hist_path}\n")
    sys.stderr.write(f"[ok] reduced table: {reduced_path}\n")
    sys.stderr.write(f"[ok] reduced_full: {reduced_full_path}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    defaults = MotifsConfig()
    ap = argparse.ArgumentParser(
        description="Seed-and-wobble motif discovery from probe intensities + kmer occurrence index."
    )
    ap.add_argument("--intensities", required=True, help="Path to probe intensity file")
    ap.add_argument(
        "--array",
        dest="array",
        required=False,
        default=defaults.array,
        help="Path to k-mer array file mapping k-mers to genomic regions",
    )
    ap.add_argument(
        "--kmers", dest="array", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    ap.add_argument(
        "--mask",
        default=defaults.mask,
        help=(
            "Optional explicit 0/1 sequence mask for masked motif discovery, e.g. "
            "11111000011111. When supplied, the mask defines the motif span and "
            "--genome is required; the ordinary k-mer seed/extension path is left unchanged."
        ),
    )
    ap.add_argument(
        "--genome",
        default=defaults.genome,
        help="Genome FASTA used for masked motif discovery (--mask mode)",
    )
    ap.add_argument(
        "--mask-candidate-probes",
        type=int,
        default=defaults.mask_candidate_probes,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--mask-max-candidates",
        type=int,
        default=defaults.mask_max_candidates,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--kmer-size", type=int, default=defaults.kmer_size, help="k-mer size (default: 8)")
    ap.add_argument(
        "--no-combine-revcomp",
        dest="combine_revcomp",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    ap.set_defaults(combine_revcomp=True)

    ap.add_argument(
        "--min-F",
        type=int,
        default=defaults.min_F,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--max-gaps",
        type=int,
        default=defaults.max_gaps,
        help="Max number of wildcard ('.') positions when searching seed patterns (default: 3; use 0 for exact k-mers only)",
    )
    ap.add_argument(
        "--min-per-base",
        type=int,
        default=defaults.min_per_base,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--min-support",
        type=int,
        default=defaults.min_support,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--pseudocount",
        type=float,
        default=defaults.pseudocount,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=defaults.beta,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--auto-max-steps",
        type=int,
        default=defaults.auto_max_steps,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--extend-left",
        type=int,
        default=defaults.extend_left,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--extend-right",
        type=int,
        default=defaults.extend_right,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--ic-stop-threshold",
        type=float,
        default=defaults.ic_stop_threshold,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--ic-stop-consecutive",
        type=int,
        default=defaults.ic_stop_consecutive,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--outdir",
        default=defaults.outdir,
        help="Output directory (default: current directory)",
    )
    ap.add_argument(
        "--prefix",
        default=defaults.prefix,
        help="Output prefix (default: affinity_motif)",
    )

    ap.add_argument(
        "--pretty-logo",
        action="store_true",
        help="Render a prettier motif logo (light gray background + fixed DNA colors)",
    )
    ap.add_argument(
        "--seed",
        default=defaults.seed,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--top-n-report",
        type=int,
        default=defaults.top_n_report,
        help=argparse.SUPPRESS,
    )

    ap.set_defaults(**asdict(defaults))
    return ap


def run_motifs(config: MotifsConfig) -> int:
    """Run motifs with validated task options; no command-line parsing."""
    config.validate()
    args = config

    # Cap max gaps used during EXTENSION to what is possible for the chosen k (anchor length = k-1).
    # (Seed search can still use up to args.max_gaps wildcards within length-k patterns.)
    max_gaps_ext = min(int(args.max_gaps), int(args.kmer_size) - 1)
    if max_gaps_ext < 0:
        max_gaps_ext = 0

    if args.kmer_size != 8:
        sys.stderr.write(
            f"[note] Using k={args.kmer_size}. Seed/wobble/extension support this, "
            "but probe support and runtime may change with k.\n"
        )

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    intensities = read_intensities(args.intensities)
    ranks = build_probe_ranks(intensities)

    if args.mask is not None:
        return run_masked_motifs(args, intensities, ranks)

    if not args.array:
        sys.stderr.write("[error] --array is required unless --mask is supplied.\n")
        return 2

    kmer_positions = read_unique_kmer_positions(args.array)

    # basic k-mer length sanity
    kmer_positions = {
        k: v for k, v in kmer_positions.items() if len(k) == args.kmer_size
    }

    kmer_to_idx = build_kmer_to_probe_idx(
        kmer_positions=kmer_positions,
        region_to_i=ranks.region_to_i,
        combine_revcomp=args.combine_revcomp,
    )

    if args.seed is None:
        es_df = choose_seed(
            kmer_to_idx, ranks.i_to_rank, min_F=args.min_F, max_gaps=args.max_gaps
        )
        if es_df.empty:
            sys.stderr.write("[error] No kmers passed min-F filter.\n")
            return 2
        seed = str(es_df.loc[0, "kmer"])

        top_path = os.path.join(args.outdir, f"{args.prefix}.top_escores.tsv")
        es_df.head(args.top_n_report).to_csv(top_path, sep="\t", index=False)
        sys.stderr.write(f"[ok] Wrote top E-score table: {top_path}\n")
    else:
        seed = args.seed.strip().upper()
        es_df = choose_seed(
            kmer_to_idx, ranks.i_to_rank, min_F=args.min_F, max_gaps=args.max_gaps
        )

    sys.stderr.write(f"[seed] {seed}\n")

    # Core seed-and-wobble
    reduced_df, core_ppm = ppm_from_seed_wobble(
        seed=seed,
        kmer_to_idx=kmer_to_idx,
        i_to_rank=ranks.i_to_rank,
        min_per_base=args.min_per_base,
        beta=args.beta,
        min_support=args.min_support,
        pseudocount=args.pseudocount,
    )

    # If the seed search used wildcards, derive a concrete core consensus to use for optional extension.
    core_seed = consensus_from_ppm(core_ppm)

    # Optional flanks
    left_flank: List[pd.Series] = []
    right_flank: List[pd.Series] = []
    left_rows: List[Dict] = []
    right_rows: List[Dict] = []

    # Extension works by shifting a k-mer window and (optionally)
    # turning low-information anchor positions into '.' wildcards to preserve support.
    #
    # By default we auto-extend (args.extend_left/right == -1) until support fails.

    window = core_seed
    window_probs = [core_ppm.loc[i] for i in core_ppm.index]

    # left
    if args.extend_left != 0:
        steps = args.auto_max_steps if args.extend_left < 0 else int(args.extend_left)
        left_flank, window, window_probs, left_rows = extend_side(
            window=window,
            window_probs=window_probs,
            kmer_to_idx=kmer_to_idx,
            i_to_rank=ranks.i_to_rank,
            beta=args.beta,
            min_per_base=args.min_per_base,
            side="left",
            max_steps=steps,
            max_gaps=int(max_gaps_ext),
            ic_stop_threshold=float(args.ic_stop_threshold),
            ic_stop_consecutive=int(args.ic_stop_consecutive),
        )

    # right (start from core again so left and right are symmetric around the core)
    window = core_seed
    window_probs = [core_ppm.loc[i] for i in core_ppm.index]

    if args.extend_right != 0:
        steps = args.auto_max_steps if args.extend_right < 0 else int(args.extend_right)
        right_flank, window, window_probs, right_rows = extend_side(
            window=window,
            window_probs=window_probs,
            kmer_to_idx=kmer_to_idx,
            i_to_rank=ranks.i_to_rank,
            beta=args.beta,
            min_per_base=args.min_per_base,
            side="right",
            max_steps=steps,
            max_gaps=int(max_gaps_ext),
            ic_stop_threshold=float(args.ic_stop_threshold),
            ic_stop_consecutive=int(args.ic_stop_consecutive),
        )
    ppm = stitch_ppm(left_flank, core_ppm, right_flank)
    consensus = consensus_from_ppm(ppm)
    sys.stderr.write(f"[consensus] {consensus}\n")

    # Write outputs
    ppm_path = os.path.join(args.outdir, f"{args.prefix}.ppm.tsv")
    meme_path = os.path.join(args.outdir, f"{args.prefix}.meme")
    logo_prob_path = os.path.join(args.outdir, f"{args.prefix}.logo_prob.png")
    logo_bits_path = os.path.join(args.outdir, f"{args.prefix}.logo_bits.png")
    logo_prob_rc_path = os.path.join(args.outdir, f"{args.prefix}.logo_prob_rc.png")
    logo_bits_rc_path = os.path.join(args.outdir, f"{args.prefix}.logo_bits_rc.png")
    # Back-compat: keep the old name as the probability logo
    logo_path = logo_prob_path
    seed_curve_path = os.path.join(
        args.outdir, f"{args.prefix}.seed_enrichment_curve.png"
    )
    seed_roc_path = os.path.join(args.outdir, f"{args.prefix}.seed_roc.png")
    seed_hist_path = os.path.join(args.outdir, f"{args.prefix}.seed_escore_hist.png")
    qc_path = os.path.join(args.outdir, f"{args.prefix}.motif_vs_E.png")
    enrich_bar_path = os.path.join(args.outdir, f"{args.prefix}.reduced_enrichment.png")
    enrich_bar_rc_path = os.path.join(
        args.outdir, f"{args.prefix}.reduced_enrichment_rc.png"
    )

    reduced_path = os.path.join(args.outdir, f"{args.prefix}.reduced.tsv")

    # Full reduced table (core wobble + flank extensions) for recreating Fig 3a-style panels.
    reduced_full_path = os.path.join(args.outdir, f"{args.prefix}.reduced_full.tsv")

    core_len = int(core_ppm.shape[0])

    core_df = reduced_df.copy()
    core_df["side"] = "core"
    core_df["step"] = core_df["pos"].astype(int)
    core_df["gaps_used"] = 0

    left_df = pd.DataFrame(left_rows)
    if not left_df.empty:
        left_df["pos"] = -(left_df["step"].astype(int) + 1)

    right_df = pd.DataFrame(right_rows)
    if not right_df.empty:
        right_df["pos"] = core_len + right_df["step"].astype(int)

    reduced_full_df = pd.concat(
        [left_df, core_df, right_df], ignore_index=True, sort=False
    )
    # stable ordering
    if not reduced_full_df.empty:
        reduced_full_df["pos"] = reduced_full_df["pos"].astype(int)
        reduced_full_df.sort_values(["pos", "side", "base"], inplace=True)

    ppm.to_csv(ppm_path, sep="\t")
    reduced_df.to_csv(reduced_path, sep="\t", index=False)
    reduced_full_df.to_csv(reduced_full_path, sep="\t", index=False)

    ppm_normal = ppm.reset_index(drop=True)
    ppm_rc = reverse_complement_ppm(ppm_normal)
    write_meme(ppm_normal, meme_path, motif_name=consensus)
    plot_logo(
        ppm_normal,
        logo_prob_path,
        title=f"{consensus} (seed={seed})",
        pretty_logo=bool(args.pretty_logo),
        mode="prob",
    )
    plot_logo(
        ppm_normal,
        logo_bits_path,
        title=f"{consensus} (seed={seed})",
        pretty_logo=bool(args.pretty_logo),
        mode="bits",
    )
    plot_logo(
        ppm_rc,
        logo_prob_rc_path,
        title=f"{consensus} (seed={seed}) [RC]",
        pretty_logo=bool(args.pretty_logo),
        mode="prob",
    )
    plot_logo(
        ppm_rc,
        logo_bits_rc_path,
        title=f"{consensus} (seed={seed}) [RC]",
        pretty_logo=bool(args.pretty_logo),
        mode="bits",
    )
    plot_enrichment_bars(
        reduced_full_df, enrich_bar_path, title=None, pretty_logo=bool(args.pretty_logo)
    )
    reduced_full_df_rc = reverse_complement_reduced_df(reduced_full_df)
    plot_enrichment_bars(
        reduced_full_df_rc,
        enrich_bar_rc_path,
        title=None,
        pretty_logo=bool(args.pretty_logo),
    )

    # Seed enrichment plots + histogram of candidate E-scores
    fg_seed = fg_indices_for_pattern(seed, kmer_to_idx)
    if fg_seed.size > 0:
        plot_seed_enrichment_curve(seed, fg_seed, ranks, seed_curve_path)
        plot_seed_enrichment_roc(seed, fg_seed, ranks, seed_roc_path)
    plot_escore_histogram(
        es_df, seed_hist_path, title="Seed candidate E-score distribution"
    )

    # QC plot (only if motif length == kmer_size)
    try:
        es_df_exact = es_df[es_df.get("gaps", 0) == 0].copy()
        plot_motif_vs_escore(es_df_exact, ppm.reset_index(drop=True), qc_path)
    except Exception:
        pass

    sys.stderr.write(f"[ok] PPM:   {ppm_path}\n")
    sys.stderr.write(f"[ok] MEME:  {meme_path}\n")
    sys.stderr.write(f"[ok] logo:  {logo_path}\n")
    sys.stderr.write(f"[ok] reduced enrichment plot: {enrich_bar_path}\n")
    sys.stderr.write(f"[ok] seed curve: {seed_curve_path}\n")
    sys.stderr.write(f"[ok] seed ROC:   {seed_roc_path}\n")
    sys.stderr.write(f"[ok] E-score hist: {seed_hist_path}\n")
    sys.stderr.write(f"[ok] reduced table: {reduced_path}\n")
    sys.stderr.write(f"[ok] reduced_full: {reduced_full_path}\n")

    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = MotifsConfig.from_values(vars(args))
        return run_motifs(config)
    except TaskConfigError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
