#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict
from eubar.task_config import SnvConfig, TaskConfigError

# EUbar SNV regressions are tiny; multithreaded BLAS adds substantial overhead.
# Default to one numerical-library thread per process. Advanced users can override
# this before launch with EUBAR_BLAS_THREADS (for example, EUBAR_BLAS_THREADS=2).
import os
_BLAS_THREADS = os.environ.get("EUBAR_BLAS_THREADS", "1")
for _env_name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_env_name] = _BLAS_THREADS

import argparse
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from eubar.core.data import IntensityTable, KmerIndex
from eubar.core.sequence import SnvWindow
from eubar.core.matching import MotifMatcher
from eubar.core.masking import MaskedMotifMatcher
from eubar.core.patterns import SequenceMask
from eubar.core.design import DesignBuilder
from eubar.core.regression import RegressionEngine
from eubar.core.analyze import analyze_snv
from eubar.core.rand import RandRegressor, RandSampler
from eubar.core.reporting import print_motif_effect_table


from eubar.core.snv_results import (
    _extract_stat as _extract_stat,
    _safe_float as _safe_float,
    _parse_ref_alt_from_snv as _parse_ref_alt_from_snv,
    _holm_adjust as _holm_adjust,
    _apply_holm_within_snv as _apply_holm_within_snv,
    _best_supported_motif_pos as _best_supported_motif_pos,
    _best_pval_summary_rows as _best_pval_summary_rows,
)
from eubar.core.reporting import (
    _format_pval as _format_pval,
    _print_best_pval_table as _print_best_pval_table,
    _print_motif_effect_table_holm as _print_motif_effect_table_holm,
    _format_cell as _format_cell,
    _append_long_tsv as _append_long_tsv,
)


# Parent-loaded objects are inherited cheaply by fork-based workers on Linux/WSL.
# Spawn-based platforms fall back to loading the two input tables once per worker.
_PARENT_SHARED_STATE = None
_WORKER_STATE = None


def _make_rand_sampler_for_matcher(matcher):
    if hasattr(matcher, "make_rand_sampler"):
        return matcher.make_rand_sampler()
    return RandSampler(matcher.kmer_positions)


def _init_snv_worker(
    intensities_path, array_path, genome_path, config, prebuilt_matcher=None
):
    """Initialize one SNP worker and keep heavy objects resident for its lifetime."""
    global _WORKER_STATE

    # Masked matching warms the FASTA cache in the parent. Forked workers must
    # reopen it: inherited file descriptors share a seek position across processes.
    from eubar.core.sequence import _open_fasta
    _open_fasta.cache_clear()

    shared = _PARENT_SHARED_STATE
    if shared is not None:
        matcher = shared["matcher"]
        design = shared["design"]
    else:
        intens = IntensityTable.from_file(intensities_path)
        design = DesignBuilder(intens.values)
        if prebuilt_matcher is not None:
            matcher = prebuilt_matcher
        else:
            kmers = KmerIndex.from_file(array_path)
            matcher = MotifMatcher(kmers.kmers)

    engine = RegressionEngine()
    no_rand = bool(config["no_rand"])
    rand_sampler = None if no_rand else _make_rand_sampler_for_matcher(matcher)
    rand_regressor = None if no_rand else RandRegressor(
        design.intensities, engine=engine
    )

    _WORKER_STATE = {
        "matcher": matcher,
        "design": design,
        "engine": engine,
        "rand_sampler": rand_sampler,
        "rand_regressor": rand_regressor,
        "genome": genome_path,
        "config": dict(config),
    }


def _analyze_one_snv_worker(snv_str):
    """Process one SNV and return data to the parent; never print from workers."""
    state = _WORKER_STATE
    if state is None:
        return snv_str, None, None, "worker was not initialized"

    cfg = state["config"]
    try:
        analysis_k = int(cfg["analysis_k"])
        snv = SnvWindow.from_snv(
            snv_str,
            state["genome"],
            k=analysis_k,
            debug=bool(cfg["debug"]),
        )
        rows = analyze_snv(
            snv=snv,
            snv_str=snv_str,
            matcher=state["matcher"],
            design=state["design"],
            engine=state["engine"],
            k=analysis_k,
            mode=cfg["mode"],
            include_covariates=bool(cfg["include_covariates"]),
            fold_half=bool(cfg["fold_half"]),
            rand_n=int(cfg["rand_n"]),
            max_probes=cfg["max_probes"],
            seed=int(cfg["seed"]),
            rand_sampler=state["rand_sampler"],
            rand_regressor=state["rand_regressor"],
        )
        return snv_str, snv, rows, None
    except Exception as exc:
        return snv_str, None, None, str(exc)


def _preferred_mp_context():
    """Prefer fork on POSIX so the large read-only k-mer index is copy-on-write."""
    try:
        if "fork" in mp.get_all_start_methods():
            return mp.get_context("fork")
    except Exception:
        pass
    return mp.get_context("spawn")


def build_parser() -> argparse.ArgumentParser:
    defaults = SnvConfig()
    p = argparse.ArgumentParser(description="Run motif regression on SNV(s)")
    group = p.add_mutually_exclusive_group(required=True)
    p.add_argument("--intensities", required=True, help="Path to probe intensity file")
    p.add_argument(
        "--array", dest="array", required=False,
        help=(
            "Path to k-mer array file mapping k-mers to genomic regions. Required "
            "for normal contiguous mode; masked mode builds its probe matcher from "
            "the intensity-region coordinates and reference genome."
        ),
    )
    p.add_argument(
        "--kmerPositions", dest="array", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("--genome", required=True, help="Path to reference genome in FASTA format")
    group.add_argument("--snv-list", help="Comma-separated list of SNVs in chr:pos:ref>alt format")
    group.add_argument(
        "--snv-list-file",
        help="Optional file with SNVs, one per line in chr:pos:ref>alt format",
    )
    p.add_argument(
        "--kmer-size", dest="kmer_size", type=int, default=defaults.kmer_size,
        help="K-mer size for contiguous mode (default: 8). With --mask, mask span is used instead.",
    )
    p.add_argument(
        "--kmer_size", dest="kmer_size", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--mask", type=str, default=defaults.mask,
        help=(
            "Explicit spaced/gapped sequence mask, e.g. 11111000011111. "
            "1 positions are matched; 0 positions are ignored. The SNP is tested "
            "only at informative (1) positions."
        ),
    )
    p.add_argument(
        "--rand-n", type=int, default=defaults.rand_n,
        help="Number of random probes to use for RAND regression (default: 500)",
    )
    p.add_argument(
        "--no-rand", action="store_true", help=argparse.SUPPRESS,
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
            "ALT AFF p-values are corrected across tested positions, and REF+ALT "
            "RAND p-values are corrected together across tested positions. RAND "
            "support then uses Holm p < 0.05. Output pval is Holm-adjusted and "
            "raw_pval is appended."
        ),
    )
    p.add_argument(
        "--diagnostics", action="store_true",
        help=(
            "Append diagnostic columns (motif_pos, wildcard_kmer, n_probes, n_allele) "
            "to the --best-pval output. Requires --best-pval."
        ),
    )
    # Legacy/diagnostic model switches are retained internally for reproducibility
    # but are intentionally hidden from the normal public CLI. OLS + folded lp +
    # covariates remain the supported defaults.
    p.add_argument(
        "--mode", choices=["nb", "ols"], default=defaults.mode, help=argparse.SUPPRESS,
    )
    p.add_argument("--no-covariates", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--raw-lp", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--seed", type=int, default=defaults.seed,
        help="Seed for max-probes subsampling (default: 0); RAND retains deterministic per-SNV sampling.",
    )
    p.add_argument(
        "--max-probes", type=int, default=defaults.max_probes, dest="max_probes",
        help="Subsample to at most this many probes per window before fitting.",
    )
    p.add_argument(
        "--jobs", type=int, default=defaults.jobs,
        help=(
            "Number of SNVs to process in parallel (default: 1). Each worker "
            "uses one BLAS thread by default to avoid oversubscription."
        ),
    )
    p.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(**asdict(defaults))
    return p


def run_snv(config: SnvConfig) -> int:
    """Run snv with validated task options; no command-line parsing."""
    config.validate()
    args = config

    pattern = SequenceMask.from_options(args.kmer_size, args.mask)
    mask_obj = pattern if args.mask is not None else None
    analysis_k = pattern.span

    if args.snv_list_file:
        with open(args.snv_list_file) as f:
            snvs = [line.strip() for line in f if line.strip()]
    else:
        snvs = [s.strip() for s in (args.snv_list or "").split(",") if s.strip()]
    if not snvs:
        raise TaskConfigError("no SNVs supplied")

    intens = IntensityTable.from_file(args.intensities)

    # Warn if intensities appear to be on an intensity_like scale rather than
    # resid_log. resid_log values are centered near 0 with negative values.
    _vals = [float(v) for v in intens.values.values() if np.isfinite(v)]
    if _vals and min(_vals) >= 0:
        print(
            "[WARNING] Intensity values appear to be non-negative (min="
            f"{min(_vals):.3f}). EUbar expects resid_log intensities "
            "(centered near 0, with negative values). Please regenerate the "
            "intensity file using the current default residualized output.",
            file=sys.stderr,
        )

    design = DesignBuilder(intens.values)
    engine = RegressionEngine()
    prebuilt_windows = {}

    if mask_obj is None:
        kmers = KmerIndex.from_file(args.array)
        matcher = MotifMatcher(kmers.kmers)
    else:
        # Validate/fetch the SNV contexts once before scanning the probe universe.
        valid_snvs = []
        for snv_str in snvs:
            try:
                w = SnvWindow.from_snv(
                    snv_str, args.genome, k=analysis_k, debug=args.debug,
                )
            except Exception as exc:
                print(f"[ERROR] {snv_str}: {exc}", file=sys.stderr)
                continue
            valid_snvs.append(snv_str)
            prebuilt_windows[snv_str] = w
        snvs = valid_snvs
        if not snvs:
            return 1

        probe_regions = [
            region for region, value in intens.values.items()
            if value is not None and np.isfinite(value)
        ]
        print(
            f"[MASK] building matcher mask={mask_obj.pattern} "
            f"span={mask_obj.span} informative={mask_obj.n_informative} "
            f"probes={len(probe_regions):,}",
            file=sys.stderr,
        )
        matcher = MaskedMotifMatcher.build(
            mask=mask_obj,
            snv_windows=[prebuilt_windows[s] for s in snvs],
            probe_regions=probe_regions,
            genome_fasta=args.genome,
        )
        st = matcher.stats
        print(
            f"[MASK] indexed {st.n_windows:,} probe windows; "
            f"target signatures={st.n_target_signatures:,}; "
            f"target hits={st.n_target_region_hits:,}; "
            f"RAND signatures={st.n_rand_signatures:,}",
            file=sys.stderr,
        )
        if args.array:
            print(
                "[MASK] note: --array is not used for masked matching; "
                "the matcher is built from intensity-region coordinates + genome.",
                file=sys.stderr,
            )

    rand_sampler = None if args.no_rand else _make_rand_sampler_for_matcher(matcher)
    rand_regressor = None if args.no_rand else RandRegressor(
        design.intensities, engine=engine
    )
    fold_half = not args.raw_lp

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

    def emit_result(snv_str, snv, rows):
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
                k=analysis_k,
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
                print_motif_effect_table(
                    snv_str, snv.chrom, snv.pos, snv.seq, rows
                )

    if args.jobs == 1 or len(snvs) <= 1:
        for snv_str in snvs:
            try:
                snv = prebuilt_windows.get(snv_str)
                if snv is None:
                    snv = SnvWindow.from_snv(
                        snv_str, args.genome, k=analysis_k, debug=args.debug,
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
                k=analysis_k,
                mode=args.mode,
                include_covariates=(not args.no_covariates),
                fold_half=fold_half,
                rand_n=(0 if args.no_rand else args.rand_n),
                max_probes=args.max_probes,
                seed=args.seed,
                rand_sampler=rand_sampler,
                rand_regressor=rand_regressor,
            )
            emit_result(snv_str, snv, rows)
    else:
        # On Linux/WSL, fork lets workers inherit the large read-only matcher
        # copy-on-write. Spawn platforms receive the prebuilt masked matcher as
        # an initializer argument; ordinary contiguous mode keeps its old load-once
        # fallback behavior.
        global _PARENT_SHARED_STATE
        _PARENT_SHARED_STATE = {
            "matcher": matcher,
            "design": design,
        }

        worker_cfg = {
            "analysis_k": analysis_k,
            "mode": args.mode,
            "include_covariates": (not args.no_covariates),
            "fold_half": fold_half,
            "rand_n": (0 if args.no_rand else args.rand_n),
            "max_probes": args.max_probes,
            "seed": args.seed,
            "no_rand": args.no_rand,
            "debug": args.debug,
        }
        n_workers = min(int(args.jobs), len(snvs))
        ctx = _preferred_mp_context()

        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_snv_worker,
                initargs=(
                    args.intensities,
                    args.array,
                    args.genome,
                    worker_cfg,
                    (matcher if mask_obj is not None else None),
                ),
            ) as pool:
                # executor.map preserves input order, so stdout remains deterministic.
                for snv_str, snv, rows, err in pool.map(
                    _analyze_one_snv_worker, snvs, chunksize=1
                ):
                    if err is not None:
                        print(f"[ERROR] {snv_str}: {err}", file=sys.stderr)
                        continue
                    emit_result(snv_str, snv, rows)
        finally:
            _PARENT_SHARED_STATE = None

    return 0



def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = SnvConfig.from_values(vars(args))
        return run_snv(config)
    except TaskConfigError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())