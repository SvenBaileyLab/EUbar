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
from dataclasses import asdict, replace
from eubar.task_config import CalibrationConfig
import os
import sys
import time
from typing import Optional, Sequence

import numpy as np

from eubar.core.calibration import (
    parse_caps,
    parse_masks,
    run_mask_calibration as compute_mask_calibration,
    run_max_probes_calibration,
)
from eubar.core.data import IntensityTable, KmerIndex
from eubar.core.calibration_reporting import (
    write_max_probe_report, write_mask_report,
    _write_max_plot as _write_max_plot, _write_mask_plot as _write_mask_plot,
)


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


def build_parser() -> argparse.ArgumentParser:
    defaults = CalibrationConfig()
    p = argparse.ArgumentParser(
        description=(
            "Calibrate EUbar analysis parameters from probe representation and effect stability. "
            "Use 'max-probes' for the production probe cap or 'mask' to compare explicit masks."
        )
    )
    p.add_argument(
        "calibration",
        nargs="?",
        default=defaults.calibration,
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
    p.add_argument("--genome", dest="genome", default=defaults.genome,
                   help="Genome FASTA (required for mask calibration)")
    p.add_argument(
        "--genome-fasta", dest="genome", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("--masks", default=defaults.masks,
                   help="Comma-separated explicit 0/1 masks for mask calibration")
    p.add_argument("--mask-file", default=defaults.mask_file,
                   help="Text file containing one or more candidate 0/1 masks")
    p.add_argument(
        "--control-masks",
        default=defaults.control_masks,
        help=(
            "Comma-separated diagnostic control masks. Controls are fully evaluated and "
            "reported but are excluded from the automatic recommendation."
        ),
    )
    p.add_argument(
        "--control-mask-file",
        default=defaults.control_mask_file,
        help=(
            "Text file containing diagnostic control masks. Controls are evaluated but "
            "excluded from the automatic recommendation."
        ),
    )
    p.add_argument(
        "--random-controls",
        type=int,
        default=defaults.random_controls,
        help=(
            "Mask calibration only: generate this many random diagnostic masks matched "
            "to the recommended candidate's span and number of informative positions. "
            "First/last positions remain informative. Random controls are excluded from "
            "the recommendation (default: 0)."
        ),
    )
    p.add_argument("--stability-probes", type=int, default=defaults.stability_probes,
                   help="Fixed probe count used to compare mask effect stability (default: 100)")
    p.add_argument("--candidate-multiplier", type=int, default=defaults.candidate_multiplier, help=argparse.SUPPRESS)
    p.add_argument("--kmer-size", type=int, default=defaults.kmer_size)
    p.add_argument(
        "--n-families",
        dest="n_families",
        type=int,
        default=defaults.n_families,
        help=(
            "Representative/evaluated families. Defaults: 100 for max-probes; "
            "500 for mask calibration."
        ),
    )
    p.add_argument(
        "--n-kmers", dest="n_families", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    p.add_argument("--repeats", type=int, default=defaults.repeats, help="Subsamples per family and cap")
    p.add_argument("--caps", type=_caps_arg, default=defaults.caps)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--min-probes", type=int, default=defaults.min_probes, help="Minimum usable probes per family")
    # Retained for reproducibility/development, but not part of the normal public interface.
    p.add_argument("--no-covariates", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--raw-lp", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--max-effect-sd", type=float, default=defaults.max_effect_sd, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--max-effect-bias", type=float, default=defaults.max_effect_bias, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--min-eligible-fraction", type=float, default=defaults.min_eligible_fraction, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--borderline-relative-tolerance", type=float, default=defaults.borderline_relative_tolerance, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--near-optimal-relative-tolerance", type=float, default=defaults.near_optimal_relative_tolerance, help=argparse.SUPPRESS
    )
    p.add_argument("--plot", action="store_true", help="Also write a diagnostic PDF")
    p.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    p.set_defaults(**asdict(defaults))
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

    return write_max_probe_report(result, args.out_prefix, plot=args.plot)


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

    result = compute_mask_calibration(
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

    return write_mask_report(result, args.out_prefix, plot=args.plot)


def run_calibration(config: CalibrationConfig) -> int:
    """Load inputs and run the requested calibration."""
    config.validate()
    out_dir = os.path.dirname(os.path.abspath(config.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"[load] intensities: {config.intensities}", file=sys.stderr)
    intens = IntensityTable.from_file(config.intensities)
    if config.calibration == "max-probes":
        return _run_max_probes(config, intens)
    return _run_mask(config, intens)


def run_max_probe_calibration(config: CalibrationConfig) -> int:
    return run_calibration(replace(config, calibration="max-probes"))


def run_mask_calibration(config: CalibrationConfig) -> int:
    return run_calibration(replace(config, calibration="mask"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_calibration(CalibrationConfig.from_values(vars(args)))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
