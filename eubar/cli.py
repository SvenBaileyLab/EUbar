#!/usr/bin/env python3
from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*A NumPy version >=.* and <.* is required for this version of SciPy.*",
    category=UserWarning,
)

import sys
import importlib


# command -> (module_name, description)
COMMANDS: dict[str, tuple[str, str]] = {
    "array": ("array", "Build k-mer to region index from BED + genome"),
    "intensities": (
        "intensities",
        "Compute per-region intensities from signal track(s)",
    ),
    "scan": ("scan", "Scan a genomic region for motif effects via regression"),
    "snv": ("snv", "Run SNV-anchored motif regression across a list of variants"),
    "calibrate": ("calibrate", "Calibrate analysis parameters from stability and representation"),
    "motifs": ("motifs", "Seed-and-wobble motif discovery with extension"),
    "run": ("run", "Run one or more EUbar YAML configuration files"),
    "template": ("template", "Print or copy a commented YAML template"),
}


def _print_help() -> None:
    pad = max(len(k) for k in COMMANDS)

    lines: list[str] = []
    lines.append("eubar: run EUBAR tools as a suite\n")
    lines.append("Usage:")
    lines.append("  eubar <command> [args...]")
    lines.append("  eubar <command>          # shows that command's help")
    lines.append("  eubar list               # list commands")
    lines.append("")

    lines.append("Commands:")
    lines.append("  Preprocessing")
    for cmd in ("array", "intensities"):
        _, desc = COMMANDS[cmd]
        lines.append(f"    {cmd.ljust(pad)}  {desc}")

    lines.append("  Analysis")
    for cmd in ("scan", "snv", "calibrate"):
        _, desc = COMMANDS[cmd]
        lines.append(f"    {cmd.ljust(pad)}  {desc}")

    lines.append("  Motifs")
    _, desc = COMMANDS["motifs"]
    lines.append(f"    {'motifs'.ljust(pad)}  {desc}")

    lines.append("  Workflow")
    for cmd in ("run", "template"):
        _, desc = COMMANDS[cmd]
        lines.append(f"    {cmd.ljust(pad)}  {desc}")

    lines.append("")
    lines.append("Examples:")
    lines.append("  eubar array --help")
    lines.append("  eubar intensities --help")
    lines.append("  eubar scan --help")
    lines.append("  eubar snv --help")
    lines.append("  eubar calibrate --help")
    lines.append("  eubar motifs --help")
    lines.append("  eubar run snv.yaml")
    lines.append("  eubar template snv --output snv.yaml")
    lines.append("")
    sys.stderr.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    cmd = argv[0]

    if cmd in {"-v", "--version"}:
        from eubar import __version__

        sys.stdout.write(f"eubar {__version__}\n")
        return 0

    if cmd == "list":
        for name in COMMANDS:
            sys.stdout.write(name + "\n")
        return 0

    if cmd not in COMMANDS:
        sys.stderr.write(f"[eubar] Unknown command: {cmd}\n\n")
        _print_help()
        return 2

    module_name, _ = COMMANDS[cmd]
    forwarded = argv[1:] or ["--help"]

    # Clean dispatch: every sub-tool exposes main(argv=None) and parses argv explicitly.
    mod = importlib.import_module(f"eubar.{module_name}")
    tool_main = getattr(mod, "main", None)
    if not callable(tool_main):
        sys.stderr.write(f"[eubar] Tool module has no main(): eubar.{module_name}\n")
        return 2

    rc = tool_main(list(forwarded))
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
