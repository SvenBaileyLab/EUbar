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
import subprocess
from typing import Callable, Optional


# command -> (module_name, description)
COMMANDS: dict[str, tuple[str, str]] = {
    "array": ("array", "Build k-mer → region index from BED + genome"),
    "intensities": ("intensities", "Compute per-region intensities from signal track(s)"),
    "scan": ("scan", "Scan a genomic region for motif effects via regression"),
    "snv": ("snv", "Run SNV-anchored motif regression across a list of variants"),
    "motifs": ("motifs", "Seed-and-wobble motif discovery with extension"),
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
    for cmd in ("scan", "snv"):
        _, desc = COMMANDS[cmd]
        lines.append(f"    {cmd.ljust(pad)}  {desc}")

    lines.append("  Motifs")
    _, desc = COMMANDS["motifs"]
    lines.append(f"    {'motifs'.ljust(pad)}  {desc}")

    lines.append("")
    lines.append("Examples:")
    lines.append("  eubar array --help")
    lines.append("  eubar intensities --help")
    lines.append("  eubar scan --help")
    lines.append("  eubar snv --help")
    lines.append("  eubar motifs --help")
    lines.append("")
    sys.stderr.write("\n".join(lines) + "\n")


def _get_module_main(module_name: str) -> Optional[Callable[[list[str]], int]]:
    """
    Try to import eubar.<module_name> and return its main(argv) callable if present.
    Expected signature: main(argv: list[str] | None = None) -> int
    """
    mod = importlib.import_module(f"eubar.{module_name}")
    main = getattr(mod, "main", None)
    if callable(main):
        return main
    return None


def _run_module_as_subprocess(module_name: str, argv: list[str]) -> int:
    """
    Fallback if a module doesn't expose main(): run it as `python -m eubar.<module>`.
    """
    cmdline = [sys.executable, "-m", f"eubar.{module_name}", *argv]
    return subprocess.call(cmdline)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    cmd = argv[0]

    if cmd in {"-v", "--version"}:
        sys.stdout.write("eubar (dev)\n")
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
    forwarded = argv[1:] or ["--help"]  # `eubar <cmd>` shows that command's help

    import inspect

    mod_main = _get_module_main(module_name)
    if mod_main is not None:
        try:
            sig = inspect.signature(mod_main)
            if len(sig.parameters) == 0:
                # module parses sys.argv itself
                return int(mod_main() or 0)
            else:
                # module accepts argv (recommended)
                return int(mod_main(forwarded) or 0)
        except (TypeError, ValueError):
            # if signature introspection fails, just try argv then fallback
            try:
                return int(mod_main(forwarded) or 0)
            except TypeError:
                return int(mod_main() or 0)


    # Otherwise fallback to `python -m eubar.<module>`
    return _run_module_as_subprocess(module_name, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
