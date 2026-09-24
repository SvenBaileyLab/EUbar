from __future__ import annotations

"""Print or copy bundled EUbar YAML templates."""

import argparse
from pathlib import Path
import shutil
import sys
from typing import Optional, Sequence


TEMPLATE_NAMES = (
    "array",
    "intensities",
    "scan",
    "snv",
    "motifs",
    "calibrate_max_probes",
    "calibrate_mask",
    "pipeline",
)


def _template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Print or copy a commented EUbar YAML template."
    )
    p.add_argument("template", nargs="?", choices=TEMPLATE_NAMES)
    p.add_argument(
        "--output", "-o", default=None,
        help="Write template to this file instead of stdout",
    )
    p.add_argument("--list", action="store_true", help="List available templates")
    args = p.parse_args(argv)

    if args.list or args.template is None:
        for name in TEMPLATE_NAMES:
            print(name)
        return 0

    src = _template_dir() / f"{args.template}.yaml"
    if not src.exists():
        print(f"[eubar template] missing bundled template: {src}", file=sys.stderr)
        return 2

    if args.output:
        dst = Path(args.output).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"[eubar template] wrote: {dst}", file=sys.stderr)
    else:
        sys.stdout.write(src.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
