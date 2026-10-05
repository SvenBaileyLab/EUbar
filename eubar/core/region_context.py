"""Per-region sequence context used as an AFF covariate.

CpG observed/expected (o/e) of each accessible region:

    cpg_oe = n_CG * length / (n_C * n_G)        (0 when the region has no C or no G)

CpG-island promoters give high ChIP signal for most factors. When the REF and ALT
8-mers of a variant occur in regions with different CpG density (common for variants
that create or destroy a CpG), the AFF REF-vs-ALT contrast can pick up that regional
difference instead of an allele effect. Adding cpg_oe as a covariate compares REF and
ALT probes within similar CpG context. GC content is not added here: it is already
removed from the intensities by `eubar intensities` (resid_log).

Values are computed once from the genome and cached as a two-column TSV.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


def _parse_region(region: str):
    chrom, coords = region.rsplit(":", 1)
    start_s, end_s = coords.split("-")
    return chrom, int(start_s), int(end_s)


def cpg_oe(seq: str) -> float:
    s = seq.upper()
    n_c, n_g = s.count("C"), s.count("G")
    if n_c == 0 or n_g == 0:
        return 0.0
    return s.count("CG") * len(s) / (n_c * n_g)


def load_region_cpg_oe(
    regions: Iterable[str],
    genome_fasta: str,
    cache_path: Optional[str] = None,
) -> Dict[str, float]:
    """Return {region: cpg_oe} for every region, using/refreshing a TSV cache.

    Region strings are "chrom:start-end" with BED-style coordinates (0-based start,
    end exclusive), as in EUbar intensity files.
    """
    regions = list(dict.fromkeys(str(r) for r in regions))
    out: Dict[str, float] = {}
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.is_file():
        with open(cache) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    try:
                        out[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
    missing = [r for r in regions if r not in out]
    if missing:
        from .sequence import _open_fasta

        fa = _open_fasta(genome_fasta)
        n_bad = 0
        for r in missing:
            try:
                chrom, start, end = _parse_region(r)
                out[r] = cpg_oe(fa[chrom][start:end].seq)
            except Exception:
                out[r] = math.nan
                n_bad += 1
        if n_bad:
            print(f"[CpG] {n_bad:,} regions could not be read from the genome; "
                  "they get the median cpg_oe in each window", file=sys.stderr)
        if cache is not None:
            # write to a temporary file and rename, so parallel workers never read a half-written cache
            tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "w") as fh:
                    for r, v in out.items():
                        fh.write(f"{r}\t{v}\n")
                os.replace(tmp, cache)
            except OSError as exc:
                print(f"[CpG] could not write cache {cache}: {exc}", file=sys.stderr)
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return {r: out[r] for r in regions}


def default_cache_path(intensities_path: str) -> str:
    return f"{intensities_path}.cpg_oe.tsv"
