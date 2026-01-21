import argparse
from pybedtools import BedTool
import os
import tempfile
import pyBigWig
import math


BIGWIG_EXTS = (".bw", ".bigwig", ".bigWig", ".BW")


def deduplicate_bed_by_coords(bed_path):
    temp_bed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".bed").name
    seen = set()
    with open(bed_path) as infile, open(temp_bed_path, "w") as outfile:
        for line in infile:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            key = tuple(fields[:3])  # chrom, start, end
            if key not in seen:
                seen.add(key)
                outfile.write(line)
    print(f"[Info] Deduplicated BED written to temporary file: {temp_bed_path}")
    return temp_bed_path


def is_bigwig(path: str) -> bool:
    return path.endswith(BIGWIG_EXTS)


def safe_max_stat(bw, chrom, start, end):
    """
    Returns max signal in [start, end) from BigWig.
    If missing/None/NaN, returns 0.0.
    """
    try:
        # pyBigWig uses 0-based half-open coordinates like BED
        v = bw.stats(chrom, start, end, type="max")
        if not v:
            return 0.0
        val = v[0]
        if val is None:
            return 0.0
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return 0.0
        return float(val)
    except RuntimeError:
        # e.g., chrom not in bigwig
        return 0.0


def run_bigwig_max_per_interval(a_bed, bigwig_path, output_path, genome=None, keep_temp=False):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dedup_a_bed = deduplicate_bed_by_coords(a_bed)

    # Sort A for consistent output order (not required for BigWig querying)
    a = BedTool(dedup_a_bed)
    a = a.sort(g=genome) if genome else a.sort()

    with pyBigWig.open(bigwig_path) as bw, open(output_path, "w") as out:
        bw_chroms = bw.chroms()  # dict of chrom -> length

        for interval in a:
            chrom = interval.chrom
            start = int(interval.start)
            end = int(interval.end)

            # fast check: if chrom missing, signal is 0
            if chrom not in bw_chroms:
                max_signal = 0.0
            else:
                # clamp to chrom length just in case
                clen = int(bw_chroms[chrom])
                s = max(0, min(start, clen))
                e = max(0, min(end, clen))
                if e <= s:
                    max_signal = 0.0
                else:
                    max_signal = safe_max_stat(bw, chrom, s, e)

            out.write(f"{chrom}:{start}-{end}\t{max_signal}\n")

    print(f"[Done] Output written to: {output_path}")

    if not keep_temp:
        os.remove(dedup_a_bed)
        print(f"[Cleanup] Removed temporary deduplicated BED: {dedup_a_bed}")


def run_bedtools_map_pybedtools(a_bed, b_bedgraph, output_path, genome=None, keep_temp=False):
    """
    BedGraph path (non-BigWig). We sort BOTH A and B to avoid bedtools map sort errors.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dedup_a_bed = deduplicate_bed_by_coords(a_bed)

    a = BedTool(dedup_a_bed)
    b = BedTool(b_bedgraph)

    a = a.sort(g=genome) if genome else a.sort()
    b = b.sort(g=genome) if genome else b.sort()

    result = a.map(b=b, c=4, o="max")

    with open(output_path, "w") as out:
        for interval in result:
            fields = interval.fields
            chrom, start, end = fields[0], fields[1], fields[2]
            max_signal = fields[-1] if fields[-1] != "." else "0"
            out.write(f"{chrom}:{start}-{end}\t{max_signal}\n")

    print(f"[Done] Output written to: {output_path}")

    if not keep_temp:
        os.remove(dedup_a_bed)
        print(f"[Cleanup] Removed temporary deduplicated BED: {dedup_a_bed}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract max ChIP signal over DNase/ATAC regions from either BigWig or BedGraph."
    )
    parser.add_argument("--bed", required=True, help="Path to DNase-seq/ATAC-seq BED file (-a)")
    parser.add_argument("--signal", required=True,
                        help="Path to ChIP-seq BedGraph OR BigWig file (-b)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--genome", default=None,
                        help="Optional genome chrom sizes file for bedtools sort (-g)")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")

    args = parser.parse_args(argv)

    if is_bigwig(args.signal):
        run_bigwig_max_per_interval(
            args.bed, args.signal, args.output,
            genome=args.genome, keep_temp=args.keep_temp
        )
    else:
        run_bedtools_map_pybedtools(
            args.bed, args.signal, args.output,
            genome=args.genome, keep_temp=args.keep_temp
        )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
