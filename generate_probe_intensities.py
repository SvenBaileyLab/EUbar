import argparse
from pybedtools import BedTool
import os
import tempfile
import pyBigWig


def convert_bigwig_to_bedgraph_py(bigwig_path):
    temp_bedgraph = tempfile.NamedTemporaryFile(delete=False, suffix=".bedgraph")
    with pyBigWig.open(bigwig_path) as bw, open(temp_bedgraph.name, "w") as out:
        for chrom in bw.chroms().keys():
            intervals = bw.intervals(chrom)
            if intervals:
                for start, end, value in intervals:
                    out.write(f"{chrom}\t{start}\t{end}\t{value}\n")
    print(f"[Info] Converted BigWig to BedGraph using pyBigWig: {temp_bedgraph.name}")
    return temp_bedgraph.name


def deduplicate_bed_by_coords(bed_path):
    temp_bed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".bed").name
    seen = set()
    with open(bed_path) as infile, open(temp_bed_path, "w") as outfile:
        for line in infile:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            key = tuple(fields[:3])  # chrom, start, end
            if key not in seen:
                seen.add(key)
                outfile.write(line)
    print(f"[Info] Deduplicated BED written to temporary file: {temp_bed_path}")
    return temp_bed_path


def run_bedtools_map_pybedtools(a_bed, b_bedgraph, output_path, keep_temp=False):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


    # Deduplicate BED file by first 3 columns
    dedup_a_bed = deduplicate_bed_by_coords(a_bed)

    # Handle BigWig input
    temp_bedgraph_path = None
    if b_bedgraph.endswith((".bw", ".bigwig", ".bigWig", ".BW")):
        temp_bedgraph_path = convert_bigwig_to_bedgraph_py(b_bedgraph)
        b_bedgraph = temp_bedgraph_path

    a = BedTool(dedup_a_bed).sort()
    b = BedTool(b_bedgraph)

    # Map and trim output to: chrom, start, end, mapped max value
    result = a.map(b=b, c=4, o="max")

    with open(output_path, "w") as out:
        for interval in result:
            fields = interval.fields
            chrom, start, end = fields[0], fields[1], fields[2]
            max_signal = fields[-1] if fields[-1] != "." else "0"
            out.write(f"{chrom}:{start}-{end}\t{max_signal}\n")
    print(f"[Done] Output written to: {output_path}")

    # Cleanup
    if temp_bedgraph_path and not keep_temp:
        os.remove(temp_bedgraph_path)
        print(f"[Cleanup] Removed temporary BedGraph file: {temp_bedgraph_path}")

    if not keep_temp:
        os.remove(dedup_a_bed)
        print(f"[Cleanup] Removed temporary deduplicated BED: {dedup_a_bed}")


def main():
    parser = argparse.ArgumentParser(
        description="Pybedtools wrapper for bedtools map to extract max ChIP signal in DNase regions."
    )
    parser.add_argument("--bed", required=True, help="Path to DNase-seq BED file (-a)")
    parser.add_argument("--bedgraph", required=True,
                        help="Path to ChIP-seq BEDGRAPH or BigWig file (-b)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary files")

    args = parser.parse_args()
    run_bedtools_map_pybedtools(args.bed, args.bedgraph, args.output, keep_temp=args.keep_temp)


if __name__ == "__main__":
    main()
