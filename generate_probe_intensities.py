import argparse
from pybedtools import BedTool
import os


def run_bedtools_map_pybedtools(a_bed, b_bedgraph, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    a = BedTool(a_bed)
    b = BedTool(b_bedgraph)

    # Use bedtools map via pybedtools
    result = a.map(b=b, c=4, o="max")

    # Save the result
    result.saveas(output_path)
    print(f"[Done] Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pybedtools wrapper for bedtools map to extract max ChIP signal in DNase regions."
    )
    parser.add_argument(
        "--a_bed", required=True, help="Path to DNase-seq BED file (-a)"
    )
    parser.add_argument(
        "--b_bedgraph", required=True, help="Path to ChIP-seq BEDGRAPH file (-b)"
    )
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()
    run_bedtools_map_pybedtools(args.a_bed, args.b_bedgraph, args.output)


if __name__ == "__main__":
    main()
