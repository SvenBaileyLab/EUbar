import argparse
from utils import (
    read_intensities,
    scan_motif_kmers,
    read_unique_kmer_positions,
    get_sequence_from_fasta,
    print_rows_as_tsv,
    plot_aff_motif_effects,
    run_snv_regression,
    reverse_complement,
)


def main():
    parser = argparse.ArgumentParser(description="Run motif regression on a region")
    parser.add_argument(
        "--intensities", required=True, help="Path to probe intensity file"
    )
    parser.add_argument(
        "--kmerPositions", required=True, help="Path to k-mer position file"
    )
    parser.add_argument("--genome", required=True, help="FASTA genome file")
    parser.add_argument(
        "--region", required=True, help="Genomic region, e.g. chr6:1295220-1295228"
    )
    parser.add_argument(
        "--kmer_size", type=int, default=8, help="K-mer size (default: 8)"
    )
    parser.add_argument(
        "--mode",
        choices=["nb", "ols"],
        default="nb",
        help="Regression mode (nb or ols)",
    )
    parser.add_argument(
        "--no-covariates", action="store_true", help="Disable lp and sl covariates"
    )
    parser.add_argument(
        "--save-figure", type=str, help="Filename to save figure (e.g. motif_plot.png)"
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Use reverse complement of the sequence"
    )
    args = parser.parse_args()

    intensities = read_intensities(args.intensities)
    kmers = read_unique_kmer_positions(args.kmerPositions)

    chrom, coords = args.region.split(":")
    start, end = map(int, coords.split("-"))
    region_seq = get_sequence_from_fasta(chrom, start, end, args.genome)
    if args.reverse:
        region_seq = reverse_complement(region_seq)

    allele_region_offsets, allele_matched_kmers = scan_motif_kmers(
        region_seq=region_seq,
        kmer_size=args.kmer_size,
        kmer_positions=kmers,
    )

    all_rows = []

    for motif_pos in sorted(allele_region_offsets):
        for snv_index in sorted(allele_region_offsets[motif_pos]):
            rows = run_snv_regression(
                motif_pos,
                snv_index,
                allele_region_offsets,
                allele_matched_kmers,
                intensities,
                model_type=args.mode,
                include_covariates=not args.no_covariates,
                region_seq=region_seq,
            )
            all_rows.extend(rows)

    print_rows_as_tsv(all_rows)

    if args.save_figure:
        plot_aff_motif_effects(all_rows, save_path=args.save_figure)


if __name__ == "__main__":
    main()
