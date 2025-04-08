import argparse
import math
from utils import (
    read_intensities,
    read_kmer_positions,
    get_sequence_from_fasta,
    match_all_kmers_to_wildcards,
    regression_driver,
    get_all_kmer_wildcards,
    print_rows_as_tsv,
    plot_aff_motif_effects,
)


def main():
    parser = argparse.ArgumentParser(
        description="Scan a genomic region for motif effects (eubar_scan.py)"
    )
    parser.add_argument("--intensities", required=True, help="Probe intensity file")
    parser.add_argument("--kmerPositions", required=True, help="K-mer position file")
    parser.add_argument("--genome", required=True, help="FASTA genome file")
    parser.add_argument(
        "--region",
        required=True,
        help="Region in format chr:start-end (e.g. chr6:41071098-41071113)",
    )
    parser.add_argument("--kmer_size", type=int, default=8, help="K-mer length")
    parser.add_argument(
        "--dhs", action="store_true", help="Include DHS-related covariates"
    )
    # parser.add_argument("--num-random", type=int, default=500, help="Number of random probes")
    # parser.add_argument("--use-percentage", action="store_true", help="Use percentage instead of fixed number of random probes")
    # parser.add_argument("--random-percent", type=float, default=0.1, help="Percentage of random probes (used if --use-percentage is enabled)")
    parser.add_argument(
        "--mode",
        type=str,
        default="neg-binomial",
        help="Model mode for regression (default: neg-binomial)",
    )
    parser.add_argument(
        "--save-figure", type=str, help="Filename to save figure (e.g. motif_plot.png)"
    )

    args = parser.parse_args()

    intensities = read_intensities(args.intensities)
    kmers = read_kmer_positions(args.kmerPositions)
    used_regions = set(kmers.keys())

    # if args.use_percentage:
    # total_kmer_regions = sum(len(regions) for regions in kmers.values())
    # args.num_random = int(args.random_percent * total_kmer_regions)

    chrom, coords = args.region.split(":")
    start, end = map(int, coords.split("-"))
    region_seq = get_sequence_from_fasta(chrom, start, end, args.genome)

    rows = []

    for i in range(len(region_seq) - args.kmer_size + 1):
        kmer_seq = region_seq[i : i + args.kmer_size]
        motif_start = start + i
        # print(f"\n[Window {i}] kmer_seq = {kmer_seq}")

        # === Step 2: Generate all possible wildcards at every position ===
        wildcard_variants = get_all_kmer_wildcards(kmer_seq)
        # print("  Wildcards:")
        # for pos, wildcard in wildcard_variants.items():
        # print(f"    Position {pos}: {wildcard}")

        # === Step 3: Match wildcards to genome-wide k-mers ===
        matched = match_all_kmers_to_wildcards(kmers, wildcard_variants)

        # print("  Matched wildcard positions:")
        for motif_pos in matched:
            for allele in matched[motif_pos]:
                count = len(matched[motif_pos][allele])
                # print(f"    Position {motif_pos}, Allele {allele}: {count} matches")

        # === Step 4: Run regression for each wildcard position and allele ===
        for motif_pos in matched:
            for allele in matched[motif_pos]:
                matched_block = {
                    motif_pos: {motif_pos: {allele: matched[motif_pos][allele]}}
                }

                try:
                    aff_stats = regression_driver(
                        matched_block,
                        probe_intensities=intensities,
                        exclude_ref_allele=False,
                        dhs=args.dhs,
                        mode=args.mode,
                    )
                except Exception as e:
                    print(
                        f"[Warning] Regression failed for Window {i}, Pos {motif_pos}, Allele {allele}: {e}"
                    )
                    aff_stats = {allele: (math.nan, None, math.nan)}

                coef, _, pval = aff_stats.get(allele, (math.nan, None, math.nan))
                absolute_pos = i + motif_pos

                wildcard_kmer = wildcard_variants[motif_pos]
                filled_kmer = list(wildcard_kmer)
                filled_kmer[motif_pos] = allele
                filled_kmer = "".join(filled_kmer)

                rows.append(
                    [
                        wildcard_kmer,
                        filled_kmer,
                        i,
                        motif_pos,
                        "AFF",
                        allele,
                        coef,
                        pval,
                        absolute_pos,
                    ]
                )

    print_rows_as_tsv(rows)

    if args.save_figure:
        plot_aff_motif_effects(rows, save_path=args.save_figure)


if __name__ == "__main__":
    main()
