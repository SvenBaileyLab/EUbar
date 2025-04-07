import argparse
import math
from utils import (
    read_intensities,
    read_kmer_positions,
    get_sequence_from_fasta,
    get_kmer_variants,
    match_kmers_to_wildcards,
    select_random_probes,
    project_kmers_to_random_probes,
    regression_driver,
    get_all_kmer_wildcards,
    print_rows_as_tsv
)

def main():
    parser = argparse.ArgumentParser(description="Scan a genomic region for motif effects (eubar_scan.py)")
    parser.add_argument("--intensities", required=True, help="Probe intensity file")
    parser.add_argument("--kmerPositions", required=True, help="K-mer position file")
    parser.add_argument("--genome", required=True, help="FASTA genome file")
    parser.add_argument("--region", required=True, help="Region in format chr:start-end (e.g. chr6:41071098-41071113)")
    parser.add_argument("--kmer_size", type=int, default=8, help="K-mer length")
    parser.add_argument("--dhs", action="store_true", help="Include DHS-related covariates")
    parser.add_argument("--num-random", type=int, default=500, help="Number of random probes")
    parser.add_argument("--use-percentage", action="store_true", help="Use percentage instead of fixed number of random probes")
    parser.add_argument("--random-percent", type=float, default=0.1, help="Percentage of random probes (used if --use-percentage is enabled)")
    parser.add_argument("--mode", type=str, default="neg-binomial", help="Model mode for regression (default: neg-binomial)")

    args = parser.parse_args()

    intensities = read_intensities(args.intensities)
    kmers = read_kmer_positions(args.kmerPositions)
    used_regions = set(kmers.keys())

    if args.use_percentage:
        total_kmer_regions = sum(len(regions) for regions in kmers.values())
        args.num_random = int(args.random_percent * total_kmer_regions)

    chrom, coords = args.region.split(":")
    start, end = map(int, coords.split("-"))
    region_seq = get_sequence_from_fasta(chrom, start, end, args.genome)

    rows = []
    
    for i in range(len(region_seq) - args.kmer_size + 1):
        kmer_seq = region_seq[i:i + args.kmer_size]
        motif_start = start + i

        kmer_list, original_wildcards, snp_indices = get_kmer_variants(kmer_seq, args.kmer_size)
        comp_alleles, matched = match_kmers_to_wildcards(kmers, original_wildcards, snp_indices, kmer_seq)

        rand_seed = hash((args.region, motif_start)) % (2**32)
        rand_probes = select_random_probes(kmers, used_regions, num_random=args.num_random, seed=rand_seed)
        split_random = project_kmers_to_random_probes(rand_probes, original_wildcards, kmer_list, kmers)

        for motif_pos in comp_alleles:
            for snp_index in comp_alleles[motif_pos]:
                ref_allele = comp_alleles[motif_pos][snp_index]

                matched_block = {motif_pos: {snp_index: matched[motif_pos]}}
                comp_block    = {motif_pos: {snp_index: ref_allele}}

                # Always run AFF
                aff_stats = regression_driver(
                    matched_block,
                    probe_intensities=intensities,
                    exclude_ref_allele=False,
                    dhs=args.dhs,
                    mode=args.mode
                )

                # Try RAND, gracefully handle failure
                try:
                    random_block = {motif_pos: {snp_index: split_random[motif_pos]}}
                    rand_stats = regression_driver(
                        random_block,
                        probe_intensities=intensities,
                        exclude_ref_allele=False,
                        dhs=args.dhs,
                        mode=args.mode
                    )
                except KeyError:
                    rand_stats = {}
                    print(
                        f"[Warning] No matching random probes found for window {i}. "
                        f"Introducing NaN values in RAND. "
                        f"Consider increasing `num_random` above {args.num_random} to reduce missing data."
                    )

                # Wildcard and fill variants
                wildcard_variants = get_all_kmer_wildcards(kmer_seq)
                for snp_i, wildcard in wildcard_variants.items():
                    for base in "ACGT":
                        filled = list(wildcard)
                        filled[snp_i] = base
                        filled_kmer = ''.join(filled)

                        # AFF
                        if base in aff_stats:
                            coef, _, pval = aff_stats[base]
                        else:
                            coef, pval = math.nan, math.nan
                        rows.append([wildcard, filled_kmer, i, snp_i, "AFF", base, coef, pval])

                        # RAND
                        if base in rand_stats:
                            coef, _, pval = rand_stats[base]
                        else:
                            coef, pval = math.nan, math.nan
                        rows.append([wildcard, filled_kmer, i, snp_i, "RAND", base, coef, pval])

    print_rows_as_tsv(rows)


if __name__ == "__main__":
    main()
