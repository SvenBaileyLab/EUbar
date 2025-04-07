#!/usr/bin/env python3

import argparse
from utils import (
        read_intensities,
        read_kmer_positions, 
        get_mutation_sequence,
        analyze_motif_effects, 
        print_motif_effect_table,
        determine_num_random
    )

def main():
    parser = argparse.ArgumentParser(description="INVPBM motif walker (Python version)")
    parser.add_argument("--intensities", required=True, help="Probe intensity file")
    parser.add_argument("--kmerPositions", required=True, help="K-mer position file")
    parser.add_argument("--genome", required=True, help="FASTA genome file")
    parser.add_argument("--snv-list", required=True, help="Comma-separated list of SNVs (e.g. chr6:41071106:C>T)")
    parser.add_argument("--kmer_size", type=int, default=8, help="K-mer length")
    parser.add_argument("--dhs", action="store_true", help="Include DHS-related covariates")
    parser.add_argument("--num-random", type=int, default=500, help="Number of random probes")
    parser.add_argument("--use-percentage", action="store_true", help="Use percentage of total probes instead of fixed number")
    parser.add_argument("--percentage", type=float, default=0.10, help="Percentage of probes to use when --use-percentage is set (0.10 = 10%)")


    args = parser.parse_args()

    # Load data
    intensities = read_intensities(args.intensities)
    kmers = read_kmer_positions(args.kmerPositions)
    
    # Get the first k-mer from the kmers dictionary
    any_kmer = next(iter(kmers))
    actual_kmer_len = len(any_kmer)

    if actual_kmer_len != args.kmer_size:
        raise ValueError(
            f"K-mer size mismatch: --kmer_size = {args.kmer_size}, but kmerPositions file contains {actual_kmer_len}-mers."
        )

    genome = args.genome

    # Used regions for random probe exclusion
    used_regions = set(kmers.keys())

    num_random = determine_num_random(
        kmers,
        use_percentage=args.use_percentage,
        percentage=args.percentage,
        default=args.num_random
    )

    for snv in args.snv_list.split(","):

        chrom, pos, refalt = snv.split(":")
        ref, alt = refalt.split(">")
        pos = int(pos)

        motif = get_mutation_sequence(chrom, pos, ref, genome, args.kmer_size)

        results = analyze_motif_effects(
            snv=snv,
            kmers=kmers,
            kmer_size=args.kmer_size,
            used_regions=used_regions,
            intensities=intensities,
            genome=genome,
            num_random=num_random,
            dhs=args.dhs
        )

        print_motif_effect_table(
            snv=snv,
            chrom=chrom,
            pos=pos,
            motif=motif,
            results=results
        )

if __name__ == "__main__":
    main()

