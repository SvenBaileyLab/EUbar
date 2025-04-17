import argparse

from utils import (
    read_intensities,
    read_unique_kmer_positions,
    parse_snv_string,
    normalize_snv_region,
    get_snv_aligned_wildcards,
    match_snv_aligned_kmers,
    run_per_motif_regression,
    sample_rand_kmers_per_allele,
    run_rand_regression_from_region_map,
    print_motif_effect_table
)
    
def main():
    parser = argparse.ArgumentParser(description="Run motif regression on SNV(s)")
    parser.add_argument("--intensities", required=True)
    parser.add_argument("--kmerPositions", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--snv-list", required=True)
    parser.add_argument("--kmer_size", type=int, default=8)
    parser.add_argument("--mode", choices=["nb", "ols"], default="nb")
    parser.add_argument("--no-covariates", action="store_true")
    parser.add_argument("--rand_n", type=int, default=5000)

    args = parser.parse_args()

    intensities = read_intensities(args.intensities)
    kmers = read_unique_kmer_positions(args.kmerPositions)

    for snv_str in args.snv_list.split(","):
        chrom, snv_pos, ref, alt = parse_snv_string(snv_str)
        snv_info = normalize_snv_region(chrom, snv_pos, ref, alt, args.genome, args.kmer_size)
        snv_info["snv_str"] = snv_str
        snv_info["wildcards"] = dict(get_snv_aligned_wildcards(snv_info, args.kmer_size))

        allele_region_offsets, matched_regions = match_snv_aligned_kmers(snv_info, kmers, args.kmer_size)

        results_aff = run_per_motif_regression(
            allele_region_offsets,
            intensities,
            snv_info,
            model_type=args.mode,
            include_covariates=not args.no_covariates,
        )

        rand_regions_per_allele, _ = sample_rand_kmers_per_allele(
            kmers,
            matched_regions,
            snv_str=snv_info["snv_str"],
            kmer_size=args.kmer_size,
            rand_n=args.rand_n,
        )

        results_rand = run_rand_regression_from_region_map(
            rand_regions_per_allele=rand_regions_per_allele,
            probe_intensities=intensities,
            snv_str=snv_info["snv_str"],
            model_type=args.mode,
        )

        print_motif_effect_table(
            snv_str=snv_info["snv_str"],
            chrom=snv_info["chrom"],
            pos=snv_info["pos"],
            region_seq=snv_info["region_seq"],
            results=results_aff + results_rand,
        )

if __name__ == "__main__":
    main()
