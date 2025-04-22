import argparse
import re
import sys

from utils import (
    read_intensities,
    read_unique_kmer_positions,
    parse_snv_string,
    normalize_snv_region,
    get_snv_aligned_wildcards,
    # match_snv_aligned_kmers,
    run_per_motif_regression,
    sample_rand_kmers_per_allele,
    run_rand_regression_from_region_map,
    print_motif_effect_table
)

def wildcard_match(kmer, wildcard):
    """
    Returns True if the kmer matches the wildcard pattern (dot as any base).
    Much faster than regex.
    """
    for k, w in zip(kmer, wildcard):
        if w != '.' and k != w:
            return False
    return True

def match_snv_aligned_kmers(snv_info, kmer_positions, kmer_size):
    """
    Optimized version of match_snv_aligned_kmers using fast character-based matching.
    """
    allele_region_offsets = {}
    matched_regions = set()
    wildcards = get_snv_aligned_wildcards(snv_info, kmer_size)

    for motif_pos, wildcard in wildcards:
        snv_index = wildcard.index(".")
        region_to_alleles = {}
        region_kmer_hits = {}

        for kmer, region_dict in kmer_positions.items():
            if len(kmer) != kmer_size:
                continue  # skip invalid kmer lengths

            if wildcard_match(kmer, wildcard):
                allele = kmer[snv_index]
                for region_id, offset in region_dict.items():
                    region_to_alleles.setdefault(region_id, set()).add(allele)
                    region_kmer_hits.setdefault(region_id, {})[allele] = (offset, kmer)
                    matched_regions.add(region_id)

        for region_id, alleles in region_to_alleles.items():
            if len(alleles) == 1:
                allele = next(iter(alleles))
                offset, _ = region_kmer_hits[region_id][allele]
                allele_region_offsets.setdefault(motif_pos, {}).setdefault(allele, {})[region_id] = offset

    return allele_region_offsets, wildcards, matched_regions

    
def main():
    parser = argparse.ArgumentParser(description="Run motif regression on SNV(s)")
    parser.add_argument("--intensities", required=True, help="Path to probe intensity file")
    parser.add_argument("--kmerPositions", required=True, help="Path to k-mer array file mapping kmers to genomic regions")
    parser.add_argument("--genome", required=True, help="Path to reference genome in FASTA format")
    parser.add_argument("--snv-list", required=False, help="Comma-separated list of SNVs in chr:pos:ref>alt format")
    parser.add_argument("--snv-list-file", help="Optional file with SNVs, one per line in chr:pos:ref>alt format")
    parser.add_argument("--kmer_size", type=int, default=8, help="Size of the k-mers to analyze (default: 8)")
    parser.add_argument("--mode", choices=["nb", "ols"], default="nb", help="Regression type: negative binomial ('nb') or ordinary least squares ('ols')")
    parser.add_argument("--no-covariates", action="store_true", help="Disable use of location and length covariates in the regression model")
    parser.add_argument("--rand_n", type=int, default=5000, help="Number of random probes to use for RAND regression (default: 5000)")

    args = parser.parse_args()

    intensities = read_intensities(args.intensities)
    kmers = read_unique_kmer_positions(args.kmerPositions)

    if args.snv_list_file:
        with open(args.snv_list_file) as f:
            snvs = [line.strip() for line in f if line.strip()]
    else:
        snvs = args.snv_list.split(",")

    for snv_str in snvs:
        try:
            chrom, snv_pos, ref, alt = parse_snv_string(snv_str)
            snv_info = normalize_snv_region(chrom, snv_pos, ref, alt, args.genome, args.kmer_size)
            snv_info["snv_str"] = snv_str
            snv_info["wildcards"] = dict(get_snv_aligned_wildcards(snv_info, args.kmer_size))

            allele_region_offsets, wildcards, matched_regions = match_snv_aligned_kmers(snv_info, kmers, args.kmer_size)

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

        except ValueError as e:
            print(f"[WARNING] Skipping {snv_str}: {e}", file=sys.stderr)
            continue

if __name__ == "__main__":
    main()
