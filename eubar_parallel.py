import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice

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

# Shared globals inside workers
global_kmers = None
global_intensities = None

# Worker initializer
def init_worker(kmers, intensities):
    global global_kmers, global_intensities
    global_kmers = kmers
    global_intensities = intensities

# Helper to split SNVs into chunks
def chunked_iterable(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    iterable = iter(iterable)
    return iter(lambda: list(islice(iterable, n)), [])

# Worker job: process a batch of SNVs
def process_snv_batch(snv_batch, genome_path, kmer_size, mode, no_covariates, rand_n):
    global global_kmers, global_intensities
    results = []
    for snv_str in snv_batch:
        try:
            chrom, snv_pos, ref, alt = parse_snv_string(snv_str)
            snv_info = normalize_snv_region(chrom, snv_pos, ref, alt, genome_path, kmer_size)
            snv_info["snv_str"] = snv_str
            snv_info["wildcards"] = dict(get_snv_aligned_wildcards(snv_info, kmer_size))

            allele_region_offsets, wildcards, matched_regions = match_snv_aligned_kmers(
                snv_info, global_kmers, kmer_size
            )

            results_aff = run_per_motif_regression(
                allele_region_offsets,
                global_intensities,
                snv_info,
                model_type=mode,
                include_covariates=not no_covariates,
            )

            rand_regions_per_allele, _ = sample_rand_kmers_per_allele(
                global_kmers,
                matched_regions,
                snv_str=snv_info["snv_str"],
                kmer_size=kmer_size,
                rand_n=rand_n,
            )

            results_rand = run_rand_regression_from_region_map(
                rand_regions_per_allele=rand_regions_per_allele,
                probe_intensities=global_intensities,
                snv_str=snv_info["snv_str"],
                model_type=mode,
            )

            results.append((snv_info, results_aff + results_rand))
        except Exception as e:
            print(f"Error processing SNV {snv_str}: {e}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run motif regression on SNV(s)")
    parser.add_argument("--intensities", required=True, help="Path to probe intensity file")
    parser.add_argument("--kmerPositions", required=True, help="Path to k-mer array file mapping kmers to genomic regions")
    parser.add_argument("--genome", required=True, help="Path to reference genome in FASTA format")
    parser.add_argument("--snv-list", required=False, help="Comma-separated list of SNVs in chr:pos:ref>alt format")
    parser.add_argument("--snv-list-file", help="Optional file with SNVs, one per line in chr:pos:ref>alt format")
    parser.add_argument("--kmer_size", type=int, default=8, help="Size of the k-mers to analyze (default: 8)")
    parser.add_argument("--mode", choices=["nb", "ols"], default="nb", help="Regression type")
    parser.add_argument("--no-covariates", action="store_true", help="Disable covariates in regression")
    parser.add_argument("--rand_n", type=int, default=5000, help="Random k-mers for control regression")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    # Load once and share
    intensities = read_intensities(args.intensities)
    kmers = read_unique_kmer_positions(args.kmerPositions)

    if args.snv_list_file:
        with open(args.snv_list_file) as f:
            snvs = [line.strip() for line in f if line.strip()]
    else:
        snvs = args.snv_list.split(",")

    print(f"Loaded {len(snvs)} SNVs. Processing with {args.threads} threads...")

    # Divide SNVs into chunks per worker
    batch_size = len(snvs) // args.threads + 1
    snv_batches = chunked_iterable(snvs, batch_size)

    # Start parallel processing
    with ProcessPoolExecutor(
        max_workers=args.threads,
        initializer=init_worker,
        initargs=(kmers, intensities)
    ) as executor:
        futures = [
            executor.submit(
                process_snv_batch,
                batch,
                args.genome,
                args.kmer_size,
                args.mode,
                args.no_covariates,
                args.rand_n
            )
            for batch in snv_batches
        ]

        for future in as_completed(futures):
            try:
                batch_results = future.result()
                for snv_info, results in batch_results:
                    print_motif_effect_table(
                        snv_str=snv_info["snv_str"],
                        chrom=snv_info["chrom"],
                        pos=snv_info["pos"],
                        region_seq=snv_info["region_seq"],
                        results=results,
                    )
            except Exception as e:
                print(f"Batch processing failed: {e}")

if __name__ == "__main__":
    main()
