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
    print_motif_effect_table,
    reverse_complement,
)


def wildcard_match(kmer, wildcard):
    """
    Returns True if the kmer matches the wildcard pattern (dot as any base).
    Much faster than regex.
    """
    for k, w in zip(kmer, wildcard):
        if w != "." and k != w:
            return False
    return True


def match_snv_aligned_kmers(snv_info, kmer_positions, kmer_size, include_revcomp=True):
    from collections import defaultdict

    def get_matches(wildcard, strand):
        region_to_alleles = defaultdict(set)
        region_kmer_hits = {}

        for kmer in kmer_positions:
            if len(kmer) != kmer_size:
                continue
            test_kmer = kmer if strand == "forward" else reverse_complement(kmer)
            if wildcard_match(test_kmer, wildcard):
                allele = test_kmer[wildcard.index(".")]
                for region_id, offset in kmer_positions[kmer].items():
                    region_to_alleles[region_id].add(allele)
                    region_kmer_hits.setdefault(region_id, {})[allele] = (offset, kmer)

        return region_to_alleles, region_kmer_hits

    allele_region_offsets = {}
    matched_regions = set()
    wildcards = get_snv_aligned_wildcards(snv_info, kmer_size)

    for motif_pos, wildcard in wildcards:
        fw_to_alleles, fw_hits = get_matches(wildcard, strand="forward")
        rc_to_alleles, rc_hits = (
            get_matches(wildcard, strand="reverse") if include_revcomp else ({}, {})
        )

        combined = defaultdict(set)
        for region, alleles in fw_to_alleles.items():
            combined[region].update(alleles)
        for region, alleles in rc_to_alleles.items():
            combined[region].update(alleles)

        # Now handle unambiguous
        for region, alleles in combined.items():
            if len(alleles) == 1:
                allele = next(iter(alleles))
                if region in fw_hits and allele in fw_hits[region]:
                    offset, _ = fw_hits[region][allele]
                elif region in rc_hits and allele in rc_hits[region]:
                    offset, _ = rc_hits[region][allele]
                else:
                    continue  # shouldn't happen, but safety

                allele_region_offsets.setdefault(motif_pos, {}).setdefault(allele, {})[
                    region
                ] = offset
                matched_regions.add(region)

    return allele_region_offsets, wildcards, matched_regions


def run_aff_regression(
    motif_pos,
    snv_index,
    allele_region_offsets,
    allele_matched_kmers,
    intensities,
    region_seq=None,
    model_type="nb",
    include_covariates=True,
    return_matrix=False,
):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import math
    from utils import extract_covariates

    rows = []
    ref_allele = region_seq[motif_pos + snv_index]
    alleles = list(allele_region_offsets.get(motif_pos, {}).keys())
    alt_alleles = [a for a in alleles if a != ref_allele]

    all_regions = set()
    for a in alleles:
        all_regions.update(allele_region_offsets[motif_pos][a].keys())

    design = []
    y_values = []
    region_list = []

    for region in all_regions:
        row = {a: 0 for a in alt_alleles}
        # Count how many alleles (alt + ref) this region maps to
        present_alleles = [
            a for a in alleles if region in allele_region_offsets[motif_pos][a]
        ]
        if len(present_alleles) == 1 and region in intensities:
            only_allele = present_alleles[0]
            if only_allele in alt_alleles:
                row[only_allele] = 1
            # Else, it's the ref allele → all zeros
            design.append(row)
            y_values.append(intensities[region])
            region_list.append(region)

    if not design:
        return []

    X = pd.DataFrame(design, index=region_list)
    y = pd.Series(y_values, index=region_list)

    if include_covariates:
        kmer_pos = {
            region: allele_region_offsets[motif_pos][a][region]
            for a in alleles
            for region in allele_region_offsets[motif_pos][a]
        }
        lp, sl = extract_covariates(region_list, kmer_pos)
        X["lp"] = lp
        X["sl"] = sl

    X_const = sm.add_constant(X)
    model = (
        sm.GLM(y, X_const, family=sm.families.NegativeBinomial())
        if model_type == "nb"
        else sm.OLS(np.log1p(y), X_const)
    )
    results = model.fit()

    seen = set()
    for a in alt_alleles:
        coef = results.params.get(a, math.nan)
        pval = results.pvalues.get(a, math.nan)
        for wildcard_kmer, filled_kmer, _ in allele_matched_kmers[motif_pos][
            snv_index
        ].get(a, []):
            key = (wildcard_kmer, filled_kmer)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "wildcard_kmer": wildcard_kmer,
                    "filled_kmer": filled_kmer,
                    "motif_pos": motif_pos,
                    "snp_index": snv_index,
                    "type": "AFF",
                    "allele": a,
                    "coef": coef,
                    "pval": pval,
                    "absolute_pos": motif_pos + snv_index,
                }
            )

    # Add ref allele rows
    for wildcard_kmer, filled_kmer, _ in allele_matched_kmers[motif_pos][snv_index].get(
        ref_allele, []
    ):
        key = (wildcard_kmer, filled_kmer)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "wildcard_kmer": wildcard_kmer,
                "filled_kmer": filled_kmer,
                "motif_pos": motif_pos,
                "snp_index": snv_index,
                "type": "AFF",
                "allele": ref_allele,
                "coef": "NA",
                "pval": "NA",
                "absolute_pos": motif_pos + snv_index,
            }
        )

    return (rows, X, y) if return_matrix else rows


def main():
    parser = argparse.ArgumentParser(description="Run motif regression on SNV(s)")
    parser.add_argument(
        "--intensities", required=True, help="Path to probe intensity file"
    )
    parser.add_argument(
        "--kmerPositions",
        required=True,
        help="Path to k-mer array file mapping kmers to genomic regions",
    )
    parser.add_argument(
        "--genome", required=True, help="Path to reference genome in FASTA format"
    )
    parser.add_argument(
        "--snv-list",
        required=False,
        help="Comma-separated list of SNVs in chr:pos:ref>alt format",
    )
    parser.add_argument(
        "--snv-list-file",
        help="Optional file with SNVs, one per line in chr:pos:ref>alt format",
    )
    parser.add_argument(
        "--kmer_size",
        type=int,
        default=8,
        help="Size of the k-mers to analyze (default: 8)",
    )
    parser.add_argument(
        "--mode",
        choices=["nb", "ols"],
        default="nb",
        help="Regression type: negative binomial ('nb') or ordinary least squares ('ols')",
    )
    parser.add_argument(
        "--no-covariates",
        action="store_true",
        help="Disable use of location and length covariates in the regression model",
    )
    parser.add_argument(
        "--rand_n",
        type=int,
        default=5000,
        help="Number of random probes to use for RAND regression (default: 5000)",
    )

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
            snv_info = normalize_snv_region(
                chrom, snv_pos, ref, alt, args.genome, args.kmer_size
            )
            snv_info["snv_str"] = snv_str
            snv_info["wildcards"] = dict(
                get_snv_aligned_wildcards(snv_info, args.kmer_size)
            )

            allele_region_offsets, wildcards, matched_regions = match_snv_aligned_kmers(
                snv_info, kmers, args.kmer_size
            )

            # Step 1: build allele_matched_kmers
            allele_matched_kmers = {}
            for motif_pos, wildcard in wildcards:
                snv_index = wildcard.index(".")
                allele_matched_kmers.setdefault(motif_pos, {}).setdefault(snv_index, {})
                for kmer in kmers:
                    if wildcard_match(kmer, wildcard):
                        allele = kmer[snv_index]
                        for region_id in kmers[kmer]:
                            allele_matched_kmers[motif_pos][snv_index].setdefault(
                                allele, []
                            ).append((wildcard, kmer, region_id))

            # Step 2: run AFF regression across motif
            results_aff = []
            for motif_pos, snv_dict in allele_matched_kmers.items():
                for snv_index in snv_dict:
                    rows = run_aff_regression(
                        motif_pos=motif_pos,
                        snv_index=snv_index,
                        allele_region_offsets=allele_region_offsets,
                        allele_matched_kmers=allele_matched_kmers,
                        intensities=intensities,
                        region_seq=snv_info["region_seq"],
                        model_type=args.mode,
                        include_covariates=not args.no_covariates,
                    )
                    results_aff.extend(rows)

            # Step 3: RAND
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

            # Step 4: Print results
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
