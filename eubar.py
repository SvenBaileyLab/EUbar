import argparse
import math
import sys
import pandas as pd
import numpy as np
import re
import math
import pandas as pd
import statsmodels.api as sm
from numba import njit


from utils import (
    read_intensities,
    read_unique_kmer_positions,
    get_sequence_from_fasta,
#     reverse_complement,
#     parse_snv_string,
    extract_covariates,
#     normalize_snv_region,
)

def parse_snv_string(snv_str):
    chrom, pos, change = snv_str.split(":")
    ref, alt = change.split(">")
    return chrom, int(pos), ref.upper(), alt.upper()

def reverse_complement(seq):
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]

def normalize_snv_region(chrom, snv_pos, ref_allele, alt_allele, genome, kmer_size, debug=False):
    """
    Normalize SNV region to a window of size 2*kmer_size - 1, centered on the SNV.
    Automatically reverse complements if the ref allele doesn't match the genome.
    """
    half_window = kmer_size - 1
    region_start = snv_pos - half_window
    region_end = snv_pos + half_window  # exclusive

    region_seq = get_sequence_from_fasta(chrom, region_start, region_end, genome)
    snv_index = snv_pos - region_start
    genome_base = region_seq[snv_index]

    if debug:
        sys.stderr.write(f"\nWindowed region: {chrom}:{region_start}-{region_end}\n")
        sys.stderr.write(f"Sequence: {region_seq}\n")
        sys.stderr.write(f"SNV should be at position {snv_index} → base: {genome_base}\n")

    if genome_base == ref_allele:
        if debug:
            sys.stderr.write("[*] Reference allele matches genome — no reverse complement needed.\n")
    else:
        if debug:
            sys.stderr.write(f"[!] Reference allele mismatch: genome has '{genome_base}', SNV claims '{ref_allele}'\n")
            sys.stderr.write("[*] Trying reverse complement...\n")

        region_seq = reverse_complement(region_seq)
        snv_index = len(region_seq) - 1 - snv_index
        
        genome_base = region_seq[snv_index]

        if debug:
            sys.stderr.write(f"[debug] Reverse complemented sequence: {region_seq}\n")
            sys.stderr.write(f"[debug] Reverse complemented snv_index: {snv_index}\n")
            sys.stderr.write(f"[debug] Reverse complemented genome_base: {genome_base}\n")
            sys.stderr.write(f"[debug] Reverse complemented ref_allele: {ref_allele}\n")
            sys.stderr.write(f"[debug] reverse_complement('C') = {reverse_complement('C')}\n")

        if genome_base != ref_allele:
            raise ValueError(
                f"[ERROR] Reference allele mismatch even after reverse complement. "
                f"Genome has '{genome_base}', expected '{ref_allele}' at index {snv_index}"
            )

        if debug:
            sys.stderr.write("[*] Using reverse complement:\n")
            sys.stderr.write(f"    Updated sequence: {region_seq}\n")
            sys.stderr.write(f"    Updated ref: {ref_allele}, alt: {alt_allele}\n")

    # Final debug before returning
    if debug:
        sys.stderr.write("\n[returning]\n")
        sys.stderr.write(f"  region_seq: {region_seq}\n")
        sys.stderr.write(f"  snv_index: {snv_index}\n")
        sys.stderr.write(f"  ref_allele: {ref_allele}\n")
        sys.stderr.write(f"  genome_base (final): {region_seq[snv_index]}\n")

    return {
        "chrom": chrom,
        "pos": snv_pos,
        "ref": ref_allele,
        "alt": alt_allele,
        "region_start": region_start,
        "region_end": region_end,
        "region_seq": region_seq,
        "snv_index": snv_index,
    }

def get_snv_aligned_wildcards(snv_info, kmer_size):
    wildcards = []
    for start in range(len(snv_info["region_seq"]) - kmer_size + 1):
        end = start + kmer_size
        if start <= snv_info["snv_index"] < end:
            rel_pos = snv_info["snv_index"] - start
            kmer = snv_info["region_seq"][start:end]
            wildcard = kmer[:rel_pos] + "." + kmer[rel_pos + 1:]
            wildcards.append((start, wildcard))
    return wildcards


def match_snv_aligned_kmers(snv_info, kmer_positions, kmer_size, debug=False):
    """
    Match k-mers from kmer_positions to SNV-aligned wildcards.
    Returns:
        allele_region_offsets: {motif_pos: {allele: {region: offset}}}
        allele_matched_kmers: {motif_pos: {allele: [matched_kmers]}}
    """
    allele_region_offsets = {}
    allele_matched_kmers = {}

    wildcards = get_snv_aligned_wildcards(snv_info, kmer_size)

    for motif_pos, wildcard in wildcards:
        pattern = re.compile("^" + wildcard.replace(".", "[ACGT]") + "$")
        snv_index = wildcard.index(".")

        # Temporary: region → alleles map for deduplication
        region_to_alleles = {}
        region_kmer_hits = {}

        # First pass: collect all matches
        for kmer, region_dict in kmer_positions.items():
            if pattern.fullmatch(kmer):
                allele = kmer[snv_index]

                for region_id, offset in region_dict.items():
                    region_to_alleles.setdefault(region_id, set()).add(allele)
                    region_kmer_hits.setdefault(region_id, {})[allele] = (offset, kmer)

        # Second pass: keep only regions with a single allele match
        for region_id, alleles in region_to_alleles.items():
            if len(alleles) == 1:
                allele = next(iter(alleles))
                offset, kmer = region_kmer_hits[region_id][allele]
                allele_region_offsets.setdefault(motif_pos, {}).setdefault(allele, {})[region_id] = offset
                allele_matched_kmers.setdefault(motif_pos, {}).setdefault(allele, []).append(kmer)

    if debug:
        sys.stderr.write(f"\n=== [match_snv_aligned_kmers] DEBUG SUMMARY ===\n")
        sys.stderr.write(f"SNV: {snv_info['chrom']}:{snv_info['pos']} {snv_info['ref']}>{snv_info['alt']}\n")
        sys.stderr.write(f"Region: {snv_info['region_start']}-{snv_info['region_end']} ({snv_info['region_seq']})\n")
        sys.stderr.write(f"Number of wildcard kmers generated: {len(wildcards)}\n")

        total_hits = 0
        for motif_pos, wildcard in wildcards:
            if motif_pos not in allele_region_offsets:
                continue
            sys.stderr.write(f"\nMotif position {motif_pos} — Wildcard: {wildcard}\n")
            for allele, regions in allele_region_offsets[motif_pos].items():
                n_regions = len(regions)
                matched_kmers = sorted(set(allele_matched_kmers[motif_pos][allele]))
                total_hits += n_regions
                sys.stderr.write(f"  Allele {allele}: {n_regions} region(s), {len(matched_kmers)} unique k-mer(s)\n")
                for k in matched_kmers:
                    sys.stderr.write(f"    ↳ {k}\n")

        sys.stderr.write(f"\nTotal matched regions (deduplicated): {total_hits}\n")
        sys.stderr.write(f"===============================================\n")

    return allele_region_offsets, allele_matched_kmers


def build_allele_matrix(probe_intensities, allele_region_offsets, exclude_alleles=None):
    """
    Builds the binary matrix X (probes × alleles) and intensity vector y.

    Returns:
        X: np.ndarray (N x A)
        y: np.ndarray (N,)
        allele_list: list of alleles (column order)
        used_regions: list of regions (row order)
    """
    if exclude_alleles is None:
        exclude_alleles = set()

    X = []
    y = []
    used_regions = []
    region_set = set()

    # Get full list of alleles (columns)
    allele_list = sorted({
        allele
        for pos in allele_region_offsets.values()
        for allele in pos
        if allele not in exclude_alleles
    })

    # Build region-to-alleles lookup
    region_to_alleles = {}
    for pos_dict in allele_region_offsets.values():
        for allele, region_dict in pos_dict.items():
            if allele in exclude_alleles:
                continue
            for region in region_dict:
                region_to_alleles.setdefault(region, set()).add(allele)

    # Build X and y matrices
    for region, alleles_present in region_to_alleles.items():
        if region not in probe_intensities or region in region_set:
            continue
        region_set.add(region)
        used_regions.append(region)
        row = [1 if allele in alleles_present else 0 for allele in allele_list]
        X.append(row)
        y.append(probe_intensities[region])

    return np.array(X), np.array(y), allele_list, used_regions

def run_per_motif_regression(
    allele_region_offsets,
    probe_intensities,
    snv_info,
    model_type="neg-binomial",  # or "ols"
    include_covariates=True,
    debug=False
):
    results = []

    for motif_pos, allele_dict in allele_region_offsets.items():
        if debug:
            sys.stderr.write(f"\n=== Running regression for motif position {motif_pos} ===\n")

        # Build allele matrix (excluding ref allele)
        X, y, alleles, used_regions  = build_allele_matrix(
            probe_intensities,
            {motif_pos: allele_dict},
            exclude_alleles={snv_info["ref"]}
        )

        if len(y) < 10 or X.shape[1] == 0:
            if debug:
                sys.stderr.write("  [!] Not enough data. Skipping.\n")
            continue

        X = pd.DataFrame(X, columns=alleles)

        if include_covariates:
            # Build region → offset dict for current motif
            region_to_offset = {
                region: offset
                for allele_regions in allele_dict.values()
                for region, offset in allele_regions.items()
                if region in probe_intensities
            }

            lp, sl = extract_covariates(used_regions, region_to_offset)

            X["lp"] = lp
            X["sl"] = sl

        X_const = sm.add_constant(X)

        if model_type == "ols":
            y_transformed = np.log1p(y)
            model = sm.OLS(y_transformed, X_const)
        else:
            model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial())

        fit = model.fit()

        if debug:
            sys.stderr.write(f"  Shape of X: {X.shape}\n")
            sys.stderr.write(f"  Alleles tested: {alleles}\n")
            sys.stderr.write(f"  Coefs: {fit.params.tolist()}\n")
            sys.stderr.write(f"  P-values: {fit.pvalues.tolist()}\n")

        for i, allele in enumerate(alleles):
            results.append({
                "snv_str" : snv_info["snv_str"],
                "motif_pos": motif_pos,
                "allele": allele,
                "coef": fit.params[i + 1],  # +1 because of intercept
                "pval": fit.pvalues[i + 1],
                "n": int((X[allele] == 1).sum())
            })

    return results

def print_motif_effect_table(snv_str, chrom, pos, region_seq, results):
    ref_allele = snv_str.split(":")[2].split(">")[0]
    motif_positions = sorted(set(r["motif_pos"] for r in results))

    print(f"{snv_str}\t{chrom}\t{pos}\t{region_seq}")

    for group in ["AFF"]:  # You can add "RAND" if you extend later
        for allele in ["A", "C", "G", "T"]:
            coefs = []
            pvals = []

            for i in motif_positions:
                match = next((r for r in results if r["motif_pos"] == i and r["allele"] == allele), None)

                if group == "AFF" and allele == ref_allele:
                    coefs.append(0)
                    pvals.append("NA")
                elif match:
                    coefs.append(match["coef"])
                    pvals.append(match["pval"])
                else:
                    coefs.append(0)
                    pvals.append("NA")

            pos_str = ",".join(map(str, motif_positions))
            coef_str = ",".join(f"{x:.7g}" for x in coefs)
            pval_str = ",".join(f"{x:.7g}" if isinstance(x, float) else x for x in pvals)

            print(f"{group}\t{snv_str}\t{allele}\t{pos_str}\t{coef_str}\t{pval_str}")



def print_rows_as_tsv_scan_style_with_snv(
    rows,
    snv_str,
    header=(
        "snv",
        "wildcard_kmer",
        "filled_kmer",
        "window_index",
        "snv_index",
        "type",
        "allele",
        "coef",
        "pval",
    ),
):
    """
    Print regression results in scan-style TSV format with SNV string as first column.
    Replaces NaNs with the string "NaN".
    """
    df_rows = []
    for row in rows:
        formatted = [snv_str]  # Add SNV column first
        for item in row[:8]:  # Only include scan-style fields
            if isinstance(item, float) and math.isnan(item):
                formatted.append("NaN")
            elif item == "":
                formatted.append("NaN")
            else:
                formatted.append(item)
        df_rows.append(formatted)

    df = pd.DataFrame(df_rows, columns=header)
    print(df.to_csv(sep="\t", index=False))
    
    
def main():
    parser = argparse.ArgumentParser(description="Run motif regression on SNV(s)")
    parser.add_argument("--intensities", required=True)
    parser.add_argument("--kmerPositions", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--snv-list", required=True, help="One or more SNVs (e.g. chr5:1234:G>A or comma-separated)")
    parser.add_argument("--kmer_size", type=int, default=8)
    parser.add_argument("--mode", choices=["nb", "ols"], default="nb")
    parser.add_argument("--no-covariates", action="store_true")
    
    args = parser.parse_args()

    intensities = read_intensities(args.intensities)
    kmers = read_unique_kmer_positions(args.kmerPositions)

    all_rows = []

    for snv_str in args.snv_list.split(","):
        chrom, snv_pos, ref, alt = parse_snv_string(snv_str)
        snv_info = normalize_snv_region(chrom, snv_pos, ref, alt, args.genome, args.kmer_size)
        snv_info["snv_str"] = snv_str
        snv_info["wildcards"] = dict(get_snv_aligned_wildcards(snv_info, args.kmer_size))

        allele_region_offsets, _ = match_snv_aligned_kmers(snv_info, kmers, args.kmer_size)
        results = run_per_motif_regression(
            allele_region_offsets,
            intensities,
            snv_info,
            model_type=args.mode,
            include_covariates=not args.no_covariates
        )

        print_motif_effect_table(
            snv_str=snv_info["snv_str"],
            chrom=snv_info["chrom"],
            pos=snv_info["pos"],
            region_seq=snv_info["region_seq"],
            results=results
        )

if __name__ == "__main__":
    main()