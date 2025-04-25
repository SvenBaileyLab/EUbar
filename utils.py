import re
from collections import defaultdict
import random
import numpy as np
import statsmodels.api as sm
from pyfaidx import Fasta
import warnings
from statsmodels.discrete.discrete_model import NegativeBinomial
import hashlib
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_snv_string(snv_str):
    chrom, pos, change = snv_str.split(":")
    ref, alt = change.split(">")
    return chrom, int(pos), ref.upper(), alt.upper()


def reverse_complement(seq):
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]


def get_sequence_from_fasta(chrom, start, end, genome_fasta_path):
    from pyfaidx import Fasta

    """
    Extracts the DNA sequence from a given genomic region.

    Args:
        chrom (str): Chromosome name (e.g., "chr6")
        start (int): 1-based start position (inclusive)
        end (int): 1-based end position (inclusive)
        genome_fasta_path (str): Path to the genome FASTA file

    Returns:
        str: The DNA sequence from the region
    """
    genome = Fasta(genome_fasta_path)
    sequence = genome[chrom][start - 1 : end].seq.upper()
    return sequence


def deterministic_hash(string, max_val=2**32):
    return int(hashlib.md5(string.encode()).hexdigest(), 16) % max_val


def read_intensities(filepath):
    """
    Reads a probe intensity file and returns a dictionary.
    """
    intensities = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            region, value = line.split()
            intensities[region] = float(value)
    return intensities


def _determine_num_random(kmers, use_percentage=False, percentage=0.10, default=500):
    if use_percentage:
        total_regions = sum(len(v) for v in kmers.values())
        num_random = int(total_regions * percentage)
        # Optional: cap between reasonable bounds
        num_random = max(100, min(1000, num_random))
        return num_random
    else:
        return default


def normalize_snv_region(
    chrom, snv_pos, ref_allele, alt_allele, genome, kmer_size, debug=False
):
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
        sys.stderr.write(
            f"SNV should be at position {snv_index} → base: {genome_base}\n"
        )

    if genome_base == ref_allele:
        if debug:
            sys.stderr.write(
                "[*] Reference allele matches genome — no reverse complement needed.\n"
            )
    else:
        if debug:
            sys.stderr.write(
                f"[!] Reference allele mismatch: genome has '{genome_base}', SNV claims '{ref_allele}'\n"
            )
            sys.stderr.write("[*] Trying reverse complement...\n")

        region_seq = reverse_complement(region_seq)
        snv_index = len(region_seq) - 1 - snv_index

        genome_base = region_seq[snv_index]

        if debug:
            sys.stderr.write(f"[debug] Reverse complemented sequence: {region_seq}\n")
            sys.stderr.write(f"[debug] Reverse complemented snv_index: {snv_index}\n")
            sys.stderr.write(
                f"[debug] Reverse complemented genome_base: {genome_base}\n"
            )
            sys.stderr.write(f"[debug] Reverse complemented ref_allele: {ref_allele}\n")
            sys.stderr.write(
                f"[debug] reverse_complement('C') = {reverse_complement('C')}\n"
            )

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
            wildcard = kmer[:rel_pos] + "." + kmer[rel_pos + 1 :]
            wildcards.append((start, wildcard))
    return wildcards


def match_snv_aligned_kmers(snv_info, kmer_positions, kmer_size):
    allele_region_offsets = {}
    matched_regions = set()
    wildcards = get_snv_aligned_wildcards(snv_info, kmer_size)

    for motif_pos, wildcard in wildcards:
        pattern = re.compile("^" + wildcard.replace(".", "[ACGT]") + "$")
        snv_index = wildcard.index(".")
        region_to_alleles = {}
        region_kmer_hits = {}

        for kmer, region_dict in kmer_positions.items():
            if pattern.fullmatch(kmer):
                allele = kmer[snv_index]
                for region_id, offset in region_dict.items():
                    # track allele matches as before
                    region_to_alleles.setdefault(region_id, set()).add(allele)
                    region_kmer_hits.setdefault(region_id, {})[allele] = (offset, kmer)
                    # NEW: collect matched regions
                    matched_regions.add(region_id)

        for region_id, alleles in region_to_alleles.items():
            if len(alleles) == 1:
                allele = next(iter(alleles))
                offset, _ = region_kmer_hits[region_id][allele]
                allele_region_offsets.setdefault(motif_pos, {}).setdefault(allele, {})[
                    region_id
                ] = offset

    return allele_region_offsets, wildcards, matched_regions


def _match_snv_aligned_kmers(snv_info, kmer_positions, kmer_size, debug=False):
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
                allele_region_offsets.setdefault(motif_pos, {}).setdefault(allele, {})[
                    region_id
                ] = offset
                allele_matched_kmers.setdefault(motif_pos, {}).setdefault(
                    allele, []
                ).append(kmer)

    if debug:
        sys.stderr.write(f"\n=== [match_snv_aligned_kmers] DEBUG SUMMARY ===\n")
        sys.stderr.write(
            f"SNV: {snv_info['chrom']}:{snv_info['pos']} {snv_info['ref']}>{snv_info['alt']}\n"
        )
        sys.stderr.write(
            f"Region: {snv_info['region_start']}-{snv_info['region_end']} ({snv_info['region_seq']})\n"
        )
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
                sys.stderr.write(
                    f"  Allele {allele}: {n_regions} region(s), {len(matched_kmers)} unique k-mer(s)\n"
                )
                for k in matched_kmers:
                    sys.stderr.write(f"    ↳ {k}\n")

        sys.stderr.write(f"\nTotal matched regions (deduplicated): {total_hits}\n")
        sys.stderr.write(f"===============================================\n")

    return allele_region_offsets, allele_matched_kmers


def _read_kmer_positions(file_path):
    """
    Reads k-mer positions and labels each offset with .1, .2, ... (starting from 1),
    even for the first occurrence.
    Output:
        {
            "CTGAACTT": {
                "chr14:23094440-23096032.1": 1433,
                "chr14:23094440-23096032.2": 1569,
                ...
            }
        }
    """
    result = {}

    with open(file_path) as f:
        for line in f:
            kmer, entries = line.strip().split("\t")
            result[kmer] = {}

            for entry in entries.strip(",").split(","):
                region, count, offsets = entry.split(";")
                offset_list = offsets.strip().split()

                if len(offset_list) == 1:
                    # Only one match, use region name directly
                    result[kmer][region] = int(offset_list[0])
                else:
                    # Multiple matches, use .1, .2, ...
                    for i, offset in enumerate(offset_list, start=1):
                        region_id = f"{region}.{i}"
                        result[kmer][region_id] = int(offset)

    return result


def read_unique_kmer_positions(file_path):
    """
    Reads k-mer positions and retains only regions where the k-mer occurs exactly once (count == 1).

    Input format:
        kmer<TAB>region1;count1;offsets1,region2;count2;offsets2,...

    Output:
        {
            "CTGAACTT": {
                "chr14:23094440-23096032": 1433,
                ...
            }
        }
    """
    result = {}

    with open(file_path) as f:
        for line in f:
            kmer, entries = line.strip().split("\t")
            result[kmer] = {}

            for entry in entries.strip(",").split(","):
                try:
                    region, count, offsets = entry.split(";")
                    if int(count) != 1:
                        continue  # skip if there are multiple matches
                    offset = int(offsets.strip())
                    result[kmer][region] = offset
                except ValueError:
                    continue  # skip malformed lines

            if not result[kmer]:
                del result[kmer]  # remove empty kmer entries

    return result


def read_unique_kmer_positions_safe(file_path, max_kmers=None):
    """
    Reads k-mer positions and retains only regions where the k-mer occurs exactly once (count == 1).
    Returns a nested dict: {kmer: {region: offset}}.

    Optionally stops early if max_kmers is provided (for testing/debug).
    """
    result = {}
    kmer_count = 0

    with open(file_path) as f:
        for line_num, line in enumerate(f):
            try:
                kmer, entries = line.strip().split("\t")
            except ValueError:
                continue  # skip malformed lines

            regions = {}

            for entry in entries.strip(",").split(","):
                try:
                    region, count, offsets = entry.split(";")
                    if int(count) != 1:
                        continue
                    offset = int(offsets.strip())
                    regions[region] = offset
                except ValueError:
                    continue

            if regions:
                result[kmer] = regions
                kmer_count += 1

            if max_kmers and kmer_count >= max_kmers:
                print(f"[Info] Hit max_kmers={max_kmers} at line {line_num}")
                break

    return result


def _filter_unique_kmer_hits(kmer_positions):
    """
    Removes k-mer entries that map to the same region more than once (i.e., with .1, .2, etc.).
    Keeps only base regions that appear exactly once.
    """
    filtered = {}

    for kmer, region_dict in kmer_positions.items():
        # Count base regions
        base_counts = {}
        for region_id in region_dict:
            base = region_id.split(".")[0]
            base_counts[base] = base_counts.get(base, 0) + 1

        # Now filter to keep only base regions with count == 1
        filtered[kmer] = {
            region_id: offset
            for region_id, offset in region_dict.items()
            if base_counts[region_id.split(".")[0]] == 1
        }

    return filtered


def _get_all_kmer_wildcards(kmer_seq):
    """
    Given a k-mer sequence, return:
    - {snp_index: wildcarded_kmer} for all positions
    """
    wildcard_variants = {}
    for i in range(len(kmer_seq)):
        wildcard = list(kmer_seq)
        wildcard[i] = "."
        wildcard_kmer = "".join(wildcard)
        wildcard_variants[i] = wildcard_kmer
    return wildcard_variants


def get_kmer_variants(motif_seed, kmer_size):
    """
    Given a motif seed and a kmer size, return:
    - list of overlapping k-mers from the motif
    - dictionary of wildcarded variants: {motif_pos: wildcarded_kmer}
    - dictionary of SNP indices: {motif_pos: index_within_kmer}
    """
    kmer_list = []
    wildcard_variants = {}
    snp_indices = {}

    motif_len = len(motif_seed)
    for i in range(motif_len - kmer_size + 1):
        kmer = motif_seed[i : i + kmer_size]
        kmer_list.append(kmer)

        snp_index = kmer_size - 1 - i  # SNP always at center of motif
        if snp_index < 0 or snp_index >= kmer_size:
            continue  # skip if SNP outside this k-mer window

        wildcarded = list(kmer)
        wildcarded[snp_index] = "."
        wildcard_kmer = "".join(wildcarded)

        wildcard_variants[i] = wildcard_kmer
        snp_indices[i] = snp_index

    return kmer_list, wildcard_variants, snp_indices


def match_kmers_to_wildcards(
    kmer_positions, wildcard_variants, snp_indices, motif_seed
):
    matched = {}
    comp_alleles = {}

    for motif_pos, wildcard_kmer in wildcard_variants.items():
        snp_index = snp_indices[motif_pos]
        ref_allele = motif_seed[motif_pos + snp_index]

        pattern = re.compile("^" + wildcard_kmer.replace(".", "[ACGT]") + "$")

        for kmer in kmer_positions:
            if pattern.fullmatch(kmer):
                allele = kmer[snp_index]

                matched.setdefault(motif_pos, {}).setdefault(allele, {}).update(
                    kmer_positions[kmer]
                )

        comp_alleles[motif_pos] = {snp_index: ref_allele}

    return comp_alleles, matched


def match_all_kmers_to_wildcards(kmer_positions, wildcard_variants):
    """
    Matches all k-mers to wildcard patterns, without requiring snp_index or reference allele.

    Returns:
        matched: {motif_pos: {allele: {region: pos}}}
    """
    matched = {}

    for motif_pos, wildcard_kmer in wildcard_variants.items():
        pattern = re.compile("^" + wildcard_kmer.replace(".", "[ACGT]") + "$")
        snp_index = wildcard_kmer.index(".")

        for kmer in kmer_positions:
            if pattern.fullmatch(kmer):
                allele = kmer[snp_index]
                matched.setdefault(motif_pos, {}).setdefault(allele, {}).update(
                    kmer_positions[kmer]
                )

    return matched


def scan_motif_kmers(region_seq, kmer_size, kmer_positions):
    """
    Slides a k-mer window across the sequence and builds:
    - allele_region_offsets[motif_pos][snv_index][allele][region_id] = offset
    - allele_matched_kmers[motif_pos][snv_index][allele] = list of (wildcard_kmer, matched_kmer, region_id)

    Returns:
        (allele_region_offsets, allele_matched_kmers)
    """

    allele_region_offsets = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    allele_matched_kmers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    seen = set()  # prevent duplicate matches per motif_pos + snv + base + region + kmer

    for motif_pos in range(len(region_seq) - kmer_size + 1):
        kmer = region_seq[motif_pos : motif_pos + kmer_size]

        for snv_index in range(kmer_size):
            # Create wildcarded k-mer
            wildcard_kmer = list(kmer)
            wildcard_kmer[snv_index] = "."
            wildcard_kmer = "".join(wildcard_kmer)

            for base in "ACGT":
                filled_kmer = list(kmer)
                filled_kmer[snv_index] = base
                filled_kmer = "".join(filled_kmer)
                filled_kmer_reverse_complement = reverse_complement(filled_kmer)

                for match_kmer in [filled_kmer, filled_kmer_reverse_complement]:
                    if match_kmer in kmer_positions:
                        for region_id, offset in kmer_positions[match_kmer].items():
                            key = (motif_pos, snv_index, base, region_id, match_kmer)
                            if key in seen:
                                continue
                            seen.add(key)

                            allele_region_offsets[motif_pos][snv_index][base][
                                region_id
                            ] = offset

                            # Always store the forward-facing version (filled_kmer)
                            allele_matched_kmers[motif_pos][snv_index][base].append(
                                (wildcard_kmer, filled_kmer, region_id)
                            )

                            # Always store both forward and reverse-facing version (match_kmer)
                            # allele_matched_kmers[motif_pos][snv_index][base].append(
                            #     (wildcard_kmer, match_kmer, region_id)
                            # )

    return allele_region_offsets, allele_matched_kmers


def select_random_probes(kmer_positions, matched_probes, num_random=500, seed=43020):
    """
    Randomly selects N probe regions not overlapping with matched probe regions.

    Args:
        kmer_positions (dict): {kmer: {region: offset, ...}, ...}
        matched_probes (set): set of regions to exclude
        num_random (int): number of probes to sample
        seed (int): for reproducibility

    Returns:
        dict: {region_id: offset}
    """
    random.seed(seed)

    available = []

    for kmer, region_dict in kmer_positions.items():
        for region, offset in region_dict.items():
            if region not in matched_probes:
                available.append((region, offset))

    if len(available) < num_random:
        raise ValueError(
            f"Only {len(available)} unmatched probes available, can't select {num_random}."
        )

    sampled = random.sample(available, num_random)
    return {region: offset for region, offset in sampled}


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
    allele_list = sorted(
        {
            allele
            for pos in allele_region_offsets.values()
            for allele in pos
            if allele not in exclude_alleles
        }
    )

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


def _OLD_build_allele_matrix(matched, probe_intensities, exclude_alleles=None):
    """
    Build binary allele matrix and matching intensity values.

    Args:
        matched (dict): from match_kmers_to_wildcards()
        probe_intensities (dict): {region: intensity}
        exclude_alleles (set): alleles to exclude (e.g., {"A"})

    Returns:
        allele_matrix (dict): {allele: [0, 1, 1, ...]}
        probe_values (list): [intensity1, intensity2, ...]
    """
    allele_matrix = {}
    probe_values = []
    used_regions = []
    seen = set()

    for pos in matched.values():
        for sub in pos.values():
            for allele in sub:
                if exclude_alleles and allele in exclude_alleles:
                    continue
                allele_matrix.setdefault(allele, [])

    for pos in matched.values():
        for sub in pos.values():
            for allele in sub:
                if exclude_alleles and allele in exclude_alleles:
                    continue
                for region in sub[allele]:
                    if region not in probe_intensities or region in seen:
                        continue
                    seen.add(region)
                    used_regions.append(region)
                    probe_values.append(probe_intensities[region])
                    for a in allele_matrix:
                        allele_matrix[a].append(1 if a == allele else 0)

    return allele_matrix, probe_values, used_regions


def run_per_motif_regression(
    allele_region_offsets,
    probe_intensities,
    snv_info,
    model_type="neg-binomial",  # or "ols"
    include_covariates=True,
    debug=False,
):
    results = []

    for motif_pos, allele_dict in allele_region_offsets.items():
        if debug:
            sys.stderr.write(
                f"\n=== Running regression for motif position {motif_pos} ===\n"
            )

        # Build allele matrix (excluding ref allele)
        X, y, alleles, used_regions = build_allele_matrix(
            probe_intensities,
            {motif_pos: allele_dict},
            exclude_alleles={snv_info["ref"]},
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
            results.append(
                {
                    "snv_str": snv_info["snv_str"],
                    "motif_pos": motif_pos,
                    "allele": allele,
                    "coef": fit.params[i + 1],  # +1 because of intercept
                    "pval": fit.pvalues[i + 1],
                    "n": int((X[allele] == 1).sum()),
                }
            )

    return results


def sample_rand_kmers_per_allele(
    kmers, matched_regions, snv_str, kmer_size=8, rand_n=5000
):
    """
    Randomly sample k-mers, tracking alleles per motif position.
    Skips any regions used in AFF.

    Returns:
    - rand_regions_per_allele: {motif_pos: {allele: {region: offset}}}
    - used_regions: set of sampled region names
    """
    random.seed(deterministic_hash(snv_str))

    all_kmer_keys = list(kmers.keys())
    used_regions = set()
    rand_regions_per_allele = defaultdict(lambda: defaultdict(dict))

    attempts = 0
    max_attempts = rand_n * 20  # extra buffer

    while len(used_regions) < rand_n and attempts < max_attempts:
        kmer = random.choice(all_kmer_keys)
        region_dict = kmers[kmer]

        # Filter regions not in AFF and not already used
        available_regions = [
            r for r in region_dict if r not in matched_regions and r not in used_regions
        ]
        if not available_regions:
            attempts += 1
            continue

        region = random.choice(available_regions)
        offset = region_dict[region]
        motif_pos = random.randint(0, kmer_size - 1)
        allele = kmer[motif_pos]

        rand_regions_per_allele[motif_pos][allele][region] = offset
        used_regions.add(region)
        attempts += 1

    if len(used_regions) < rand_n:
        print(f"[WARN] Only collected {len(used_regions)} usable regions for RAND.")

    return rand_regions_per_allele, used_regions


def run_rand_regression_from_region_map(
    rand_regions_per_allele, probe_intensities, snv_str, model_type="nb"
):
    """
    Runs per-position regression using predefined allele-to-region mapping.

    Parameters:
        rand_regions_per_allele: dict of {motif_pos: {allele: set(region_ids)}}
        probe_intensities: dict of region -> intensity
        snv_str: e.g., 'chr5:1295113:C>T'
        model_type: 'nb' or 'ols'

    Returns:
        List of dicts with motif_pos, allele, coef, pval, n, label="RAND"
    """
    results = []
    all_alleles = ["A", "C", "G", "T"]

    for motif_pos in sorted(rand_regions_per_allele):
        region_sets = rand_regions_per_allele[motif_pos]

        # Combine all regions
        all_regions = set()
        for allele in all_alleles:
            all_regions.update(region_sets.get(allele, set()))

        # Prepare covariates
        y, lp_cov, sl_cov, X_alleles = [], [], [], {a: [] for a in all_alleles}
        for region in all_regions:
            if region not in probe_intensities:
                continue
            y.append(probe_intensities[region])
            try:
                _, coords = region.split(":")
                start, end = map(int, coords.split("-"))
                length = end - start
            except:
                length = 200
            offset = 1  # dummy
            lp_cov.append(offset / length)
            sl_cov.append(length)

            for allele in all_alleles:
                X_alleles[allele].append(
                    1 if region in region_sets.get(allele, set()) else 0
                )

        if len(y) < 10:
            continue

        # Build design matrix
        X_df = pd.DataFrame({a: X_alleles[a] for a in all_alleles})
        X_df["lp"] = lp_cov
        X_df["sl"] = sl_cov
        X_const = sm.add_constant(X_df)
        y = np.array(y)

        if model_type == "ols":
            y_transformed = np.log1p(y)
            model = sm.OLS(y_transformed, X_const)
        else:
            model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial())

        fit = model.fit()

        for allele in all_alleles:
            results.append(
                {
                    "snv_str": snv_str,
                    "motif_pos": motif_pos,
                    "allele": allele,
                    "coef": fit.params.get(allele, 0.0),
                    "pval": fit.pvalues.get(allele, np.nan),
                    "n": len(region_sets.get(allele, [])),
                    "label": "RAND",
                }
            )

    return results


def print_motif_effect_table(snv_str, chrom, pos, region_seq, results):
    ref_allele = snv_str.split(":")[2].split(">")[0]
    motif_positions = sorted(set(r["motif_pos"] for r in results))

    print(f"{snv_str}\t{chrom}\t{pos}\t{region_seq}")

    for group in ["AFF", "RAND"]:
        for allele in ["A", "C", "G", "T"]:
            coefs = []
            pvals = []

            for i in motif_positions:
                match = next(
                    (
                        r
                        for r in results
                        if r["motif_pos"] == i
                        and r["allele"] == allele
                        and r.get("label", "AFF") == group
                    ),
                    None,
                )

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
            pval_str = ",".join(
                f"{x:.7g}" if isinstance(x, float) else x for x in pvals
            )

            print(f"{group}\t{snv_str}\t{allele}\t{pos_str}\t{coef_str}\t{pval_str}")


def _print_rows_as_tsv_scan_style_with_snv(
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


def _project_kmers_to_random_probes(rand_probes, wildcard_variants, kmer_list, kmers):
    projected = defaultdict(lambda: defaultdict(dict))
    bases = ["A", "C", "G", "T"]

    for motif_pos, wildcard in wildcard_variants.items():
        for snp_index in range(len(kmer_list[0])):  # all kmers are same length
            if wildcard[snp_index] != ".":
                continue  # skip if not SNP site
            for base in bases:
                variant_kmer = wildcard[:snp_index] + base + wildcard[snp_index + 1 :]
                if variant_kmer not in kmers:
                    continue
                for region, offset in kmers[variant_kmer].items():
                    if region in rand_probes:
                        projected[motif_pos][base][region] = offset
    return dict(projected)


def _dump_glm_inputs(outfile, allele_matrix, y, extra_covariates):
    """
    Write GLM input vectors in a Perl-style debug format to file.

    Args:
        outfile (str): Output path
        allele_matrix (dict): {allele: list of 0/1}
        y (list or np.array): Response values
        extra_covariates (dict): e.g., {"lp": [...], "sl": [...]}
    """
    with open(outfile, "w") as f:
        f.write("=== GLM INPUT VECTORS ===\n")
        for allele, values in allele_matrix.items():
            f.write(f"{allele}: " + " ".join(map(str, values)) + "\n")

        for cov_name, values in extra_covariates.items():
            f.write(f"{cov_name}: " + " ".join(map(str, values)) + "\n")

        if isinstance(y, np.ndarray):
            y = y.tolist()
        f.write("y: " + " ".join(map(str, y)) + "\n")


def extract_covariates(regions, region_to_kmer_pos):
    lp = []
    sl = []
    for region in regions:
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))
        size = end - start
        kmer_pos = region_to_kmer_pos.get(region, 0)
        lp_val = kmer_pos / size if size > 0 else 0.5
        lp.append(lp_val)
        sl.append(size)
    return lp, sl


def _run_regression(
    allele_matrix, probe_values, mode="neg-binomial", extra_covariates=None
):
    """
    Run regression to model allele effects on probe intensity.

    Args:
        allele_matrix (dict): {allele: binary vector}
        probe_values (list): list of intensity values
        mode (str): "log-linear" or "neg-binomial"
        extra_covariates (dict): must include "lp" (optional: "sl")

    Returns:
        dict: {allele or covariate: (coef, stderr, pvalue)}
    """
    y = np.array(probe_values)
    alleles = list(allele_matrix.keys())
    X_cols = [allele_matrix[a] for a in alleles]

    if not extra_covariates or "lp" not in extra_covariates:
        raise ValueError("Missing required covariate: 'lp'")

    covariate_names = []
    for name, values in extra_covariates.items():
        X_cols.append(values)
        covariate_names.append(name)

    # dump_glm_inputs("glm_vectors.txt", allele_matrix, y, extra_covariates)

    X = np.column_stack(X_cols)
    X = sm.add_constant(X)

    if mode == "log-linear":
        y = np.log1p(y)
        model = sm.OLS(y, X)
    elif mode == "neg-binomial":
        # model = NegativeBinomial(y, X)
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    result = model.fit()

    stats = {}
    param_names = alleles + covariate_names
    for i, name in enumerate(param_names):
        try:
            stats[name] = (
                result.params[i + 1],
                result.bse[i + 1],
                result.pvalues[i + 1],
            )
        except IndexError:
            stats[name] = (float("nan"), float("nan"), float("nan"))

    return stats


def run_snv_regression(
    motif_pos,
    snv_index,
    allele_region_offsets,
    allele_matched_kmers,
    intensities,
    region_seq=None,
    model_type="nb",
    include_covariates=True,
):
    import pandas as pd
    import numpy as np
    import math
    import statsmodels.api as sm
    from collections import defaultdict
    import time
    from utils import extract_covariates  # assumes your extract_covariates is defined

    start = time.time()
    rows = []

    # [Step 1] Get alleles and reference
    ref_allele = region_seq[motif_pos + snv_index]
    region_lookup = allele_region_offsets[motif_pos][snv_index]
    all_alleles = sorted(region_lookup.keys())
    alt_alleles = [a for a in all_alleles if a != ref_allele]

    # [Step 2] Build region → allele map
    region_to_alleles = defaultdict(set)
    for allele, regions in region_lookup.items():
        for region in regions:
            region_to_alleles[region].add(allele)

    # [Step 3] Select unambiguous regions with intensity
    unambiguous_regions = [
        region
        for region, allele_set in region_to_alleles.items()
        if len(allele_set) == 1 and region in intensities
    ]
    if not unambiguous_regions:
        return []

    # [Step 4] Build design matrix with numpy
    allele_index = {a: i for i, a in enumerate(alt_alleles)}
    design_array = np.zeros(
        (len(unambiguous_regions), len(alt_alleles)), dtype=np.uint8
    )
    y_values = []
    region_list = []

    for i, region in enumerate(unambiguous_regions):
        assigned = next(iter(region_to_alleles[region]))
        if assigned in allele_index:
            design_array[i, allele_index[assigned]] = 1
        y_values.append(intensities[region])
        region_list.append(region)

    X = pd.DataFrame(design_array, index=region_list, columns=alt_alleles)
    y = pd.Series(y_values, index=region_list)

    # [Step 5] Covariates
    if include_covariates:
        kmer_pos = {
            region: region_lookup[allele][region]
            for allele in all_alleles
            for region in region_lookup[allele]
            if region in region_list
        }
        lp, sl = extract_covariates(region_list, kmer_pos)
        X["lp"] = lp
        X["sl"] = sl

    # [Step 6] Regression
    X_const = sm.add_constant(X)
    if model_type == "ols":
        model = sm.OLS(y.apply(math.log1p), X_const)
    else:
        model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial())

    results = model.fit()

    # [Step 7] Output rows
    seen = set()
    for allele in alt_alleles:
        coef = results.params.get(allele, math.nan)
        pval = results.pvalues.get(allele, math.nan)
        for wildcard_kmer, filled_kmer, _ in allele_matched_kmers[motif_pos][
            snv_index
        ].get(allele, []):
            key = (wildcard_kmer, filled_kmer)
            if key not in seen:
                seen.add(key)
                rows.append(
                    [
                        wildcard_kmer,
                        filled_kmer,
                        motif_pos,
                        snv_index,
                        "AFF",
                        allele,
                        coef,
                        pval,
                        motif_pos + snv_index,
                    ]
                )

    # [Step 8] Add ref allele with NA
    for wildcard_kmer, filled_kmer, _ in allele_matched_kmers[motif_pos][snv_index].get(
        ref_allele, []
    ):
        key = (wildcard_kmer, filled_kmer)
        if key not in seen:
            seen.add(key)
            rows.append(
                [
                    wildcard_kmer,
                    filled_kmer,
                    motif_pos,
                    snv_index,
                    "AFF",
                    ref_allele,
                    "NA",
                    "NA",
                    motif_pos + snv_index,
                ]
            )

    # print(f"[✓] Regression done for window {motif_pos}:{snv_index} in {time.time() - start:.2f}s")
    return rows


def _old_run_snv_regression(
    motif_pos,
    snv_index,
    allele_region_offsets,
    allele_matched_kmers,
    intensities,
    model_type="nb",
    include_covariates=True,
):
    rows = []
    alleles = allele_region_offsets[motif_pos][snv_index]
    all_regions = set()
    for region_dict in alleles.values():
        all_regions.update(region_dict.keys())

    design = []
    y_values = []
    region_list = []

    for region in all_regions:
        row = {a: 0 for a in alleles}
        for a in alleles:
            if region in alleles[a]:
                row[a] = 1
        if region in intensities:
            design.append(row)
            y_values.append(intensities[region])
            region_list.append(region)

    if not design:
        return []

    X = pd.DataFrame(design)
    y = pd.Series(y_values, index=region_list)
    X.index = y.index

    if include_covariates:
        kmer_pos = {
            region: allele_region_offsets[motif_pos][snv_index][allele][region]
            for allele in allele_region_offsets[motif_pos][snv_index]
            for region in allele_region_offsets[motif_pos][snv_index][allele]
        }
        lp, sl = extract_covariates(region_list, kmer_pos)
        X["lp"] = lp
        X["sl"] = sl

    X_const = sm.add_constant(X)

    if model_type == "ols":
        model = sm.OLS(y.apply(math.log1p), X_const)
    else:
        model = sm.GLM(y, X_const, family=sm.families.NegativeBinomial())

    results = model.fit()

    for allele in alleles:
        coef = results.params.get(allele, math.nan)
        pval = results.pvalues.get(allele, math.nan)
        for wildcard_kmer, filled_kmer in set(
            allele_matched_kmers[motif_pos][snv_index][allele]
        ):
            row = [
                wildcard_kmer,
                filled_kmer,
                motif_pos,
                snv_index,
                "AFF",
                allele,
                coef,
                pval,
                motif_pos + snv_index,
            ]
            rows.append(row)

    return rows


def _regression_driver(
    probe_dict,
    probe_intensities,
    exclude_ref_allele=False,
    comp_alleles=None,
    dhs=False,
    mode="neg-binomial",
):
    """
    Generalized regression function for either AFF or RAND.

    Args:
        probe_dict (dict): matched or random dictionary
        probe_intensities (dict): Region -> intensity
        exclude_ref_allele (bool): If True, exclude the reference allele
        comp_alleles (dict or None): Required if exclude_ref_allele is True
        dhs (bool): Whether to include size covariate
        mode (str): Regression mode

    Returns:
        dict: Regression stats per allele
    """

    def extract_covariates(regions, region_to_kmer_pos):
        lp = []
        sl = []
        for region in regions:
            chrom, coords = region.split(":")
            start, end = map(int, coords.split("-"))
            size = end - start
            kmer_pos = region_to_kmer_pos.get(region, 0)
            lp_val = kmer_pos / size if size > 0 else 0.5
            lp.append(lp_val)
            sl.append(size)
        return lp, sl

    motif_pos = list(probe_dict.keys())[0]
    snp_index = list(probe_dict[motif_pos].keys())[0]

    exclude = set()
    if exclude_ref_allele:
        if comp_alleles is None:
            raise ValueError("comp_alleles must be provided when excluding ref allele")
        ref_allele = comp_alleles[motif_pos][snp_index]
        exclude = {ref_allele}

    matrix, values, used = build_allele_matrix(
        probe_dict, probe_intensities, exclude_alleles=exclude
    )

    kmer_pos = {
        r: probe_dict[motif_pos][snp_index][allele][r]
        for allele in probe_dict[motif_pos][snp_index]
        for r in probe_dict[motif_pos][snp_index][allele]
    }

    lp, sl = extract_covariates(used, kmer_pos)
    covars = {"lp": lp}
    if dhs:
        covars["sl"] = sl

    stats = run_regression(matrix, values, mode=mode, extra_covariates=covars)
    return stats


def _analyze_motif_effects(
    snv,
    kmers,
    kmer_size,
    used_regions,
    intensities,
    genome,
    num_random=500,
    dhs=False,
    mode="neg-binomial",
):
    chrom, pos, refalt = snv.split(":")
    ref, alt = refalt.split(">")
    pos = int(pos)

    # Step 1: get motif and variants
    motif_seed = get_mutation_sequence(chrom, pos, ref, genome, kmer_size)
    kmer_list, wildcard_variants, snp_indices = get_kmer_variants(motif_seed, kmer_size)

    # Step 2: match wildcards to genome kmers
    comp_alleles, matched = match_kmers_to_wildcards(
        kmers, wildcard_variants, snp_indices, motif_seed
    )

    # Step 3: select and split random probes ONCE per SNP
    rand_seed = deterministic_hash(snv)
    rand_probes = select_random_probes(
        kmers, used_regions, num_random=num_random, seed=rand_seed
    )
    split_random = project_kmers_to_random_probes(
        rand_probes, wildcard_variants, kmer_list, kmers
    )

    # Step 4: run regression for each motif position
    regression_results = {}

    for motif_pos in comp_alleles:
        for snp_index in comp_alleles[motif_pos]:
            ref_allele = comp_alleles[motif_pos][snp_index]

            matched_block = {motif_pos: {snp_index: matched[motif_pos]}}
            comp_block = {motif_pos: {snp_index: ref_allele}}

            # === AFF: exclude ref allele ===
            aff_stats = regression_driver(
                matched_block,
                probe_intensities=intensities,
                exclude_ref_allele=True,
                comp_alleles=comp_block,
                dhs=dhs,
                mode=mode,
            )

            # === RAND: keep all alleles, handle failures ===
            try:
                random_block = {motif_pos: {snp_index: split_random[motif_pos]}}
                rand_stats = regression_driver(
                    random_block,
                    probe_intensities=intensities,
                    exclude_ref_allele=False,
                    comp_alleles=comp_block,
                    dhs=dhs,
                    mode=mode,
                )
            except KeyError:
                rand_stats = {}
                print(
                    f"[Warning] No matching random probes found for {snv} at position {motif_pos}. "
                    f"Introducing NaN values in RAND. "
                    f"Consider increasing `num_random` above {num_random} to reduce missing data."
                )

            regression_results[motif_pos] = {"aff": aff_stats, "rand": rand_stats}

    return regression_results


def _print_motif_effect_table(snv, chrom, pos, motif, results):
    ref_allele = snv.split(":")[2].split(">")[0]
    positions = sorted(results.keys())  # 0–7

    print(f"{snv}\t{chrom}\t{pos}\t{motif}")

    for group in ["AFF", "RAND"]:
        group_key = group.lower()

        for allele in ["A", "C", "G", "T"]:
            coefs = []
            pvals = []

            for i in positions:
                entry = results[i][group_key]
                if group == "AFF" and allele == ref_allele:
                    coefs.append(0)
                    pvals.append("NA")
                else:
                    coef, _, pval = entry.get(allele, (0, 0, "NA"))
                    coefs.append(coef)
                    pvals.append(pval)

            pos_str = ",".join(map(str, positions))
            coef_str = ",".join(f"{x:.7g}" for x in coefs)
            pval_str = ",".join(
                f"{x:.7g}" if isinstance(x, float) else x for x in pvals
            )

            print(f"{group}\t{snv}\t{allele}\t{pos_str}\t{coef_str}\t{pval_str}")


def _get_mutation_sequence(chrom, pos, ref_allele, genome_fasta_path, flank=8):
    """
    Extracts the motif sequence around a given SNV.

    Args:
        chrom (str): e.g. "chr6"
        pos (int): position of the SNV (1-based)
        ref_allele (str): reference allele to confirm
        genome_fasta_path (str): path to .fa or .fasta
        flank (int): number of bases upstream and downstream to grab

    Returns:
        str: motif seed (length = 2 * flank)
    """
    genome = Fasta(genome_fasta_path)
    start = pos - flank
    end = pos + flank - 1

    seq = genome[chrom][start:end].seq.upper()

    # Optional: check the reference matches
    center_base = genome[chrom][pos - 1].seq.upper()
    if center_base != ref_allele:
        raise ValueError(
            f"Reference allele mismatch at {chrom}:{pos}: expected {ref_allele}, found {center_base}"
        )

    return seq


def _get_kmer_motif_matches(kmers, motif_seed, kmer_size):
    """
    Convenience function to get matched k-mers and comp_alleles.

    Args:
        kmers (dict): from read_kmer_positions()
        motif_seed (str): sequence containing SNV
        kmer_size (int): e.g. 8

    Returns:
        matched: dict
        comp_alleles: dict
    """
    _, wildcard_variants = get_kmer_variants(motif_seed, kmer_size)
    comp_alleles, matched = match_kmers_to_wildcards(kmers, wildcard_variants)
    return matched


def _fix_random_block_structure(rand_probes, pos):
    return {
        pos: {
            allele: {region: offset for region, offset in rand_probes.items()}
            for allele in "ACGT"
        }
    }


def print_rows_as_tsv(
    rows,
    header=(
        "wildcard_kmer",
        "filled_kmer",
        "window_index",
        "snp_index",
        "type",
        "allele",
        "coef",
        "pval",
        "absolute_pos",
    ),
):
    """
    Print a list of rows as a TSV to stdout, replacing NaNs with "NaN" string explicitly.

    Args:
        rows (list of list): Data rows to print
        header (tuple): Optional column headers for the TSV
    """
    df_rows = []
    for row in rows:
        formatted = []
        for item in row:
            if isinstance(item, float) and math.isnan(item):
                formatted.append("NaN")
            elif item == "":
                formatted.append("NaN")
            else:
                formatted.append(item)
        df_rows.append(formatted)

    df = pd.DataFrame(
        df_rows,
        columns=[
            "wildcard_kmer",
            "filled_kmer",
            "window_index",
            "snp_index",
            "type",
            "allele",
            "coef",
            "pval",
            "absolute_pos",
        ],
    )
    print(df.to_csv(sep="\t", index=False))


def plot_aff_motif_effects(rows, save_path):
    import matplotlib

    matplotlib.use("Agg")
    df = pd.DataFrame(
        rows,
        columns=[
            "wildcard_kmer",
            "filled_kmer",
            "window_index",
            "snp_index",
            "type",
            "allele",
            "coef",
            "pval",
            "absolute_pos",
        ],
    )

    df = df[df["type"] == "AFF"].copy()

    df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
    df = df.dropna(subset=["coef", "pval"])
    df["-log10(pval)"] = -np.log10(df["pval"])
    df["absolute_position"] = df["window_index"] + df["snp_index"] + 1
    region_length = df["absolute_position"].max()

    allele_colors = {"A": "black", "C": "red", "G": "green", "T": "blue"}

    fig_width = max(12, region_length * 0.15)
    fig, axs = plt.subplots(2, 1, figsize=(fig_width, 6), sharex=True)

    for metric, ax in zip(["coef", "-log10(pval)"], axs):
        for _, row in df.iterrows():
            x = int(row["absolute_position"])
            y = row[metric]
            allele = row["allele"]
            color = allele_colors.get(allele, "gray")
            if pd.notna(y):
                ax.text(
                    x,
                    y,
                    allele,
                    color=color,
                    fontsize=12,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        ymin = df[metric].min()
        ymax = df[metric].max()
        yrange = ymax - ymin if ymax != ymin else 1
        padding = yrange * 0.1
        ax.set_ylim(ymin - padding, ymax + padding)
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.4)

    axs[1].set_xlabel("Genomic Position")
    axs[0].set_title("Motif Scan: Coefficients and -log10(p-values)")

    step = 10 if region_length > 80 else 5 if region_length > 40 else 1
    axs[1].set_xticks(range(1, region_length + 1, step))
    axs[1].set_xlim(0.5, region_length + 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Info] Figure saved to: {save_path}")
