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


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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


def determine_num_random(kmers, use_percentage=False, percentage=0.10, default=500):
    if use_percentage:
        total_regions = sum(len(v) for v in kmers.values())
        num_random = int(total_regions * percentage)
        # Optional: cap between reasonable bounds
        num_random = max(100, min(1000, num_random))
        return num_random
    else:
        return default


def read_kmer_positions(file_path):
    """
    Reads k-mer positions and handles multiple offsets per region.
    Output:
        {
            "CTGAACTT": {
                "chr14:23094440-23096032": 1433,
                "chr14:23094440-23096032.1": 1569,
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

                for i, offset in enumerate(offset_list):
                    suffix = f".{i}" if i > 0 else ""
                    region_id = f"{region}{suffix}"
                    result[kmer][region_id] = int(offset)

    return result


def get_all_kmer_wildcards(kmer_seq):
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


def build_allele_matrix(matched, probe_intensities, exclude_alleles=None):
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


def project_kmers_to_random_probes(rand_probes, wildcard_variants, kmer_list, kmers):
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


def dump_glm_inputs(outfile, allele_matrix, y, extra_covariates):
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


def run_regression(
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


def regression_driver(
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


def analyze_motif_effects(
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


def print_motif_effect_table(snv, chrom, pos, motif, results):
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


def get_mutation_sequence(chrom, pos, ref_allele, genome_fasta_path, flank=8):
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


def get_kmer_motif_matches(kmers, motif_seed, kmer_size):
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


def fix_random_block_structure(rand_probes, pos):
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
