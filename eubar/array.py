from __future__ import annotations
import argparse
from pyfaidx import Fasta
from collections import defaultdict
import itertools
from numba import njit
import numpy as np


def parse_bed(file_path):
    with open(file_path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            fields = line.strip().split()
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            yield chrom, start, end


@njit
def is_valid_kmer(kmer):
    for c in kmer:
        if c not in "ACGT":
            return False
    return True


@njit
def extract_kmers_array(seq, kmer_size):
    """
    Fast Numba-based function: returns list of (kmer, offset) pairs.
    """
    seq = seq.upper()
    max_i = len(seq) - kmer_size + 1
    result = []

    for i in range(max_i):
        kmer = seq[i : i + kmer_size]
        if is_valid_kmer(kmer):
            result.append((kmer, i))

    return result


def extract_kmers_from_sequence(seq, kmer_size):
    """
    Wrapper around Numba-accelerated version. Returns dict of {kmer: [offsets]}.
    """
    found = defaultdict(list)
    result = extract_kmers_array(seq, kmer_size)
    for kmer, offset in result:
        found[kmer].append(offset)
    return found


def verify_kmers(all_kmers_dict, kmer_size):
    actual_kmers = set(all_kmers_dict.keys())
    missing_count = 0
    max_print = 10

    for kmer_tuple in itertools.product("ACGT", repeat=kmer_size):
        kmer = "".join(kmer_tuple)
        if kmer not in actual_kmers:
            if missing_count < max_print:
                print(f"Missing: {kmer}")
            missing_count += 1

    if missing_count == 0:
        print(f"All kmers represented! N={len(actual_kmers)}")
    else:
        print(
            f"WARNING - {len(actual_kmers)} kmers found. {missing_count} missing (showing first {min(missing_count, max_print)})."
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a k-mer array file from a BED file and reference genome FASTA."
    )
    parser.add_argument(
        "--bed", required=True, help="Input BED file of genomic regions"
    )
    parser.add_argument("--genome", required=True, help="Reference genome FASTA")
    parser.add_argument(
        "--kmer_size",
        "--kmer-size",
        dest="kmer_size",
        type=int,
        default=8,
        help="k-mer size (default: 8)",
    )
    parser.add_argument(
        "--output", "--out", dest="output", required=True, help="Output array file"
    )

    args = parser.parse_args(argv)

    fasta = Fasta(args.genome)
    all_kmers = defaultdict(list)

    seen_regions = set()
    total_regions = 0
    skipped_duplicates = 0
    skipped_invalid = 0

    for chrom, start, end in parse_bed(args.bed):
        total_regions += 1
        region_key = (chrom, start, end)
        if region_key in seen_regions:
            skipped_duplicates += 1
            continue
        seen_regions.add(region_key)

        region_str = f"{chrom}:{start}-{end}"
        try:
            seq = fasta[chrom][start:end].seq
        except KeyError:
            print(f"[Warning] Chromosome {chrom} not found in genome. Skipping.")
            continue

        found_kmers = extract_kmers_from_sequence(seq, args.kmer_size)

        for kmer, offsets in found_kmers.items():
            if not is_valid_kmer(kmer):
                skipped_invalid += 1
                continue
            all_kmers[kmer].append(
                f"{region_str};{len(offsets)};{' '.join(str(i + 1) for i in offsets)}"
            )

    with open(args.output, "w") as out:
        for kmer in sorted(all_kmers):
            index_line = f"{kmer}\t{','.join(all_kmers[kmer])}"
            out.write(index_line + "\n")

    print(f"[Done] K-mer index written to {args.output}")
    print(f"[Done] Skipped {skipped_invalid} k-mers with N or ambiguous bases")
    print(
        f"[Summary] Processed {len(seen_regions)} unique regions (skipped {skipped_duplicates} duplicates from {total_regions} lines)"
    )
    verify_kmers(all_kmers, args.kmer_size)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
