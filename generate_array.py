import argparse
from pyfaidx import Fasta
from collections import defaultdict
import re
import itertools

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

def extract_kmers_from_sequence(seq, region_str, kmer_size):
    """
    Returns a dictionary of {kmer: [offsets]} within a region.
    """
    found = defaultdict(list)
    for i in range(len(seq) - kmer_size + 1):
        kmer = seq[i:i + kmer_size].upper()
        if re.fullmatch("[ACGT]+", kmer):
            found[kmer].append(i)
    return found

def generate_all_kmers(k):
    return {''.join(p) for p in itertools.product('ACGT', repeat=k)}

def verify_kmers(all_kmers_dict, kmer_size):
    expected_kmers = generate_all_kmers(kmer_size)
    actual_kmers = set(all_kmers_dict.keys())
    missing = expected_kmers - actual_kmers

    if not missing:
        print(f"All kmers represented! N={len(actual_kmers)}")
    else:
        print(f"WARNING - Only {len(actual_kmers)} kmers found!!")
        print("Missing kmers:")
        for kmer in sorted(missing):
            print(kmer)

def main():
    parser = argparse.ArgumentParser(description="Generate k-mer index from ATAC/DNase-seq BED and genome FASTA.")
    parser.add_argument("--bed", required=True, help="Input BED file with regions (e.g., DNase/ATAC-seq peaks)")
    parser.add_argument("--genome", required=True, help="Reference genome in FASTA format")
    parser.add_argument("--kmer_size", type=int, default=8, help="Length of k-mers to extract")
    parser.add_argument("--output", required=True, help="Output file for the k-mer index")

    args = parser.parse_args()

    fasta = Fasta(args.genome)
    all_kmers = defaultdict(list)

    for chrom, start, end in parse_bed(args.bed):
        region_str = f"{chrom}:{start}-{end}"
        try:
            seq = fasta[chrom][start:end].seq
        except KeyError:
            print(f"[Warning] Chromosome {chrom} not found in genome. Skipping.")
            continue

        found_kmers = extract_kmers_from_sequence(seq, region_str, args.kmer_size)

        for kmer, offsets in found_kmers.items():
            all_kmers[kmer].append(f"{region_str};{len(offsets)};{' '.join(str(i + 1) for i in offsets)}")

    with open(args.output, "w") as out:
        for kmer in sorted(all_kmers):
            index_line = f"{kmer}\t{','.join(all_kmers[kmer])}"
            out.write(index_line + "\n")

    print(f"[Done] K-mer index written to {args.output}")
    verify_kmers(all_kmers, args.kmer_size)

if __name__ == "__main__":
    main()



