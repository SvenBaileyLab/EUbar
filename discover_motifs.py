#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import logomaker
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path
from os.path import join
from utils import (
    read_intensities,
    read_unique_kmer_positions
)

# ----------------- Utilities -----------------

def revcomp(seq):
    trans = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(trans)[::-1]

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

# ----------------- Step 1: motifSeed.py -----------------

def compute_kmer_means(kmers, probes):
    kmer_stats = {}
    seen = set()
    for k in kmers:
        r = revcomp(k)
        if k in seen or r in seen:
            continue
        seen.add(k)
        seen.add(r)

        signal = [probes[p] for p in kmers[k] if p in probes]
        mu = np.mean(signal) if signal else np.nan

        if k != r and r in kmers:
            rsignal = [probes[p] for p in kmers[r] if p in probes]
            rmu = np.mean(rsignal) if rsignal else np.nan
            combined = signal + rsignal
        else:
            rmu = np.nan
            combined = signal

        cmu = np.mean(combined) if combined else np.nan
        kmer_stats[k] = cmu
    return kmer_stats

# ----------------- Step 2: suggest_seeds.py -----------------

def get_top_kmer(kmer_means):
    return sorted(kmer_means.items(), key=lambda x: x[1], reverse=True)[0][0]

# ----------------- Step 3: motif_walk.py -----------------

def walk(kmer_means, seed, threshold, direction=1, kmer_size=8):
    max_steps = 100
    size = kmer_size
    step = 1
    ext_seq = seed
    tmp_kmer = seed
    max_signal = threshold + 1

    steps = 0
    while max_signal > threshold and steps < max_steps:
        steps += 1
        print(f"  Step {steps}: {tmp_kmer}")
        sub_kmer = tmp_kmer[1:size] if direction == 1 else tmp_kmer[0:size-1]
        r_sub_kmer = revcomp(sub_kmer)
        candidates = {}

        for k in kmer_means:
            if sub_kmer in k or r_sub_kmer in revcomp(k):
                pos = k.find(sub_kmer)
                if (direction == 1 and pos == 0) or (direction == -1 and pos == 1):
                    candidates[k] = kmer_means[k]

        print(f"    Found {len(candidates)} candidates")
        if not candidates:
            break

        max_signal = max(candidates.values())
        if max_signal < threshold:
            break

        for k, v in candidates.items():
            if v == max_signal:
                next_base = k[size - 1] if direction == 1 else k[0]
                ext_seq = ext_seq + next_base if direction == 1 else next_base + ext_seq
                tmp_kmer = sub_kmer + next_base if direction == 1 else next_base + sub_kmer
                break

    return ext_seq

# ----------------- Step 4: motif_logo.py -----------------

def find_similar_kmers(motif, kmer_means, max_mismatches=2, kmer_size=8):
    k = kmer_size
    all_similar = set()
    for i in range(len(motif) - k + 1):
        window = motif[i:i+k]
        for kmer in kmer_means:
            if hamming_distance(kmer, window) <= max_mismatches:
                all_similar.add(kmer)
    return list(all_similar)

def build_pfm(kmers, motif=None):
    if not motif:
        pfm = {base: [0]*len(kmers[0]) for base in "ACGT"}
        for kmer in kmers:
            for i, base in enumerate(kmer):
                pfm[base][i] += 1
        return pd.DataFrame(pfm).T

    # Align kmers to the motif and build full-length PFM
    pfm = {base: [0]*len(motif) for base in "ACGT"}
    k = len(kmers[0])
    for kmer in kmers:
        for i in range(len(motif) - k + 1):
            window = motif[i:i+k]
            if hamming_distance(kmer, window) <= 2:
                for j, base in enumerate(kmer):
                    pfm[base][i + j] += 1
                break
    return pd.DataFrame(pfm).T

def plot_logo(pfm, motif, out_file, use_bits=True, label=None):
    ppm = pfm.div(pfm.sum(axis=0), axis=1)
    matrix = logomaker.transform_matrix(ppm.T, from_type='probability', to_type='information') if use_bits else ppm.T
    fig, ax = plt.subplots(figsize=(len(matrix) * 0.5, 2.5))
    logomaker.Logo(matrix, ax=ax, color_scheme='classic')
    ax.set_title(f"{label}: {motif}" if label else motif)    
    ax.set_xticks(range(len(matrix)))
    ax.set_xticklabels([str(i + 1) for i in range(len(matrix))])
    ax.set_xlabel("Position")
    ax.set_ylabel("Bits" if use_bits else "Probability")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved {out_file}")


def save_meme_pwm(ppm, motif_name, out_file):
    with open(out_file, 'w') as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies:\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        f.write(f"MOTIF {motif_name}\n")
        f.write("letter-probability matrix: alength= 4 w= {}\n".format(ppm.shape[1]))
        for i in range(ppm.shape[1]):
            row = [ppm.loc[base, i] for base in "ACGT"]
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    print(f"Saved {out_file}")

# ----------------- MAIN -----------------

def main():
    parser = argparse.ArgumentParser(description="Full motif discovery pipeline.")
    parser.add_argument("--intensities", required=True)
    parser.add_argument("--kmerPositions", required=True)
    parser.add_argument("--seed", help="Optional k-mer seed to walk from")
    parser.add_argument("--thres", type=float, help="Signal threshold; auto if not provided")
    parser.add_argument("--percentile", type=float, default=80.0)
    parser.add_argument("--mismatches", type=int, default=1)
    parser.add_argument("--kmer_size", type=int, default=8, help="k-mer size (default: 8)")
    parser.add_argument("--logo", action='store_true', help="Generate sequence logo")
    parser.add_argument("--prob", action='store_true', help="Plot logo using probabilities instead of bits")
    parser.add_argument("--savePWM", action='store_true', help="Save position weight matrix in MEME format")
    parser.add_argument("--out", default="motif")
    parser.add_argument("--outdir", default=".", help="Directory to save outputs (default: current dir)")
    parser.add_argument("--keep_stats", action='store_true', help="Save k-mer intensity stats to TSV")
    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    probes = read_intensities(args.intensities)
    kmers = read_unique_kmer_positions(args.kmerPositions)
    kmer_means = compute_kmer_means(kmers, probes)

    if args.keep_stats:
        stats_path = join(args.outdir, f"{args.out}_kmer_stats.tsv")
        with open(stats_path, 'w') as f:
            f.write("KMER	rKMER	MEAN	MEDIAN	fMEAN	rMEAN	fMEDIAN	rMEDIAN\n")

            seen = set()
            for k in sorted(kmers.keys()):
                r = revcomp(k)
                if k in seen or r in seen:
                    continue
                seen.add(k)
                seen.add(r)

                signal = [probes[p] for p in kmers[k] if p in probes]
                mu = np.mean(signal) if signal else np.nan
                med = np.median(signal) if signal else np.nan

                if k != r and r in kmers:
                    rsignal = [probes[p] for p in kmers[r] if p in probes]
                    rmu = np.mean(rsignal) if rsignal else np.nan
                    rmed = np.median(rsignal) if rsignal else np.nan
                    combined = signal + rsignal
                else:
                    rmu = np.nan
                    rmed = np.nan
                    combined = signal

                cmu = np.mean(combined) if combined else np.nan
                cmed = np.median(combined) if combined else np.nan

                f.write(f"{k}	{r}	{cmu:.4f}	{cmed:.4f}	{mu:.4f}	{rmu:.4f}	{med:.4f}	{rmed:.4f}\n")
        print(f"Saved k-mer stats to {stats_path}")


    if args.thres is None:
        args.thres = np.percentile(list(kmer_means.values()), args.percentile)
        print(f"Auto threshold = {args.thres:.4f}")

    seed = args.seed if args.seed else get_top_kmer(kmer_means)

    print(f"\n=== Walking seed: {seed} ===")
    right = walk(kmer_means, seed, args.thres, direction=1, kmer_size=args.kmer_size)
    full_motif = walk(kmer_means, right, args.thres, direction=-1, kmer_size=args.kmer_size)
    print(f"Motif: {full_motif}")

    similar = find_similar_kmers(full_motif, kmer_means, max_mismatches=args.mismatches, kmer_size=args.kmer_size)
    if not similar:
        print("No similar kmers found for logo/PWM.")
        return

    pfm = build_pfm(similar, motif=full_motif)
    ppm = pfm.div(pfm.sum(axis=0), axis=1)

    if args.logo:
        logo_filename = join(args.outdir, f"{args.out}_logo.png")
        plot_logo(pfm, full_motif, logo_filename, use_bits=not args.prob, label=args.out)

    if args.savePWM:
        pwm_filename = join(args.outdir, f"{args.out}.meme")
        save_meme_pwm(ppm, full_motif, pwm_filename)

if __name__ == "__main__":
    main()
