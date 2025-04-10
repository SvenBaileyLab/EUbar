# Motif Discovery from k-mer Signal: A Seed-and-Extend Pipeline

This Python pipeline performs **motif discovery** using a signal-guided, k-mer-based seed-and-extend approach. It is designed for use with transcription factor (TF) ChIP-seq signal intensities over **open chromatin regions** (ATAC-seq or DNase-seq), using a fully enumerated k-mer array as input. The result is a biologically grounded motif built from high-signal k-mers and visualized as a sequence logo.

---

## Installation

Make sure you have Python 3.7+ and the following packages installed:

```bash
pip install numpy pandas matplotlib logomaker
```

Clone this repository:

```bash
git clone https://github.com/yourusername/motif-discovery.git
cd motif-discovery
```

---

## Input Files

You need two files:

### 1. `--intensities`  
A TSV file of probe regions with TF ChIP-seq signal intensities:
```
chr1:827460-827554   12.3
chr1:15323200-15323380   8.7
...
```

### 2. `--kmerPositions`  
A file mapping each k-mer to the probe regions it occurs in. One line per k-mer:
```
AAAACGTC   chr1:827460-827554;1;68,chr1:15323200-15323380;1;165,...
ACGTACGT   chr12:567832-567950;1;43,...
...
```
Only k-mers that occur **once** per region are used.

---

## Usage

```bash
python discover_motifs.py \
  --intensities path/to/intensities.bed \
  --kmerPositions path/to/kmer_positions.txt \
  --out GABPA \
  --outdir results/ \
  --logo \
  --savePWM \
  --keep_stats
```

---

## Arguments

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--intensities`  | Path to ChIP-seq intensity file for probes (required)                       |
| `--kmerPositions`| Path to k-mer to region mapping file (required)                             |
| `--seed`         | Optional starting k-mer (default = auto-selected from highest signal)       |
| `--thres`        | Signal threshold for motif walking (default = auto from `--percentile`)     |
| `--percentile`   | If no `--thres` is given, threshold is set to this percentile (default: 80) |
| `--kmer_size`    | K-mer length (default: 8)                                                   |
| `--mismatches`   | Max mismatches allowed for logo kmers (default: 1)                          |
| `--logo`         | Generate a sequence logo (PNG)                                              |
| `--prob`         | Use probabilities instead of bits in logo                                   |
| `--savePWM`      | Output a MEME-format motif PWM                                              |
| `--keep_stats`   | Save TSV file of all k-mer signal statistics                                |
| `--out`          | Output filename prefix (default: `motif`)                                   |
| `--outdir`       | Output directory (default: current directory)                               |

---

## How It Works

1. **Compute k-mer signal**: Each k-mer is assigned a mean intensity from the ChIP-seq signal of regions where it appears.

2. **Seed selection**: The k-mer with the highest mean intensity is chosen as the **seed** (or provided via `--seed`).

3. **Motif walking**:
   - The motif is extended **left and right** by overlapping k-mers
   - Only k-mers with signal ≥ threshold are used

4. **K-mer collection**:
   - All k-mers from the full dataset within a certain **Hamming distance** (e.g. ≤1) of any part of the motif are collected

5. **Motif building**:
   - K-mers are aligned to the motif
   - A **Position Frequency Matrix (PFM)** is built from base counts
   - The PFM is normalized into a **PPM** or **PWM**

6. **Visualization & output**:
   - Sequence logo (via `logomaker`)
   - MEME-format motif (optional)
   - Raw motif stats and matrices

---

## Outputs

| File                              | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `GABPA_logo.png`                  | Motif logo plot (either bits or probabilities)                              |
| `GABPA.meme`                      | MEME-format motif for downstream analysis                                   |
| `GABPA_kmer_stats.tsv`            | Full table of k-mer and reverse complement statistics (if `--keep_stats`)   |

---

## Example Output

```
Motif: TGGGGGTTGTAGTCGTTGGTA
Saved: results/GABPA_logo.png
Saved: results/GABPA.meme
Saved: results/GABPA_kmer_stats.tsv
```

---

## Example Run

```bash
python discover_motifs.py \
  --intensities data/GABPA_MCF7_chip.txt \
  --kmerPositions data/GABPA_Array_ATAC.txt \
  --out GABPA \
  --outdir results/ \
  --logo --savePWM --keep_stats
```
--- 

## Citing

If this pipeline contributes to your research, please consider citing:

```
FILL
```

---