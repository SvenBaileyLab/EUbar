# EUbar

EUbar predicts the effect of noncoding single nucleotide variants on transcription factor binding affinity using accessible chromatin regions as sequence probes and matched ChIP-seq signal as a measure of binding intensity. It also supports whole-region scanning for mutational effect landscapes and affinity-based motif discovery.

![EUbar method summary](docs/eubar_summary.png)

---

## Installation

```bash
git clone https://github.com/SvenBaileyLab/EUbar
cd EUbar
pip install .
```

Confirm the install:

```bash
eubar --help
```

---

## Workflow overview

Every EUbar analysis follows three steps:

1. **Build an array file** — index all k-mers in your accessible regions
2. **Compute probe intensities** — summarise ChIP-seq signal across those regions
3. **Run analysis** — predict SNV effects, scan a region, or discover motifs

---

## Commands

| Command | Description |
|---------|-------------|
| `array` | Build a k-mer index from a BED file and genome FASTA |
| `intensities` | Compute GC-corrected probe intensities from a BigWig or BedGraph signal track |
| `snv` | Predict the effect of one or more SNVs on TF binding |
| `scan` | Scan a genomic region for predicted binding effects at every position |
| `motifs` | Derive an affinity-based TF binding motif from probe intensities |

Command-specific help is available with `eubar <command> --help`.

---

## Quick start

```bash
# 1. Build array
eubar array   --bed regions.bed   --genome hg38.fa   --kmer-size 8   --output regions_8mer.txt

# 2. Compute intensities
eubar intensities   --bed regions.bed   --signal tf_chipseq.bw   --genome-fasta hg38.fa   --output probe_intensities.tsv

# 3. Predict SNV effect
eubar snv   --intensities probe_intensities.tsv   --array regions_8mer.txt   --genome hg38.fa   --snv-list "chr5:1295113:C>T"   --best-pval
```

---

## Tutorials

Step-by-step tutorials using real ENCODE data (MCF7 DNase-seq + GABPA ChIP-seq):

- [Data preparation](docs/01_data_prep.md) — download data, build array, compute intensities
- [SNV analysis](docs/02_snv.md) — predict allelic effects on TF binding
- [Scan analysis](docs/03_scan.md) — scan a genomic region for binding effects
- [Motif discovery](docs/04_motifs.md) — derive an affinity-based binding motif

---

## Citation

If you use EUbar in your research, please cite:

> [manuscript citation — to be added on publication]

---

## License

GNU General Public License v3.0.