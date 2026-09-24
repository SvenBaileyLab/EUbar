# EUbar

EUbar predicts the effect of noncoding single nucleotide variants on transcription factor binding affinity using accessible chromatin regions as sequence probes and matched ChIP-seq signal as a measure of binding intensity. It also supports whole-region scanning for mutational effect landscapes and affinity-based motif discovery.

![EUbar method summary](docs/eubar_summary.png)

---

## Installation

```bash
pip install eubar

# or

git clone https://github.com/SvenBaileyLab/EUbar
cd EUbar
pip install .
```

Signal extraction also requires the external `bedtools` executable. For YAML
jobs, install the optional dependency with `pip install 'eubar[yaml]'`.

Confirm the install:

```bash
eubar --help
```

---

## Workflow overview

For contiguous sequence analyses:

1. **Build an array file** — index all k-mers in your accessible regions
2. **Compute probe intensities** — summarise ChIP-seq signal across those regions
3. **Run analysis** — predict SNV effects, scan a region, or discover motifs

Masked SNV and motif analyses use probe intensities and genome sequence directly,
without an array file. [Calibration](docs/06_masks_and_calibration.md) compares
probe caps or candidate masks before you choose settings for an analysis.

---

## Commands

| Command | Description |
|---------|-------------|
| `array` | Build a k-mer index from a BED file and genome FASTA |
| `intensities` | Compute GC-corrected probe intensities from a BigWig or BedGraph signal track |
| `snv` | Predict the effect of one or more SNVs on TF binding |
| `scan` | Scan a genomic region for predicted binding effects at every position |
| `motifs` | Derive an affinity-based TF binding motif from probe intensities |
| `calibrate max-probes` | Assess coefficient stability across probe caps |
| `calibrate mask` | Compare mask representation and coefficient stability |
| `run` | Execute a YAML task or pipeline |

Command-specific help is available with `eubar <command> --help`.

---

## Quick start

```bash
# 1. Build array
eubar array \
  --bed regions.bed --genome hg38.fa \
  --kmer-size 8 --output regions_8mer.txt

# 2. Compute intensities
eubar intensities \
  --bed regions.bed --signal tf_chipseq.bw \
  --genome-fasta hg38.fa --output probe_intensities.tsv

# 3. Predict SNV effect
eubar snv \
  --intensities probe_intensities.tsv --array regions_8mer.txt \
  --genome hg38.fa --snv-list "chr5:1295113:C>T" \
  --best-pval --holm --diagnostics
```

---

## Tutorials

Step-by-step tutorials using real ENCODE data (MCF7 DNase-seq + GABPA ChIP-seq):

- [Data preparation](docs/01_data_prep.md) — download data, build array, compute intensities
- [SNV analysis](docs/02_snv.md) — predict allelic effects on TF binding
- [Scan analysis](docs/03_scan.md) — scan a genomic region for binding effects
- [Motif discovery](docs/04_motifs.md) — contiguous and masked motifs
- [Masks and calibration](docs/06_masks_and_calibration.md) — spacing, probe limits and recommendation tables
- [YAML jobs](docs/07_yaml.md) — reusable tasks and pipelines

---

## Citation

If you use EUbar in your research, please cite:

> [manuscript citation — to be added on publication]

---

## License

GNU General Public License v3.0.

## Python API and tests

See [the Python task API](docs/05_python_api.md) for calling tasks directly and
[the test suite](eubar/tests/README.md) for regression checks.
