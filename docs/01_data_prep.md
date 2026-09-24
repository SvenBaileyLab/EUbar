# Data Preparation

This tutorial walks through downloading example data from ENCODE and building the **array file** and **intensity file** used for contiguous analyses.
Masked SNV and motif analyses use the intensity file and reference genome directly.

We use MCF7 DNase-seq and GABPA ChIP-seq data from ENCODE, the same datasets used in the EUbar manuscript, as a worked example. The same workflow applies to any cell type or TF with matched DNase-seq/ATAC-seq and ChIP-seq data available.

---

## 1. Download example data

Create a working directory and download the required files from ENCODE.

```bash
mkdir eubar_tutorial && cd eubar_tutorial
mkdir data results

# MCF7 DNase-seq peaks (BED file)
wget -O data/MCF7_DNase.bed.gz \
  https://www.encodeproject.org/files/ENCFF835KCG/@@download/ENCFF835KCG.bed.gz

# GABPA ChIP-seq signal (BigWig)
wget -O data/GABPA_MCF7.bw \
  https://www.encodeproject.org/files/ENCFF676BAJ/@@download/ENCFF676BAJ.bigWig

# Reference genome (hg38) 
wget -O data/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip data/hg38.fa.gz
```

Decompress the DNase BED file:

```bash
gunzip data/MCF7_DNase.bed.gz
```

You should now have:

``` text
eubar_tutorial/
├── data/
│   ├── MCF7_DNase.bed
│   ├── GABPA_MCF7.bw
│   └── hg38.fa
└── results/
```

---

## 2. Build the array file

The array file indexes every k-mer in the accessible genome; it records which accessible regions contain each k-mer and at what position. Rebuild it when the probe BED, reference genome or k-mer size changes.
Use the same probe regions for the array and intensity files. BED coordinates
are zero-based, half-open; SNV and scan command coordinates are one-based.

```bash
eubar array \
  --bed data/MCF7_DNase.bed \
  --genome data/hg38.fa \
  --kmer-size 8 \
  --output results/MCF7_DNase_8mer.txt
```

This will take a few minutes depending on the number of accessible regions. When complete you should see:

``` bash
[Done] K-mer index written to results/MCF7_DNase_8mer.txt
All kmers represented! N=65536
```

There are 4^8 = 65,536 possible 8-mers, but complete representation is not
guaranteed. Missing or sparsely represented contexts limit the analyses they
can support; inspect coverage before changing the peak set.

---

## 3. Compute probe intensities

The intensity file summarizes the GABPA ChIP-seq signal across each accessible region and applies GC content correction. This produces the probe-level binding intensity used in all downstream analyses.

```bash
eubar intensities \
  --bed data/MCF7_DNase.bed \
  --signal data/GABPA_MCF7.bw \
  --genome-fasta data/hg38.fa \
  --output results/GABPA_MCF7_intensities.tsv
```

By default, EUbar extracts the **maximum signal over each BED interval**,
clips negative extracted values to zero before `log1p`, and residualizes the
transformed signal against GC content. The output format is `resid_log`: signed
residuals are expected. Use the default OLS mode for SNV/ordinary scan analyses
and this residualized input for calibration. The advanced pooled scan has a
separate response transformation; see [Scan analysis](03_scan.md).

The output is a two-column tab-delimited file:

``` tsv
chr1:100-200     0.342
chr1:500-700    -0.118
chr2:300-450     1.204
...
```

---

## Signal options and input checks

Install the external `bedtools` executable as well as the Python package for
signal extraction. Keep the BED, signal track and FASTA on the same assembly,
with matching chromosome names. An intensity row is a region ID
(`chr:start-end`, using BED coordinates) and a numeric value, with no header.

| Option | Default | Meaning |
| --- | --- | --- |
| `--summary` | `max` | `max` or `mean` over the interval; `center_max` or `center_mean` over a centered window |
| `--window-bp` | 100 | Window width for `center_*` summaries; ignored for full-interval summaries |
| `--no-residualize` | Off | Output extracted signal without GC residualization; FASTA is then optional |
| `--genome-size-file` | None | Chromosome sizes for chromosome ordering in `bedtools sort` |
| `--keep-temp` | Off | Retain temporary extraction files |

Advanced residualization controls remain available: `--resid-use-length` adds
length, `--resid-open` defaults to `off`, and `--resid-output` selects
`resid_log` (default), `log_corrected`, or `intensity_like`. These change the
meaning or scale of the input to later fits. Keep them fixed when comparing
analyses; the examples here use the defaults.

## What you have now

| File | Description |
|------|-------------|
| `results/MCF7_DNase_8mer.txt` | Array file — k-mer index for the MCF7 accessible genome |
| `results/GABPA_MCF7_intensities.tsv` | Intensity file — GC-corrected GABPA probe intensities |

These files support contiguous SNV prediction, scanning and motif discovery.
Masked SNV and motif runs instead read sequences using the intensity-file
region IDs and genome FASTA; they do not require a precomputed masked array.

---

**Next:** [SNV analysis →](02_snv.md)
