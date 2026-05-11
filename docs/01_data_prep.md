# Data Preparation

This tutorial walks through downloading example data from ENCODE and building the two input files required for all EUbar analyses: the **array file** and the **intensity file**.

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

The array file indexes every k-mer in the accessible genome; it records which accessible regions contain each k-mer and at what position. This only needs to be built once per cell type and k-mer size.

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

All 4^8 = 65,536 possible 8-mers should be present. If any are missing, consider using a larger or more permissive peak set.

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

This outputs GC-corrected log-space residuals (`resid_log` format), which is the default and the correct format for all EUbar analyses.

The output is a two-column tab-delimited file:

``` tsv
chr1:100-200     0.342
chr1:500-700    -0.118
chr2:300-450     1.204
...
```

---

## What you have now

| File | Description |
|------|-------------|
| `results/MCF7_DNase_8mer.txt` | Array file — k-mer index for the MCF7 accessible genome |
| `results/GABPA_MCF7_intensities.tsv` | Intensity file — GC-corrected GABPA probe intensities |

These two files are the inputs for all three analysis modes: SNV prediction, genomic scanning, and motif discovery. Continue to the next tutorial pages to run each analysis.

---

**Next:** [SNV analysis →](02_snv.md)
