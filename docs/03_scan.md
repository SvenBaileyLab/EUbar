# Scan Analysis

This tutorial demonstrates how to use EUbar to scan a genomic region and evaluate the predicted effect of every possible single-nucleotide change on TF binding. We scan the TERT promoter region (chr5:1,295,105-1,295,140, hg38) for GABPA binding effects — the same analysis used in the EUbar manuscript to localise the two recurrent cancer driver mutations.

This tutorial assumes you have already completed [Data Preparation](01_data_prep.md) and have the following files:

``` text
results/MCF7_DNase_8mer.txt
results/GABPA_MCF7_intensities.tsv
```

---

## 1. Run the scan

Use `scan` with a genomic region in `chr:start-end` format. Results are written to stdout — redirect to a file for downstream use.

```bash
eubar scan   --intensities results/GABPA_MCF7_intensities.tsv   --array results/MCF7_DNase_8mer.txt   --genome data/hg38.fa   --region "chr5:1295105-1295140"   --mode ols   --save-figure results/tert_scan.png   > results/tert_scan.tsv
```

The `--save-figure` flag saves a plot of the scan results. `--mode ols` uses ordinary least squares regression, which is recommended for GC-corrected residual intensities.

The output is a TSV written to stdout with one row per overlapping k-mer window per allele across the scanned interval. Each position in the region is evaluated in multiple overlapping windows, capturing how each possible change interacts with its local sequence context.

---

## 2. Get a best-window summary per position

For a cleaner view of the scan — one result per genomic position — use `--best-pval`. This selects the window with the strongest predicted effect at each position and produces a figure that is easier to interpret.

```bash
eubar scan   --intensities results/GABPA_MCF7_intensities.tsv   --array results/MCF7_DNase_8mer.txt   --genome data/hg38.fa   --region "chr5:1295105-1295140"   --mode ols   --best-pval   --save-figure results/tert_scan_best.png   > results/tert_scan_best.tsv
```

The figures produced by these two commands are shown below.

All windows:

![TERT scan all windows](tert_promoter.png)

Best window per position:

![TERT scan best pval](tert_promoter-best_pval.png)

---

## 3. Understanding the output

The scan output is written to stdout and can be redirected to a file. Each row represents one allele at one k-mer window position.

| Column | Description |
|--------|-------------|
| `wildcard_kmer` | The 8-mer with `.` marking the evaluated position |
| `filled_kmer` | The 8-mer with the specific allele substituted at the variant position |
| `window_index` | Start position of the k-mer window within the scanned sequence |
| `snp_index` | Position of the variant within the k-mer window (0-7) |
| `type` | `AFF` = allelic effect regression |
| `allele` | The allele being evaluated at this position |
| `coef` | Regression coefficient — predicted change in GC-corrected log-space ChIP-seq intensity relative to the reference allele |
| `pval` | P-value for the predicted allelic effect |
| `absolute_pos` | Absolute position within the scanned region (0-based) |
| `method` | Regression method used; `ols|best_pval` when `--best-pval` is set |

### Interpreting the results

Each position in the scanned region is tested for all three possible single-nucleotide changes. Positions with large positive coefficients and small p-values are predicted to gain TF binding if mutated to that allele; large negative coefficients indicate predicted loss.

In the TERT promoter scan, the two hotspot positions (−124C>T and −146C>T) stand out as the strongest predicted gain-of-binding events across the entire interval, with no other position reaching comparable effect sizes. This is consistent with their established role in creating de novo GABPA binding sites.

---

**Previous:** [SNV analysis](02_snv.md)
**Next:** [Motif discovery](04_motifs.md)
