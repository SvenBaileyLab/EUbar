# SNV Analysis

This tutorial demonstrates how to use EUbar to predict the effect of a single nucleotide variant on TF binding affinity. We use the recurrent TERT promoter mutation −124C>T (chr5:1,295,113 C>T, hg38) as an example — a well-characterised variant known to create a de novo GABPA binding site.

This tutorial assumes you have already completed [Data Preparation](01_data_prep.md) and have the following files:

``` bash
results/MCF7_DNase_8mer.txt
results/GABPA_MCF7_intensities.tsv
```

---

## 1. Evaluate a single SNV

Use `snv` to predict the effect of −124C>T on GABPA binding.

```bash
eubar snv   --intensities results/GABPA_MCF7_intensities.tsv   --array results/MCF7_DNase_8mer.txt   --genome data/hg38.fa   --snv-list "chr5:1295113:C>T"
```

This returns one row per overlapping 8-mer window for each allele and regression type (AFF and RAND). For an 8-mer analysis, a single SNV generates 8 overlapping windows, each placing the variant at a different position within the k-mer.

Expected output:

```  text
snv                     type    allele  motif_pos  wildcard_kmer   effect          pval
chr5:1295113:C>T        AFF     T       0          CAGCCCC.        -0.03443606     1.318e-01
chr5:1295113:C>T        AFF     T       1          AGCCCC.T         0.03919671     1.842e-01
chr5:1295113:C>T        AFF     T       2          GCCCC.TC         0.1554104      1.969e-08
chr5:1295113:C>T        AFF     T       3          CCCC.TCC         0.0901687      1.353e-05
chr5:1295113:C>T        AFF     T       4          CCC.TCCG         0.4309754      8.004e-22
chr5:1295113:C>T        AFF     T       5          CC.TCCGG         0.5909847      2.575e-38
chr5:1295113:C>T        AFF     T       6          C.TCCGGG         1.006108       8.756e-136
chr5:1295113:C>T        AFF     T       7          .TCCGGGC         0.5811603      1.609e-35
chr5:1295113:C>T        RAND    C       0          NA               0.01321019     6.427e-01
chr5:1295113:C>T        RAND    C       1          NA              -0.04817633     1.470e-01
...
chr5:1295113:C>T        RAND    T       6          NA               1.164914       6.363e-139
chr5:1295113:C>T        RAND    T       7          NA               0.6924163      2.447e-43
```

### Output columns

| Column | Description |
|--------|-------------|
| `snv` | Variant in `chr:pos:ref>alt` format |
| `type` | `AFF` = allelic effect regression; `RAND` = background enrichment regression |
| `allele` | Allele being evaluated |
| `motif_pos` | Position of the k-mer window start relative to the SNV context (0 = leftmost window) |
| `wildcard_kmer` | The 8-mer sequence with `.` marking the variant position; `NA` for RAND rows |
| `effect` | Regression coefficient — difference in GC-corrected log-space ChIP-seq intensity between the queried allele and the reference |
| `pval` | P-value for the allelic effect |

### Interpreting the AFF rows

Each AFF row corresponds to one overlapping k-mer window. The effect and significance vary across windows because each window places the variant in a different sequence context, capturing how the SNV interacts with its local sequence environment. In this example, window 6 (wildcard kmer `C.TCCGGG`) shows the strongest predicted gain of GABPA binding (effect = 1.01, p = 8.8e-136), reflecting the creation of an ETS-family binding motif by the alternate allele.

A positive effect means the alternate allele is associated with higher TF occupancy; predicted gain of binding. A negative effect means predicted loss of binding.

---

## 2. Get a single best-window summary

For a concise per-SNV summary, use `--best-pval`. This selects the window with the strongest predicted effect supported by RAND background evidence and returns three rows per SNV: the AFF result and both RAND alleles at that window position.

```bash
eubar snv   --intensities results/GABPA_MCF7_intensities.tsv   --array results/MCF7_DNase_8mer.txt   --genome data/hg38.fa   --snv-list "chr5:1295113:C>T"   --best-pval
```

Expected output:

``` tsv
snv                     type    allele  effect      pval
chr5:1295113:C>T        AFF     T       1.006108    8.756e-136
chr5:1295113:C>T        RAND    C       0.190231    3.249e-05
chr5:1295113:C>T        RAND    T       1.164914    6.363e-139
```

This is the recommended output format for large-scale analyses and downstream filtering.

---

## 3. Evaluate a list of SNVs

For multiple variants, provide a file with one variant per line in `chr:pos:ref>alt` format.

``` bash
eubar snv   --intensities results/GABPA_MCF7_intensities.tsv   --array results/MCF7_DNase_8mer.txt   --genome data/hg38.fa   --snv-list-file data/snvs.txt   --best-pval
```

Where `data/snvs.txt` contains one variant per line:

``` text
chr5:1295113:C>T
chr5:1295135:C>T
```

---

## 4. Understanding RAND

The RAND rows are a critical part of interpreting EUbar results. For each SNV, EUbar samples a background set of probes from accessible regions that do not match the queried k-mer sequence. It then fits the same regression model to these background probes, asking whether either allele sequence is enriched among more strongly bound regions across the array as a whole.

**Why RAND matters:** The AFF regression tells you whether the alternate allele probes have higher intensity than the reference allele probes in the local k-mer context. But this difference is only meaningful if the variant context is actually engaged by the TF. RAND provides this confirmation; a significant positive RAND coefficient (p < 0.05, effect > 0) for either allele indicates that sequences containing that k-mer are preferentially enriched in highly occupied regions across the accessible genome, confirming that the TF actively binds this sequence context.

**Interpreting RAND in the example above:**

- RAND C (effect = 0.19, p = 3.2e-05): the reference allele context is enriched among more strongly bound probes; the locus is already engaged by GABPA.
- RAND T (effect = 1.16, p = 6.4e-139): the alternate allele context shows even stronger enrichment; consistent with the variant creating a higher-affinity binding sequence.

Together, AFF and RAND support the interpretation that −124C>T creates a stronger GABPA binding site at an already-accessible locus, consistent with its established role as a driver mutation.

**Filtering on RAND:** When processing large numbers of SNVs, a standard filter is to retain predictions where at least one RAND allele has p < 0.05 and a positive effect, confirming active TF engagement at the variant site.

---

**Previous:** [Data Preparation](01_data_prep.md)
**Next:** [Scan analysis](03_scan.md)
