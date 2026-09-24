# Motif Discovery

This tutorial demonstrates how to use EUbar to derive an affinity-based TF binding motif directly from ChIP-seq probe intensities. Unlike frequency-based motifs that summarise aligned sequences within ChIP-seq peaks, EUbar motifs are built by learning which k-mer sequences are preferentially associated with higher binding intensity across the entire accessible genome.

We use the same MCF7 GABPA data from [Data Preparation](01_data_prep.md).

---

## 1. Run motif discovery

```bash
eubar motifs \
  --intensities results/GABPA_MCF7_intensities.tsv \
  --array results/MCF7_DNase_8mer.txt \
  --kmer-size 8 \
  --outdir results/GABPA_motif --prefix GABPA
```

EUbar selects an enriched seed pattern, estimates base preferences at each
position, and attempts to extend the motif on both sides. By default, seed
search includes patterns with up to three wildcard positions (`--max-gaps 3`).
Use `--max-gaps 0` for exact k-mer seeds only. All output files are written to the specified directory with the given prefix.

---

## 2. Output files

| File | Description |
|------|-------------|
| `GABPA.meme` | Motif in MEME format for use with downstream tools |
| `GABPA.ppm.tsv` | Position probability matrix as a tab-delimited table |
| `GABPA.reduced.tsv` | Base-specific enrichment scores per motif position |
| `GABPA.reduced_full.tsv` | Full enrichment table including extended flanking positions |
| `GABPA.top_escores.tsv` | Top-ranked k-mers by enrichment score used during seed selection |
| `GABPA.logo_bits.png` | Sequence logo scaled by information content (bits) |
| `GABPA.logo_prob.png` | Sequence logo scaled by base probability |
| `GABPA.logo_bits_rc.png` | Reverse complement logo (bits) |
| `GABPA.logo_prob_rc.png` | Reverse complement logo (probability) |
| `GABPA.reduced_enrichment.png` | Bar plot of base-specific enrichment scores per motif position |
| `GABPA.reduced_enrichment_rc.png` | Reverse complement enrichment bar plot |
| `GABPA.seed_enrichment_curve.png` | Seed-hit enrichment along the ranked probe list |
| `GABPA.seed_roc.png` | ROC-style enrichment plot for the seed k-mer |
| `GABPA.seed_escore_hist.png` | Distribution of E-scores across all candidate k-mers |

---

## 3. Interpreting the output

### Seed selection

EUbar scores represented seed candidates that meet the support filter by
their enrichment (E = AUC - 0.5). With default settings these include gapped
patterns as well as exact k-mers; the search is not restricted to 65,536 exact
8-mers. The distribution of E-scores is shown below. Most k-mers cluster near 0, with a small number of highly enriched sequences standing out.

![E-score distribution](GABPA/GABPA.seed_escore_hist.png)

The highest-scoring eligible pattern becomes the seed. The saved GABPA
example shown here used the seed **ACTTCCGG** with E = 0.432, consistent with the known ETS-family core motif. The ROC plot below shows how well this seed separates highly bound probes from background (AUC = 0.932, E = 0.432):

![Seed ROC](GABPA/GABPA.seed_roc.png)

The E-score is defined as AUC - 0.5, where AUC is computed by ranking all probes from highest to lowest intensity and measuring how well probes containing the seed k-mer are concentrated at the top of that ranking. An E-score of 0.432 indicates strong preferential enrichment of ACTTCCGG-containing probes among the most highly bound accessible regions.

### Base-specific enrichment scores

After seed selection, EUbar evaluates all four possible bases at each position of the motif, comparing probes containing each base against probes containing the other three. The enrichment score at each position reflects the relative preference for each base among highly bound probes.

![Enrichment scores per position](GABPA/GABPA.reduced_enrichment.png)

Positions with a strongly dominant bar (one base clearly above the others) are highly informative positions of the motif. Positions where bars are small and mixed are low-information flanking positions.

### Sequence logos

The final motif is reported as a position probability matrix (PPM) and visualised as sequence logos. Two representations are available:

**Information content (bits) — highlights the most specific positions:**

![GABPA motif logo bits](GABPA/GABPA.logo_bits.png)

**Base probability — shows all positions including low-specificity flanks:**

![GABPA motif logo probability](GABPA/GABPA.logo_prob.png)

These saved GABPA logos illustrate a strongly specified core and weaker
flanks. The precise seed and extension depend on the input files, support
filters and search options; the images are examples rather than a guarantee
of identical output from every run.

### MEME format output

The motif is also exported in MEME format for direct use with FIMO, TOMTOM, or other tools from the MEME suite:

```
MEME version 4

ALPHABET= ACGT

strands: + -

MOTIF CACTTCCGGT
letter-probability matrix: alength= 4 w= 10 nsites= 10 E= 0
0.156 0.501 0.248 0.095
1.000 0.000 0.000 0.000
0.005 0.948 0.019 0.029
...
```

### The reduced_full table

`GABPA.reduced_full.tsv` is the underlying data behind the motif — it contains the base-specific enrichment scores at every position of the final motif, including the extended flanking positions. Each row represents one base at one position.

| Column | Description |
|--------|-------------|
| `step` | Extension step — 0 is the position closest to the seed, increasing outward |
| `side` | `core` = seed positions; `left` / `right` = extended flanking positions |
| `gaps_used` | Number of wildcard positions used when matching this k-mer |
| `base` | The base being evaluated at this position |
| `variant` | The full k-mer sequence with this base substituted at the tested position |
| `E_reduced` | Enrichment score (AUC - 0.5) comparing probes containing this base against probes containing the other three bases at this position |
| `p` | P-value for the enrichment (Mann-Whitney U approximation) |
| `F` | Number of foreground probes — probes containing this base at this position |
| `B` | Number of background probes — probes containing the other three bases |
| `IC` | Information content contribution of this position (bits); populated for flanking positions |
| `pos` | Absolute position in the final motif (0-based from seed start; negative = left flank) |

The `E_reduced` values at core positions are converted to base probabilities
via softmax, with `--beta 10` by default. These are score-derived motif weights,
not observed base frequencies or calibrated binding probabilities. Positions where one base has a strongly positive `E_reduced` and the others are negative are highly specific positions of the motif. Flanking positions with low `E_reduced` values and low IC contribute little information and represent the degenerate edges of the motif.

---

## 4. Controlling the search

| Option | Default | Meaning |
| --- | --- | --- |
| `--kmer-size` | 8 | Seed/window length for the contiguous array path |
| `--max-gaps` | 3 | Maximum wildcard positions in seed search and extension retries |
| `--seed ACTTCCGG` | Automatic | Force a sequence seed; this is not a numeric random seed |
| `--top-n-report` | 50 | Number of ranked seed candidates written to the table |
| `--pretty-logo` | Off | Alternate logo styling |

The following retained tuning options are hidden from normal help:

| Option | Default | Meaning |
| --- | --- | --- |
| `--min-F` | 20 | Minimum seed foreground support |
| `--min-per-base` | 20 | Per-base support threshold for reduced comparisons and extension |
| `--min-support` | 1 | Minimum support for reduced base comparisons |
| `--beta` | 10 | Softmax scaling of reduced E-scores |
| `--pseudocount` | 0 | Smoothing added when constructing probabilities |
| `--extend-left`, `--extend-right` | -1 | Automatic extension; 0 disables that side, positive values cap steps |
| `--auto-max-steps` | 20 | Maximum steps per side in automatic extension |
| `--ic-stop-threshold` | 0.2 | Low-information threshold for extension stopping |
| `--ic-stop-consecutive` | 2 | Consecutive low-information steps before stopping |
| `--no-combine-revcomp` | Off | Disable reverse-complement pooling in the ordinary array path |

Extension can stop early when support is insufficient. To inspect the seed
without flanking extension, add `--extend-left 0 --extend-right 0`.

## 5. Motifs with a fixed mask

A mask describes separated matched blocks, such as five matched bases, four
spacers and five matched bases. See [Masks and calibration](06_masks_and_calibration.md)
for how to compare candidate patterns.

```bash
eubar motifs \
  --intensities results/GABPA_MCF7_intensities.tsv \
  --genome data/hg38.fa \
  --mask 11111000011111 \
  --outdir results/GABPA_masked_motif --prefix GABPA_masked
```

This is a syntax example, not a mask recommendation for GABPA. The intensity
file supplies probe coordinates, and FASTA supplies sequence. `--array` is not
used. Candidates are collected from the highest-intensity probes, then scored
against the full probe collection. `--mask-candidate-probes` defaults to 5000;
`--mask-max-candidates` defaults to 10000. These bound candidate generation,
not the number of probes used for final scoring.

The motif has the fixed mask span. Base preferences are estimated only at
informative positions. Spacer columns are uniform (0.25 for each base), have
zero information content, and are marked `masked_spacer` in the reduced tables.
Masked motif patterns display spacers as `.`, whereas masked SNV output uses
`x` for spacers and reserves `.` for the tested base.

Masked discovery does not run the ordinary flanking-extension procedure.
`--max-gaps`, `--extend-left`, `--extend-right` and the ordinary extension
stopping options do not determine its span. Both strands are scanned in this
path; `--no-combine-revcomp` does not disable that behavior.

To force a masked seed, supply either all informative bases or a full-span
pattern, for example `--seed TTGGCGCCAA` or `--seed 'TTGGC....GCCAA'` for the
14-base mask above. Informative positions must contain A, C, G or T. The example
shows the accepted syntax, not a seed selected from the GABPA data.

Motif discovery has no `--max-probes` option. Its candidate limits and support
filters serve different purposes from the SNV regression cap.

**Previous:** [Scan analysis](03_scan.md) · **Related:** [Masks and calibration](06_masks_and_calibration.md)
